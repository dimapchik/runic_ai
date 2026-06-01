import logging
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, Trainer
from peft import LoraConfig, get_peft_model
import os
import re
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import TrainerCallback
import sacrebleu
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
import nltk

logger = logging.getLogger(__name__)

# Download required resources for METEOR
try:
    nltk.data.find('tokenizers/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# >>> ClearML
from clearml import Task
from transformers.integrations import ClearMLCallback
# <<< ClearML

# ======================
# CONFIG
# ======================

# ClearML task init
# task = Task.init(
#     project_name="d-prokopev/runic_ai",
#     task_name="qwen-3.5-0.8B-lora-dating-regression",
#     task_type=Task.TaskTypes.training,
#     tags=["qwen-3.5-0.8B", "lora", "regression", "runic", "dating"],
# )

task = Task.init(
    project_name="d-prokopev/runic_ai",
    task_name="qwen-3.5-0.8B-lora-dating-regression",
    task_type=Task.TaskTypes.training,
    tags=["qwen-3.5-0.8B", "lora", "regression", "runic", "dating"],
)

# Пути к данным
TRAIN_FILE = "/train_model/dating_dataset/train.csv"
VAL_FILE = "/train_model/dating_dataset/val.csv"
TEST_FILE = "/train_model/dating_dataset/test.csv"

# Путь для сохранения модели
OUTPUT_DIR = "/train_model/gemma_dating_trained"

# Путь к базовой модели
MODEL_NAME = "/models/"

# Параметры данных
MAX_SEQ_LENGTH = 512

# Custom callback для логирования метрик
class MetricsLoggerCallback(TrainerCallback):
    """Callback для логирования train/eval метрик в JSON файл"""
    
    def __init__(self, log_path="dating_training_metrics.json"):
        self.log_path = log_path
        self.metrics_history = {
            "train_loss": [],
            "eval_loss": [],
            "eval_mae": [],
            "eval_rmse": [],
            "learning_rate": [],
            "epoch": []
        }
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Логируем train метрики
            if "loss" in logs:
                self.metrics_history["train_loss"].append(logs["loss"])
            if "learning_rate" in logs:
                self.metrics_history["learning_rate"].append(logs["learning_rate"])
            if "epoch" in logs:
                self.metrics_history["epoch"].append(logs["epoch"])
            
            # Печатаем в красивом формате
            print("\n" + "="*60)
            print(f"📊 METRICS (step {state.global_step})")
            print("="*60)
            print(f"  Epoch:           {logs.get('epoch', 'N/A')}")
            train_loss = logs.get('loss')
            eval_loss = logs.get('eval_loss')
            lr = logs.get('learning_rate')
            print(f"  Loss (train):    {train_loss:.4f}" if isinstance(train_loss, (int, float)) else f"  Loss (train):    {train_loss if train_loss is not None else 'N/A'}")
            print(f"  Loss (eval):     {eval_loss:.4f}" if isinstance(eval_loss, (int, float)) else f"  Loss (eval):     {eval_loss if eval_loss is not None else 'N/A'}")
            print(f"  Learning rate:   {lr:.6f}" if isinstance(lr, (int, float)) else f"  Learning rate:   {lr if lr is not None else 'N/A'}")
            print("="*60)
            
            # Сохраняем в JSON
            with open(self.log_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            # Логируем eval метрики
            if "eval_loss" in metrics:
                self.metrics_history["eval_loss"].append(metrics["eval_loss"])
            if "eval_mae" in metrics:
                self.metrics_history["eval_mae"].append(metrics["eval_mae"])
            if "eval_rmse" in metrics:
                self.metrics_history["eval_rmse"].append(metrics["eval_rmse"])
            
            print("\n" + "="*60)
            print(f"📈 EVALUATION RESULTS (step {state.global_step})")
            print("="*60)
            # Основные метрики — в начале, отдельно
            print(f"  eval_loss          {metrics.get('eval_loss', 'N/A'):.4f}" if isinstance(metrics.get('eval_loss'), (int, float)) else f"  eval_loss          {metrics.get('eval_loss', 'N/A')}")
            print(f"  eval_mae           {metrics.get('eval_mae', 'N/A'):.2f}" if isinstance(metrics.get('eval_mae'), (int, float)) else f"  eval_mae           {metrics.get('eval_mae', 'N/A')}")
            print(f"  eval_rmse          {metrics.get('eval_rmse', 'N/A'):.2f}" if isinstance(metrics.get('eval_rmse'), (int, float)) else f"  eval_rmse          {metrics.get('eval_rmse', 'N/A')}")
            # Остальные метрики
            for key, value in metrics.items():
                if key not in ['eval_loss', 'eval_mae', 'eval_rmse'] and isinstance(value, (int, float)):
                    print(f"  {key:20} {value:.4f}")
            print("="*60)
            
            # Сохраняем в JSON
            with open(self.log_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)

# Инициализируем callback
metrics_callback = MetricsLoggerCallback()


# ======================
# DATA PROCESSING
# ======================

def load_dating_data(file_path: str) -> pd.DataFrame:
    """
    Load dating data from CSV file.
    Expected columns: transliteration, translation, dating, actual_date, spread
    
    Handles edge cases like quotes in spread values (e.g., '187.5"')
    """
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found")
        return pd.DataFrame()
    
    # Load CSV file
    df = pd.read_csv(file_path)
    
    # Ensure required columns exist
    required_cols = ['transliteration', 'translation', 'dating', 'actual_date', 'spread']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️  Missing columns: {missing_cols}")
        return pd.DataFrame()
    
    # Clean and process data
    data = []
    for idx, row in df.iterrows():
        try:
            # Parse actual_date
            actual_date = float(row['actual_date'])
            
            # Parse spread - handle cases like '187.5"' with quotes
            spread_str = str(row['spread']).strip().rstrip('"').rstrip("'")
            spread = float(spread_str)
            
            if actual_date > 0:  # Only include valid dates
                data.append({
                    'transliteration': str(row['transliteration']).strip(),
                    'translation': str(row['translation']).strip(),
                    'dating': str(row['dating']).strip(),
                    'actual_date': actual_date,
                    'spread': spread
                })
        except (ValueError, KeyError) as e:
            print(f"⚠️  Skipping row {idx} (invalid data): {e}")
            continue
    
    return pd.DataFrame(data)


class DatingDataset(Dataset):
    """PyTorch Dataset for dating prediction."""
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512, 
                 mean_date: float = None, std_date: float = None):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Compute normalization statistics using actual_date
        self.dating_values = df['actual_date'].tolist()
        
        # Use provided stats (from train) or compute from current dataset
        if mean_date is not None and std_date is not None:
            self.mean_date = mean_date
            self.std_date = std_date
        else:
            self.mean_date = np.mean(self.dating_values)
            self.std_date = np.std(self.dating_values) + 1e-6  # Avoid division by zero
        
        print(f"\n📊 Dataset statistics:")
        print(f"   Total samples: {len(df)}")
        print(f"   Date range: {min(self.dating_values):.1f} - {max(self.dating_values):.1f}")
        print(f"   Mean date: {self.mean_date:.1f}, Std: {self.std_date:.1f}")
        if mean_date is not None:
            print(f"   ⚠️ Using normalization stats from training set")
        
        # Also track spread statistics
        if 'spread' in df.columns:
            self.mean_spread = np.mean(df['spread'].tolist())
            self.std_spread = np.std(df['spread'].tolist()) + 1e-6
            print(f"   Spread range: {min(df['spread']):.1f} - {max(df['spread']):.1f}")
            print(f"   Mean spread: {self.mean_spread:.1f}, Std: {self.std_spread:.1f}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Create input text
        input_text = f"Transliteration: {row['transliteration']}\nTranslation: {row['translation']}"
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Normalize target value using actual_date
        normalized_date = (row['actual_date'] - self.mean_date) / self.std_date
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(normalized_date, dtype=torch.float32)
        }


# ======================
# REGRESSION MODEL
# ======================

class DatingRegressionModel(nn.Module):
    """
    Wrapper model that adds a regression head on top of a causal LM.
    """
    
    def __init__(self, base_model, hidden_size: int):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        
        # Regression head: 2-layer MLP
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        # Get last hidden state
        last_hidden_state = outputs.hidden_states[-1]  # (batch, seq_len, hidden)
        
        # Pool: use mean of non-padded tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            sum_hidden = (last_hidden_state * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1)
            pooled = sum_hidden / count
        else:
            pooled = last_hidden_state.mean(dim=1)
        
        # Regression prediction
        prediction = self.regression_head(pooled).squeeze(-1)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(prediction, labels)
        
        return {
            'loss': loss,
            'logits': prediction,
            'hidden_states': outputs.hidden_states
        }


# ======================
# METRICS
# ======================

def compute_metrics(eval_pred, mean_date: float, std_date: float):
    """Compute regression metrics (denormalized)."""
    predictions, labels = eval_pred
    
    # Denormalize
    predictions_denorm = predictions * std_date + mean_date
    labels_denorm = labels * std_date + mean_date
    
    mae = mean_absolute_error(labels_denorm, predictions_denorm)
    mse = mean_squared_error(labels_denorm, predictions_denorm)
    rmse = np.sqrt(mse)
    
    # Trainer автоматически добавит префикс 'eval_' к этим ключам
    return {
        'eval_mae': mae,
        'eval_mse': mse,
        'eval_rmse': rmse,
    }


class DatingTrainer(Trainer):
    """Custom trainer that passes normalization stats to compute_metrics and handles Gemma2 tied weights."""
    
    def __init__(self, *args, mean_date: float, std_date: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean_date = mean_date
        self.std_date = std_date
    
    def compute_metrics(self, eval_pred):
        return compute_metrics(eval_pred, self.mean_date, self.std_date)
    
    def _save(self, output_dir=None, state_dict=None):
        """
        Override _save to handle tied weights in Gemma2.
        Uses safe_serialization=False to avoid shared memory issues.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        
        # Сохраняем базовую модель с LoRA (Gemma2 с tied weights)
        # safe_serialization=False обходит ошибку shared memory
        self.model.base_model.save_pretrained(
            output_dir,
            safe_serialization=False,
        )
        
        # Сохраняем регрессионную голову отдельно
        regression_head_path = os.path.join(output_dir, 'regression_head.pt')
        torch.save(self.model.regression_head.state_dict(), regression_head_path)
        
        # Сохраняем токенизатор (используем processing_class для совместимости)
        if hasattr(self, 'processing_class') and self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        
        # Сохраняем training_args
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


# ======================
# MAIN TRAINING
# ======================

print("🚀 Starting dating prediction training...")

# Шаг 1: Загрузка модели и токенизатора
print("\n📥 Loading model and tokenizer...")
model_name = MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side='right',
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda:0",
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
print("✅ Model loaded")

# Шаг 2: Загрузка данных
print("\n📥 Loading dating datasets...")
train_df = load_dating_data(TRAIN_FILE)
val_df = load_dating_data(VAL_FILE)
test_df = load_dating_data(TEST_FILE)

print(f"\n📊 Data statistics:")
print(f"   Train samples: {len(train_df)}")
print(f"   Val samples:   {len(val_df)}")
print(f"   Test samples:  {len(test_df)}")

# Создание датасетов (val/test используют stats от train для консистентности)
train_dataset = DatingDataset(train_df, tokenizer, MAX_SEQ_LENGTH)
val_dataset = DatingDataset(val_df, tokenizer, MAX_SEQ_LENGTH, 
                             mean_date=train_dataset.mean_date, 
                             std_date=train_dataset.std_date)

# Сохранение статистик нормализации
normalization_stats = {
    'mean_date': float(train_dataset.mean_date),
    'std_date': float(train_dataset.std_date)
}
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, 'normalization_stats.json'), 'w') as f:
    json.dump(normalization_stats, f, indent=2)
print(f"\n✅ Normalization stats saved to {OUTPUT_DIR}/normalization_stats.json")

# Шаг 3: Настройка LoRA
print("\n⚙️ Configuring LoRA...")

# LoRA для Gemma
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,  # Можно увеличить для Gemma
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # target_modules=["gate_proj", "up_proj", "down_proj", "q_proj", "v_proj"],
)

model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# Обёртка с регрессионной головой
hidden_size = model.base_model.config.hidden_size
model = DatingRegressionModel(model, hidden_size)
print("✅ Regression head added")

# Шаг 4: Настройка параметров обучения
print("\n⚙️ Configuring training arguments...")

# Вычисляем warmup_steps из warmup_ratio (т.к. warmup_ratio deprecated в новых transformers)
# Примерное количество шагов = (кол-во_samples / batch_size) * epochs / gradient_accumulation
# Для warmup_ratio=0.05 → 5% от общего числа шагов
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100,  # ~5% от ожидаемых шагов
    weight_decay=0.01,
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
    num_train_epochs=20,              # 20 эпох = ~4K шагов (компромисс для 3K samples)
    save_steps=500,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
)

# Прокинем конфиг в ClearML
task.connect({
    "model_name": MODEL_NAME,
    "data_paths": {
        "train": TRAIN_FILE,
        "val": VAL_FILE,
        "test": TEST_FILE,
    },
    "max_seq_length": MAX_SEQ_LENGTH,
    "normalization_stats": normalization_stats,
    "lora": {
        "r": lora_cfg.r,
        "lora_alpha": lora_cfg.lora_alpha,
        "lora_dropout": lora_cfg.lora_dropout,
        "target_modules": list(lora_cfg.target_modules),
    },
    "training": {
        "output_dir": training_args.output_dir,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "learning_rate": training_args.learning_rate,
        "num_train_epochs": training_args.num_train_epochs,
        "warmup_steps": training_args.warmup_steps,
    }
})

# Шаг 5: Создание тренера
print("\n🔧 Initializing trainer...")
trainer = DatingTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    mean_date=train_dataset.mean_date,
    std_date=train_dataset.std_date,
    callbacks=[metrics_callback, ClearMLCallback()],
)

# Шаг 6: Обучение
print("\n" + "="*60)
print("🚀 STARTING TRAINING")
print("="*60)
trainer.train()

# Шаг 7: Сохранение модели
# Используем подход как в train_model.py — сохраняем через trainer.save_model()
# но с кастомной логикой для DatingRegressionModel
print("\n💾 Saving model...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Сохраняем базовую модель с LoRA (Gemma2 + LoRA адаптеры)
# Для Gemma2 с tied weights используем safe_serialization=False
model.base_model.save_pretrained(
    OUTPUT_DIR,
    safe_serialization=False,  # Обход ошибки shared memory для Gemma2
)
print(f"✅ Base model (with LoRA) saved to {OUTPUT_DIR}")

# 2. Сохраняем регрессионную голову отдельно
regression_head_path = os.path.join(OUTPUT_DIR, 'regression_head.pt')
torch.save(model.regression_head.state_dict(), regression_head_path)
print(f"✅ Regression head saved to {regression_head_path}")

# 3. Сохраняем токенизатор
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Tokenizer saved to {OUTPUT_DIR}")

# 4. Сохраняем stats нормализации (уже сохранены выше, но продублируем для ясности)
with open(os.path.join(OUTPUT_DIR, 'normalization_stats.json'), 'w') as f:
    json.dump(normalization_stats, f, indent=2)
print(f"✅ Normalization stats saved to {OUTPUT_DIR}/normalization_stats.json")

# Шаг 8: Оценка на тестовом наборе
if len(test_df) > 0:
    print("\n" + "="*60)
    print("📊 EVALUATION ON TEST SET")
    print("="*60)
    
    test_dataset = DatingDataset(test_df, tokenizer, MAX_SEQ_LENGTH,
                                  mean_date=train_dataset.mean_date,
                                  std_date=train_dataset.std_date)
    test_results = trainer.evaluate(test_dataset)
    
    print("\n📈 Test Results:")
    for key, value in test_results.items():
        if isinstance(value, (int, float)):
            print(f"  {key:20} {value:.2f}")

# Шаг 9: Финальные метрики
print("\n" + "="*60)
print("📊 FINAL TRAINING METRICS")
print("="*60)
print(f"  Total steps:     {trainer.state.global_step}")
print(f"  Final epoch:     {trainer.state.epoch:.4f}")
if metrics_callback.metrics_history['train_loss']:
    print(f"  Final train loss: {metrics_callback.metrics_history['train_loss'][-1]:.4f}")
if metrics_callback.metrics_history['eval_loss']:
    print(f"  Final eval loss:  {metrics_callback.metrics_history['eval_loss'][-1]:.4f}")
if metrics_callback.metrics_history['eval_mae']:
    print(f"  Final MAE:        {metrics_callback.metrics_history['eval_mae'][-1]:.2f} years")
if metrics_callback.metrics_history['eval_rmse']:
    print(f"  Final RMSE:       {metrics_callback.metrics_history['eval_rmse'][-1]:.2f} years")
print("="*60)

# Логируем финальные метрики в ClearML
task.get_logger().report_scalar(
    title="Final Metrics",
    series="train_loss",
    value=metrics_callback.metrics_history['train_loss'][-1] if metrics_callback.metrics_history['train_loss'] else 0,
    iteration=trainer.state.global_step
)
if metrics_callback.metrics_history['eval_loss']:
    task.get_logger().report_scalar(
        title="Final Metrics",
        series="eval_loss",
        value=metrics_callback.metrics_history['eval_loss'][-1],
        iteration=trainer.state.global_step
    )
if metrics_callback.metrics_history['eval_mae']:
    task.get_logger().report_scalar(
        title="Final Metrics",
        series="mae",
        value=metrics_callback.metrics_history['eval_mae'][-1],
        iteration=trainer.state.global_step
    )

# Добавляем output directory как артефакт
task.upload_artifact(name="dating_model", artifact_object=OUTPUT_DIR)

print(f"\n✅ Training complete! Model saved to {OUTPUT_DIR}")
print(f"✅ Total samples used: {len(train_df) + len(val_df) + len(test_df)}")
print(f"✅ ClearML task: {task.name} (ID: {task.id})")


# ======================
# INFERENCE & EVALUATION
# ======================

print("\n" + "="*60)
print("🔮 RUNNING INFERENCE ON TEST SET")
print("="*60)


class DatingPredictor:
    """Predictor for runic inscription dating (inference only)."""
    
    def __init__(self, model_path: str, device: str = 'cuda:0'):
        self.model_path = model_path
        self.device = device
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        # Load regression head
        hidden_size = self.base_model.config.hidden_size
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
        regression_head_path = os.path.join(model_path, 'regression_head.pt')
        if os.path.exists(regression_head_path):
            self.regression_head.load_state_dict(torch.load(regression_head_path, map_location=device))
            print(f"✅ Loaded regression head from {regression_head_path}")
        
        self.regression_head.to(device)
        self.regression_head.eval()
        self.base_model.eval()
        
        # Load normalization stats
        self.normalization_stats = self._load_normalization_stats(model_path)
        print(f"✅ Normalization: mean={self.normalization_stats['mean_date']:.1f}, std={self.normalization_stats['std_date']:.1f}")
    
    def _load_normalization_stats(self, model_path: str) -> dict:
        stats_path = os.path.join(model_path, 'normalization_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                return json.load(f)
        return {'mean_date': 900.0, 'std_date': 300.0}
    
    def predict(self, transliteration: str, translation: str = "") -> float:
        """Predict date for a single inscription."""
        input_text = f"Transliteration: {transliteration}\nTranslation: {translation}"
        
        encoding = self.tokenizer(
            input_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            last_hidden = outputs.hidden_states[-1]
            mask = attention_mask.unsqueeze(-1).float()
            sum_hidden = (last_hidden * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1)
            pooled = sum_hidden / count
            
            predicted_normalized = self.regression_head(pooled)
        
        # Denormalize
        predicted_year = predicted_normalized.item() * self.normalization_stats['std_date'] + self.normalization_stats['mean_date']
        return predicted_year
    
    def predict_batch(self, data: list) -> list:
        """Predict dates for multiple inscriptions."""
        results = []
        for item in data:
            pred_year = self.predict(
                item.get('transliteration', ''),
                item.get('translation', '')
            )
            results.append({
                'transliteration': item.get('transliteration', ''),
                'translation': item.get('translation', ''),
                'actual_dating': item.get('dating', ''),
                'actual_year': item.get('actual_date', 0),
                'spread': item.get('spread', 0),
                'predicted_year': pred_year,
                'error': abs(pred_year - item.get('actual_date', 0))
            })
        return results


def year_to_dating_string(year: float) -> str:
    """Convert predicted year to dating string format for text metrics."""
    year_int = int(round(year))
    # Return as range format similar to training data
    return f"{year_int - 50} - {year_int + 50}"


def evaluate_predictions(predictions: list) -> dict:
    """Calculate evaluation metrics (regression + text-based)."""
    actual_years = [p['actual_year'] for p in predictions if p['actual_year'] > 0]
    predicted_years = [p['predicted_year'] for p in predictions if p['actual_year'] > 0]
    
    if not actual_years:
        return {}
    
    actual = np.array(actual_years)
    predicted = np.array(predicted_years)
    
    # ===== Regression Metrics =====
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    
    # Additional metrics
    median_error = np.median(np.abs(predicted - actual))
    mean_error = np.mean(np.abs(predicted - actual))
    std_error = np.std(np.abs(predicted - actual))
    
    # Percentage errors
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-6))) * 100
    
    # ===== Text-based Metrics (BLEU, chrF, ROUGE-L, METEOR) =====
    # Convert to string format for text comparison
    predicted_strings = [year_to_dating_string(p) for p in predicted]
    actual_strings = [year_to_dating_string(a) for a in actual]
    
    # BLEU
    bleu_score = 0.0
    try:
        bleu = sacrebleu.corpus_bleu(predicted_strings, [[a] for a in actual_strings])
        bleu_score = bleu.score
    except Exception as e:
        print(f"⚠️  BLEU calculation error: {e}")
    
    # chrF
    chrf_score = 0.0
    try:
        chrf = sacrebleu.corpus_chrf(predicted_strings, [[a] for a in actual_strings], word_order=1)
        chrf_score = chrf.score
    except Exception as e:
        print(f"⚠️  chrF calculation error: {e}")
    
    # chrF++
    chrf_plusplus_score = 0.0
    try:
        chrf_pp = sacrebleu.corpus_chrf(predicted_strings, [[a] for a in actual_strings], word_order=2)
        chrf_plusplus_score = chrf_pp.score
    except Exception as e:
        print(f"⚠️  chrF++ calculation error: {e}")
    
    # ROUGE-L
    rouge_l_score = 0.0
    try:
        rouge = Rouge()
        rouge_scores = rouge.get_scores(predicted_strings, actual_strings, avg=True)
        rouge_l_score = rouge_scores['rouge-l']['f'] * 100
    except Exception as e:
        print(f"⚠️  ROUGE-L calculation error: {e}")
    
    # METEOR
    meteor_score_val = 0.0
    try:
        references_tokenized = [[a.split()] for a in actual_strings]
        predictions_tokenized = [p.split() for p in predicted_strings]
        meteor_score_val = meteor_score(references_tokenized, predictions_tokenized) * 100
    except Exception as e:
        print(f"⚠️  METEOR calculation error: {e}")
    
    return {
        # Regression metrics
        'mae': mae,
        'rmse': rmse,
        'mse': mse,
        'median_error': median_error,
        'mean_error': mean_error,
        'std_error': std_error,
        'mape': mape,
        # Text-based metrics
        'bleu': bleu_score,
        'chrf': chrf_score,
        'chrf++': chrf_plusplus_score,
        'rouge_l': rouge_l_score,
        'meteor': meteor_score_val,
    }


def run_inference_evaluation(model_path: str, test_df: pd.DataFrame, output_file: str = "/train_model/dating_inference_results.csv"):
    """Run full inference evaluation on test set."""
    
    predictor = DatingPredictor(model_path)
    
    # Convert DataFrame to list of dicts
    test_data = test_df.to_dict('records')
    
    print(f"\n📊 Running inference on {len(test_data)} test samples...")
    
    # Run predictions
    predictions = predictor.predict_batch(test_data)
    
    # Calculate metrics
    metrics = evaluate_predictions(predictions)
    
    print("\n" + "="*60)
    print("📈 INFERENCE EVALUATION METRICS")
    print("="*60)
    
    if metrics:
        print("\n  === REGRESSION METRICS ===")
        print(f"  MAE (years):         {metrics['mae']:.2f}")
        print(f"  RMSE (years):        {metrics['rmse']:.2f}")
        print(f"  MSE:                 {metrics['mse']:.2f}")
        print(f"  Median Error (years):{metrics['median_error']:.2f}")
        print(f"  Mean Error (years):  {metrics['mean_error']:.2f}")
        print(f"  Std Error (years):   {metrics['std_error']:.2f}")
        print(f"  MAPE (%):            {metrics['mape']:.2f}%")
        
        # print("\n  === TEXT-BASED METRICS ===")
        # print(f"  BLEU:                {metrics['bleu']:.2f}")
        # print(f"  chrF:                {metrics['chrf']:.2f}")
        # print(f"  chrF++:              {metrics['chrf++']:.2f}")
        # print(f"  ROUGE-L:             {metrics['rouge_l']:.2f}")
        # print(f"  METEOR:              {metrics['meteor']:.2f}")
    else:
        print("  No valid predictions to evaluate")
    
    print("="*60)
    
    # Save results to CSV
    results_df = pd.DataFrame(predictions)
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to {output_file}")
    
    # Log to ClearML
    if metrics:
        # Regression metrics
        task.get_logger().report_scalar(
            title="Inference Metrics",
            series="mae",
            value=metrics['mae'],
            iteration=trainer.state.global_step
        )
        task.get_logger().report_scalar(
            title="Inference Metrics",
            series="rmse",
            value=metrics['rmse'],
            iteration=trainer.state.global_step
        )
        task.get_logger().report_scalar(
            title="Inference Metrics",
            series="mape",
            value=metrics['mape'],
            iteration=trainer.state.global_step
        )
        
        # Text-based metrics
        # task.get_logger().report_scalar(
        #     title="Inference Metrics",
        #     series="bleu",
        #     value=metrics['bleu'],
        #     iteration=trainer.state.global_step
        # )
        # task.get_logger().report_scalar(
        #     title="Inference Metrics",
        #     series="chrf++",
        #     value=metrics['chrf++'],
        #     iteration=trainer.state.global_step
        # )
        # task.get_logger().report_scalar(
        #     title="Inference Metrics",
        #     series="rouge_l",
        #     value=metrics['rouge_l'],
        #     iteration=trainer.state.global_step
        # )
        # task.get_logger().report_scalar(
        #     title="Inference Metrics",
        #     series="meteor",
        #     value=metrics['meteor'],
        #     iteration=trainer.state.global_step
        # )
    
    # Save metrics to JSON
    metrics_file = "/train_model/dating_inference_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to {metrics_file}")
    
    # Print sample predictions
    print("\n" + "="*60)
    print("📋 SAMPLE PREDICTIONS (first 10)")
    print("="*60)
    for i, pred in enumerate(predictions[:10]):
        print(f"\n  {i+1}. Transliteration: {pred['transliteration'][:50]}...")
        print(f"     Actual: {pred['actual_dating']} ({pred['actual_year']:.1f})")
        print(f"     Predicted: {pred['predicted_year']:.1f}")
        print(f"     Error: {pred['error']:.1f} years")
    
    return metrics


# Run inference evaluation
if len(test_df) > 0:
    inference_metrics = run_inference_evaluation(OUTPUT_DIR, test_df)
else:
    print("\n⚠️  No test data available for inference evaluation")
    inference_metrics = {}

print("\n" + "="*60)
print("🎉 ALL TASKS COMPLETE")
print("="*60)
print(f"  ✅ Training:   {OUTPUT_DIR}")
print(f"  ✅ Metrics:    dating_inference_metrics.json")
print(f"  ✅ Results:    dating_inference_results.csv")
print(f"  ✅ ClearML:    {task.name} (ID: {task.id})")
print("="*60)
