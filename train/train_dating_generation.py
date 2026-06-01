import logging
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import TrainerCallback
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
import nltk
import re

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
task = Task.init(
    project_name="d-prokopev/runic_ai",
    task_name="gemma-2-2b-lora-dating-generation",
    task_type=Task.TaskTypes.training,
    tags=["gemma-2-2b-it", "lora", "generation", "runic", "dating"],
)

# Пути к данным
TRAIN_FILE = "/train_model/dating_dataset/train.csv"
VAL_FILE = "/train_model/dating_dataset/val.csv"
TEST_FILE = "/train_model/dating_dataset/test.csv"

# Путь для сохранения модели
OUTPUT_DIR = "/train_model/gemma_dating_generation_trained"

# Путь к базовой модели
MODEL_NAME = "/models/"

# Параметры данных
MAX_SEQ_LENGTH = 512

# Промпт для генерации года
DATING_PROMPT = """### Instruction:
Analyze the following runic inscription transliteration to predict the year it was created.
Provide ONLY the predicted year as a number (e.g., 850). Do not include any labels, explanations, or additional text.

### Transliteration:
{transliteration}

### Response:
The predicted year is: """

# Custom callback для логирования метрик
class MetricsLoggerCallback(TrainerCallback):
    """Callback для логирования train/eval метрик в JSON файл"""
    
    def __init__(self, log_path="/train_model/dating_generation_metrics.json"):
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


class DatingGenerationDataset(Dataset):
    """PyTorch Dataset for dating prediction via generation."""
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Statistics for logging
        self.dating_values = df['actual_date'].tolist()
        
        print(f"\n📊 Dataset statistics:")
        print(f"   Total samples: {len(df)}")
        print(f"   Date range: {min(self.dating_values):.1f} - {max(self.dating_values):.1f}")
        print(f"   Mean date: {np.mean(self.dating_values):.1f}, Std: {np.std(self.dating_values):.1f}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Create input prompt (translation removed)
        prompt = DATING_PROMPT.format(
            transliteration=row['transliteration']
        )
        
        # Create target: just the year number
        target = str(int(row['actual_date']))
        
        # Full text for training: prompt + target
        full_text = prompt + target
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels: -100 for prompt tokens, actual tokens for target
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Tokenize prompt separately to find where target starts
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        prompt_length = prompt_encoding['input_ids'].shape[1]
        
        # Create labels: mask prompt tokens with -100
        labels = input_ids.clone()
        labels[:prompt_length] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


# ======================
# METRICS
# ======================

def extract_year_from_text(text: str) -> float:
    """Extract year number from generated text."""
    # Try to find 4-digit or 3-digit number
    match = re.search(r'\b(\d{3,4})\b', text)
    if match:
        return float(match.group(1))
    # Fallback: try to find any number
    match = re.search(r'\b(\d+(?:\.\d+)?)\b', text)
    if match:
        return float(match.group(1))
    return None


def compute_metrics(eval_pred, tokenizer):
    """Compute regression metrics from generated text."""
    predictions, labels = eval_pred
    
    # Replace -100 with tokenizer.pad_token_id for decoding
    labels_copy = labels.copy()
    labels_copy[labels_copy == -100] = tokenizer.pad_token_id
    
    predicted_years = []
    actual_years = []
    
    for pred_ids, label_ids in zip(predictions, labels_copy):
        # Decode prediction
        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
        
        # Extract year from prediction
        pred_year = extract_year_from_text(pred_text)
        
        # Extract year from label (it's just the number)
        label_text = tokenizer.decode(label_ids, skip_special_tokens=True).strip()
        label_year = extract_year_from_text(label_text)
        
        if pred_year is not None and label_year is not None:
            predicted_years.append(pred_year)
            actual_years.append(label_year)
    
    if not predicted_years:
        return {'eval_mae': 999.0, 'eval_rmse': 999.0}
    
    mae = mean_absolute_error(actual_years, predicted_years)
    mse = mean_squared_error(actual_years, predicted_years)
    rmse = np.sqrt(mse)
    
    return {
        'eval_mae': mae,
        'eval_mse': mse,
        'eval_rmse': rmse,
    }


class DatingGenerationTrainer(Trainer):
    """Custom trainer for generation-based dating."""
    
    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
    
    def compute_metrics(self, eval_pred):
        return compute_metrics(eval_pred, self.tokenizer)


# ======================
# MAIN TRAINING
# ======================

print("🚀 Starting dating prediction (generation) training...")

# Шаг 1: Загрузка модели и токенизатора
print("\n📥 Loading model and tokenizer...")

# Используем локальный путь напрямую
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side='right',
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cuda:0",       # ← явно указываем GPU
    local_files_only=True,
    trust_remote_code=True,
    dtype=torch.float16,
)
print("✅ Model loaded")

# Подготовка модели для LoRA обучения
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

# Шаг 2: Загрузка данных
print("\n📥 Loading dating datasets...")
train_df = load_dating_data(TRAIN_FILE)
val_df = load_dating_data(VAL_FILE)
test_df = load_dating_data(TEST_FILE)

print(f"\n📊 Data statistics:")
print(f"   Train samples: {len(train_df)}")
print(f"   Val samples:   {len(val_df)}")
print(f"   Test samples:  {len(test_df)}")

# Создание датасетов
train_dataset = DatingGenerationDataset(train_df, tokenizer, MAX_SEQ_LENGTH)
val_dataset = DatingGenerationDataset(val_df, tokenizer, MAX_SEQ_LENGTH)

# Шаг 3: Настройка LoRA
print("\n⚙️ Configuring LoRA...")

# LoRA для Gemma2
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# Шаг 4: Настройка параметров обучения
print("\n⚙️ Configuring training arguments...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100,
    weight_decay=0.01,
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
    num_train_epochs=20,
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
    "prompt_template": DATING_PROMPT,
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
trainer = DatingGenerationTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    callbacks=[metrics_callback, ClearMLCallback()],
)

# Шаг 6: Обучение
print("\n" + "="*60)
print("🚀 STARTING TRAINING")
print("="*60)
trainer.train()

# Шаг 7: Сохранение модели
print("\n💾 Saving model...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Сохраняем модель с LoRA
model.save_pretrained(OUTPUT_DIR)
print(f"✅ Model (with LoRA) saved to {OUTPUT_DIR}")

# Сохраняем токенизатор
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Tokenizer saved to {OUTPUT_DIR}")

# Сохраняем промпт для инференса
with open(os.path.join(OUTPUT_DIR, 'prompt_template.txt'), 'w') as f:
    f.write(DATING_PROMPT)
print(f"✅ Prompt template saved to {OUTPUT_DIR}/prompt_template.txt")

# Шаг 8: Оценка на тестовом наборе
if len(test_df) > 0:
    print("\n" + "="*60)
    print("📊 EVALUATION ON TEST SET")
    print("="*60)
    
    test_dataset = DatingGenerationDataset(test_df, tokenizer, MAX_SEQ_LENGTH)
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
task.upload_artifact(name="dating_generation_model", artifact_object=OUTPUT_DIR)

print(f"\n✅ Training complete! Model saved to {OUTPUT_DIR}")
print(f"✅ ClearML task: {task.name} (ID: {task.id})")


# ======================
# INFERENCE & EVALUATION
# ======================

print("\n" + "="*60)
print("🔮 RUNNING INFERENCE ON TEST SET")
print("="*60)


class DatingGenerationPredictor:
    """Predictor for runic inscription dating via text generation."""
    
    def __init__(self, model_path: str, device: str = 'cuda:0'):
        self.model_path = model_path
        self.device = device
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        self.model.eval()
        
        # Load prompt template
        prompt_path = os.path.join(model_path, '/train_model/prompt_template.txt')
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r') as f:
                self.prompt_template = f.read()
        else:
            self.prompt_template = DATING_PROMPT
        
        print(f"✅ Loaded model from {model_path}")
    
    def predict(self, transliteration: str, max_new_tokens: int = 10) -> dict:
        """Predict date for a single inscription."""
        prompt = self.prompt_template.format(
            transliteration=transliteration
        )
        
        encoding = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=False,  # Greedy decoding for deterministic output
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract year
        predicted_year = extract_year_from_text(generated_text)
        
        return {
            'prompt': prompt,
            'generated_text': generated_text,
            'predicted_year': predicted_year
        }
    
    def predict_batch(self, data: list, max_new_tokens: int = 10) -> list:
        """Predict dates for multiple inscriptions."""
        results = []
        for item in data:
            pred = self.predict(
                item.get('transliteration', ''),
                max_new_tokens
            )
            results.append({
                'transliteration': item.get('transliteration', ''),
                'actual_dating': item.get('dating', ''),
                'actual_year': item.get('actual_date', 0),
                'spread': item.get('spread', 0),
                'predicted_year': pred['predicted_year'],
                'generated_text': pred['generated_text'],
                'error': abs(pred['predicted_year'] - item.get('actual_date', 0)) if pred['predicted_year'] else None
            })
        return results


def evaluate_predictions(predictions: list) -> dict:
    """Calculate evaluation metrics."""
    # Filter valid predictions
    valid_preds = [p for p in predictions if p['predicted_year'] is not None and p['actual_year'] > 0]
    
    if not valid_preds:
        return {}
    
    actual = np.array([p['actual_year'] for p in valid_preds])
    predicted = np.array([p['predicted_year'] for p in valid_preds])
    
    # Regression metrics
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    
    # Additional metrics
    median_error = np.median(np.abs(predicted - actual))
    mean_error = np.mean(np.abs(predicted - actual))
    std_error = np.std(np.abs(predicted - actual))
    
    # Percentage errors
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-6))) * 100
    
    # Accuracy within ranges
    accuracy_50 = np.mean(np.abs(predicted - actual) <= 50) * 100
    accuracy_100 = np.mean(np.abs(predicted - actual) <= 100) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mse': mse,
        'median_error': median_error,
        'mean_error': mean_error,
        'std_error': std_error,
        'mape': mape,
        'accuracy_50': accuracy_50,
        'accuracy_100': accuracy_100,
        'valid_predictions': len(valid_preds),
        'total_predictions': len(predictions),
    }


def run_inference_evaluation(model_path: str, test_df: pd.DataFrame, output_file: str = "/train_model/dating_generation_results.csv"):
    """Run full inference evaluation on test set."""
    
    predictor = DatingGenerationPredictor(model_path)
    
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
        print(f"  Accuracy ±50 years:  {metrics['accuracy_50']:.1f}%")
        print(f"  Accuracy ±100 years: {metrics['accuracy_100']:.1f}%")
        print(f"  Valid predictions:   {metrics['valid_predictions']}/{metrics['total_predictions']}")
    else:
        print("  No valid predictions to evaluate")
    
    print("="*60)
    
    # Save results to CSV
    results_df = pd.DataFrame(predictions)
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to {output_file}")
    
    # Log to ClearML
    if metrics:
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
            series="accuracy_50",
            value=metrics['accuracy_50'],
            iteration=trainer.state.global_step
        )
    
    # Save metrics to JSON
    metrics_file = "/train_model/dating_generation_inference_metrics.json"
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
        print(f"     Predicted: {pred['predicted_year']}")
        print(f"     Generated: {pred['generated_text'][-50:]}")
        if pred['error'] is not None:
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
print(f"  ✅ Metrics:    /train_model/dating_generation_inference_metrics.json")
print(f"  ✅ Results:    /train_model/dating_generation_results.csv")
print(f"  ✅ ClearML:    {task.name} (ID: {task.id})")
print("="*60)
