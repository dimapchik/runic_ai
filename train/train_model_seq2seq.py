"""
Seq2Seq Model Training Script for Runic AI
==========================================
Обучение Seq2Seq моделей (M2M100, NLLB, MADLAD) с LoRA адаптерами.

Особенности:
- Поддержка M2M100, NLLB-200, MADLAD-400
- LoRA fine-tuning для эффективного обучения
- Многоязычный перевод с указанием целевого языка
- Интеграция с ClearML для логирования
"""

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import json

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
    task_name="madlad400-3b-mt-lora-sft-multi-lang",
    task_type=Task.TaskTypes.training,
    tags=["madlad400-3b-mt", "lora", "seq2seq", "runic", "multi-lang"],
)

# Маппинг языков и их ID
LANGUAGE_TO_ID = {'en': 0, 'de': 1, 'sv': 2}

# Маппинг языков на токены модели
# Для M2M100: короткие коды (en, de, sv)
# Для NLLB-200: полные коды (eng_Latn, deu_Latn, swe_Latn)
# Для MADLAD-400: instruction-based формат
# 
# MADLAD-400 использует инструктивный формат промптов:
# "translate {source_lang} to {target_lang}: {text}"
# https://arxiv.org/abs/2310.03025
LANGUAGE_TO_TOKEN = {
    # MADLAD-400: полные названия языков (lowercase)
    "en": "english",
    "de": "german",
    "sv": "swedish",
    # Дополнительные языки при необходимости:
    # "da": "danish",
    # "no": "norwegian",
    # "fi": "finnish",
    # "ru": "russian",
    # "es": "spanish",
    # "fr": "french",
    # "it": "italian",
}

# Источник для MADLAD (фиксированный)
SOURCE_LANGUAGE = "old norse"  # или "runic" / "old_norse"

# Языки для обучения
TARGET_LANGUAGES = ["en", "de", "sv"]

# Путь к параллельному корпусу
DATA_PATH = "/train_model/parallel_corpus_train.csv"

# Путь для сохранения модели
OUTPUT_DIR = "/train_model/model_trained"

# Путь к модели (локальный или HuggingFace)
# Варианты:
# - "facebook/m2m100-418M" (меньше, быстрее)
# - "facebook/m2m100-1.2B" (больше, качественнее)
# - "google/madlad400-3b-mt" (очень большая, 400 языков)
# - "facebook/nllb-200-1.3B" (NLLB версия)
# - "/models/" (локальная модель)
MODEL_NAME = "/models/"

# Custom callback для логирования метрик
class MetricsLoggerCallback(TrainerCallback):
    """Callback для логирования train/eval метрик в JSON файл"""
    
    def __init__(self, log_path="/train_model/training_metrics_seq2seq.json"):
        self.log_path = log_path
        self.metrics_history = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
            "grad_norm": [],
            "epoch": []
        }
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Логируем train метрики
            if "loss" in logs:
                self.metrics_history["train_loss"].append(logs["loss"])
            if "learning_rate" in logs:
                self.metrics_history["learning_rate"].append(logs["learning_rate"])
            if "grad_norm" in logs:
                self.metrics_history["grad_norm"].append(logs["grad_norm"])
            if "epoch" in logs:
                self.metrics_history["epoch"].append(logs["epoch"])
            
            # Печатаем в красивом формате
            print("\n" + "="*60)
            print(f"📊 METRICS (step {state.global_step})")
            print("="*60)
            print(f"  Epoch:           {logs.get('epoch', 'N/A')}")
            print(f"  Loss (train):    {logs.get('loss', 'N/A')}")
            print(f"  Loss (eval):     {logs.get('eval_loss', 'N/A')}")
            print(f"  Grad norm:       {logs.get('grad_norm', 'N/A')}")
            print(f"  Learning rate:   {logs.get('learning_rate', 'N/A')}")
            print("="*60)
            
            # Сохраняем в JSON
            with open(self.log_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            # Логируем eval метрики
            if "eval_loss" in metrics:
                self.metrics_history["eval_loss"].append(metrics["eval_loss"])
            
            print("\n" + "="*60)
            print(f"📈 EVALUATION RESULTS (step {state.global_step})")
            print("="*60)
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key:20} {value:.4f}")
            print("="*60)
            
            # Сохраняем в JSON
            with open(self.log_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)

# Инициализируем callback
metrics_callback = MetricsLoggerCallback()

# ======================
# MODEL LOADING
# ======================

print("="*80)
print("📥 Loading Seq2Seq Model")
print("="*80)

print(f"\nModel: {MODEL_NAME}")
print(f"Device: CUDA")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    device_map="cuda:0",
    local_files_only=False,  # True если модель локальная
    trust_remote_code=True,
    dtype=torch.float16,  # Новый API (torch_dtype устарел)
)
print("✓ Model loaded")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

# Для M2M100: установить язык источника по умолчанию
if hasattr(tokenizer, 'src_lang'):
    tokenizer.src_lang = "en"
    print(f"✓ Source language set to: {tokenizer.src_lang}")

# Добавить pad_token если нет
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("✓ Pad token added")

# Подготовка модели для LoRA обучения
print("\n🔧 Preparing model for LoRA training...")
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
print("✓ Model prepared for LoRA")

# ======================
# DATA LOADING
# ======================

print("\n" + "="*80)
print("📂 Loading Dataset")
print("="*80)

def load_parallel_corpus_multi(path, target_languages):
    """
    Load parallel corpus from CSV and filter by target languages.
    Returns dataset with source text and target translation.
    """
    df = pd.read_csv(path)
    
    # Filter by target languages
    df = df[df['language'].isin(target_languages)]
    
    print(f"\n📊 Dataset statistics:")
    print(f"   Total samples: {len(df)}")
    for lang in target_languages:
        lang_count = len(df[df['language'] == lang])
        print(f"   {lang.upper()}: {lang_count} samples")
    
    examples = {
        "source_text": [],
        "target_translation": [],
        "target_language": [],
        "target_token": [],
    }
    
    for _, row in df.iterrows():
        src = row['source_text']
        tgt = row['target_translation']
        lang = row['language']
        
        # Получить токен целевого языка
        target_token = LANGUAGE_TO_TOKEN.get(lang, "en")
        
        examples["source_text"].append(src)
        examples["target_translation"].append(tgt)
        examples["target_language"].append(lang)
        examples["target_token"].append(target_token)
    
    return Dataset.from_dict(examples)

dataset = load_parallel_corpus_multi(DATA_PATH, target_languages=TARGET_LANGUAGES)

# ======================
# DATA PREPROCESSING
# ======================

print("\n" + "="*80)
print("🔄 Preprocessing Data")
print("="*80)

def preprocess_function(example):
    """
    Preprocess examples for Seq2Seq training.
    
    Для M2M100 формат:
    - Input: {source_text}
    - Target: __{lang}__ {target_text}
    
    Для NLLB-200 формат:
    - Input: {source_text}
    - Target: __{lang_token}__ {target_text}
    
    Для MADLAD-400 формат (instruction-based):
    - Input: translate {source_lang} to {target_lang}: {source_text}
    - Target: {target_text}
    """
    source_text = example["source_text"]
    target_translation = example["target_translation"]
    target_token = example["target_token"]
    
    # MADLAD-400: instruction-based формат
    # Input: "translate old norse to english: {source_text}"
    # Target: "{target_translation}"
    formatted_input = f"translate {SOURCE_LANGUAGE} to {target_token}: {source_text}"
    formatted_target = target_translation
    
    # Токенизация
    model_inputs = tokenizer(
        formatted_input,
        max_length=128,
        padding=False,
        truncation=True,
    )
    
    labels = tokenizer(
        formatted_target,
        max_length=128,
        padding=False,
        truncation=True,
    )
    
    # Добавить labels (decoder input)
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

# Распределение по языкам ДО токенизации (для статистики)
print(f"\n📊 Language distribution (before tokenization):")
df_before = pd.DataFrame(dataset)
for lang in TARGET_LANGUAGES:
    lang_count = len(df_before[df_before['target_language'] == lang])
    print(f"   {lang.upper()}: {lang_count} samples ({lang_count/len(dataset)*100:.1f}%)")

# Применить предобработку
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=False,
    remove_columns=dataset.column_names,  # ← Удаляем исходные колонки после токенизации
    desc="Tokenizing dataset"
)

# Разделение на train и val
split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

print(f"\n✓ Dataset preprocessed")
print(f"   Train samples: {len(train_dataset)}")
print(f"   Val samples: {len(val_dataset)}")

# Проверить пустые labels
empty_labels = sum(1 for x in train_dataset if len(x["labels"]) == 0)
print(f"Empty labels in train: {empty_labels}")
# Проверить формат target
sample = train_dataset[0]
print(f"Source: {tokenizer.decode(sample['input_ids'])}")
print(f"Target: {tokenizer.decode(sample['labels'])}")
# ======================
# LORA CONFIGURATION
# ======================

print("\n" + "="*80)
print("⚙️  LoRA Configuration")
print("="*80)

# Target modules зависят от архитектуры модели
# M2M100: ["q_proj", "k_proj", "v_proj", "out_proj"]
# NLLB: ["q_proj", "v_proj"]
# MADLAD: ["q", "k", "v", "o"] (проверить в config)

# Автоматическое определение target modules
def get_target_modules(model):
    """Определить подходящие слои для LoRA"""
    model_type = model.config.model_type
    
    if "m2m_100" in model_type:
        return ["q_proj", "k_proj", "v_proj", "out_proj"]
    elif "nllb" in model_type:
        return ["q_proj", "v_proj"]
    elif "madlad" in model_type or "t5" in model_type:
        return ["q", "k", "v", "o", "wi", "wo"]
    else:
        # По умолчанию
        return ["q_proj", "v_proj"]

target_modules = get_target_modules(model)
print(f"Model type: {model.config.model_type}")
print(f"Target modules: {target_modules}")

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM",  # ← Seq2Seq task type
    target_modules=target_modules,
)

model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ======================
# TRAINING CONFIGURATION
# ======================

print("\n" + "="*80)
print("⚙️  Training Configuration")
print("="*80)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    num_train_epochs=40,
    save_steps=400,              # ← Кратно eval_steps (200 * 2 = 400)
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    predict_with_generate=True,  # ← Важно для Seq2Seq
    generation_max_length=128,
    fp16=True,
    save_total_limit=2,
    dataloader_pin_memory=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False
)

# Прокинем конфиг в ClearML
task.connect({
    "model_name": MODEL_NAME,
    "data_path": DATA_PATH,
    "target_languages": TARGET_LANGUAGES,
    "language_to_id": LANGUAGE_TO_ID,
    "language_to_token": LANGUAGE_TO_TOKEN,
    "lora": {
        "r": lora_cfg.r,
        "lora_alpha": lora_cfg.lora_alpha,
        "lora_dropout": lora_cfg.lora_dropout,
        "target_modules": lora_cfg.target_modules,
        "task_type": lora_cfg.task_type,
    },
    "training": {
        "output_dir": training_args.output_dir,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "learning_rate": training_args.learning_rate,
        "num_train_epochs": training_args.num_train_epochs,
        "warmup_steps": training_args.warmup_steps,
        "max_grad_norm": training_args.max_grad_norm,
    }
})

# Data collator для Seq2Seq
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True,
)

# ======================
# TRAINER
# ======================

print("\n" + "="*80)
print("🚀 Creating Trainer")
print("="*80)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[metrics_callback, ClearMLCallback()],
)

print("✓ Trainer created")

# ======================
# TRAINING
# ======================

print("\n" + "="*80)
print("🎯 Starting Training")
print("="*80)

trainer.train()

# ======================
# SAVE MODEL
# ======================

print("\n" + "="*80)
print("💾 Saving Model")
print("="*80)

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✓ Model saved to {OUTPUT_DIR}")

# Сохранение финальных метрик
print("\n" + "="*60)
print("📊 FINAL TRAINING METRICS")
print("="*60)
print(f"  Total steps:     {trainer.state.global_step}")
print(f"  Final epoch:     {trainer.state.epoch:.4f}")
print(f"  Train loss:      {metrics_callback.metrics_history['train_loss'][-1]:.4f}")
if metrics_callback.metrics_history['eval_loss']:
    print(f"  Eval loss:       {metrics_callback.metrics_history['eval_loss'][-1]:.4f}")
print("="*60)
print(f"\n✓ Metrics saved to {metrics_callback.log_path}")

# Логируем финальные метрики в ClearML
task.get_logger().report_scalar(
    title="Final Metrics",
    series="train_loss",
    value=metrics_callback.metrics_history['train_loss'][-1],
    iteration=trainer.state.global_step
)
if metrics_callback.metrics_history['eval_loss']:
    task.get_logger().report_scalar(
        title="Final Metrics",
        series="eval_loss",
        value=metrics_callback.metrics_history['eval_loss'][-1],
        iteration=trainer.state.global_step
    )

# Добавить output directory как артефакт
task.upload_artifact(name="fine_tuned_model", artifact_object=OUTPUT_DIR)

print(f"\n✓ Training complete!")
print(f"✓ Model: {MODEL_NAME}")
print(f"✓ Total samples: {len(dataset)}")
print(f"✓ Languages: {', '.join(TARGET_LANGUAGES)}")
print(f"✓ ClearML task: {task.name} (ID: {task.id})")
print("="*80)
