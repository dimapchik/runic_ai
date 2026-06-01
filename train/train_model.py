import pandas as pd
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import json
from transformers import TrainerCallback

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
    task_name="qwen3.5-4b-lora-sft-multi-lang",
    task_type=Task.TaskTypes.training,
    tags=["qwen3.5", "lora", "sft", "runic", "multi-lang"],
)

# Маппинг языков и их ID
LANGUAGE_TO_ID = {'en': 0, 'de': 1, 'sv': 2}  # ← добавь языки и их ID по необходимости

# Языки для обучения — можно указать несколько!
TARGET_LANGUAGES = ["en", "de", "sv"]  # ← Укажи нужные языки

# Промпты для каждого языка (детальные, как в inference_lora.py)
# Формат соответствует inference промптам для консистентности training/inference
TRANSLATION_PROMPTS = {
    "en": """### Instruction:
Translate the following Old Norse/Runic transliteration to English.
Provide ONLY the English translation text. Do not include any labels, explanations, or additional text.

### Input:
{text}

### Response:
""",
    "de": """### Instruction:
Übersetze die folgende altnordische/Runen-Transliteration ins Deutsche.
Gib NUR die deutsche Übersetzung aus. Keine Labels, Erklärungen oder zusätzlichen Text.

### Input:
{text}

### Response:
""",
    "sv": """### Instruction:
Översätt följande fornnordiska/runtranslitteration till svenska.
Ange ENDAST den svenska översättningstexten. Inga etiketter, förklaringar eller extra text.

### Input:
{text}

### Response:
"""
}

# Короткие инструкции для логирования
INSTRUCTION_DESCRIPTIONS = {
    "en": "Translate the following Old Norse/Runic transliteration to English.",
    "de": "Übersetze die folgende altnordische/Runen-Transliteration ins Deutsche.",
    "sv": "Översätt följande fornnordiska/runtranslitteration till svenska.",
}

# Путь к параллельному корпусу (из parsers/collect_parallel_data.py)
# В контейнере файлы копируются в /train_model/
# Используем train версию (test выделен для финальной оценки)
DATA_PATH = "/train_model/parallel_corpus_train.csv"

# Путь для сохранения модели
OUTPUT_DIR = "/train_model/model_trained"

# Custom callback для логирования метрик
class MetricsLoggerCallback(TrainerCallback):
    """Callback для логирования train/eval метрик в JSON файл"""
    
    def __init__(self, log_path="/train_model/training_metrics.json"):
        self.log_path = log_path
        self.metrics_history = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
            "grad_norm": [],
            "mean_token_accuracy": [],
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
            if "mean_token_accuracy" in logs:
                self.metrics_history["mean_token_accuracy"].append(logs["mean_token_accuracy"])
            if "epoch" in logs:
                self.metrics_history["epoch"].append(logs["epoch"])
            
            # Печатаем в красивом формате
            print("\n" + "="*60)
            print(f"📊 METRICS (step {state.global_step})")
            print("="*60)
            print(f"  Epoch:           {logs.get('epoch', 'N/A')}")
            print(f"  Loss (train):    {logs.get('loss', 'N/A')}")
            print(f"  Loss (eval):     {logs.get('eval_loss', 'N/A')}")
            print(f"  Accuracy:        {logs.get('mean_token_accuracy', 'N/A')}")
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

# Шаг 1: Загрузка модели и токенизатора
# Модель скачивается в /models/ через model_registry в .ml-job-preset.yml
# A100 80GB позволяет загрузить модель без 4-битного квантования
model_name = "/models/"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda:0",          # ← явно указываем GPU
    local_files_only=True,
    trust_remote_code=True,
    dtype=torch.float16,    # ← новый API (torch_dtype устарел)
)
print("Модель загружена, прообразую модель")

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
)

# Добавляем pad_token если его нет
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Подготовка модели для LoRA обучения (без квантования)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
print("загружаю датасет")

# Шаг 2: Подготовка датасета
def load_parallel_corpus_multi(path, target_languages):
    """
    Load parallel corpus from CSV and filter by multiple target languages.
    Returns dataset with instruction in target language.
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
        "instruction": [],
        "input": [],
        "output": [],
        "target_language": []
    }
    
    for _, row in df.iterrows():
        lang = row['language']
        src = row['source_text']
        tgt = row['target_translation']
        
        # Get full prompt in target language
        prompt_template = TRANSLATION_PROMPTS.get(lang, TRANSLATION_PROMPTS["en"])
        # Формируем полный промпт с input текстом
        full_prompt = prompt_template.format(text=src)
        
        examples["instruction"].append(full_prompt)
        examples["input"].append(src)
        examples["output"].append(tgt)
        examples["target_language"].append(lang)
    
    return Dataset.from_dict(examples)

dataset = load_parallel_corpus_multi(DATA_PATH, target_languages=TARGET_LANGUAGES)


def formatting_prompts_func(example):
    """
    Format examples for SFT training.
    Returns a single formatted text string (not batched).
    
    Примечание: instruction уже содержит полный промпт с input,
    поэтому добавляем только Response с output.
    """
    # instruction уже содержит полный промпт включая ### Input: {text}
    # Поэтому просто добавляем output в конце
    instruction = example["instruction"]  # Полный промпт с ### Instruction, ### Input, ### Response:
    output = example["output"]
    
    # Формат: instruction уже содержит "### Response:\n" в конце, добавляем output
    text = f"{instruction}{output}\n"
    
    return text  # ← возвращаем строку, не dict

# Разделение на train и val
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

print(f"\nдатасет загружен")
print(f"train samples: {len(train_dataset)}, val samples: {len(val_dataset)}")

# Распределение по языкам в train
train_df = pd.DataFrame(train_dataset)
print(f"\n📊 Train distribution:")
for lang in TARGET_LANGUAGES:
    lang_count = len(train_df[train_df['target_language'] == lang])
    print(f"   {lang.upper()}: {lang_count} samples ({lang_count/len(train_df)*100:.1f}%)")

print("настраиваю конфиги обучения")

# Шаг 3: Настройка параметров LoRA
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_cfg)

# Шаг 4: Настройка параметров обучения
# Оптимизировано для A100 80GB — максимальная утилизация GPU
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,      # ↑ увеличили для загрузки GPU
    gradient_accumulation_steps=2,      # ↓ уменьшили (effective batch = 8)
    learning_rate=1e-4,
    max_grad_norm=1.0,
    gradient_checkpointing=False,       # ← отключили для скорости (A100 80GB хватит)
    lr_scheduler_type="cosine",
    warmup_steps=100,
    num_train_epochs=20,
    save_steps=100,
    logging_steps=10,                    # ← логирование каждый шаг (было 10)
    eval_strategy="steps",              # ← eval каждые eval_steps (новое имя в transformers 4.40+)
    eval_steps=200,                     # ← каждые 200 шагов
    eval_on_start=True,                   # ← eval в начале обучения
    fp16=True,
    save_total_limit=2,
    dataloader_pin_memory=True,
)

# Прокинем конфиг в ClearML
task.connect({
    "model_name": model_name,
    "data_path": DATA_PATH,
    "target_languages": TARGET_LANGUAGES,
    "language_to_id": LANGUAGE_TO_ID,
    "translation_prompts": TRANSLATION_PROMPTS,
    "instruction_descriptions": INSTRUCTION_DESCRIPTIONS,
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
        "max_grad_norm": training_args.max_grad_norm,
    }
})

# Шаг 5: Создание тренера и обучение
# SFTTrainer автоматически обрабатывает токенизацию и создание labels
# formatting_func применяется к датасету перед токенизацией

print("конфиги настроены, начинаю настраивать тренера")

# В новых версиях SFTTrainer tokenizer передаётся через processing_class
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,           # ← tokenizer через processing_class
    formatting_func=formatting_prompts_func,  # ← применяется к данным
    callbacks=[metrics_callback, ClearMLCallback()],  # ← ClearML + кастомный callback
)
print("тренер создан, начинаю обучение модели")
trainer.train()
print("обучение завершено, сохраняю модель")

# Шаг 6: Сохранение обученной модели
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)  # сохраняем также токенизатор

# Сохраняем финальные метрики
print("\n" + "="*60)
print("📊 FINAL TRAINING METRICS")
print("="*60)
print(f"  Total steps:     {trainer.state.global_step}")
print(f"  Final epoch:     {trainer.state.epoch:.4f}")
print(f"  Train loss:      {metrics_callback.metrics_history['train_loss'][-1]:.4f}")
if metrics_callback.metrics_history['eval_loss']:
    print(f"  Eval loss:       {metrics_callback.metrics_history['eval_loss'][-1]:.4f}")
print(f"  Final accuracy:  {metrics_callback.metrics_history['mean_token_accuracy'][-1]:.4f}")
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
task.get_logger().report_scalar(
    title="Final Metrics",
    series="accuracy",
    value=metrics_callback.metrics_history['mean_token_accuracy'][-1],
    iteration=trainer.state.global_step
)

# Добавляем output directory как артефакт
task.upload_artifact(name="fine_tuned_model", artifact_object=OUTPUT_DIR)

print(f"\n✓ Training complete! Model saved to {OUTPUT_DIR}")
print(f"✓ Total samples used: {len(dataset)}")
print(f"✓ Languages: {', '.join(TARGET_LANGUAGES)}")
print(f"✓ ClearML task: {task.name} (ID: {task.id})")
