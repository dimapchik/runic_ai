import pandas as pd
from unsloth import FastLanguageModel
from datasets import Dataset, load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# >>> ClearML
from clearml import Task
from transformers.integrations import ClearMLCallback
# <<< ClearML

# ======================
# CONFIG
# ======================

# Путь к параллельному корпусу (из parsers/collect_parallel_data.py)
DATA_PATH = "parallel_corpus.csv"

# Язык целевых переводов для обучения
# LANGUAGE_TO_ID = {'en': 0, 'de': 1, 'sv': 2, 'unknown': 3}
TARGET_LANGUAGE = "en"  # английский для перевода
TARGET_LANGUAGE_ID = 0  # соответствующий ID

# Путь для сохранения модели
OUTPUT_DIR = "./qwen3.5-finetuned"

# ======================
# ClearML task init
# ======================
task = Task.init(
    project_name="runic_ai",
    task_name="qwen3.5-0.8b-lora-sft",
)

# Шаг 1: Загрузка модели и токенизатора
model_name = "Qwen/Qwen3.5-0.8B"  # замените на нужную модель
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)

# Шаг 2: Подготовка датасета
def load_parallel_corpus(path, target_language="en"):
    """
    Load parallel corpus from CSV and filter by target language.
    
    Args:
        path: Path to CSV file (parallel_corpus.csv)
        target_language: Target language code ('en', 'de', 'sv')
    
    Returns:
        Dataset: HuggingFace Dataset with formatted examples
    """
    df = pd.read_csv(path)
    
    # Filter by target language
    df = df[df['language'] == target_language]
    
    # Create instruction/input/output format
    examples = {
        "instruction": ["Translate the following runic transliteration to English."] * len(df),
        "input": df["source_text"].tolist(),
        "output": df["target_translation"].tolist(),
        "language_id": [TARGET_LANGUAGE_ID] * len(df)
    }
    
    return Dataset.from_dict(examples)

dataset = load_parallel_corpus(DATA_PATH, target_language=TARGET_LANGUAGE)

def formatting_prompts_func(examples):
    """
    Format examples for SFT training.
    Creates prompt-response pairs for translation task.
    """
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}
"""
        texts.append(text)
    
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# Шаг 3: Настройка параметров LoRA
lora_cfg = dict(
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
)
model = FastLanguageModel.get_peft_model(model, **lora_cfg)

# Шаг 4: Настройка параметров обучения
training_cfg = dict(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="constant",
    num_train_epochs=3,
    save_steps=100,
    logging_steps=10,
    fp16=True,
    push_to_hub=False,

    # >>> ClearML: включаем репортинг метрик из Trainer
    report_to=["clearml"],
    # (опционально) удобное имя запуска
    run_name=task.id,
    # <<< ClearML
)
training_args = TrainingArguments(**training_cfg)

# Прокинем конфиг в ClearML (чтобы был виден в UI)
task.connect({
    "model_name": model_name,
    "data_path": DATA_PATH,
    "target_language": TARGET_LANGUAGE,
    "target_language_id": TARGET_LANGUAGE_ID,
    "lora": lora_cfg,
    "training": training_cfg,
})

# Шаг 5: Создание тренера и обучение
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,

    # >>> ClearML callback (доп. гарантия логирования)
    callbacks=[ClearMLCallback()],
    # <<< ClearML
)

trainer.train()

# Шаг 6: Сохранение обученной модели
trainer.save_model(OUTPUT_DIR)

# (опционально) залить папку с моделью как артефакт
task.upload_artifact(name="finetuned_model_dir", artifact_object=OUTPUT_DIR)

# Логирование статистики датасета в ClearML
task.upload_artifact(
    name="dataset_stats",
    artifact_object={
        "total_samples": len(dataset),
        "target_language": TARGET_LANGUAGE,
        "language_id": TARGET_LANGUAGE_ID,
        "source_file": DATA_PATH
    }
)

task.close()

print(f"\n✓ Training complete! Model saved to {OUTPUT_DIR}")
print(f"✓ Total samples used: {len(dataset)}")
