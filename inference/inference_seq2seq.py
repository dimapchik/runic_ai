"""
Seq2Seq Model Inference Script for Runic AI
============================================
Запуск Seq2Seq моделей (NLLB-200, M2M100, MADLAD) без Ollama.
Прямая загрузка моделей через transformers.

Особенности:
- Поддержка NLLB-200, M2M100, MADLAD-400
- Генерация с указанием целевого языка через forced_bos_token_id
- Генерация переводов на тестовой выборке
- Расчёт метрик (BLEU, chrF, chrF++, ROUGE-L, METEOR)
- Сохранение результатов и графиков
"""

import pandas as pd
from tqdm import tqdm
import sacrebleu
import json
import random
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from rouge import Rouge
import os
import matplotlib.pyplot as plt

# Альтернативный расчёт METEOR без NLTK (используем evaluate library)
try:
    from evaluate import load as load_metric
    meteor_metric = load_metric("meteor")
    METEOR_AVAILABLE = True
    print("✓ METEOR metric loaded (evaluate library)")
except Exception as e:
    print(f"⚠️  METEOR not available: {e}")
    print("   Install with: pip install evaluate")
    METEOR_AVAILABLE = False
    meteor_metric = None

# ======================
# CONFIG - MODEL SETTINGS
# ======================

# Список Seq2Seq моделей для оценки
# Каждая модель: {"model_path": path, "adapter_name": name, "model_type": "nllb"|"m2m100"|"madlad"}
#
# model_type определяет формат языковых токенов:
# - "nllb": eng_Latn, deu_Latn, swe_Latn
# - "m2m100": en, de, sv
# - "madlad": english, german, swedish (instruction-based)
#
MODELS_CONFIG = [
    {
        "model_path": "../nllb-200-3.3B-ft_multi-lang_20_epochs",
        "adapter_name": "nllb-200-3.3B-ft_multi-lang_20_epochs",
        "model_type": "nllb"  # NLLB-200 с языковыми токенами
    },
    # Пример M2M100:
    # {
    #     "model_path": "../m2m100-1.2B-ft",
    #     "adapter_name": "m2m100-1.2B-ft",
    #     "model_type": "m2m100"
    # },
    # Пример MADLAD:
    # {
    #     "model_path": "../madlad400-3b-mt-ft",
    #     "adapter_name": "madlad400-3b-mt-ft",
    #     "model_type": "madlad"
    # },
]

# Директория для сохранения результатов
RESULTS_DIR = "nllb-200-3.3B-ft_multi-lang_20_epochs"

# Директория для графиков (в корне проекта)
PLOTS_DIR = "../plots/nllb-200-3.3B-ft_multi-lang_20_epochs"

# Устройство для вычислений
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Путь к тестовому параллельному корпусу
DATA_PATH = "parallel_corpus_test.csv"

# Языки для инференса
TARGET_LANGUAGES = ["en", "de", "sv"]

# Маппинг языков на токены для разных моделей
LANGUAGE_TO_TOKEN = {
    # NLLB-200 токены (формат: {language}_{script})
    "nllb": {
        "en": "eng_Latn",
        "de": "deu_Latn",
        "sv": "swe_Latn",
    },
    # M2M100 токены (короткие коды)
    "m2m100": {
        "en": "en",
        "de": "de",
        "sv": "sv",
    },
    # MADLAD токены (полные названия)
    "madlad": {
        "en": "english",
        "de": "german",
        "sv": "swedish",
    },
}

# Источник для MADLAD (instruction-based формат)
SOURCE_LANGUAGE = "old norse"

# Параметры инференса
MIN_WORDS = 6  # минимальное количество слов в исходном тексте
NUM_SAMPLES_PER_LANGUAGE = 100  # количество сэмплов на язык
RANDOM_SEED = 42  # для воспроизводимости
MAX_LENGTH = 128  # максимум токенов для генерации
BATCH_SIZE = 1  # размер батча (1 для последовательной генерации)


# ======================
# MODEL LOADING
# ======================

def load_model_and_tokenizer(model_path, model_type="nllb"):
    """
    Загружает Seq2Seq модель и токенизатор.
    
    Args:
        model_path: Путь к модели
        model_type: Тип модели (nllb, m2m100, madlad)
    
    Returns:
        model, tokenizer: Загруженная модель и токенизатор
    """
    print(f"\n📥 Loading Seq2Seq model from {model_path}...")
    print(f"   Model type: {model_type}")
    print(f"   Device: {DEVICE}")
    
    # Загружаем модель
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        device_map=DEVICE,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
    )
    
    model.eval()
    
    # Загружаем токенизатор
    print(f"📖 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    
    # Для M2M100: установить язык источника по умолчанию
    if hasattr(tokenizer, 'src_lang'):
        tokenizer.src_lang = "en"
        print(f"✓ Source language set to: {tokenizer.src_lang}")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Model loaded!")
    return model, tokenizer


# ======================
# DATA LOADING
# ======================

def load_data(path, target_language="en"):
    """
    Загружает параллельный корпус из CSV файла.
    
    Args:
        path: Путь к CSV файлу
        target_language: Целевой язык для фильтрации
    
    Returns:
        sources, references: Списки исходных текстов и референсных переводов
    """
    df = pd.read_csv(path)
    
    # Фильтруем по целевому языку
    if target_language:
        df = df[df['language'] == target_language]
    
    sources = []
    references = []
    
    for _, row in df.iterrows():
        src = row['source_text']
        tgt = row['target_translation']
        
        # Фильтр по количеству слов
        word_count = len(src.split())
        if word_count < MIN_WORDS:
            continue
        
        sources.append(src)
        references.append(tgt)

    return sources, references


# ======================
# TRANSLATION GENERATION
# ======================

def clean_translation_output(text):
    """
    Очищает вывод модели от лишних префиксов и шаблонов.
    """
    if not text:
        return ""
    
    text = text.strip()
    
    import re
    patterns_to_remove = [
        r"^English translation:\s*",
        r"^Translation:\s*",
        r"^Here is the translation:\s*",
        r"^The translation is:\s*",
        r"^Deutsche Übersetzung:\s*",
        r"^Übersetzung:\s*",
        r"^Svensk översättning:\s*",
        r"^Översättning:\s*",
        r"^```[\w]*\n",
        r"\n```$",
        r"^\*\*Translation:\*\*\s*",
        r"^\[Translation\]\s*",
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # Берём только первую строку если их несколько
    lines = text.split('\n')
    if len(lines) > 1:
        first_line = lines[0].strip()
        if len(first_line) > 10:
            text = first_line
    
    return text.strip()


def translate_with_model(model, tokenizer, text, target_lang="en", model_type="nllb"):
    """
    Генерирует перевод для одного текста используя Seq2Seq модель.
    
    Args:
        model: Seq2Seq модель
        tokenizer: Токенизатор
        text: Исходный текст
        target_lang: Целевой язык
        model_type: Тип модели (nllb, m2m100, madlad)
    
    Returns:
        translation: Сгенерированный перевод
    """
    # Получить токен целевого языка
    lang_tokens = LANGUAGE_TO_TOKEN.get(model_type, LANGUAGE_TO_TOKEN["nllb"])
    target_token = lang_tokens.get(target_lang, lang_tokens["en"])
    
    if model_type == "madlad":
        # MADLAD использует instruction-based формат
        # Input: "translate old norse to english: {text}"
        input_text = f"translate {SOURCE_LANGUAGE} to {target_token}: {text}"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    else:
        # NLLB и M2M100 используют простой формат с языковым токеном
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        if model_type in ["nllb", "m2m100"]:
            # NLLB и M2M100: использовать forced_bos_token_id для указания целевого языка
            forced_token_id = tokenizer.lang_code_to_id.get(target_token, tokenizer.lang_code_to_id["eng_Latn"])
            
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_token_id,  # ← Ключевой параметр для NLLB/M2M100
                max_length=MAX_LENGTH,
                num_beams=5,
                early_stopping=True,
            )
        else:
            # MADLAD: стандартная генерация
            outputs = model.generate(
                **inputs,
                max_length=MAX_LENGTH,
                num_beams=5,
                early_stopping=True,
            )
    
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    translation = clean_translation_output(translation)
    return translation


def evaluate_samples(sample_sources, sample_references, model, tokenizer, model_name, model_type, target_lang="en"):
    """
    Оценивает модель на сэмплах для конкретного языка.
    
    Returns:
        results: Список словарей с результатами
    """
    results = []
    
    for idx, (source, reference) in enumerate(tqdm(
        zip(sample_sources, sample_references),
        total=len(sample_sources),
        desc=f"  [{target_lang}] [{model_name}]"
    )):
        prediction = translate_with_model(model, tokenizer, source, target_lang, model_type)
        
        row = {
            "Sample_ID": idx + 1,
            "Source": source,
            "Reference": reference,
            "Target_Language": target_lang,
            "Model": model_name,
            "Prediction": prediction
        }
        results.append(row)
    
    return results


# ======================
# METRICS CALCULATION
# ======================

def calculate_metrics(predictions, references, model_name, target_lang):
    """
    Вычисляет все метрики для модели на конкретном языке.
    """
    # Фильтруем пустые предсказания
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p and p.strip()]
    
    if not valid_pairs:
        print(f"⚠️  {model_name}: No valid predictions")
        return {
            "Model": model_name,
            "Target_Language": target_lang,
            "BLEU": 0.0,
            "chrF": 0.0,
            "chrF++": 0.0,
            "ROUGE-L": 0.0,
            "METEOR": 0.0,
            "Valid_Predictions": 0
        }
    
    valid_predictions = [p for p, r in valid_pairs]
    valid_references = [r for p, r in valid_pairs]
    
    # BLEU
    bleu = sacrebleu.corpus_bleu(valid_predictions, [valid_references])
    
    # chrF++
    chrf_plusplus = sacrebleu.corpus_chrf(valid_predictions, [valid_references], word_order=2)
    chrf = sacrebleu.corpus_chrf(valid_predictions, [valid_references], word_order=1)
    
    # ROUGE-L
    rouge_l = 0.0
    try:
        rouge = Rouge()
        non_empty_pairs = [(p, r) for p, r in zip(valid_predictions, valid_references) 
                          if p.strip() and len(p.strip()) > 0]
        if non_empty_pairs:
            non_empty_preds = [p for p, r in non_empty_pairs]
            non_empty_refs = [r for p, r in non_empty_pairs]
            rouge_scores = rouge.get_scores(non_empty_preds, non_empty_refs, avg=True)
            rouge_l = rouge_scores['rouge-l']['f'] * 100
    except Exception as e:
        print(f"⚠️  ROUGE calculation error: {e}")
        rouge_l = 0.0
    
    # METEOR
    meteor = 0.0
    if METEOR_AVAILABLE and meteor_metric:
        try:
            meteor_result = meteor_metric.compute(predictions=valid_predictions, references=valid_references)
            meteor = meteor_result["meteor"] * 100
        except Exception as e:
            print(f"⚠️  METEOR calculation error: {e}")
            meteor = 0.0
    
    return {
        "Model": model_name,
        "Target_Language": target_lang,
        "BLEU": round(bleu.score, 2),
        "chrF": round(chrf.score, 2),
        "chrF++": round(chrf_plusplus.score, 2),
        "ROUGE-L": round(rouge_l, 2),
        "METEOR": round(meteor, 2),
        "Valid_Predictions": len(valid_predictions)
    }


# ======================
# PLOTTING
# ======================

def plot_metrics_for_language(df_metrics, target_lang, output_dir="plots"):
    """
    Строит графики BLEU и chrF++ для конкретного языка.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if df_metrics.empty:
        print(f"⚠️  No metrics to plot for {target_lang}")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    models = df_metrics['Model'].tolist()
    bleu_scores = df_metrics['BLEU'].tolist()
    chrf_scores = df_metrics['chrF++'].tolist()
    
    # BLEU plot
    bars1 = ax1.bar(models, bleu_scores, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('BLEU Score', fontsize=12)
    ax1.set_title(f'BLEU Scores - {target_lang.upper()}', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim(0, max(bleu_scores) * 1.2 if bleu_scores else 10)
    
    for bar, score in zip(bars1, bleu_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{score:.1f}', ha='center', va='bottom', fontsize=9)
    
    # chrF++ plot
    bars2 = ax2.bar(models, chrf_scores, color='coral', alpha=0.8)
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('chrF++ Score', fontsize=12)
    ax2.set_title(f'chrF++ Scores - {target_lang.upper()}', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(0, max(chrf_scores) * 1.2 if chrf_scores else 10)
    
    for bar, score in zip(bars2, chrf_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{score:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f"metrics_comparison_{target_lang}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Metrics plot saved to {output_file}")


def plot_aggregated_comparison(df_metrics, output_dir="plots"):
    """
    Строит агрегированные метрики по всем языкам для всех моделей.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Агрегируем по моделям (усредняем по языкам)
    agg_metrics = df_metrics.groupby('Model').agg({
        'BLEU': 'mean',
        'chrF': 'mean',
        'chrF++': 'mean',
        'ROUGE-L': 'mean',
        'METEOR': 'mean'
    }).reset_index()
    
    if agg_metrics.empty:
        print("⚠️  No aggregated metrics to plot")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    metrics_to_plot = [
        ('BLEU', 'steelblue'),
        ('chrF', 'coral'),
        ('chrF++', 'green'),
        ('ROUGE-L', 'purple'),
        ('METEOR', 'orange')
    ]
    
    models = agg_metrics['Model'].tolist()
    
    for idx, (metric_name, color) in enumerate(metrics_to_plot):
        ax = axes[idx]
        scores = agg_metrics[metric_name].tolist()
        
        bars = ax.bar(models, scores, color=color, alpha=0.8)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(f'{metric_name} Score (avg)', fontsize=12)
        ax.set_title(f'{metric_name} - Average Across Languages', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, max(scores) * 1.2 if scores else 10)
        
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{score:.1f}', ha='center', va='bottom', fontsize=9)
    
    axes[5].axis('off')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "metrics_aggregated_comparison.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Aggregated metrics plot saved to {output_file}")


# ======================
# MAIN
# ======================

def main():
    print("=" * 80)
    print("Seq2Seq Model Inference - Runic Text Translation Evaluation")
    print("=" * 80)
    
    # Проверка устройства
    print(f"\n🖥️  Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    elif DEVICE == "mps":
        print(f"   ✓ MPS (Metal Performance Shaders) enabled")
        print(f"   ✓ Apple Silicon acceleration active")
    else:
        print(f"   ⚠️  Running on CPU (slow!)")
        print(f"   Install PyTorch with CUDA/MPS support for faster inference")
    
    print(f"\n📋 Models to evaluate: {len(MODELS_CONFIG)}")
    for i, cfg in enumerate(MODELS_CONFIG, 1):
        print(f"   {i}. {cfg['adapter_name']} ({cfg['model_type']})")
    
    # Загрузка данных
    print("\n" + "=" * 80)
    print("📂 Loading dataset...")
    print("=" * 80)
    
    all_language_data = {}
    for target_lang in TARGET_LANGUAGES:
        sources, references = load_data(DATA_PATH, target_language=target_lang)
        if len(sources) == 0:
            print(f"⚠️  No data available for {target_lang}, skipping...")
            continue
        
        # Выбираем случайные сэмплы
        num_samples = min(NUM_SAMPLES_PER_LANGUAGE, len(sources))
        random.seed(RANDOM_SEED)
        sample_indices = random.sample(range(len(sources)), num_samples)
        sample_sources = [sources[i] for i in sample_indices]
        sample_references = [references[i] for i in sample_indices]
        
        all_language_data[target_lang] = {
            "sources": sample_sources,
            "references": sample_references
        }
        print(f"✓ Loaded {num_samples} samples for {target_lang.upper()}")
    
    # Обработка каждой модели
    all_results = []
    all_metrics = []
    
    for model_idx, model_cfg in enumerate(MODELS_CONFIG, 1):
        print("\n" + "=" * 80)
        print(f"🤖 EVALUATING MODEL {model_idx}/{len(MODELS_CONFIG)}: {model_cfg['adapter_name']}")
        print("=" * 80)
        
        # Загрузка модели
        print(f"\n📥 Loading model...")
        print(f"   Model path: {model_cfg['model_path']}")
        print(f"   Model type: {model_cfg['model_type']}")
        print("-" * 80)
        
        try:
            model, tokenizer = load_model_and_tokenizer(
                model_cfg['model_path'],
                model_cfg['model_type']
            )
        except Exception as e:
            print(f"❌ Error loading model {model_cfg['adapter_name']}: {e}")
            print("   Skipping this model and continuing with next...")
            continue
        
        # Обработка каждого языка
        model_results = []
        model_metrics = []
        
        for target_lang in TARGET_LANGUAGES:
            if target_lang not in all_language_data:
                continue
            
            lang_data = all_language_data[target_lang]
            sample_sources = lang_data["sources"]
            sample_references = lang_data["references"]
            
            print("\n" + "-" * 80)
            print(f"🌍 Processing language: {target_lang.upper()}")
            print("-" * 80)
            
            # Генерация переводов
            print(f"\n🔄 Running inference on {len(sample_sources)} samples...")
            results_list = evaluate_samples(
                sample_sources,
                sample_references,
                model,
                tokenizer,
                model_cfg['adapter_name'],
                model_cfg['model_type'],
                target_lang
            )
            
            # Вычисление метрик
            print(f"\n{'=' * 80}")
            print(f"=== METRICS FOR {target_lang.upper()} ===")
            print(f"{'=' * 80}")
            
            predictions = [r.get("Prediction", "") for r in results_list]
            
            metrics = calculate_metrics(
                predictions,
                sample_references,
                model_cfg['adapter_name'],
                target_lang
            )
            
            # Печать метрик
            if METEOR_AVAILABLE:
                print(f"✓ {model_cfg['adapter_name']:35} | BLEU: {metrics['BLEU']:6.2f} | chrF: {metrics['chrF']:6.2f} | chrF++: {metrics['chrF++']:6.2f} | ROUGE-L: {metrics['ROUGE-L']:6.2f} | METEOR: {metrics['METEOR']:6.2f}")
            else:
                print(f"✓ {model_cfg['adapter_name']:35} | BLEU: {metrics['BLEU']:6.2f} | chrF: {metrics['chrF']:6.2f} | chrF++: {metrics['chrF++']:6.2f} | ROUGE-L: {metrics['ROUGE-L']:6.2f}")
            
            # Сохранение результатов
            model_results.extend(results_list)
            model_metrics.append(metrics)
        
        # Сохранение всех результатов модели
        all_results.extend(model_results)
        all_metrics.extend(model_metrics)
        
        # Сохранение промежуточных результатов
        print(f"\n💾 Saving results for {model_cfg['adapter_name']}...")
        model_output_dir = os.path.join(RESULTS_DIR, model_cfg['adapter_name'])
        os.makedirs(model_output_dir, exist_ok=True)
        
        df_model_results = pd.DataFrame(model_results)
        df_model_metrics = pd.DataFrame(model_metrics)
        
        df_model_results.to_csv(
            os.path.join(model_output_dir, "translation_samples.csv"),
            index=False,
            encoding='utf-8'
        )
        df_model_metrics.to_csv(
            os.path.join(model_output_dir, "metrics_summary.csv"),
            index=False,
            encoding='utf-8'
        )
        print(f"✓ Results saved to {model_output_dir}/")
        
        # Построение графиков для этой модели
        model_plots_dir = os.path.join(PLOTS_DIR, model_cfg['adapter_name'])
        os.makedirs(model_plots_dir, exist_ok=True)
        
        for target_lang in TARGET_LANGUAGES:
            lang_metrics = [m for m in model_metrics if m['Target_Language'] == target_lang]
            if lang_metrics:
                df_lang = pd.DataFrame([lang_metrics[0]])
                plot_metrics_for_language(df_lang, target_lang, output_dir=model_plots_dir)
        
        # Очистка памяти GPU
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE == "mps":
            torch.mps.empty_cache()
        
        # Удаление модели для освобождения памяти
        del model
        del tokenizer
    
    # Объединение всех результатов
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_metrics = pd.DataFrame(all_metrics)
        
        # Построение агрегированных графиков
        print(f"\n📊 Generating aggregated comparison plots...")
        plot_aggregated_comparison(df_metrics, output_dir=PLOTS_DIR)
        
        # Сохранение общих результатов
        os.makedirs(PLOTS_DIR, exist_ok=True)
        
        df_results.to_csv(
            os.path.join(PLOTS_DIR, "all_translation_samples.csv"),
            index=False,
            encoding='utf-8'
        )
        df_metrics.to_csv(
            os.path.join(PLOTS_DIR, "all_metrics_summary.csv"),
            index=False,
            encoding='utf-8'
        )
        
        print(f"\n✓ All results saved to {PLOTS_DIR}/")
        
        # Печать сводки
        print("\n" + "=" * 80)
        print("=== SUMMARY ===")
        print("=" * 80)
        
        for model_name in df_metrics['Model'].unique():
            print(f"\n🤖 {model_name}:")
            for lang in TARGET_LANGUAGES:
                lang_metrics = df_metrics[
                    (df_metrics['Model'] == model_name) &
                    (df_metrics['Target_Language'] == lang)
                ]
                if not lang_metrics.empty:
                    bleu = lang_metrics['BLEU'].values[0]
                    chrf = lang_metrics['chrF++'].values[0]
                    print(f"   {lang.upper()}: BLEU = {bleu:.2f}, chrF++ = {chrf:.2f}")
        
        # Лучшие модели
        print("\n\n" + "=" * 80)
        print("=== BEST MODEL BY AVG BLEU ===")
        print("=" * 80)
        
        avg_bleu = df_metrics.groupby('Model')['BLEU'].mean().reset_index()
        avg_bleu = avg_bleu.sort_values('BLEU', ascending=False)
        
        if not avg_bleu.empty:
            best_model = avg_bleu.iloc[0]['Model']
            best_bleu = avg_bleu.iloc[0]['BLEU']
            print(f"🏆 {best_model}: AVG BLEU = {best_bleu:.2f}")
            
            print("\nRanking:")
            for idx, row in avg_bleu.iterrows():
                print(f"   {idx+1}. {row['Model']}: {row['BLEU']:.2f}")
        
        print("\n" + "=" * 80)
        print("✓ Evaluation complete!")
        print(f"✓ Results directory: {RESULTS_DIR}")
        print(f"✓ Plots directory: {PLOTS_DIR}")
        print("=" * 80)
    else:
        print("\n⚠️  No results generated. Check model paths and try again.")


if __name__ == "__main__":
    main()
