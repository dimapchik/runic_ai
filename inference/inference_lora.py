import pandas as pd
from tqdm import tqdm
import sacrebleu
import json
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rouge import Rouge
import os

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
# CONFIG - MULTIPLE MODELS
# ======================

# Список моделей для оценки
# Каждая модель: {"base_model": path, "lora_adapter": path, "adapter_name": name}
# 
# Чтобы перезаписать только одну модель, оставь только её в списке.
# Остальные модели будут загружены из существующих CSV файлов.
#
MODELS_CONFIG = [
    {
        "base_model": "../gemma-2-2b-it",
        "lora_adapter": "",
        "adapter_name": "gemma-2-2b-it"
    },
]

# Директории с существующими результатами других моделей (будут загружены автоматически)
EXISTING_MODELS = [
    # "qwen3.5-0.8B-ft_en",
    # "qwen3.5-0.8B-ft_sv",
]

# Директория для графиков (в корне проекта)
PLOTS_DIR = "../plots/gemma-2-2b-it"

# Устройство для вычислений (MPS для Apple Silicon)
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# Путь к твоему параллельному файлу (test для финальной оценки)
# Файл лежит в той же папке inference/
DATA_PATH = "parallel_corpus_test.csv"

# Языки для инференса
TARGET_LANGUAGES = ["en", "de", "sv"]

# Промпт для каждого языка с чётким форматом вывода
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

MIN_WORDS = 6  # минимальное количество слов в исходном тексте
NUM_SAMPLES_PER_LANGUAGE = 100  # количество пар для инференса на каждый язык
RANDOM_SEED = 42  # для воспроизводимости
MAX_NEW_TOKENS = 256  # максимум токенов для генерации
TEMPERATURE = 0.3  # температура для генерации


# ======================
# LOAD MODEL
# ======================

def load_model_and_tokenizer(base_model_path, lora_adapter_path, adapter_name):
    """
    Загружает базовую модель и применяет LoRA адаптер
    """
    print(f"📥 Loading base model from {base_model_path}...")
    print(f"   Device: {DEVICE}")
    
    # Загружаем базовую модель
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map=DEVICE,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
    )
    
    # Применяем LoRA адаптер
    print(f"🔧 Applying LoRA adapter from {lora_adapter_path}...")
    model = PeftModel.from_pretrained(
        model,
        lora_adapter_path,
        device_map=DEVICE,
    )
    
    # Merge LoRA weights в базовую модель для скорости
    print("🔗 Merging LoRA weights...")
    model = model.merge_and_unload()
    
    model.eval()
    
    # Загружаем токенизатор
    print(f"📖 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Model '{adapter_name}' loaded!")
    return model, tokenizer


# ======================
# LOAD DATA
# ======================

def load_data(path, target_language="en"):
    """
    Load data from CSV file with parallel corpus.
    """
    df = pd.read_csv(path)
    
    # Filter by target language
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
# GENERATE TRANSLATIONS
# ======================

def clean_translation_output(text):
    """
    Clean up model output to extract just the translation.
    """
    if not text:
        return ""
    
    text = text.strip()
    
    # Remove common patterns
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
    
    # Take only first line if multiple
    lines = text.split('\n')
    if len(lines) > 1:
        first_line = lines[0].strip()
        if len(first_line) > 10:
            text = first_line
    
    return text.strip()


def translate_with_model(model, tokenizer, text, target_lang="en"):
    """
    Translate text using the loaded model
    """
    prompt = TRANSLATION_PROMPTS.get(target_lang, TRANSLATION_PROMPTS["en"]).format(text=text)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=TEMPERATURE > 0.0,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_length:]
    raw_translation = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    translation = clean_translation_output(raw_translation)
    return translation


def evaluate_samples(sample_sources, sample_references, model, tokenizer, model_name, target_lang="en"):
    """
    Evaluate model on samples for a specific language
    """
    results = []
    
    for idx, (source, reference) in enumerate(zip(sample_sources, sample_references)):
        print(f"  [{target_lang}] [{model_name}] Translating sample {idx+1}/{len(sample_sources)}...")
        prediction = translate_with_model(model, tokenizer, source, target_lang)
        
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
    Calculate all metrics for a model on a specific language
    """
    # Фильтруем пустые предсказания и соответствующие референсы
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
    
    # Вычисляем все метрики
    bleu = sacrebleu.corpus_bleu(valid_predictions, [valid_references])
    chrf_plusplus = sacrebleu.corpus_chrf(valid_predictions, [valid_references], word_order=2)
    chrf = sacrebleu.corpus_chrf(valid_predictions, [valid_references], word_order=1)
    
    # ROUGE (с обработкой пустых предсказаний)
    rouge_l = 0.0
    try:
        rouge = Rouge()
        # Фильтруем полностью пустые строки для ROUGE
        non_empty_pairs = [(p, r) for p, r in zip(valid_predictions, valid_references) if p.strip() and len(p.strip()) > 0]
        if non_empty_pairs:
            non_empty_preds = [p for p, r in non_empty_pairs]
            non_empty_refs = [r for p, r in non_empty_pairs]
            rouge_scores = rouge.get_scores(non_empty_preds, non_empty_refs, avg=True)
            rouge_l = rouge_scores['rouge-l']['f'] * 100  # конвертируем в проценты как BLEU
    except Exception as e:
        print(f"⚠️  ROUGE calculation error: {e}")
        rouge_l = 0.0
    
    # METEOR (альтернативный расчёт через evaluate library)
    meteor = 0.0
    if METEOR_AVAILABLE and meteor_metric:
        try:
            meteor_result = meteor_metric.compute(predictions=valid_predictions, references=valid_references)
            meteor = meteor_result["meteor"] * 100  # конвертируем в проценты
        except Exception as e:
            print(f"⚠️  METEOR calculation error: {e}")
            meteor = 0.0
    else:
        meteor = 0.0  # METEOR недоступен
    
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
# PLOT METRICS
# ======================

def plot_metrics_for_language(df_metrics, target_lang, output_dir="plots"):
    """
    Plot BLEU and chrF++ metrics for a specific language (multiple models comparison)
    """
    import matplotlib.pyplot as plt
    
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
    
    # Add value labels on bars
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
    
    # Add value labels on bars
    for bar, score in zip(bars2, chrf_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{score:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f"metrics_comparison_{target_lang}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Metrics plot saved to {output_file}")


def load_existing_model_results(model_dir, target_languages):
    """
    Load existing results from CSV files for a model that was already evaluated.
    Returns (results_list, metrics_list) or (None, None) if files don't exist.
    """
    samples_file = f"{model_dir}/translation_samples_lora.csv"
    metrics_file = f"{model_dir}/metrics_summary_lora.csv"
    
    if not os.path.exists(samples_file) or not os.path.exists(metrics_file):
        print(f"⚠️  No existing results found for {model_dir}")
        return None, None
    
    print(f"📂 Loading existing results for {model_dir}...")
    
    # Load metrics
    df_metrics = pd.read_csv(metrics_file)
    metrics_list = df_metrics.to_dict('records')
    
    # Load samples
    df_samples = pd.read_csv(samples_file)
    results_list = df_samples.to_dict('records')
    
    print(f"✓ Loaded {len(results_list)} samples and {len(metrics_list)} metrics for {model_dir}")
    return results_list, metrics_list


def plot_aggregated_comparison(df_metrics, output_dir="plots"):
    """
    Plot aggregated metrics across all languages for all models
    """
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Aggregate by model (average across languages)
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
        
        # Add value labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{score:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Hide empty subplot
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
    print("LoRA Fine-tuned Models - Runic Text Translation Evaluation (Multi-Model)")
    print("=" * 80)
    
    # Check device
    print(f"\n🖥️  Device: {DEVICE}")
    if DEVICE == "mps":
        print(f"   ✓ MPS (Metal Performance Shaders) enabled")
        print(f"   ✓ Apple Silicon acceleration active")
    elif DEVICE == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    else:
        print(f"   ⚠️  Running on CPU (slow!)")
        print(f"   Install PyTorch with MPS support for faster inference")
    
    print(f"\n📋 Models to evaluate: {len(MODELS_CONFIG)}")
    for i, cfg in enumerate(MODELS_CONFIG, 1):
        print(f"   {i}. {cfg['adapter_name']}")
    
    # Load data once for all models
    print("\n" + "=" * 80)
    print("📂 Loading dataset...")
    print("=" * 80)
    
    all_language_data = {}
    for target_lang in TARGET_LANGUAGES:
        sources, references = load_data(DATA_PATH, target_language=target_lang)
        if len(sources) == 0:
            print(f"⚠️  No data available for {target_lang}, skipping...")
            continue
        
        # Select random samples
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
    
    # Process each model
    all_results = []
    all_metrics = []
    
    # 1. First, load existing results from EXISTING_MODELS
    print("\n" + "=" * 80)
    print("📂 Loading existing model results...")
    print("=" * 80)
    
    for existing_model in EXISTING_MODELS:
        results, metrics = load_existing_model_results(existing_model, TARGET_LANGUAGES)
        if results is not None and metrics is not None:
            all_results.extend(results)
            all_metrics.extend(metrics)
            print(f"✓ Loaded existing results for {existing_model}")
        else:
            print(f"⚠️  Could not load existing results for {existing_model}")
    
    # 2. Now evaluate new models from MODELS_CONFIG
    for model_idx, model_cfg in enumerate(MODELS_CONFIG, 1):
        print("\n" + "=" * 80)
        print(f"🤖 EVALUATING MODEL {model_idx}/{len(MODELS_CONFIG)}: {model_cfg['adapter_name']}")
        print("=" * 80)
        
        # Load model
        print(f"\n📥 Loading model...")
        print(f"   Base model: {model_cfg['base_model']}")
        print(f"   LoRA adapter: {model_cfg['lora_adapter']}")
        print("-" * 80)
        
        try:
            model, tokenizer = load_model_and_tokenizer(
                model_cfg['base_model'],
                model_cfg['lora_adapter'],
                model_cfg['adapter_name']
            )
        except Exception as e:
            print(f"❌ Error loading model {model_cfg['adapter_name']}: {e}")
            print("   Skipping this model and continuing with next...")
            continue
        
        # Process each language
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
            
            # Evaluate
            print(f"\n🔄 Running inference on {len(sample_sources)} samples...")
            results_list = evaluate_samples(
                sample_sources,
                sample_references,
                model,
                tokenizer,
                model_cfg['adapter_name'],
                target_lang
            )
            
            # Calculate metrics
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
            
            # Print metrics
            if METEOR_AVAILABLE:
                print(f"✓ {model_cfg['adapter_name']:25} | BLEU: {metrics['BLEU']:6.2f} | chrF: {metrics['chrF']:6.2f} | chrF++: {metrics['chrF++']:6.2f} | ROUGE-L: {metrics['ROUGE-L']:6.2f} | METEOR: {metrics['METEOR']:6.2f}")
            else:
                print(f"✓ {model_cfg['adapter_name']:25} | BLEU: {metrics['BLEU']:6.2f} | chrF: {metrics['chrF']:6.2f} | chrF++: {metrics['chrF++']:6.2f} | ROUGE-L: {metrics['ROUGE-L']:6.2f}")
            
            # Store results
            model_results.extend(results_list)
            model_metrics.append(metrics)
            
            # Plot metrics for this language
            df_metrics_lang = pd.DataFrame([metrics])
            plot_metrics_for_language(df_metrics_lang, target_lang, output_dir=f"{PLOTS_DIR}/{model_cfg['adapter_name']}")
        
        # Store all results
        all_results.extend(model_results)
        all_metrics.extend(model_metrics)
        
        # Save intermediate results after each model
        print(f"\n💾 Saving intermediate results for {model_cfg['adapter_name']}...")
        os.makedirs(model_cfg['adapter_name'], exist_ok=True)
        
        df_model_results = pd.DataFrame(model_results)
        df_model_metrics = pd.DataFrame(model_metrics)
        
        df_model_results.to_csv(
            f"{model_cfg['adapter_name']}/translation_samples_lora.csv",
            index=False,
            encoding='utf-8'
        )
        df_model_metrics.to_csv(
            f"{model_cfg['adapter_name']}/metrics_summary_lora.csv",
            index=False,
            encoding='utf-8'
        )
        print(f"✓ Results saved to {model_cfg['adapter_name']}/")
        
        # Clear GPU memory
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE == "mps":
            torch.mps.empty_cache()
    
    # Combine all results
    df_results = pd.DataFrame(all_results)
    df_metrics = pd.DataFrame(all_metrics)
    
    # Display sample results
    print("\n" + "=" * 80)
    print("=== SAMPLE DETAILED RESULTS ===")
    print("=" * 80)
    
    for idx, row in df_results.head(10).iterrows():
        print(f"\n📌 Sample {row['Sample_ID']} ({row['Target_Language']}) - Model: {row['Model']}")
        print(f"   📖 Source: {row['Source'][:100]}...")
        print(f"   ✅ Reference: {row['Reference'][:100]}...")
        print(f"   🤖 Prediction: {row['Prediction'][:100] if row['Prediction'] else 'N/A'}...")
    
    # Save all results
    print(f"\n\n" + "=" * 80)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    all_samples_file = f"{PLOTS_DIR}/all_translation_samples.csv"
    all_metrics_file = f"{PLOTS_DIR}/all_metrics_summary.csv"
    
    df_results.to_csv(all_samples_file, index=False, encoding='utf-8')
    df_metrics.to_csv(all_metrics_file, index=False, encoding='utf-8')
    
    print(f"✓ All sample translations saved to {all_samples_file}")
    print(f"✓ All metrics summary saved to {all_metrics_file}")
    
    # Plot aggregated comparison
    print(f"\n📊 Generating aggregated comparison plots...")
    plot_aggregated_comparison(df_metrics, output_dir=PLOTS_DIR)
    
    # Print summary
    print(f"\n\n" + "=" * 80)
    print("=== SUMMARY ACROSS ALL MODELS AND LANGUAGES ===")
    print("=" * 80)
    
    # Group by model and language (all models, including existing)
    all_model_names = set(df_metrics['Model'].tolist())
    for model_name in all_model_names:
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
    
    # Best model summary
    print(f"\n\n" + "=" * 80)
    print("=== BEST MODEL BY AVG BLEU ===")
    print("=" * 80)
    
    # Get all unique model names (including existing)
    all_model_names = list(set(df_metrics['Model'].tolist()))
    
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
    print(f"✓ Results directory: {PLOTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
