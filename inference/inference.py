import pandas as pd
from tqdm import tqdm
import sacrebleu
import requests
import json
import os
import random
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
import nltk

# Скачайте необходимые ресурсы для METEOR
try:
    nltk.data.find('tokenizers/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# ======================
# CONFIG
# ======================

# Ollama API endpoint (по умолчанию localhost:11434)
OLLAMA_BASE_URL = "http://localhost:11434"

# Укажи свои модели для Ollama
# Пример: mistral, neural-chat, dolphin-mixtral, etc.
MODELS = [
    # local/light models
    # "mistral",               # 7B small
    # "translategemma:latest",        # translation-focused
    "gemma2:2b",                 # general-purpose (локальная модель из ./gemma-2-2b-it)
    # "gemma2:7b",                 # general-purpose
    # "phi4:mini",           # 3B, very fast for quick checks
    # cloud models for comparison
    # "minimax-m2.5:cloud",    # 13B-ish, higher quality
    # "qwen3.5:0.8b",
    # "qwen3.5:0.8b-ft",
    # "qwen3.5:2b",
    # "qwen3.5:4b",
    # "qwen3.5:9b"
    # "qwen3.5:cloud",
    # "dolphin-mixtral",     # local large model
    # еще модели...
]

# Путь к твоему параллельному файлу
# Формат: CSV с колонками: source_text, target_translation, source_name, source_file, language, language_id
# Используется ПРЕДВАРИТЕЛЬНО ОТФИЛЬТРОВАННЫЙ test датасет (из split_dataset.py)
# Примечание: файл лежит в той же папке inference/
DATA_PATH = "parallel_corpus_test.csv"

# Директория для сохранения результатов (аналогично inference_lora.py)
RESULTS_DIR = "gemma-2-2b-it"
PLOTS_DIR = "../plots/gemma-2-2b-it"

# Имя для сохранения результатов модели
MODEL_RESULTS_NAME = "gemma-2-2b-it"

# Языки для инференса (можно фильтровать по language или language_id)
# LANGUAGE_TO_ID = {'en': 0, 'de': 1, 'sv': 2, 'unknown': 3}
# Теперь поддерживаем все 3 языка
TARGET_LANGUAGES = ["en", "de", "sv"]  # английский, немецкий, шведский

# Промпт для каждого языка с чётким форматом вывода
TRANSLATION_PROMPTS = {
    "en": """Translate the following Old Norse/Runic transliteration to English.
Provide ONLY the English translation text. Do not include any labels, explanations, or additional text.
Your response must be just the translation, nothing else.

{text}""",
    "de": """Übersetze die folgende altnordische/Runen-Transliteration ins Deutsche.
Gib NUR die deutsche Übersetzung aus. Keine Labels, Erklärungen oder zusätzlichen Text.
Deine Antwort muss nur die Übersetzung sein, nichts anderes.

{text}""",
    "sv": """Översätt följande fornnordiska/runtranslitteration till svenska.
Ange ENDAST den svenska översättningstexten. Inga etiketter, förklaringar eller extra text.
Ditt svar måste vara bara översättningen, inget annat.

{text}"""
}

MIN_WORDS = 6  # минимальное количество слов (фильтр уже применён в split_dataset.py)
NUM_SAMPLES_PER_LANGUAGE = 100  # количество пар для инференса на каждый язык
REQUEST_TIMEOUT = 120  # timeout на запрос к Ollama
RANDOM_SEED = 42  # для воспроизводимости

# Примечание: фильтр по MIN_WORDS уже применён в split_dataset.py при создании test датасета
# Поэтому здесь дополнительная фильтрация может быть отключена при необходимости
FILTER_BY_WORD_COUNT = False  # ← True для двойной проверки, False для ускорения

# ======================
# LOAD DATA
# ======================

def load_data(path, target_language="en"):
    """
    Load data from CSV file with parallel corpus.
    
    Args:
        path: Path to CSV file
        target_language: Filter by target language (e.g., 'en', 'de', 'sv')
    
    Returns:
        sources, references: Lists of source texts and reference translations
    """
    import pandas as pd
    
    df = pd.read_csv(path)
    
    # Filter by target language if specified
    if target_language:
        df = df[df['language'] == target_language]
    
    sources = []
    references = []
    
    for _, row in df.iterrows():
        src = row['source_text']
        tgt = row['target_translation']
        
        # Фильтр по количеству слов (опционально, т.к. уже применён в split_dataset.py)
        if FILTER_BY_WORD_COUNT:
            word_count = len(src.split())
            if word_count < MIN_WORDS:
                continue
        
        sources.append(src)
        references.append(tgt)

    return sources, references


# ======================
# GENERATE TRANSLATIONS WITH OLLAMA
# ======================

def clean_translation_output(text, target_lang="en"):
    """
    Clean up model output to extract just the translation.
    Removes common prefixes, labels, and extra text.
    """
    if not text:
        return ""
    
    text = text.strip()
    
    # Common patterns to remove
    patterns_to_remove = [
        # English
        r"^English translation:\s*",
        r"^Translation:\s*",
        r"^Here is the translation:\s*",
        r"^The translation is:\s*",
        # German
        r"^Deutsche Übersetzung:\s*",
        r"^Übersetzung:\s*",
        # Swedish
        r"^Svensk översättning:\s*",
        r"^Översättning:\s*",
        # Generic
        r"^```[\w]*\n",
        r"\n```$",
        r"^\*\*Translation:\*\*\s*",
        r"^\[Translation\]\s*",
    ]
    
    import re
    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # Take only the first line if there are multiple lines (models sometimes add explanations)
    lines = text.split('\n')
    if len(lines) > 1:
        # Check if first line looks like a complete translation
        first_line = lines[0].strip()
        if len(first_line) > 10:  # Reasonable translation length
            text = first_line
    
    return text.strip()


def translate_with_ollama(model_name, text, target_lang="en"):
    """
    Translate text using Ollama model
    """
    prompt = TRANSLATION_PROMPTS.get(target_lang, TRANSLATION_PROMPTS["en"]).format(text=text)
    
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.3,  # low temperature for consistency
                "think": False  # disable "thinking..." placeholder
            },
            # timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        result = response.json()
        
        # Extract and clean translation
        raw_translation = result.get("response", "").strip()
        translation = clean_translation_output(raw_translation, target_lang)
        
        return translation
    except Exception as e:
        print(f"Error translating with {model_name}: {e}")
        return ""


def translate_batch(model_name, texts):
    """
    Translate a batch of texts using Ollama
    """
    predictions = []
    for text in texts:
        pred = translate_with_ollama(model_name, text)
        predictions.append(pred)
    return predictions


def evaluate_samples(sample_sources, sample_references, models_list, target_lang="en"):
    """
    Evaluate all models on the same set of samples for a specific target language
    Returns a list of dicts with source, reference, and predictions from each model
    """
    results = []
    
    for idx, (source, reference) in enumerate(zip(sample_sources, sample_references)):
        row = {
            "Sample_ID": idx + 1,
            "Source": source,
            "Reference": reference,
            "Target_Language": target_lang
        }
        
        for model_name in models_list:
            print(f"  [{target_lang}] Translating sample {idx+1}/{len(sample_sources)} with {model_name}...")
            prediction = translate_with_ollama(model_name, source, target_lang)
            row[f"{model_name}_translation"] = prediction
        
        results.append(row)
    
    return results


def save_model_results(df_results, df_metrics, model_name, target_languages, results_dir=RESULTS_DIR):
    """
    Save results and metrics for a specific model (analogous to inference_lora.py)
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Create model-specific folder
    model_folder = os.path.join(results_dir, model_name.replace(":", "_"))
    os.makedirs(model_folder, exist_ok=True)
    
    # Filter results for this model
    model_results = df_results[df_results['Target_Language'].isin(target_languages)]
    model_metrics = df_metrics[df_metrics['Model'] == model_name]
    
    # Save model-specific CSVs
    model_samples_file = os.path.join(model_folder, "translation_samples.csv")
    model_metrics_file = os.path.join(model_folder, "metrics_summary.csv")
    
    model_results.to_csv(model_samples_file, index=False, encoding='utf-8')
    model_metrics.to_csv(model_metrics_file, index=False, encoding='utf-8')
    
    print(f"✓ Results for {model_name} saved to {model_folder}/")
    return model_folder


# ======================
# MAIN
# ======================

def check_ollama_connection():
    """Check if Ollama is running and list available models"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        available_models = [m.get("name", "unknown") for m in models]
        return True, available_models
    except Exception as e:
        return False, []


def plot_metrics_for_language(df_metrics, target_lang, model_name=None, output_dir="plots"):
    """
    Plot BLEU and chrF++ metrics for a specific language
    If model_name is provided, save in model-specific folder
    """
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    if df_metrics.empty:
        print(f"⚠️  No metrics to plot for {target_lang}")
        return
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
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
    
    if model_name:
        # Save in model-specific folder
        model_output_dir = os.path.join(output_dir, model_name.replace(":", "_"))
        os.makedirs(model_output_dir, exist_ok=True)
        output_file = os.path.join(model_output_dir, f"metrics_{target_lang}.png")
    else:
        output_file = os.path.join(output_dir, f"metrics_{target_lang}.png")
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Metrics plot saved to {output_file}")


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


def main():
    print("=" * 60)
    print("Ollama-based Runic Text Translation Evaluation")
    print("=" * 60)
    
    # Check Ollama connection
    print(f"\n🔌 Connecting to Ollama at {OLLAMA_BASE_URL}...")
    connected, available_models = check_ollama_connection()
    
    if not connected:
        print(f"❌ Error: Cannot connect to Ollama at {OLLAMA_BASE_URL}")
        print("Make sure Ollama is running:")
        print("  1. Install Ollama from https://ollama.ai")
        print("  2. Run 'ollama serve' in another terminal")
        print("  3. Run 'ollama pull <model_name>' to download models")
        return
    
    print(f"✓ Connected to Ollama!")
    print(f"Available models: {', '.join(available_models) if available_models else 'None'}")
    
    # Check if requested models are available
    print(f"\n📋 Models to evaluate: {', '.join(MODELS)}")
    available_models_to_use = [m for m in MODELS if m in available_models]
    missing_models = [m for m in MODELS if m not in available_models]
    
    if missing_models:
        print(f"⚠️  Warning: These models are not available: {', '.join(missing_models)}")
        print("Download them with: ollama pull <model_name>\n")
    
    if not available_models_to_use:
        print("❌ Error: No available models to evaluate!")
        return
    
    # Process each target language
    all_results = []
    all_metrics = []
    
    for target_lang in TARGET_LANGUAGES:
        print("\n" + "=" * 80)
        print(f"🌍 Processing language: {target_lang.upper()}")
        print("=" * 80)
        
        # Load data for this language
        print(f"\n📂 Loading dataset for {target_lang}...")
        sources, references = load_data(DATA_PATH, target_language=target_lang)
        print(f"✓ Loaded {len(sources)} source-reference pairs for {target_lang}")
        
        if len(sources) == 0:
            print(f"⚠️  No data available for {target_lang}, skipping...")
            continue
        
        # Используем ВСЕ доступные сэмплы из test датасета (уже 100 на язык)
        # или ограничиваем NUM_SAMPLES_PER_LANGUAGE если нужно меньше
        num_samples = min(NUM_SAMPLES_PER_LANGUAGE, len(sources))
        print(f"\n📊 Using {num_samples} test samples for {target_lang}...")
        
        if num_samples == len(sources):
            # Берём все сэмплы из test (рекомендуется для честной оценки)
            sample_sources = sources
            sample_references = references
            print(f"✓ Using ALL test samples (no random selection)")
        else:
            # Выбираем случайные сэмплы если их больше чем нужно
            random.seed(RANDOM_SEED)
            sample_indices = random.sample(range(len(sources)), num_samples)
            sample_sources = [sources[i] for i in sample_indices]
            sample_references = [references[i] for i in sample_indices]
            print(f"✓ Selected {num_samples} random samples from {len(sources)} available")
        
        # Evaluate samples with all available models
        print(f"\n🔄 Running inference on {num_samples} samples with {len(available_models_to_use)} model(s)...")
        print("-" * 80)
        
        results_list = evaluate_samples(sample_sources, sample_references, available_models_to_use, target_lang)
        
        # Calculate metrics for each model for this language
        print(f"\n{'=' * 80}")
        print(f"=== METRICS FOR {target_lang.upper()} ===")
        print(f"{'=' * 80}")
        
        metrics_list = []
        
        for model_name in available_models_to_use:
            predictions = [r.get(f"{model_name}_translation", "") for r in results_list]
            valid_predictions = [p for p in predictions if p and p.strip()]
            
            if not valid_predictions:
                print(f"⚠️  {model_name}: No valid predictions")
                metrics_list.append({
                    "Model": model_name,
                    "Target_Language": target_lang,
                    "BLEU": 0.0,
                    "chrF++": 0.0,
                    "Valid_Predictions": 0
                })
                continue
            
            # Вычисляем все метрики
            bleu = sacrebleu.corpus_bleu(valid_predictions, [sample_references[:len(valid_predictions)]])
            chrf_plusplus = sacrebleu.corpus_chrf(valid_predictions, [sample_references[:len(valid_predictions)]], word_order=2)
            chrf = sacrebleu.corpus_chrf(valid_predictions, [sample_references[:len(valid_predictions)]], word_order=1)
            
            # ROUGE (с обработкой пустых предсказаний)
            rouge_l = 0.0
            try:
                rouge = Rouge()
                # Фильтруем полностью пустые строки для ROUGE
                non_empty_pairs = [(p, r) for p, r in zip(valid_predictions, sample_references[:len(valid_predictions)]) if p.strip() and len(p.strip()) > 0]
                if non_empty_pairs:
                    non_empty_preds = [p for p, r in non_empty_pairs]
                    non_empty_refs = [r for p, r in non_empty_pairs]
                    rouge_scores = rouge.get_scores(non_empty_preds, non_empty_refs, avg=True)
                    rouge_l = rouge_scores['rouge-l']['f'] * 100  # конвертируем в проценты как BLEU
            except Exception as e:
                print(f"⚠️  ROUGE calculation error: {e}")
                rouge_l = 0.0
            
            # METEOR
            meteor = 0.0
            try:
                references_tokenized = [[ref.split() for ref in sample_references[:len(valid_predictions)]]]
                predictions_tokenized = [pred.split() for pred in valid_predictions]
                meteor = meteor_score(references_tokenized, predictions_tokenized) * 100  # конвертируем в проценты
            except Exception as e:
                print(f"⚠️  METEOR calculation error: {e}")
                meteor = 0.0
            
            metrics_list.append({
                "Model": model_name,
                "Target_Language": target_lang,
                "BLEU": round(bleu.score, 2),
                "chrF": round(chrf.score, 2),
                "chrF++": round(chrf_plusplus.score, 2),
                "ROUGE-L": round(rouge_l, 2),
                "METEOR": round(meteor, 2),
                "Valid_Predictions": len(valid_predictions)
            })
            
            print(f"✓ {model_name:20} | BLEU: {round(bleu.score, 2):6.2f} | chrF: {round(chrf.score, 2):6.2f} | chrF++: {round(chrf_plusplus.score, 2):6.2f} | ROUGE-L: {rouge_l:6.2f} | METEOR: {meteor:6.2f}")
        
    # Store results
    all_results.extend(results_list)
    all_metrics.extend(metrics_list)
    
    # Plot metrics for this language
    df_metrics_lang = pd.DataFrame(metrics_list)
    plot_metrics_for_language(df_metrics_lang, target_lang, output_dir=PLOTS_DIR)
    
    # Save intermediate results after each language
    print(f"\n💾 Saving intermediate results for {target_lang}...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Save per-language intermediate files
    lang_samples_file = os.path.join(RESULTS_DIR, f"translation_samples_{target_lang}.csv")
    lang_metrics_file = os.path.join(RESULTS_DIR, f"metrics_summary_{target_lang}.csv")
    
    pd.DataFrame(results_list).to_csv(lang_samples_file, index=False, encoding='utf-8')
    pd.DataFrame(metrics_list).to_csv(lang_metrics_file, index=False, encoding='utf-8')
    print(f"✓ Intermediate results saved to {RESULTS_DIR}/{target_lang}/")
    
    # Combine all results
    df_results = pd.DataFrame(all_results)
    df_metrics = pd.DataFrame(all_metrics)
    
    # Display detailed results (sample)
    print("\n" + "=" * 80)
    print("=== SAMPLE DETAILED RESULTS ===")
    print("=" * 80)
    
    for idx, row in df_results.head(10).iterrows():
        print(f"\n📌 Sample {row['Sample_ID']} ({row['Target_Language']}):")
        print(f"   📖 Source: {row['Source'][:100]}...")
        print(f"   ✅ Reference: {row['Reference'][:100]}...")
        for model_name in available_models_to_use:
            translation = row.get(f"{model_name}_translation", "N/A")
            print(f"   🤖 {model_name}: {translation[:100]}...")
    
    # Save results per model (аналогично inference_lora.py)
    print(f"\n\n" + "=" * 80)
    print("💾 Saving results...")
    print("=" * 80)
    
    for model_name in available_models_to_use:
        # Filter results for this model - restructure data for inference_lora.py format
        model_results = []
        for _, row in df_results.iterrows():
            translation = row.get(f"{model_name}_translation", "")
            if translation:
                model_results.append({
                    "Sample_ID": row['Sample_ID'],
                    "Source": row['Source'],
                    "Reference": row['Reference'],
                    "Target_Language": row['Target_Language'],
                    "Model": model_name,
                    "Prediction": translation
                })
        
        df_model_results = pd.DataFrame(model_results)
        model_metrics = df_metrics[df_metrics['Model'] == model_name]
        
        # Create model-specific folder
        model_folder = os.path.join(RESULTS_DIR, model_name.replace(":", "_"))
        os.makedirs(model_folder, exist_ok=True)
        
        # Save model-specific CSVs (matching inference_lora.py format)
        model_samples_file = os.path.join(model_folder, "translation_samples.csv")
        model_metrics_file = os.path.join(model_folder, "metrics_summary.csv")
        
        df_model_results.to_csv(model_samples_file, index=False, encoding='utf-8')
        model_metrics.to_csv(model_metrics_file, index=False, encoding='utf-8')
        
        print(f"✓ Results for {model_name} saved to {model_folder}/")
    
    # Also save combined results in PLOTS_DIR
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    all_samples_file = os.path.join(PLOTS_DIR, "all_translation_samples.csv")
    all_metrics_file = os.path.join(PLOTS_DIR, "all_metrics_summary.csv")
    
    df_results.to_csv(all_samples_file, index=False, encoding='utf-8')
    df_metrics.to_csv(all_metrics_file, index=False, encoding='utf-8')
    
    print(f"✓ All sample translations saved to {all_samples_file}")
    print(f"✓ All metrics summary saved to {all_metrics_file}")
    
    # Plot aggregated comparison
    # print(f"\n📊 Generating aggregated comparison plots...")
    # plot_aggregated_comparison(df_metrics, output_dir=PLOTS_DIR)
    
    # Print summary across all languages
    print(f"\n\n" + "=" * 80)
    print("=== SUMMARY ACROSS ALL LANGUAGES ===")
    print("=" * 80)
    
    for model_name in available_models_to_use:
        print(f"\n{model_name}:")
        for lang in TARGET_LANGUAGES:
            lang_metrics = df_metrics[(df_metrics['Model'] == model_name) & (df_metrics['Target_Language'] == lang)]
            if not lang_metrics.empty:
                bleu = lang_metrics['BLEU'].values[0]
                chrf = lang_metrics['chrF++'].values[0]
                print(f"   {lang.upper()}: BLEU = {bleu:.2f}, chrF++ = {chrf:.2f}")
    
    # Best model summary
    print(f"\n\n" + "=" * 80)
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


if __name__ == "__main__":
    main()
