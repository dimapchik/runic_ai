import pandas as pd
from tqdm import tqdm
import sacrebleu
import requests
import json
import random

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
    # "gemma2:latest",                 # general-purpose
    # "phi3:mini",           # 3B, very fast for quick checks
    # cloud models for comparison
    # "minimax-m2.5:cloud",    # 13B-ish, higher quality
    "qwen3.5:0.8b",
    "qwen3.5:2b",
    "qwen3.5:4b",
    "qwen3.5:9b"
    # "qwen3.5:cloud",
    # "dolphin-mixtral",     # local large model
    # еще модели...
]

# Путь к твоему параллельному файлу
# Формат: CSV с колонками: source_text, target_translation, source_name, source_file, language, language_id
DATA_PATH = "parallel_corpus.csv"

# Языки для инференса (можно фильтровать по language или language_id)
# LANGUAGE_TO_ID = {'en': 0, 'de': 1, 'sv': 2, 'unknown': 3}
TARGET_LANGUAGE = "en"  # английский для перевода

MIN_WORDS = 6  # минимальное количество слов в исходном тексте
NUM_SAMPLES = 100  # количество пар для инференса
REQUEST_TIMEOUT = 120  # timeout на запрос к Ollama
RANDOM_SEED = 42  # для воспроизводимости

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
        
        # Фильтр по количеству слов в исходном тексте
        word_count = len(src.split())
        if word_count < MIN_WORDS:
            continue
        
        sources.append(src)
        references.append(tgt)

    return sources, references


# ======================
# GENERATE TRANSLATIONS WITH OLLAMA
# ======================

def translate_with_ollama(model_name, text):
    """
    Translate text using Ollama model
    """
    # the prompt only requests the translation; model should respond with just
    # the English text, no extra labels or commentary
    prompt = f"""Translate the following Old Norse/Runic transliteration to
English. Provide only the English translation, without any additional text."

{text}"""
    
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.3,  # low temperature for consistency,
                "think": False  # disable "thinking..." placeholder
            },
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        result = response.json()
        
        # Extract translation (remove prompt from response)
        translation = result.get("response", "").strip()
        # Remove the prompt part if it's included
        if "English translation:" in translation:
            translation = translation.split("English translation:")[-1].strip()
        
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


def evaluate_samples(sample_sources, sample_references, models_list):
    """
    Evaluate all models on the same set of samples
    Returns a list of dicts with source, reference, and predictions from each model
    """
    results = []
    
    for idx, (source, reference) in enumerate(zip(sample_sources, sample_references)):
        row = {
            "Sample_ID": idx + 1,
            "Source": source,
            "Reference": reference
        }
        
        for model_name in models_list:
            print(f"  Translating sample {idx+1}/{len(sample_sources)} with {model_name}...")
            prediction = translate_with_ollama(model_name, source)
            row[f"{model_name}_translation"] = prediction
        
        results.append(row)
    
    return results


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
    
    # Load data
    print(f"\n📂 Loading dataset from {DATA_PATH}...")
    print(f"   (Filtering: language={TARGET_LANGUAGE}, min_words={MIN_WORDS})")
    sources, references = load_data(DATA_PATH, target_language=TARGET_LANGUAGE)
    print(f"✓ Loaded {len(sources)} source-reference pairs after filtering")
    
    # Select random samples
    print(f"\n🎲 Selecting {NUM_SAMPLES} random samples...")
    random.seed(RANDOM_SEED)
    sample_indices = random.sample(range(len(sources)), min(NUM_SAMPLES, len(sources)))
    sample_sources = [sources[i] for i in sample_indices]
    sample_references = [references[i] for i in sample_indices]
    print(f"✓ Selected samples: {NUM_SAMPLES}")
    
    # Evaluate samples with all available models
    print(f"\n🔄 Running inference on {NUM_SAMPLES} samples with {len(available_models_to_use)} model(s)...")
    print("=" * 80)
    
    results_list = evaluate_samples(sample_sources, sample_references, available_models_to_use)
    df_results = pd.DataFrame(results_list)
    
    # Display detailed results
    print("\n" + "=" * 80)
    print("=== DETAILED RESULTS ===")
    print("=" * 80)
    
    for idx, row in df_results.iterrows():
        print(f"\n📌 Sample {row['Sample_ID']}:")
        print(f"   📖 Source: {row['Source'][:100]}...")
        print(f"   ✅ Reference: {row['Reference'][:100]}...")
        for model_name in available_models_to_use:
            translation = row.get(f"{model_name}_translation", "N/A")
            print(f"   🤖 {model_name}: {translation[:100]}...")
    
    # Calculate metrics for each model
    print(f"\n\n" + "=" * 80)
    print("=== METRICS ===")
    print("=" * 80)
    
    metrics_list = []
    
    for model_name in available_models_to_use:
        predictions = df_results[f"{model_name}_translation"].tolist()
        valid_predictions = [p for p in predictions if p and p.strip()]
        
        if not valid_predictions:
            print(f"⚠️  {model_name}: No valid predictions")
            metrics_list.append({
                "Model": model_name,
                "BLEU": 0.0,
                "chrF++": 0.0,
                "Valid_Predictions": 0
            })
            continue
        
        bleu = sacrebleu.corpus_bleu(valid_predictions, [sample_references[:len(valid_predictions)]])
        chrf = sacrebleu.corpus_chrf(valid_predictions, [sample_references[:len(valid_predictions)]], word_order=2)
        
        metrics_list.append({
            "Model": model_name,
            "BLEU": round(bleu.score, 2),
            "chrF++": round(chrf.score, 2),
            "Valid_Predictions": len(valid_predictions)
        })
        
        print(f"✓ {model_name:20} | BLEU: {round(bleu.score, 2):6.2f} | chrF++: {round(chrf.score, 2):6.2f}")
    
    df_metrics = pd.DataFrame(metrics_list)
    
    # Save results
    print(f"\n\n" + "=" * 80)
    samples_output_file = "translation_samples.csv"
    metrics_output_file = "metrics_summary.csv"
    
    df_results.to_csv(samples_output_file, index=False, encoding='utf-8')
    df_metrics.to_csv(metrics_output_file, index=False, encoding='utf-8')
    
    print(f"✓ Sample translations saved to {samples_output_file}")
    print(f"✓ Metrics summary saved to {metrics_output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
