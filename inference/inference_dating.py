import pandas as pd
from tqdm import tqdm
import sacrebleu
import json
import random
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rouge import Rouge
import os
import re

# Альтернативный расчёт METEOR (используем evaluate library)
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
# CONFIG
# ======================

# Модель для оценки
# Формат: {"base_model": path, "lora_adapter": path, "adapter_name": name}
MODELS_CONFIG = [
    {
        "base_model": "../qwen3.5-0.8B",
        "lora_adapter": "",
        "adapter_name": "qwen3.5-0.8B-dating"
    },
    {
        "base_model": "../gemma-2-2b-it",
        "lora_adapter": "",
        "adapter_name": "gemma-2-2b-dating"
    },
    # {
    #     "base_model": "../qwen3.5-4B",
    #     "lora_adapter": "",
    #     "adapter_name": "qwen3.5-4B-dating"
    # },
    {
        "base_model": "../qwen3.5-0.8B",
        "lora_adapter": "../qwen3.5-0.8B-ft_multi-lang_20_epochs",
        "adapter_name": "qwen3.5-0.8B-ft_multi-lang_20_epochs-dating"
    },
    {
        "base_model": "../gemma-2-2b-it",
        "lora_adapter": "../gemma-2-2b-it-ft_20_epochs",
        "adapter_name": "gemma-2-2b-it-ft_20_epochs-dating"
    },
    # {
    #     "base_model": "../qwen3.5-4B",
    #     "lora_adapter": "../qwen3.5-4B-ft_multi-lang_20_epochs",
    #     "adapter_name": "qwen3.5-4B-ft_multi-lang_20_epochs-dating"
    # }
]

# Директории с существующими результатами других моделей (будут загружены автоматически)
EXISTING_MODELS = [
    # "path/to/existing/model/results",
]

# Директория для сохранения результатов
RESULTS_DIR = "dating_results"

# Директория для графиков
PLOTS_DIR = "../plots/dating"

# Устройство для вычислений
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# Путь к тестовому датасету
# Формат CSV: transliteration, translation, dating, actual_date, spread
DATA_PATH = "dating_dataset/test.csv"

# Количество сэмплов для инференса
NUM_SAMPLES = 100  # количество пар для инференса
RANDOM_SEED = 42  # для воспроизводимости

# Параметры нормализации (будут загружены из модели)
DEFAULT_MEAN_DATE = 900.0
DEFAULT_STD_DATE = 300.0

# Режим инференса: "regression" или "generative"
# regression: использует регрессионную голову (требует дообучения)
# generative: модель генерирует год текстом (zero-shot, без дообучения)
INFERENCE_MODE = "generative"

# Промпт для dating prediction
DATING_PROMPT = """### Instruction:
You are a dating prediction model for runic inscriptions.
Your task is to predict the year (as a single number) when the inscription was created.

Analyze the transliteration and translation, considering:
- Linguistic features and language evolution
- Historical context from the translation
- Runestone style indicators

Provide ONLY a single number representing the predicted year (e.g., 950, 1050, 1200).
Do not include any explanations, labels, or additional text.

### Input:
Transliteration: {transliteration}
Translation: {translation}

### Response:
"""


# ======================
# LOAD MODEL
# ======================

class DatingRegressionHead(nn.Module):
    """Regression head for dating prediction."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, pooled_hidden):
        return self.regression_head(pooled_hidden).squeeze(-1)


def load_model_and_tokenizer(base_model_path, lora_adapter_path, adapter_name):
    """
    Загружает базовую модель, LoRA адаптер и регрессионную голову
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
    if lora_adapter_path and os.path.exists(lora_adapter_path):
        print(f"🔧 Applying LoRA adapter from {lora_adapter_path}...")
        model = PeftModel.from_pretrained(
            model,
            lora_adapter_path,
            device_map=DEVICE,
        )
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
    
    # Загружаем регрессионную голову
    hidden_size = model.config.hidden_size
    regression_head = DatingRegressionHead(hidden_size)
    
    regression_head_path = os.path.join(lora_adapter_path, 'regression_head.pt') if lora_adapter_path else None
    if regression_head_path and os.path.exists(regression_head_path):
        print(f"🔧 Loading regression head from {regression_head_path}...")
        regression_head.load_state_dict(torch.load(regression_head_path, map_location=DEVICE))
    else:
        print(f"⚠️  Regression head not found, using untrained head")
    
    regression_head.to(DEVICE)
    regression_head.eval()
    
    # Загружаем статистику нормализации
    normalization_stats = load_normalization_stats(lora_adapter_path)
    
    print(f"✓ Model '{adapter_name}' loaded!")
    print(f"   Normalization: mean={normalization_stats['mean_date']:.1f}, std={normalization_stats['std_date']:.1f}")
    
    return model, tokenizer, regression_head, normalization_stats


def load_normalization_stats(model_path: str) -> dict:
    """Load normalization statistics from model directory."""
    if not model_path:
        return {'mean_date': DEFAULT_MEAN_DATE, 'std_date': DEFAULT_STD_DATE}
    
    stats_path = os.path.join(model_path, 'normalization_stats.json')
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            return json.load(f)
    return {'mean_date': DEFAULT_MEAN_DATE, 'std_date': DEFAULT_STD_DATE}


# ======================
# LOAD DATA
# ======================

def parse_dating(dating_str: str) -> float:
    """
    Parse dating string like '725 - 1100' to mid-point value.
    """
    if not dating_str or pd.isna(dating_str):
        return 0.0
    
    dating_str = str(dating_str).strip()
    
    match = re.match(r'(\d+)\s*-\s*(\d+)', dating_str)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        return (start + end) / 2.0
    
    try:
        return float(dating_str)
    except ValueError:
        return 0.0


def load_data(path):
    """
    Load dating data from CSV file.
    Format: transliteration, translation, dating, actual_date, spread
    """
    data = []
    
    if not os.path.exists(path):
        print(f"⚠️  File {path} not found")
        return []
    
    df = pd.read_csv(path)
    
    for _, row in df.iterrows():
        actual_date = float(row['actual_date'])
        spread = float(row['spread'])
        
        if actual_date > 0:
            data.append({
                'transliteration': str(row['transliteration']),
                'translation': str(row['translation']),
                'dating': str(row['dating']),
                'actual_date': actual_date,
                'spread': spread
            })
    
    return data


# ======================
# PREDICT DATING
# ======================

def extract_year_from_text(text: str) -> int:
    """
    Extract 4-digit year from generated text.
    """
    # Find all 4-digit numbers
    years = re.findall(r'\b(\d{4})\b', text)
    
    if years:
        # Return the first reasonable year (between 300 and 2025)
        for year in years:
            year_int = int(year)
            if 300 <= year_int <= 2025:
                return year_int
        # If no reasonable year found, return the first one
        return int(years[0])
    
    return 0


def predict_dating_generative(model, tokenizer, 
                               transliteration: str, translation: str = "") -> int:
    """
    Predict date using text generation (zero-shot, no regression head).
    Model generates the year as text.
    """
    prompt = f"""### Instruction:
You are a dating prediction model for runic inscriptions.
Your task is to predict the year (as a 4-digit number) when the inscription was created.

Analyze the transliteration and translation, considering:
- Linguistic features and language evolution
- Historical context from the translation
- Runestone style indicators

IMPORTANT: Return ONLY a 4-digit year number (e.g., 950, 1050, 1200).
Do not include any explanations, labels, words, or additional text.
Your entire response must be just the number.

### Input:
Transliteration: {transliteration}
Translation: {translation}

### Response:
"""
    
    encoding = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=512
    )
    
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=10,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract year from generated text
    predicted_year = extract_year_from_text(generated_text)
    
    return predicted_year


def predict_dating(model, tokenizer, regression_head, normalization_stats, 
                   transliteration: str, translation: str = "") -> float:
    """
    Predict date for a single inscription using regression head.
    """
    input_text = DATING_PROMPT.format(
        transliteration=transliteration,
        translation=translation
    )
    
    encoding = tokenizer(
        input_text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )
    
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        last_hidden = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).float()
        sum_hidden = (last_hidden * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1)
        pooled = sum_hidden / count
        
        predicted_normalized = regression_head(pooled)
    
    # Denormalize
    predicted_year = predicted_normalized.item() * normalization_stats['std_date'] + normalization_stats['mean_date']
    return predicted_year


def evaluate_samples(samples, model, tokenizer, regression_head, normalization_stats, model_name, inference_mode="regression"):
    """
    Evaluate model on samples using specified inference mode.
    """
    results = []
    
    for idx, item in enumerate(tqdm(samples, desc=f"[{model_name}] Predicting ({inference_mode})")):
        if inference_mode == "generative":
            pred_year = predict_dating_generative(
                model, tokenizer,
                item['transliteration'],
                item['translation']
            )
        else:
            pred_year = predict_dating(
                model, tokenizer, regression_head, normalization_stats,
                item['transliteration'],
                item['translation']
            )
        
        row = {
            "Sample_ID": idx + 1,
            "Transliteration": item['transliteration'],
            "Translation": item['translation'],
            "Actual_Dating": item['dating'],
            "Actual_Year": item['actual_date'],
            "Spread": item['spread'],
            "Predicted_Year": pred_year,
            "Error": abs(pred_year - item['actual_date']),
            "Model": model_name,
            "Inference_Mode": inference_mode
        }
        results.append(row)
    
    return results


# ======================
# METRICS CALCULATION
# ======================

def year_to_dating_string(year: float) -> str:
    """Convert predicted year to dating string format for text metrics."""
    year_int = int(round(year))
    return f"{year_int - 50} - {year_int + 50}"


def calculate_metrics(results, model_name):
    """
    Calculate regression metrics for dating prediction.
    """
    # Filter valid predictions
    valid_results = [r for r in results if r['Actual_Year'] > 0]
    
    if not valid_results:
        print(f"⚠️  {model_name}: No valid predictions")
        return {
            "Model": model_name,
            "MAE": 0.0,
            "RMSE": 0.0,
            "Median_Error": 0.0,
            "Mean_Error": 0.0,
            "Std_Error": 0.0,
            "MAPE": 0.0,
            "Mean_Spread": 0.0,
            "Valid_Predictions": 0
        }
    
    actual_years = np.array([r['Actual_Year'] for r in valid_results])
    predicted_years = np.array([r['Predicted_Year'] for r in valid_results])
    spreads = np.array([r.get('Spread', 0) for r in valid_results])
    
    # ===== Regression Metrics =====
    mae = float(np.mean(np.abs(predicted_years - actual_years)))
    rmse = float(np.sqrt(np.mean((predicted_years - actual_years)**2)))
    median_error = float(np.median(np.abs(predicted_years - actual_years)))
    mean_error = float(np.mean(np.abs(predicted_years - actual_years)))
    std_error = float(np.std(np.abs(predicted_years - actual_years)))
    
    # MAPE (avoid division by zero)
    mape = float(np.mean(np.abs((actual_years - predicted_years) / (actual_years + 1e-6))) * 100)
    
    return {
        "Model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "Median_Error": round(median_error, 2),
        "Mean_Error": round(mean_error, 2),
        "Std_Error": round(std_error, 2),
        "MAPE": round(mape, 2),
        "Mean_Spread": round(np.mean(spreads), 2) if len(spreads) > 0 else 0.0,
        "Valid_Predictions": len(valid_results)
    }


# ======================
# PLOT METRICS
# ======================

def plot_metrics_comparison(df_metrics, output_dir="plots"):
    """
    Plot regression metrics for all models
    """
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    if df_metrics.empty:
        print(f"⚠️  No metrics to plot")
        return
    
    models = df_metrics['Model'].tolist()
    
    # Create figure with subplots for each metric
    metrics_to_plot = [
        ('MAE', 'years', 'steelblue'),
        ('RMSE', 'years', 'coral'),
        ('MAPE', '%', 'green'),
        ('BLEU', 'score', 'purple'),
        ('chrF++', 'score', 'orange'),
        ('ROUGE-L', 'score', 'brown'),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, (metric_name, unit, color) in enumerate(metrics_to_plot):
        if metric_name not in df_metrics.columns:
            continue
        
        ax = axes[idx]
        scores = df_metrics[metric_name].tolist()
        
        bars = ax.bar(models, scores, color=color, alpha=0.8)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(f'{metric_name} ({unit})', fontsize=12)
        ax.set_title(f'{metric_name} - Dating Prediction', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # Set appropriate y-axis limits
        if metric_name in ['MAE', 'RMSE', 'Mean_Error', 'Median_Error', 'Std_Error']:
            ax.set_ylim(0, max(scores) * 1.2 if scores else 10)
        elif metric_name == 'MAPE':
            ax.set_ylim(0, min(100, max(scores) * 1.2) if scores else 100)
        else:
            ax.set_ylim(0, max(scores) * 1.2 if scores else 10)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{score:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "metrics_comparison_dating.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Metrics plot saved to {output_file}")


def plot_predictions_vs_actual(results, model_name, output_dir="plots"):
    """
    Plot predicted vs actual dates
    """
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    valid_results = [r for r in results if r['Actual_Year'] > 0]
    if not valid_results:
        return
    
    actual = [r['Actual_Year'] for r in valid_results]
    predicted = [r['Predicted_Year'] for r in valid_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot
    ax1.scatter(actual, predicted, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax1.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--', linewidth=2)
    ax1.set_xlabel('Actual Year', fontsize=12)
    ax1.set_ylabel('Predicted Year', fontsize=12)
    ax1.set_title(f'Predicted vs Actual Dating - {model_name}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Error distribution
    errors = [abs(p - a) for p, a in zip(predicted, actual)]
    ax2.hist(errors, bins=30, color='steelblue', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Absolute Error (years)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Error Distribution - {model_name}', fontsize=14, fontweight='bold')
    ax2.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.1f}')
    ax2.axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.1f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f"predictions_vs_actual_{model_name.replace('/', '_')}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Predictions plot saved to {output_file}")


# ======================
# LOAD EXISTING RESULTS
# ======================

def load_existing_model_results(model_dir):
    """
    Load existing results from CSV files for a model that was already evaluated.
    """
    samples_file = os.path.join(model_dir, "dating_predictions.csv")
    metrics_file = os.path.join(model_dir, "dating_metrics.csv")
    
    if not os.path.exists(samples_file) or not os.path.exists(metrics_file):
        print(f"⚠️  No existing results found for {model_dir}")
        return None, None
    
    print(f"📂 Loading existing results for {model_dir}...")
    
    df_metrics = pd.read_csv(metrics_file)
    metrics_list = df_metrics.to_dict('records')
    
    df_samples = pd.read_csv(samples_file)
    results_list = df_samples.to_dict('records')
    
    print(f"✓ Loaded {len(results_list)} samples and {len(metrics_list)} metrics for {model_dir}")
    return results_list, metrics_list


# ======================
# MAIN
# ======================

def main():
    print("=" * 80)
    print("Dating Prediction Model Evaluation")
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
    
    print(f"\n📋 Models to evaluate: {len(MODELS_CONFIG)}")
    for i, cfg in enumerate(MODELS_CONFIG, 1):
        print(f"   {i}. {cfg['adapter_name']}")
    
    # Load data
    print("\n" + "=" * 80)
    print("📂 Loading dataset...")
    print("=" * 80)
    
    all_data = load_data(DATA_PATH)
    
    if not all_data:
        print(f"❌ No data loaded from {DATA_PATH}")
        return
    
    print(f"✓ Loaded {len(all_data)} samples from {DATA_PATH}")
    
    # Select random samples if needed
    if NUM_SAMPLES < len(all_data):
        random.seed(RANDOM_SEED)
        sample_indices = random.sample(range(len(all_data)), NUM_SAMPLES)
        samples = [all_data[i] for i in sample_indices]
        print(f"✓ Selected {NUM_SAMPLES} random samples for evaluation")
    else:
        samples = all_data
        print(f"✓ Using all {len(samples)} samples for evaluation")
    
    # Process each model
    all_results = []
    all_metrics = []
    
    # 1. First, load existing results
    print("\n" + "=" * 80)
    print("📂 Loading existing model results...")
    print("=" * 80)
    
    for existing_model in EXISTING_MODELS:
        results, metrics = load_existing_model_results(existing_model)
        if results is not None and metrics is not None:
            all_results.extend(results)
            all_metrics.extend(metrics)
            print(f"✓ Loaded existing results for {existing_model}")
    
    # 2. Now evaluate new models
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
            model, tokenizer, regression_head, normalization_stats = load_model_and_tokenizer(
                model_cfg['base_model'],
                model_cfg['lora_adapter'],
                model_cfg['adapter_name']
            )
        except Exception as e:
            print(f"❌ Error loading model {model_cfg['adapter_name']}: {e}")
            print("   Skipping this model and continuing with next...")
            continue
        
        # Determine inference mode based on adapter availability
        has_regression_head = model_cfg['lora_adapter'] and os.path.exists(
            os.path.join(model_cfg['lora_adapter'], 'regression_head.pt')
        )
        current_mode = INFERENCE_MODE if has_regression_head else "generative"
        
        if INFERENCE_MODE == "generative" and has_regression_head:
            print(f"⚠️  Model has regression head, but INFERENCE_MODE='generative' - using generative mode")
            current_mode = "generative"
        
        # Evaluate
        print(f"\n🔄 Running inference on {len(samples)} samples...")
        print(f"   Inference mode: {current_mode}")
        results_list = evaluate_samples(
            samples, model, tokenizer, regression_head, normalization_stats,
            model_cfg['adapter_name'],
            inference_mode=current_mode
        )
        
        # Calculate metrics
        print(f"\n{'=' * 80}")
        print(f"=== METRICS FOR {model_cfg['adapter_name']} ===")
        print(f"{'=' * 80}")
        
        metrics = calculate_metrics(results_list, model_cfg['adapter_name'])
        
        # Print metrics
        print(f"\n{'=' * 60}")
        print(f"📊 REGRESSION METRICS")
        print(f"{'=' * 60}")
        print(f"  MAE (years):      {metrics['MAE']:>10.2f}")
        print(f"  RMSE (years):     {metrics['RMSE']:>10.2f}")
        print(f"  Median Error:     {metrics['Median_Error']:>10.2f}")
        print(f"  Mean Error:       {metrics['Mean_Error']:>10.2f}")
        print(f"  Std Error:        {metrics['Std_Error']:>10.2f}")
        print(f"  MAPE (%):         {metrics['MAPE']:>10.2f}")
        print(f"  Mean Spread:      {metrics['Mean_Spread']:>10.2f}")
        print(f"{'=' * 60}")
        print(f"  Valid Predictions:{metrics['Valid_Predictions']:>10}")
        print(f"{'=' * 60}")
        
        # Store results
        all_results.extend(results_list)
        all_metrics.append(metrics)
        
        # Plot
        plot_predictions_vs_actual(results_list, model_cfg['adapter_name'], output_dir=PLOTS_DIR)
        
        # Save intermediate results
        print(f"\n💾 Saving results for {model_cfg['adapter_name']}...")
        os.makedirs(model_cfg['adapter_name'], exist_ok=True)
        
        df_results = pd.DataFrame(results_list)
        df_metrics = pd.DataFrame([metrics])
        
        df_results.to_csv(
            f"{model_cfg['adapter_name']}/dating_predictions.csv",
            index=False,
            encoding='utf-8'
        )
        df_metrics.to_csv(
            f"{model_cfg['adapter_name']}/dating_metrics.csv",
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
        print(f"\n📌 Sample {row['Sample_ID']} - Model: {row['Model']}")
        print(f"   📖 Transliteration: {row['Transliteration'][:80]}...")
        print(f"   ✅ Actual: {row['Actual_Dating']} ({row['Actual_Year']:.1f})")
        print(f"   🔮 Predicted: {row['Predicted_Year']:.1f}")
        print(f"   ❌ Error: {row['Error']:.1f} years")
    
    # Save all results
    print(f"\n\n" + "=" * 80)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    all_samples_file = f"{RESULTS_DIR}/all_dating_predictions.csv"
    all_metrics_file = f"{RESULTS_DIR}/all_dating_metrics.csv"
    
    df_results.to_csv(all_samples_file, index=False, encoding='utf-8')
    df_metrics.to_csv(all_metrics_file, index=False, encoding='utf-8')
    
    print(f"✓ All predictions saved to {all_samples_file}")
    print(f"✓ All metrics saved to {all_metrics_file}")
    
    # Plot aggregated comparison
    print(f"\n📊 Generating metrics comparison plots...")
    plot_metrics_comparison(df_metrics, output_dir=PLOTS_DIR)
    
    # Print summary
    print(f"\n\n" + "=" * 80)
    print("=== SUMMARY ACROSS ALL MODELS ===")
    print("=" * 80)
    
    for model_name in df_metrics['Model'].unique():
        model_metrics = df_metrics[df_metrics['Model'] == model_name].iloc[0]
        print(f"\n🤖 {model_name}:")
        print(f"   MAE:    {model_metrics['MAE']:.2f} years")
        print(f"   RMSE:   {model_metrics['RMSE']:.2f} years")
        print(f"   MAPE:   {model_metrics['MAPE']:.2f}%")
        print(f"   Median: {model_metrics['Median_Error']:.2f} years")
    
    # Best model by MAE
    print(f"\n\n" + "=" * 80)
    print("=== BEST MODEL BY MAE ===")
    print("=" * 80)
    
    if not df_metrics.empty:
        best_model = df_metrics.loc[df_metrics['MAE'].idxmin()]
        print(f"🏆 {best_model['Model']}: MAE = {best_model['MAE']:.2f} years")
        
        print("\nRanking by MAE (lower is better):")
        sorted_metrics = df_metrics.sort_values('MAE')
        for idx, row in sorted_metrics.iterrows():
            print(f"   {idx+1}. {row['Model']}: {row['MAE']:.2f} years")
    
    print("\n" + "=" * 80)
    print("✓ Evaluation complete!")
    print(f"✓ Results directory: {RESULTS_DIR}")
    print(f"✓ Plots directory: {PLOTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
