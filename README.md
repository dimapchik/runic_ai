# Runic AI — Machine Translation for Runic Inscriptions

Project for training and evaluating machine translation models for runic transliterations → modern languages (English, Swedish, German).

---

## 📁 Project Structure

```
runic_ai/
├── parsers/
│   ├── collect_parallel_data.py    # Script to collect parallel corpus
│   ├── christerhamp_gamla_runor.csv
│   ├── srd_parallel.csv
│   └── runes_parsed_parallel_1_10k.csv
├── parallel_corpus.csv             # Main parallel dataset
├── parallel_corpus_detailed.json   # Dataset with metadata
├── train_model.py                  # Model fine-tuning script
├── inference.py                    # Model evaluation script
├── plot_metrics.py                 # Metrics visualization
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install pandas torch transformers trl unsloth datasets sacrebleu clearml
```

### 2. Collect Parallel Data

```bash
python parsers/collect_parallel_data.py
```

**Output:**
- `parallel_corpus.csv` — 14,585 parallel samples (source_text, target_translation, language, language_id)
- `parallel_corpus_detailed.json` — extended version with metadata

**Language mapping:**
| Language | Code | ID |
|----------|------|----|
| English  | en   | 0  |
| German   | de   | 1  |
| Swedish  | sv   | 2  |
| Unknown  | unknown | 3 |

---

## 📚 Training

### ClearML Setup (Optional)

1. Register at [clear.ml](https://app.clear.ml)
2. Get API keys from Settings → API Keys
3. Configure locally:
   ```bash
   clearml-init
   ```

### Run Training

```bash
python train_model.py
```

**Configuration in `train_model.py`:**
```python
DATA_PATH = "parallel_corpus.csv"
TARGET_LANGUAGE = "en"  # en, de, sv
TARGET_LANGUAGE_ID = 0  # 0=en, 1=de, 2=sv

model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # HuggingFace model
OUTPUT_DIR = "./qwen3.5-finetuned"
```

---

## 🔍 Inference & Evaluation

### Ollama Setup

1. Install [Ollama](https://ollama.ai)
2. Download models:
   ```bash
   ollama pull qwen3.5:0.8b
   ollama pull qwen3.5:2b
   ```
3. Start server:
   ```bash
   ollama serve
   ```

### Run Evaluation

```bash
python inference.py
```

**Configuration in `inference.py`:**
```python
TARGET_LANGUAGE = "en"  # Filter dataset by language
MIN_WORDS = 6           # Minimum words in source text
NUM_SAMPLES = 100       # Number of samples to evaluate
MODELS = ["qwen3.5:0.8b", "qwen3.5:2b", ...]
```

**Output:**
- `translation_samples.csv` — detailed translations
- `metrics_summary.csv` — BLEU and chrF++ scores

---

## 📊 Metrics

The project uses standard translation metrics:
- **BLEU** — Bilingual Evaluation Understudy
- **chrF++** — Character n-gram F-score (word_order=2)

Visualize metrics:
```bash
python plot_metrics.py
```

---

## 📦 Data Sources

| Source | File | Language | Samples |
|--------|------|----------|---------|
| Christer Hamp | christerhamp_gamla_runor.csv | Swedish | ~2,467 |
| Scandinavian Runic Database | srd_parallel.csv | English | ~7,304 |
| RunesDB EU | runes_parsed_parallel_1_10k.csv | German | ~7,486 |

**Total:** 14,585 unique parallel samples (after deduplication)

---

## 🛠 Scripts Reference

### `parsers/collect_parallel_data.py`

Collects and consolidates parallel translation data.

```bash
python parsers/collect_parallel_data.py \
    --parsers-dir parsers \
    --output-csv parallel_corpus.csv \
    --output-json parallel_corpus_detailed.json \
    --no-dedup  # optional: disable deduplication
```

### `train_model.py`

Fine-tunes a model using LoRA + Unsloth.

**Requirements:**
- `parallel_corpus.csv` in working directory
- ClearML API keys (if using ClearML)
- GPU with CUDA (recommended)

### `inference.py`

Evaluates Ollama models on translation task.

**Requirements:**
- Ollama server running
- Models downloaded via `ollama pull`

---

## 🔧 Configuration

### Language IDs

Used for multi-language training:

```python
LANGUAGE_TO_ID = {
    'en': 0,
    'de': 1,
    'sv': 2,
    'unknown': 3
}
```

### Model Selection

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| qwen3.5:0.8b | 0.8B | Fast | Baseline |
| qwen3.5:2b   | 2B   | Medium | Good |
| qwen3.5:4b   | 4B   | Slow | Better |
| qwen3.5:9b   | 9B   | Very Slow | Best |

---

## 📈 ClearML Integration

All experiments are logged to ClearML:
- Training metrics (loss, learning rate)
- Hyperparameters
- Model artifacts
- Dataset statistics

View in ClearML UI under project **`runic_ai`**.

---

## 📝 License

MIT

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📧 Contact

For questions or collaboration, open an issue on GitHub.
