"""
Excel Results Collector for Runic AI
=====================================
Собирает все результаты инференсов в один Excel файл:
- Отдельный лист с метриками для всех моделей
- Отдельный лист с переводами для каждой модели

Запуск:
    python inference/collect_results_to_excel.py

Выход:
    inference/all_results_summary.xlsx
"""

import pandas as pd
import os
from pathlib import Path
from datetime import datetime

# Директория с результатами инференса
INFERENCE_DIR = Path(__file__).parent.parent

# Выходной Excel файл
OUTPUT_FILE = INFERENCE_DIR / "all_results_summary.xlsx"

# Паттерны для поиска результатов
# Ищем папки с metrics_summary*.csv и translation_samples*.csv
METRICS_PATTERNS = [
    "metrics_summary*.csv",
    "*/metrics_summary*.csv",
    "*/**/metrics_summary*.csv",
]

TRANSLATIONS_PATTERNS = [
    "translation_samples*.csv",
    "*/translation_samples*.csv",
    "*/**/translation_samples*.csv",
]


def find_csv_files(directory, pattern):
    """
    Рекурсивно ищет CSV файлы по паттерну в директории.
    """
    from fnmatch import fnmatch
    
    results = []
    for root, dirs, files in os.walk(directory):
        # Пропускаем служебные директории
        if any(x in root for x in ['.git', '__pycache__', '.venv', 'node_modules']):
            continue
        
        for file in files:
            if fnmatch(file, pattern):
                results.append(os.path.join(root, file))
    
    return results


def find_model_results(inference_dir):
    """
    Ищет все результаты инференсов в директории.
    Возвращает словарь: {model_name: {"metrics": path, "translations": path}}
    """
    models_results = {}
    
    # Ищем все metrics_summary*.csv файлы
    metrics_files = find_csv_files(inference_dir, "metrics_summary*.csv")
    
    for metrics_path in metrics_files:
        # Определяем имя модели из пути
        rel_path = os.path.relpath(metrics_path, inference_dir)
        parts = rel_path.split(os.sep)
        
        # Имя модели - это родительская папка или имя файла
        if len(parts) >= 2:
            model_name = parts[-2]  # Имя папки
        else:
            model_name = os.path.splitext(parts[0])[0]  # Имя файла без расширения
        
        # Пропускаем агрегированные файлы
        if 'all_' in model_name.lower():
            continue
        
        # Ищем соответствующий файл с переводами
        translations_path = None
        metrics_dir = os.path.dirname(metrics_path)
        
        for trans_pattern in ["translation_samples*.csv"]:
            trans_files = find_csv_files(metrics_dir, trans_pattern)
            if trans_files:
                translations_path = trans_files[0]
                break
        
        models_results[model_name] = {
            "metrics": metrics_path,
            "translations": translations_path
        }
    
    return models_results


def create_excel_report(models_results, output_file):
    """
    Создаёт Excel отчёт с метриками и переводами.
    """
    print(f"\n{'='*60}")
    print("📊 Creating Excel Report")
    print(f"{'='*60}")
    
    # Собираем все метрики
    all_metrics = []
    
    for model_name, paths in models_results.items():
        if paths["metrics"] and os.path.exists(paths["metrics"]):
            try:
                df_metrics = pd.read_csv(paths["metrics"])
                
                # Добавляем колонку с именем модели если нет
                if 'Model' not in df_metrics.columns:
                    df_metrics['Model'] = model_name
                
                all_metrics.append(df_metrics)
                
                print(f"  ✓ Metrics loaded: {model_name} ({len(df_metrics)} rows)")
            except Exception as e:
                print(f"  ⚠️  Error reading metrics for {model_name}: {e}")
    
    # Собираем все переводы
    all_translations = {}
    
    for model_name, paths in models_results.items():
        if paths["translations"] and os.path.exists(paths["translations"]):
            try:
                df_trans = pd.read_csv(paths["translations"])
                all_translations[model_name] = df_trans
                
                print(f"  ✓ Translations loaded: {model_name} ({len(df_trans)} rows)")
            except Exception as e:
                print(f"  ⚠️  Error reading translations for {model_name}: {e}")
    
    if not all_metrics and not all_translations:
        print("\n⚠️  No data found to export!")
        return False
    
    # Создаем Excel файл
    print(f"\n💾 Writing Excel file: {output_file}")
    
    with pd.ExcelWriter(output_file, engine='openpyxl', datetime_format='YYYY-MM-DD') as writer:
        # Лист с агрегированными метриками
        if all_metrics:
            df_all_metrics = pd.concat(all_metrics, ignore_index=True)
            
            # Переименовываем колонки для удобства
            column_rename = {
                'BLEU': 'BLEU Score',
                'chrF': 'chrF Score',
                'chrF++': 'chrF++ Score',
                'ROUGE-L': 'ROUGE-L Score',
                'METEOR': 'METEOR Score',
                'Target_Language': 'Language',
                'Valid_Predictions': 'Valid Predictions'
            }
            df_all_metrics = df_all_metrics.rename(columns=column_rename)
            
            # Добавляем дату генерации
            df_all_metrics['Generated At'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Пишем на лист "Metrics Summary"
            df_all_metrics.to_excel(writer, sheet_name='Metrics Summary', index=False)
            
            # Автоматически подбираем ширину колонок
            worksheet = writer.sheets['Metrics Summary']
            for idx, col in enumerate(df_all_metrics.columns):
                max_length = max(df_all_metrics[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.column_dimensions[chr(65 + idx)].width = min(max_length, 50)
            
            print(f"  ✓ Sheet 'Metrics Summary': {len(df_all_metrics)} rows")
        
        # Листы с переводами для каждой модели
        for model_name, df_trans in all_translations.items():
            # Очищаем имя листа от недопустимых символов (max 31 символ)
            sheet_name = f"Translations: {model_name}"[:31]
            sheet_name = "".join(x for x in sheet_name if x.isalnum() or x in ' -_')
            
            # Добавляем дату генерации
            df_trans['Generated At'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            df_trans.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  ✓ Sheet '{sheet_name}': {len(df_trans)} rows")
    
    print(f"\n{'='*60}")
    print(f"✅ Excel report created: {output_file}")
    print(f"{'='*60}")
    
    return True


def main():
    print("=" * 60)
    print("🔍 Runic AI - Results Collector")
    print("=" * 60)
    
    # Ищем результаты в директории inference/
    inference_dir = INFERENCE_DIR
    
    print(f"\n📂 Scanning directory: {inference_dir}")
    
    models_results = find_model_results(inference_dir)
    
    if not models_results:
        print("\n⚠️  No inference results found!")
        print("Looking for files matching: metrics_summary*.csv")
        return
    
    print(f"\n📋 Found {len(models_results)} model(s):")
    for model_name in models_results.keys():
        print(f"  - {model_name}")
    
    # Создаём отчёт
    success = create_excel_report(models_results, OUTPUT_FILE)
    
    if success:
        print(f"\n📊 Summary:")
        print(f"   Total models: {len(models_results)}")
        print(f"   Output file: {OUTPUT_FILE}")
        print(f"\n💡 Tip: Open the file in Excel or LibreOffice Calc")


if __name__ == "__main__":
    main()
