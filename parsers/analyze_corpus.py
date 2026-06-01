"""
Corpus Analysis Script for Runic AI
====================================
Анализ параллельного корпуса: подсчёт общего количества записей
и записей с заполненными полями findplace, country, dating, object_class.

Запуск:
    python parsers/analyze_corpus.py

Вывод:
    - Общее количество записей
    - Количество записей с каждым полем
    - Процент заполненности
    - Детальная статистика по комбинациям полей
"""

import pandas as pd
import os

# Пути к файлам для анализа
CORPUS_FILES = [
    "christerhamp_gamla_runor.csv",
]

# Поля для анализа
FIELDS_TO_CHECK = ["runic", "transliteration", "translation"]


def analyze_csv(filepath):
    """
    Анализирует CSV файл и возвращает статистику по полям.
    """
    if not os.path.exists(filepath):
        print(f"⚠️  File not found: {filepath}")
        return None
    
    print(f"\n{'='*60}")
    print(f"📂 Analyzing: {filepath}")
    print(f"{'='*60}")
    
    # Читаем CSV
    df = pd.read_csv(filepath)
    
    # Общее количество записей
    total_records = len(df)
    print(f"\n📊 Total records: {total_records:,}")
    
    # Статистика по полям
    print(f"\n📋 Field statistics:")
    print("-" * 60)
    
    field_stats = {}
    for field in FIELDS_TO_CHECK:
        if field in df.columns:
            # Количество непустых значений
            non_empty = df[field].notna() & (df[field].astype(str).str.strip() != "")
            count = non_empty.sum()
            percentage = (count / total_records) * 100 if total_records > 0 else 0
            
            field_stats[field] = {
                "count": count,
                "percentage": percentage,
                "total": total_records
            }
            
            print(f"  {field:20} → {count:>8,} ({percentage:>5.1f}%)")
        else:
            field_stats[field] = {
                "count": 0,
                "percentage": 0,
                "total": total_records,
                "missing": True
            }
            print(f"  {field:20} → {'COLUMN MISSING':>8}")
    
    # Комбинации полей
    print(f"\n📋 Field combinations:")
    print("-" * 60)
    
    # Сколько записей имеют все 4 поля
    all_fields_mask = pd.Series([True] * len(df))
    for field in FIELDS_TO_CHECK:
        if field in df.columns:
            all_fields_mask &= df[field].notna() & (df[field].astype(str).str.strip() != "")
    
    all_fields_count = all_fields_mask.sum()
    all_fields_pct = (all_fields_count / total_records) * 100 if total_records > 0 else 0
    print(f"  All 4 fields        → {all_fields_count:>8,} ({all_fields_pct:>5.1f}%)")
    
    # Сколько записей имеют хотя бы одно поле
    any_field_mask = pd.Series([False] * len(df))
    for field in FIELDS_TO_CHECK:
        if field in df.columns:
            any_field_mask |= df[field].notna() & (df[field].astype(str).str.strip() != "")
    
    any_field_count = any_field_mask.sum()
    any_field_pct = (any_field_count / total_records) * 100 if total_records > 0 else 0
    print(f"  At least 1 field    → {any_field_count:>8,} ({any_field_pct:>5.1f}%)")
    
    # Сколько записей не имеют ни одного поля
    no_fields_count = total_records - any_field_count
    no_fields_pct = (no_fields_count / total_records) * 100 if total_records > 0 else 0
    print(f"  No fields           → {no_fields_count:>8,} ({no_fields_pct:>5.1f}%)")
    
    # Детальная статистика по комбинациям
    print(f"\n📋 Detailed combinations:")
    print("-" * 60)
    
    from itertools import combinations
    
    for r in range(1, len(FIELDS_TO_CHECK) + 1):
        for combo in combinations(FIELDS_TO_CHECK, r):
            combo_mask = pd.Series([True] * len(df))
            for field in combo:
                if field in df.columns:
                    combo_mask &= df[field].notna() & (df[field].astype(str).str.strip() != "")
                else:
                    combo_mask = pd.Series([False] * len(df))
                    break
            
            combo_count = combo_mask.sum()
            if combo_count > 0:
                combo_pct = (combo_count / total_records) * 100 if total_records > 0 else 0
                combo_str = ", ".join(combo)
                print(f"  {combo_str:30} → {combo_count:>8,} ({combo_pct:>5.1f}%)")
    
    return {
        "filepath": filepath,
        "total": total_records,
        "field_stats": field_stats,
        "all_fields_count": all_fields_count,
        "any_field_count": any_field_count,
        "no_fields_count": no_fields_count
    }


def main():
    print("=" * 60)
    print("🔍 Runic Corpus Analysis")
    print("=" * 60)
    
    all_stats = []
    
    for filepath in CORPUS_FILES:
        stats = analyze_csv(filepath)
        if stats:
            all_stats.append(stats)
    
    # Сводная таблица
    if all_stats:
        print(f"\n\n{'='*60}")
        print("📊 SUMMARY TABLE")
        print(f"{'='*60}")
        
        print(f"\n{'File':<50} {'Total':>10} {'All 4':>10} {'Any 1':>10} {'None':>10}")
        print("-" * 90)
        
        for stats in all_stats:
            filename = os.path.basename(stats["filepath"])
            print(f"{filename:<50} {stats['total']:>10,} {stats['all_fields_count']:>10,} {stats['any_field_count']:>10,} {stats['no_fields_count']:>10,}")
        
        # Общий итог
        total_all = sum(s["total"] for s in all_stats)
        total_all_fields = sum(s["all_fields_count"] for s in all_stats)
        total_any_field = sum(s["any_field_count"] for s in all_stats)
        total_no_fields = sum(s["no_fields_count"] for s in all_stats)
        
        print("-" * 90)
        print(f"{'TOTAL':<50} {total_all:>10,} {total_all_fields:>10,} {total_any_field:>10,} {total_no_fields:>10,}")
        
        # Итоговая статистика по полям
        print(f"\n\n{'='*60}")
        print("📋 AGGREGATED FIELD STATISTICS")
        print(f"{'='*60}")
        
        # Собираем статистику по всем файлам
        aggregated = {field: {"count": 0, "total": 0} for field in FIELDS_TO_CHECK}
        
        for stats in all_stats:
            for field in FIELDS_TO_CHECK:
                if field in stats["field_stats"] and not stats["field_stats"][field].get("missing", False):
                    aggregated[field]["count"] += stats["field_stats"][field]["count"]
                    aggregated[field]["total"] += stats["field_stats"][field]["total"]
        
        print(f"\n{'Field':<20} {'Count':>12} {'Total':>12} {'Percentage':>12}")
        print("-" * 60)
        
        for field in FIELDS_TO_CHECK:
            count = aggregated[field]["count"]
            total = aggregated[field]["total"]
            pct = (count / total * 100) if total > 0 else 0
            print(f"{field:<20} {count:>12,} {total:>12,} {pct:>11.1f}%")
        
        print(f"\n{'='*60}")
        print("✓ Analysis complete!")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
