"""
Анализ распределения количества слов в датасете.
Помогает понять, сколько данных теряется при разных ограничениях.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Пути к файлам
TRAIN_PATH = "parallel_corpus.csv"

def analyze_word_count(path):
    print("=" * 80)
    print("АНАЛИЗ РАСПРЕДЕЛЕНИЯ КОЛИЧЕСТВА СЛОВ")
    print("=" * 80)
    
    # Загружаем датасет
    print(f"\n📂 Loading dataset from {path}...")
    df = pd.read_csv(path)
    print(f"✓ Total samples: {len(df)}")
    
    # Считаем количество слов
    df['word_count'] = df['source_text'].apply(lambda x: len(str(x).split()))
    
    # Статистика
    print("\n📊 WORD COUNT STATISTICS:")
    print(f"   Min:      {df['word_count'].min()} words")
    print(f"   Max:      {df['word_count'].max()} words")
    print(f"   Mean:     {df['word_count'].mean():.2f} words")
    print(f"   Median:   {df['word_count'].median():.2f} words")
    print(f"   Std:      {df['word_count'].std():.2f} words")
    
    # Процентили
    print("\n📈 PERCENTILES:")
    for p in [50, 75, 90, 95, 99]:
        print(f"   {p}th:    {np.percentile(df['word_count'], p):.1f} words")
    
    # Анализ потерь при разных лимитах
    print("\n" + "=" * 80)
    print("ПОТЕРИ ДАННЫХ ПРИ РАЗНЫХ MAX_WORDS")
    print("=" * 80)
    
    limits = [50, 75, 100, 125, 150, 200, 250, 300, 500]
    
    results = []
    for limit in limits:
        lost = len(df[df['word_count'] > limit])
        lost_pct = lost / len(df) * 100
        kept = len(df) - lost
        kept_pct = kept / len(df) * 100
        results.append({
            'max_words': limit,
            'lost': lost,
            'lost_pct': lost_pct,
            'kept': kept,
            'kept_pct': kept_pct
        })
        print(f"   MAX_WORDS={limit:3d}: lost={lost:4d} ({lost_pct:5.2f}%), kept={kept:5d} ({kept_pct:5.2f}%)")
    
    # Распределение по языкам
    print("\n" + "=" * 80)
    print("РАСПРЕДЕЛЕНИЕ ПО ЯЗЫКАМ")
    print("=" * 80)
    
    for lang in df['language'].unique():
        lang_df = df[df['language'] == lang]
        print(f"\n   {lang.upper()}:")
        print(f"      Total:   {len(lang_df)} samples")
        print(f"      Min:     {lang_df['word_count'].min()} words")
        print(f"      Max:     {lang_df['word_count'].max()} words")
        print(f"      Mean:    {lang_df['word_count'].mean():.2f} words")
        print(f"      Median:  {lang_df['word_count'].median():.2f} words")
        
        # Потери при MAX_WORDS=100
        lost_100 = len(lang_df[lang_df['word_count'] > 100])
        lost_100_pct = lost_100 / len(lang_df) * 100
        print(f"      >100:    {lost_100} samples ({lost_100_pct:.2f}%)")
    
    # Визуализация
    print("\n📊 Saving plots...")
    
    # Гистограмма
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Гистограмма распределения
    ax1 = axes[0, 0]
    ax1.hist(df['word_count'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=100, color='red', linestyle='--', linewidth=2, label='MAX_WORDS=100')
    ax1.axvline(x=df['word_count'].median(), color='green', linestyle='--', linewidth=2, label=f"Median={df['word_count'].median():.0f}")
    ax1.set_xlabel('Word Count', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Word Count Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot по языкам
    ax2 = axes[0, 1]
    languages = df['language'].unique()
    data_by_lang = [df[df['language'] == lang]['word_count'] for lang in languages]
    bp = ax2.boxplot(data_by_lang, labels=languages, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax2.set_ylabel('Word Count', fontsize=12)
    ax2.set_title('Word Count by Language', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Потери при разных лимитах
    ax3 = axes[1, 0]
    ax3.plot([r['max_words'] for r in results], 
             [r['lost_pct'] for r in results], 
             marker='o', linewidth=2, markersize=8, color='red')
    ax3.set_xlabel('MAX_WORDS Limit', fontsize=12)
    ax3.set_ylabel('Lost Data (%)', fontsize=12)
    ax3.set_title('Data Loss vs MAX_WORDS Limit', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(limits)
    ax3.set_ylim(0, max([r['lost_pct'] for r in results]) * 1.1)
    
    # Добавляем аннотации
    for r in results:
        ax3.annotate(f"{r['lost_pct']:.2f}%", 
                    (r['max_words'], r['lost_pct']),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', fontsize=8)
    
    # 4. Cumulative distribution
    ax4 = axes[1, 1]
    sorted_counts = np.sort(df['word_count'])
    cumulative_pct = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100
    ax4.plot(sorted_counts, cumulative_pct, linewidth=2, color='blue')
    ax4.axhline(y=95, color='red', linestyle='--', linewidth=2, label='95th percentile')
    ax4.axhline(y=99, color='orange', linestyle='--', linewidth=2, label='99th percentile')
    ax4.axvline(x=100, color='green', linestyle='--', linewidth=2, label='MAX_WORDS=100')
    ax4.set_xlabel('Word Count', fontsize=12)
    ax4.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax4.set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('word_count_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot saved to word_count_analysis.png")
    
    # Рекомендация
    print("\n" + "=" * 80)
    print("РЕКОМЕНДАЦИЯ")
    print("=" * 80)
    
    # Находим лучший лимит (теряем <5% данных)
    best_limit = None
    for limit in [50, 75, 100, 125, 150, 200, 250, 300]:
        lost_pct = len(df[df['word_count'] > limit]) / len(df) * 100
        if lost_pct < 5:
            best_limit = limit
            break
    
    if best_limit:
        lost_at_best = len(df[df['word_count'] > best_limit])
        lost_pct_at_best = lost_at_best / len(df) * 100
        print(f"   ✅ Рекомендуемый MAX_WORDS: {best_limit}")
        print(f"      Потери: {lost_at_best} samples ({lost_pct_at_best:.2f}%)")
    else:
        print("   ⚠️  Даже при высоком лимите теряется много данных")
        print("      Рассмотрите возможность увеличения MAX_WORDS или разделения на короткие/длинные")
    
    # Анализ конкретных потерь при 100
    lost_100 = len(df[df['word_count'] > 100])
    lost_100_pct = lost_100 / len(df) * 100
    
    print(f"\n   📊 При MAX_WORDS=100:")
    print(f"      Потери: {lost_100} samples ({lost_100_pct:.2f}%)")
    
    if lost_100_pct < 1:
        print(f"      ✅ Отлично! Потери минимальны (<1%)")
    elif lost_100_pct < 5:
        print(f"      ✅ Хорошо! Потери приемлемы (<5%)")
    elif lost_100_pct < 10:
        print(f"      ⚠️  Нормально! Потери умеренны (<10%)")
    else:
        print(f"      ❌ Плохо! Потери значительны (>10%)")
        print(f"      Рассмотрите MAX_WORDS={best_limit} или выше")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    analyze_word_count(TRAIN_PATH)
