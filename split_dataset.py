"""
Скрипт для разделения параллельного датасета на train и test.

Создаёт:
- parallel_corpus_train.csv — для обучения (все кроме test)
- parallel_corpus_test.csv — для финальной оценки (по 100 на каждый язык)
"""

import pandas as pd
import random

# Пути
INPUT_PATH = "./train/parallel_corpus.csv"
TRAIN_OUTPUT = "./train/parallel_corpus_train.csv"
TEST_OUTPUT = "./train/parallel_corpus_test.csv"

# Настройки
TEST_SAMPLES_PER_LANGUAGE = 100
RANDOM_SEED = 42

# Фильтр по длине текста (в словах)
MIN_WORDS = 6   # минимальное количество слов
MAX_WORDS = 100 # максимальное количество слов (опционально, 0 = без лимита)

# Языки
LANGUAGES = ["en", "de", "sv"]

def main():
    print("=" * 60)
    print("Dataset Splitter — Train/Test")
    print("=" * 60)
    
    # Загружаем датасет
    print(f"\n📂 Loading dataset from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"✓ Total samples: {len(df)}")
    
    # Фильтр по длине текста
    print(f"\n✂️  Filtering by word count (min={MIN_WORDS}, max={MAX_WORDS if MAX_WORDS > 0 else '∞'})...")
    
    # Считаем количество слов в source_text
    df['word_count'] = df['source_text'].apply(lambda x: len(str(x).split()))
    
    # Применяем фильтр
    original_count = len(df)
    if MAX_WORDS > 0:
        df = df[(df['word_count'] >= MIN_WORDS) & (df['word_count'] <= MAX_WORDS)]
    else:
        df = df[df['word_count'] >= MIN_WORDS]
    
    filtered_count = original_count - len(df)
    print(f"✓ Filtered out {filtered_count} samples ({filtered_count/original_count*100:.1f}%)")
    print(f"✓ Remaining: {len(df)} samples")
    
    # Удаляем временную колонку
    df = df.drop(columns=['word_count'])
    
    # Статистика по языкам
    print("\n📊 Language distribution:")
    for lang in LANGUAGES:
        count = len(df[df['language'] == lang])
        print(f"   {lang}: {count} samples")
    
    # Разделяем по языкам
    print(f"\n✂️  Splitting dataset (test: {TEST_SAMPLES_PER_LANGUAGE} per language)...")
    
    test_dfs = []
    train_dfs = []
    
    random.seed(RANDOM_SEED)
    
    for lang in LANGUAGES:
        lang_df = df[df['language'] == lang]
        
        if len(lang_df) <= TEST_SAMPLES_PER_LANGUAGE:
            print(f"⚠️  Warning: {lang} has only {len(lang_df)} samples (need {TEST_SAMPLES_PER_LANGUAGE})")
            # Берём все как test
            test_dfs.append(lang_df)
            continue
        
        # Перемешиваем
        lang_df = lang_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        
        # Выделяем test
        test_df = lang_df.head(TEST_SAMPLES_PER_LANGUAGE)
        train_df = lang_df.iloc[TEST_SAMPLES_PER_LANGUAGE:].reset_index(drop=True)
        
        test_dfs.append(test_df)
        train_dfs.append(train_df)
        
        print(f"   {lang}: {len(train_df)} train, {len(test_df)} test")
    
    # Объединяем
    df_test = pd.concat(test_dfs, ignore_index=True)
    df_train = pd.concat(train_dfs, ignore_index=True)
    
    # Сохраняем
    print(f"\n💾 Saving datasets...")
    df_train.to_csv(TRAIN_OUTPUT, index=False, encoding='utf-8')
    df_test.to_csv(TEST_OUTPUT, index=False, encoding='utf-8')
    
    print(f"✓ Train: {len(df_train)} samples → {TRAIN_OUTPUT}")
    print(f"✓ Test:  {len(df_test)} samples → {TEST_OUTPUT}")
    
    # Финальная статистика
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    print(f"Total original: {len(df)}")
    print(f"Total train:    {len(df_train)} ({len(df_train)/len(df)*100:.1f}%)")
    print(f"Total test:     {len(df_test)} ({len(df_test)/len(df)*100:.1f}%)")
    print("=" * 60)
    


if __name__ == "__main__":
    main()
