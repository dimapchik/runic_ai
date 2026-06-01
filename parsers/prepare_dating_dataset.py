#!/usr/bin/env python3
"""
Script to prepare dating prediction dataset from runes_parsed_parallel_1_10k.csv
Parses dating ranges, calculates actual_date (midpoint) and spread (half-range).
Splits data into train/val/test sets with stratified sampling.
Output format: transliteration | translation | dating | actual_date | spread
"""

import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split


def parse_dating(dating_str):
    """
    Parse dating string and extract midpoint and spread.
    
    Handles formats:
    - "1200 - 1300" → midpoint=1250, spread=50
    - "1100" → midpoint=1100, spread=0
    - "…,1100 - 1500" → midpoint=1300, spread=200 (extract last range)
    
    Args:
        dating_str: Dating string like "1200 - 1300" or "…,1100 - 1500"
    
    Returns:
        tuple: (midpoint, spread) or (None, None) if parsing fails
    """
    if not dating_str or pd.isna(dating_str):
        return None, None
    
    dating_str = str(dating_str).strip()
    
    if not dating_str:
        return None, None
    
    # Find all year ranges in the string (pattern: number - number)
    # This handles cases like "…,1100 - 1500" by finding the range
    range_matches = re.findall(r'(\d+)\s*-\s*(\d+)', dating_str)
    
    if range_matches:
        # Take the last range found (handles "…,1100 - 1500")
        start, end = range_matches[-1]
        start = int(start)
        end = int(end)
        
        midpoint = (start + end) / 2.0
        spread = (end - start) / 2.0
        
        return midpoint, spread
    
    # Try single year
    single_match = re.match(r'^(\d+)$', dating_str)
    if single_match:
        year = int(single_match.group(1))
        return float(year), 0.0
    
    return None, None


def prepare_dating_dataset(csv_path, output_dir='dating_dataset', test_samples=100):
    """
    Prepare train/val/test splits for dating prediction task.
    
    Args:
        csv_path: Path to the source CSV file
        output_dir: Directory to save the split datasets
        test_samples: Number of samples to include in test set (default: 100)
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Total records: {len(df)}")
    
    # Select only needed columns
    df = df[['transliteration', 'translation', 'dating']].copy()
    
    # Filter out records without dating
    df = df.dropna(subset=['dating'])
    df = df[df['dating'].str.strip() != '']
    
    print(f"Records with valid dating string: {len(df)}")
    
    # Filter out records without transliteration
    df = df.dropna(subset=['transliteration'])
    df = df[df['transliteration'].str.strip() != '']
    
    print(f"Records with valid transliteration: {len(df)}")
    
    # Filter out records without translation
    df = df.dropna(subset=['translation'])
    df = df[df['translation'].str.strip() != '']
    
    print(f"Records with valid translation: {len(df)}")
    
    # Filter out records with less than 6 words in translation
    df = df[df['translation'].str.split().str.len() >= 6]
    
    print(f"Records with translation >= 6 words: {len(df)}")
    
    if len(df) == 0:
        print("No valid records found!")
        return
    
    # Parse dating to extract actual_date (midpoint) and spread
    print("\nParsing dating strings...")
    parsed_results = df['dating'].apply(parse_dating)
    
    df['actual_date'] = parsed_results.apply(lambda x: x[0] if x else None)
    df['spread'] = parsed_results.apply(lambda x: x[1] if x else None)
    
    # Filter out records that couldn't be parsed
    valid_mask = df['actual_date'].notna()
    df_valid = df[valid_mask].copy()
    df_invalid = df[~valid_mask].copy()
    
    print(f"Valid parsed records: {len(df_valid)}")
    print(f"Invalid/unparseable records: {len(df_invalid)}")
    
    if len(df_valid) == 0:
        print("No valid parsed records!")
        return
    
    # Show some invalid examples for debugging
    if len(df_invalid) > 0:
        print("\nSample invalid dating strings:")
        for idx, dating in enumerate(df_invalid['dating'].head(10)):
            print(f"  {idx+1}. '{dating}'")
    
    # Show dating statistics
    print("\n" + "="*60)
    print("DATING STATISTICS")
    print("="*60)
    print(f"Date range: {df_valid['actual_date'].min():.0f} - {df_valid['actual_date'].max():.0f}")
    print(f"Mean date: {df_valid['actual_date'].mean():.1f}")
    print(f"Median date: {df_valid['actual_date'].median():.1f}")
    print(f"Mean spread: {df_valid['spread'].mean():.1f}")
    print(f"Median spread: {df_valid['spread'].median():.1f}")
    
    # Show spread distribution
    print("\nSpread distribution:")
    print(f"  spread = 0:   {len(df_valid[df_valid['spread'] == 0])} records (exact dates)")
    print(f"  spread < 50:  {len(df_valid[(df_valid['spread'] > 0) & (df_valid['spread'] < 50)])} records")
    print(f"  spread 50-100: {len(df_valid[(df_valid['spread'] >= 50) & (df_valid['spread'] <= 100)])} records")
    print(f"  spread > 100: {len(df_valid[df_valid['spread'] > 100])} records (wide ranges)")
    
    # Create date bins for stratified sampling
    # Use quantiles to ensure balanced distribution
    n_bins = min(20, len(df_valid) // 10)  # Adjust bins based on data size
    n_bins = max(5, n_bins)  # At least 5 bins
    
    df_valid['date_bin'] = pd.qcut(
        df_valid['actual_date'], 
        q=n_bins, 
        duplicates='drop',
        labels=False
    )
    
    print(f"\nCreated {df_valid['date_bin'].nunique()} date bins for stratification")
    
    # Split data
    # First: separate test set
    df_train_val, df_test = train_test_split(
        df_valid,
        test_size=test_samples,
        stratify=df_valid['date_bin'],
        random_state=42
    )
    
    # Second: split train/val (90/10 of remaining)
    val_size = max(1, len(df_train_val) // 10)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_size,
        stratify=df_train_val['date_bin'],
        random_state=42
    )
    
    # Shuffle each split
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    df_val = df_val.sample(frac=1, random_state=42).reset_index(drop=True)
    df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Print statistics
    print("\n" + "="*60)
    print("SPLIT STATISTICS")
    print("="*60)
    print(f"Train set: {len(df_train)} records ({len(df_train)/len(df_valid)*100:.1f}%)")
    print(f"Val set:   {len(df_val)} records ({len(df_val)/len(df_valid)*100:.1f}%)")
    print(f"Test set:  {len(df_test)} records ({len(df_test)/len(df_valid)*100:.1f}%)")
    
    # Verify distribution preservation
    print("\n" + "="*60)
    print("DISTRIBUTION VERIFICATION")
    print("="*60)
    
    print(f"\n{'Stat':<15} {'Train':<12} {'Val':<12} {'Test':<12} {'Total':<12}")
    print("-" * 63)
    print(f"{'Mean date':<15} {df_train['actual_date'].mean():<12.1f} {df_val['actual_date'].mean():<12.1f} {df_test['actual_date'].mean():<12.1f} {df_valid['actual_date'].mean():<12.1f}")
    print(f"{'Median date':<15} {df_train['actual_date'].median():<12.1f} {df_val['actual_date'].median():<12.1f} {df_test['actual_date'].median():<12.1f} {df_valid['actual_date'].median():<12.1f}")
    print(f"{'Std date':<15} {df_train['actual_date'].std():<12.1f} {df_val['actual_date'].std():<12.1f} {df_test['actual_date'].std():<12.1f} {df_valid['actual_date'].std():<12.1f}")
    print(f"{'Mean spread':<15} {df_train['spread'].mean():<12.1f} {df_val['spread'].mean():<12.1f} {df_test['spread'].mean():<12.1f} {df_valid['spread'].mean():<12.1f}")
    
    # Save datasets
    print("\n" + "="*60)
    print("SAVING DATASETS")
    print("="*60)
    
    def save_dataset(df_split, filename, include_header=True):
        output_path = os.path.join(output_dir, filename)
        
        # Create the formatted output: transliteration | translation | dating | actual_date | spread
        formatted = df_split.apply(
            lambda row: f"{row['transliteration']} | {row['translation']} | {row['dating']} | {row['actual_date']:.1f} | {row['spread']:.1f}",
            axis=1
        )
        formatted.to_csv(output_path, index=False, header=include_header)
        print(f"Saved {filename}: {len(df_split)} records")
        return output_path
    
    # Save TXT format (pipe-delimited)
    train_txt = save_dataset(df_train, 'train.txt', include_header=False)
    val_txt = save_dataset(df_val, 'val.txt', include_header=False)
    test_txt = save_dataset(df_test, 'test.txt', include_header=False)
    
    # Save CSV format with headers
    df_train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    df_val.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    df_test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    print(f"Saved CSV versions with headers")
    
    # Save sample of invalid records for review
    if len(df_invalid) > 0:
        invalid_path = os.path.join(output_dir, 'invalid_records.csv')
        df_invalid.to_csv(invalid_path, index=False)
        print(f"Saved {len(df_invalid)} invalid records to {invalid_path}")
    
    print("\n" + "="*60)
    print("DATASET PREPARATION COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {output_dir}/")
    print("Files created:")
    print("  - train.txt, val.txt, test.txt (pipe-delimited: translit | trans | dating | actual_date | spread)")
    print("  - train.csv, val.csv, test.csv (CSV with headers)")
    if len(df_invalid) > 0:
        print(f"  - invalid_records.csv (records that couldn't be parsed)")
    
    return {
        'train': len(df_train),
        'val': len(df_val),
        'test': len(df_test),
        'invalid': len(df_invalid)
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare dating prediction dataset')
    parser.add_argument('--input', type=str, default='runes_parsed_parallel_1_10k.csv',
                       help='Input CSV file path')
    parser.add_argument('--output', type=str, default='../inference/dating_dataset',
                       help='Output directory')
    parser.add_argument('--test-samples', type=int, default=100,
                       help='Number of test samples')
    
    args = parser.parse_args()
    
    prepare_dating_dataset(args.input, args.output, args.test_samples)
