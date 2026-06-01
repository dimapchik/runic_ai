#!/usr/bin/env python3
"""
Script for analyzing dating distribution in runes_parsed_parallel_1_10k.csv
Builds histograms and statistical summaries of runic inscriptions by time period.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import re


def parse_dating_range(dating_str):
    """
    Parse dating string like '210 - 310' into (start, end) tuple.
    Returns None if parsing fails.
    """
    if pd.isna(dating_str) or not dating_str.strip():
        return None
    
    # Match pattern like "210 - 310" or "160 - 240"
    match = re.match(r'(\d+)\s*-\s*(\d+)', str(dating_str))
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        return (start, end)
    
    return None


def get_century(year):
    """Convert a year to century string (e.g., 210 -> '3rd century')"""
    century = (year - 1) // 100 + 1
    suffix = 'th'
    if 1 <= century % 10 <= 3 and century not in [11, 12, 13]:
        suffix = ['st', 'nd', 'rd'][century % 10 - 1]
    return f"{century}{suffix} century"


def get_century_num(year):
    """Get century number from year"""
    return (year - 1) // 100 + 1


def analyze_dating_distribution(csv_path):
    """Main analysis function"""
    
    # Load data
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Total records: {len(df)}")
    
    # Parse dating information
    dating_ranges = df['dating'].apply(parse_dating_range)
    
    # Filter out records without dating
    valid_mask = dating_ranges.notna()
    df_valid = df[valid_mask].copy()
    df_valid['dating_range'] = dating_ranges[valid_mask]
    
    print(f"Records with valid dating: {len(df_valid)} ({len(df_valid)/len(df)*100:.1f}%)")
    
    if len(df_valid) == 0:
        print("No valid dating information found!")
        return
    
    # Extract start and end years
    df_valid['dating_start'] = df_valid['dating_range'].apply(lambda x: x[0])
    df_valid['dating_end'] = df_valid['dating_range'].apply(lambda x: x[1])
    df_valid['dating_mid'] = (df_valid['dating_start'] + df_valid['dating_end']) / 2
    
    # ===== Statistical Summary =====
    print("\n" + "="*60)
    print("STATISTICAL SUMMARY")
    print("="*60)
    print(f"\nDate range: {int(df_valid['dating_start'].min())} - {int(df_valid['dating_end'].max())}")
    print(f"Mean start year: {df_valid['dating_start'].mean():.1f}")
    print(f"Mean end year: {df_valid['dating_end'].mean():.1f}")
    
    # ===== Distribution by Century =====
    print("\n" + "="*60)
    print("DISTRIBUTION BY CENTURY (based on mid-point)")
    print("="*60)
    
    df_valid['century'] = df_valid['dating_mid'].apply(get_century_num)
    century_counts = df_valid['century'].value_counts().sort_index()
    
    print("\nCentury distribution:")
    for century, count in century_counts.items():
        print(f"  {century}th century: {count} inscriptions")
    
    # ===== Distribution by Date Range =====
    print("\n" + "="*60)
    print("DISTRIBUTION BY DATE RANGE")
    print("="*60)
    
    range_counts = df_valid['dating'].value_counts()
    print(f"\nTop 20 date ranges:")
    for date_range, count in range_counts.head(20).items():
        print(f"  {date_range}: {count} inscriptions")
    
    # ===== Visualization =====
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Runic Inscriptions Dating Distribution Analysis', fontsize=14, fontweight='bold')
    
    # Plot 1: Histogram by century
    ax1 = axes[0, 0]
    century_labels = [f"{c}th c." for c in century_counts.index]
    ax1.bar(century_labels, century_counts.values, color='steelblue', edgecolor='navy')
    ax1.set_xlabel('Century')
    ax1.set_ylabel('Number of Inscriptions')
    ax1.set_title('Distribution by Century')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(century_counts.values):
        ax1.text(i, v + max(century_counts.values)*0.01, str(v), ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Histogram of mid-point dates
    ax2 = axes[0, 1]
    ax2.hist(df_valid['dating_mid'], bins=30, color='coral', edgecolor='darkred', alpha=0.7)
    ax2.set_xlabel('Year (mid-point of range)')
    ax2.set_ylabel('Number of Inscriptions')
    ax2.set_title('Distribution by Date (Mid-point)')
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Date range visualization (start to end)
    ax3 = axes[1, 0]
    
    # Group by unique date ranges
    range_groups = df_valid.groupby('dating').size().reset_index(name='count')
    range_groups['start'] = range_groups['dating'].apply(lambda x: int(x.split('-')[0].strip()) if pd.notna(x) else 0)
    range_groups['end'] = range_groups['dating'].apply(lambda x: int(x.split('-')[1].strip()) if pd.notna(x) else 0)
    range_groups = range_groups.sort_values('start')
    
    # Create a scatter plot with line segments for each range
    y_positions = range(len(range_groups))
    for idx, (_, row) in enumerate(range_groups.iterrows()):
        ax3.plot([row['start'], row['end']], [idx, idx], 'b-', alpha=0.3, linewidth=2)
        ax3.plot([row['start']], [idx], 'o', color='darkblue', markersize=4)
        ax3.plot([row['end']], [idx], 'o', color='darkblue', markersize=4)
    
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Date Range')
    ax3.set_title('Date Ranges Timeline')
    ax3.grid(axis='x', alpha=0.3)
    
    # Limit displayed ranges for readability
    if len(range_groups) > 30:
        ax3.set_ylim(-0.5, 30.5)
    
    # Plot 4: Pie chart of top centuries
    ax4 = axes[1, 1]
    top_centuries = century_counts.head(7)
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_centuries)))
    wedges, texts, autotexts = ax4.pie(top_centuries.values, labels=[f"{c}th c." for c in top_centuries.index],
                                        autopct='%1.1f%%', colors=colors)
    ax4.set_title('Top Centuries Distribution')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = '../plots/dating_distribution_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()
    
    # ===== Additional: Timeline by decade =====
    print("\nGenerating decade-based analysis...")
    
    df_valid['decade'] = (df_valid['dating_mid'] // 10) * 10
    decade_counts = df_valid['decade'].value_counts().sort_index()
    
    fig2, ax = plt.subplots(figsize=(14, 6))
    ax.bar(decade_counts.index.astype(str), decade_counts.values, color='seagreen', edgecolor='darkgreen')
    ax.set_xlabel('Decade')
    ax.set_ylabel('Number of Inscriptions')
    ax.set_title('Distribution by Decade')
    ax.tick_params(axis='x', rotation=90)
    ax.grid(axis='y', alpha=0.3)
    
    # Show only every nth label if too many
    if len(decade_counts) > 20:
        step = len(decade_counts) // 20
        for i, label in enumerate(ax.get_xticklabels()):
            label.set_visible(i % step == 0)
    
    plt.tight_layout()
    output_path2 = '../plots/dating_by_decade.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Decade visualization saved to: {output_path2}")
    plt.close()
    
    # ===== Save detailed statistics =====
    stats_df = pd.DataFrame({
        'century': century_counts.index,
        'count': century_counts.values,
        'percentage': (century_counts.values / len(df_valid) * 100).round(2)
    })
    stats_df.to_csv('../plots/dating_century_stats.csv', index=False)
    print(f"Century statistics saved to: plots/dating_century_stats.csv")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    return df_valid


if __name__ == '__main__':
    csv_path = 'runes_parsed_parallel_1_10k.csv'
    analyze_dating_distribution(csv_path)
