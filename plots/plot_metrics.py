import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""Simple utility to plot BLEU and chrF++ from metrics_summary.csv"""

def load_metrics(path="metrics_summary.csv"):
    return pd.read_csv(path)


def plot_metrics(df, output_prefix="metrics"):
    sns.set(style="whitegrid")

    # bar plot for BLEU
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="Model", y="BLEU", palette="Blues_d")
    plt.title("Model BLEU Scores")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_bleu.png")
    plt.close()

    # bar plot for chrF++
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="Model", y="chrF++", palette="Greens_d")
    plt.title("Model chrF++ Scores")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_chrf.png")
    plt.close()

    print(f"Plots saved as {output_prefix}_bleu.png and {output_prefix}_chrf.png")


if __name__ == "__main__":
    df = load_metrics()
    if df.empty:
        print("No metrics data found. Run inference.py first.")
    else:
        plot_metrics(df)
