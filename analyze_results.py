"""
analyze_results.py
Combined plots for index benchmark results (build / exec / planning)
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Load results
fn = "results/index_benchmark_results.csv"
df = pd.read_csv(fn)

# derive dataset size from table name, e.g., hf_docs_10k -> 10000
def table_to_size(t):
    import re
    m = re.search(r'(\d+)(k?)', str(t))
    if not m:
        return t
    num = int(m.group(1))
    if m.group(2).lower() == 'k':
        return num * 1000
    return num

df['size'] = df['table'].apply(table_to_size)
df['size_label'] = df['size'].apply(lambda x: f"{int(x/1000)}k")

os.makedirs("results/plots", exist_ok=True)

# ---------- HELPER FUNCTION ---------- #
def get_metric_value(subset, idx, metric_col):
    sub_idx = subset[subset['index']==idx]
    if sub_idx.empty:
        return 0
    return sub_idx[metric_col].values[0]

# ---------- PLOTTING ---------- #
def plot_combined(df, out_file):
    sizes = sorted(df['size'].unique())
    indices = df['index'].unique()
    width = 0.35

    fig, axes = plt.subplots(1,3, figsize=(18,5))

    metrics = [
        ("build_time_s", "Index Build Time (s)", True),
        ("exec_ms", "Query Execution Time (ms)", False),
        ("plan_ms", "Query Planning Time (ms)", False)
    ]

    for ax, (metric_col, ylabel, log_scale) in zip(axes, metrics):
        max_val = 0
        for i, size in enumerate(sizes):
            subset = df[df['size']==size]
            for j, idx in enumerate(indices):
                val = get_metric_value(subset, idx, metric_col)
                max_val = max(max_val, val)
                ax.bar(i + j*width, val, width=width, label=idx if i==0 else "", alpha=0.8)
                ax.text(i + j*width, val + max_val*0.01 + 0.01, f"{val:.2f}",
                        ha='center', va='bottom', fontsize=8)

        ax.set_xticks([i + width/2 for i in range(len(sizes))])
        ax.set_xticklabels([f"{int(s/1000)}k" for s in sizes])
        ax.set_xlabel("Dataset size")
        ax.set_ylabel(ylabel)
        if log_scale:
            ax.set_yscale('log')
        else:
            ax.set_ylim(0, max_val*1.15)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_title(ylabel)

    axes[0].legend()
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

plot_combined(df, "results/plots/index_benchmark_combined.png")
print("✅ Combined plot saved: results/plots/index_benchmark_combined.png")
