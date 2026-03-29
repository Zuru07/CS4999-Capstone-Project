"""
analyze_results.py
Separate plots for index benchmark results (build / exec / planning)
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

# ---------- SINGLE PLOT FUNCTION ---------- #
def plot_metric(df, metric_col, ylabel, filename, log_scale=False):
    sizes = sorted(df['size'].unique())
    indices = df['index'].unique()
    width = 0.25

    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    max_val = 0
    for j, idx in enumerate(indices):
        vals = []
        for size in sizes:
            subset = df[df['size']==size]
            val = get_metric_value(subset, idx, metric_col)
            vals.append(val)
            max_val = max(max_val, val)
        # shift bar positions slightly for each index
        x_pos = np.arange(len(sizes)) + j * width
        ax.bar(x_pos, vals, width=width, label=idx, alpha=0.8)

    # x-axis formatting
    mid_positions = np.arange(len(sizes)) + width * (len(indices) - 1) / 2
    ax.set_xticks(mid_positions)
    ax.set_xticklabels([f"{int(s/1000)}k" for s in sizes])
    ax.set_xlabel("Dataset size")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    if log_scale:
        ax.set_yscale('log')
    else:
        ax.set_ylim(0, max_val * 1.2)

    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"✅ Saved: {filename}")

# ---------- GENERATE SEPARATE PLOTS ---------- #
plot_metric(df, "build_time_s", "Index Build Time (s)", "results/plots/build_time.png", log_scale=True)
plot_metric(df, "exec_ms", "Query Execution Time (ms)", "results/plots/exec_time.png", log_scale=False)
plot_metric(df, "plan_ms", "Query Planning Time (ms)", "results/plots/plan_time.png", log_scale=False)
