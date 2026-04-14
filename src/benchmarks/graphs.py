"""Generate visualizations from benchmark results."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_results():
    results_path = Path("data/results/comprehensive_benchmark.json")
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    raise FileNotFoundError("Run benchmarks first: python -m src.benchmarks.comprehensive")


def plot_latency_comparison(data, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    components = ["Embedding", "pgvector", "FAISS", "Doc Fetch"]
    latencies = [
        data["latency"]["embedding_generation"]["avg_ms"],
        data["latency"]["pgvector_search"]["avg_ms"],
        data["latency"]["faiss_search"]["avg_ms"],
        data["latency"]["document_fetch"]["avg_ms"],
    ]
    mins = [
        data["latency"]["embedding_generation"]["min_ms"],
        data["latency"]["pgvector_search"]["min_ms"],
        data["latency"]["faiss_search"]["min_ms"],
        data["latency"]["document_fetch"]["min_ms"],
    ]
    maxs = [
        data["latency"]["embedding_generation"]["max_ms"],
        data["latency"]["pgvector_search"]["max_ms"],
        data["latency"]["faiss_search"]["max_ms"],
        data["latency"]["document_fetch"]["max_ms"],
    ]
    
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    bars = ax.bar(components, latencies, color=colors, edgecolor="black", linewidth=1.2)
    
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("RAG Pipeline Component Latency", fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    
    for bar, lat in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f"{lat:.1f}ms", ha="center", va="bottom", fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    fig.savefig(output_dir / "latency_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: latency_comparison.png")


def plot_recall_at_k(data, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    k_values = [int(k) for k in data["recall"].keys()]
    recall_avg = [data["recall"][str(k)]["avg"] for k in k_values]
    recall_min = [data["recall"][str(k)]["min"] for k in k_values]
    recall_max = [data["recall"][str(k)]["max"] for k in k_values]
    
    ax.plot(k_values, recall_avg, "o-", color="#2980b9", linewidth=2.5, markersize=10, label="Average Recall")
    ax.fill_between(k_values, recall_min, recall_max, alpha=0.2, color="#2980b9")
    
    ax.set_xlabel("K", fontsize=12)
    ax.set_ylabel("Recall@K", fontsize=12)
    ax.set_title("FAISS HNSW Recall@K Performance", fontsize=14, fontweight="bold")
    ax.set_xticks(k_values)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.8, color="red", linestyle="--", alpha=0.5, label="80% threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / "recall_at_k.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: recall_at_k.png")


def plot_pipeline_breakdown(data, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    total = data["summary"]["total_pipeline_time_ms"]
    components = ["Embedding", "FAISS Search", "Doc Fetch"]
    times = [
        data["latency"]["embedding_generation"]["avg_ms"],
        data["latency"]["faiss_search"]["avg_ms"],
        data["latency"]["document_fetch"]["avg_ms"],
    ]
    percentages = [t / total * 100 for t in times]
    colors = ["#3498db", "#2ecc71", "#9b59b6"]
    
    ax1 = axes[0]
    bars = ax1.barh(components, times, color=colors, edgecolor="black")
    ax1.set_xlabel("Time (ms)", fontsize=12)
    ax1.set_title("Pipeline Component Times", fontsize=12, fontweight="bold")
    for bar, t, p in zip(bars, times, percentages):
        ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                 f"{t:.1f}ms ({p:.1f}%)", va="center", fontsize=10)
    ax1.set_xlim(0, max(times) * 1.4)
    
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(percentages, labels=components, colors=colors, autopct="%1.1f%%",
                                        startangle=90, explode=(0, 0.05, 0))
    ax2.set_title("Pipeline Time Distribution", fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    fig.savefig(output_dir / "pipeline_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: pipeline_breakdown.png")


def plot_speedup_comparison(data, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    speedup = data["summary"]["faiss_speedup_vs_pgvector"]
    
    bars = ax.bar(["pgvector", "FAISS"], [53.2, 0.4], color=["#e74c3c", "#2ecc71"], edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Search Latency (ms)", fontsize=12)
    ax.set_title(f"FAISS Speedup: {speedup:.0f}x Faster than pgvector", fontsize=14, fontweight="bold")
    
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{bar.get_height():.1f}ms", ha="center", va="bottom", fontsize=14, fontweight="bold")
    
    ax.annotate(f"{speedup:.0f}x", xy=(1, 0.4), xytext=(1.3, 20),
                fontsize=16, fontweight="bold", color="green",
                arrowprops=dict(arrowstyle="->", color="green", lw=2))
    
    plt.tight_layout()
    fig.savefig(output_dir / "speedup_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: speedup_comparison.png")


def plot_query_latencies(data, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    queries = ["DL Neural\nNetworks", "ML\nApplications", "NLP\nTransformers", 
               "CV CNNs", "RL Policy\nGradient"]
    
    embedding_times = data["latency"]["embedding_generation"]["times"]
    pgvector_times = data["latency"]["pgvector_search"]["times"]
    faiss_times = data["latency"]["faiss_search"]["times"]
    
    x = np.arange(len(queries))
    width = 0.25
    
    bars1 = ax.bar(x - width, embedding_times, width, label="Embedding", color="#3498db")
    bars2 = ax.bar(x, pgvector_times, width, label="pgvector", color="#e74c3c")
    bars3 = ax.bar(x + width, [f * 100 for f in faiss_times], width, label="FAISS (x100)", color="#2ecc71")
    
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Per-Query Latency Breakdown", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(queries)
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    fig.savefig(output_dir / "query_latencies.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: query_latencies.png")


def generate_all_graphs():
    print("=" * 50)
    print("GENERATING BENCHMARK GRAPHS")
    print("=" * 50)
    
    try:
        data = load_results()
    except FileNotFoundError:
        print("ERROR: Run benchmarks first!")
        print("  python -m src.benchmarks.comprehensive")
        return
    
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating graphs...")
    plot_latency_comparison(data, output_dir)
    plot_recall_at_k(data, output_dir)
    plot_pipeline_breakdown(data, output_dir)
    plot_speedup_comparison(data, output_dir)
    plot_query_latencies(data, output_dir)
    
    print("\n" + "=" * 50)
    print("All graphs saved to data/results/")
    print("=" * 50)


if __name__ == "__main__":
    generate_all_graphs()
