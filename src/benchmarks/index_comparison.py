"""Comprehensive benchmark: pgvector vs FAISS across all index types."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.pgvector import PGVectorDB
from src.db.faiss_index import FAISSIndex
from src.data.loader import load_sample_data
from src.config import EMBEDDING_CONFIG
from sentence_transformers import SentenceTransformer

RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Full query set for publication (50 queries)
QUERIES = [
    "deep learning neural networks",
    "machine learning optimization",
    "natural language processing",
    "computer vision techniques",
    "reinforcement learning algorithms",
    "transformer attention mechanism",
    "convolutional neural networks",
    "optimization gradient descent",
    "recurrent neural networks LSTM",
    "generative adversarial networks",
    "support vector machine",
    "bert language model",
    "image classification CNN",
    "graph neural networks",
    "federated learning",
    "meta learning few-shot",
    "attention is all you need",
    "word embedding BERT",
    "object detection YOLO",
    "neural machine translation",
    "semi-supervised learning",
    "self-supervised contrastive",
    "diffusion model generation",
    "stable diffusion",
    "large language model",
    "prompt engineering",
    "in-context learning",
    "chain of thought reasoning",
    "retrieval augmented generation",
    "vector database embedding",
    "approximate nearest neighbor",
    "hierarchical navigable small world",
    "inverted file index",
    "product quantization",
    "locality sensitive hashing",
    "semantic search embedding",
    "cross-encoder reranking",
    "dense retrieval sparse",
    "question answering system",
    "information retrieval",
    "document ranking",
    "passage retrieval",
    "semantic similarity",
    "text matching",
    "semantic embedding space",
    "neural search",
    "embedding quantization",
    "index compression",
    "memory efficient search",
]

# Number of runs for statistical significance
NUM_RUNS = 3

INDEX_CONFIGS = {
    "flat": {
        "pgvector": {"index_type": "flat", "params": {}},
        "faiss": {"index_type": "flat", "params": {}},
    },
    "ivf": {
        "pgvector": {"index_type": "ivfflat", "params": {"nlist": 100}},
        "faiss": {"index_type": "ivf", "params": {"nlist": 100, "nprobe": 10}},
    },
    "hnsw": {
        "pgvector": {"index_type": "hnsw", "params": {"ef_construction": 100}},
        "faiss": {"index_type": "hnsw", "params": {"hnsw_m": 16, "ef_construction": 100, "ef_search": 50}},
    },
}


def benchmark_pgvector(db, query_embs, index_config, limit=5, warmup=3):
    """Benchmark pgvector search."""
    idx_type = index_config["pgvector"]["index_type"]
    params = index_config["pgvector"]["params"]
    db.create_indexes(idx_type, **params)
    
    # Warm-up runs
    for i in range(warmup):
        if query_embs.shape[0] > 0:
            db.search(query_embs[0].tolist(), limit=limit)
    
    times = []
    results = []
    for emb in query_embs:
        start = time.perf_counter()
        r = db.search(emb.tolist(), limit=limit)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        results.append([x.id for x in r])
    
    return {
    "times": times, 
    "avg_ms": np.mean(times), 
    "std_ms": np.std(times),
    "min_ms": np.min(times),
    "max_ms": np.max(times),
    "results": results
}


def benchmark_faiss(embeddings, ids, query_embs, index_config, limit=5, warmup=3):
    """Benchmark FAISS search."""
    faiss_config = index_config["faiss"]
    params = faiss_config["params"]
    
    idx = FAISSIndex(
        dimension=EMBEDDING_CONFIG.dimension,
        index_type=faiss_config["index_type"],
        **{k: v for k, v in params.items() if k in ["nlist", "nprobe", "hnsw_m", "hnsw_ef_construction", "hnsw_ef_search"]}
    )
    
    build_start = time.perf_counter()
    idx.build(embeddings, ids)
    build_time = time.perf_counter() - build_start
    
    # Warm-up runs (query first few embeddings multiple times)
    for i in range(warmup):
        for j in range(min(3, query_embs.shape[0])):
            idx.search(query_embs[j], k=limit)
    
    times = []
    results = []
    for emb in query_embs:
        start = time.perf_counter()
        _, _, r = idx.search(emb, k=limit)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        results.append([x["id"] for x in r])
    
    return {
    "times": times, 
    "avg_ms": np.mean(times), 
    "std_ms": np.std(times),
    "min_ms": np.min(times),
    "max_ms": np.max(times),
    "results": results, 
    "build_time": build_time
}


def calculate_recall(predicted, ground_truth, k=5):
    """Calculate recall@K: what fraction of relevant docs did we find?"""
    recalls = []
    for pred, gt in zip(predicted, ground_truth):
        pred_set = set(pred[:k])
        gt_set = set(gt[:k])
        if len(gt_set) == 0:
            recalls.append(0.0)
        else:
            recalls.append(len(pred_set & gt_set) / len(gt_set))
    return np.mean(recalls)


def calculate_precision(predicted, ground_truth, k=5):
    """Calculate precision@K: of what we found, how many are relevant?"""
    precisions = []
    for pred, gt in zip(predicted, ground_truth):
        pred_set = set(pred[:k])
        gt_set = set(gt[:k])
        if k == 0:
            precisions.append(0.0)
        else:
            precisions.append(len(pred_set & gt_set) / k)
    return np.mean(precisions)


def calculate_mrr(predicted, ground_truth, k=5):
    """Mean Reciprocal Rank: where's the first relevant result?"""
    reciprocal_ranks = []
    for pred, gt in zip(predicted, ground_truth):
        gt_set = set(gt)
        found_rank = 0
        for i, doc_id in enumerate(pred[:k]):
            if doc_id in gt_set:
                found_rank = i + 1  # 1-indexed
                break
        if found_rank > 0:
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)
    return np.mean(reciprocal_ranks)


def calculate_f1(precision, recall):
    """F1@K: harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def aggregate_run_results(all_runs):
    """Aggregate results from multiple runs: mean ± std."""
    aggregated = {}
    
    for run_results in all_runs:
        for r in run_results:
            key = (r["engine"], r["index_type"])
            if key not in aggregated:
                aggregated[key] = {
                    "engine": r["engine"],
                    "index_type": r["index_type"],
                    "latency_ms": [],
                    "recall": [],
                    "precision": [],
                    "mrr": [],
                    "f1": []
                }
            aggregated[key]["latency_ms"].append(r["latency_ms"])
            aggregated[key]["recall"].append(r["recall"])
            aggregated[key]["precision"].append(r["precision"])
            aggregated[key]["mrr"].append(r["mrr"])
            aggregated[key]["f1"].append(r["f1"])
    
    final_results = []
    for key, data in aggregated.items():
        final_results.append({
            "engine": data["engine"],
            "index_type": data["index_type"],
            "latency_ms": np.mean(data["latency_ms"]),
            "std_ms": np.std(data["latency_ms"]),
            "recall": np.mean(data["recall"]),
            "precision": np.mean(data["precision"]),
            "mrr": np.mean(data["mrr"]),
            "f1": np.mean(data["f1"]),
            "all_runs": {
                "latency_ms": data["latency_ms"],
                "recall": data["recall"],
                "precision": data["precision"],
            }
        })
    
    return final_results


def prepare_data(results):
    """Prepare sorted data for plotting."""
    engines = ["pgvector", "FAISS"]
    index_types = ["flat", "ivf", "hnsw"]
    
    data = {eng: {} for eng in engines}
    for r in results:
        eng = r["engine"]
        idx = r["index_type"]
        data[eng][idx] = r
    
    def get_order(key):
        order = {"flat": 0, "ivf": 1, "hnsw": 2}
        return order.get(key, 99)
    
    return data, get_order


def plot_latency_graph(results, output_dir):
    """Plot latency comparison."""
    data, get_order = prepare_data(results)
    index_types = ["flat", "ivf", "hnsw"]
    colors = {"pgvector": "#e74c3c", "FAISS": "#3498db"}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(index_types))
    width = 0.35
    
    pg_lat = [data["pgvector"][k]["latency_ms"] for k in index_types]
    fa_lat = [data["FAISS"][k]["latency_ms"] for k in index_types]
    
    ax.bar(x - width/2, pg_lat, width, label="pgvector", color=colors["pgvector"])
    ax.bar(x + width/2, fa_lat, width, label="FAISS", color=colors["FAISS"])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Search Latency Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(["Flat", "IVF", "HNSW"])
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(output_dir / "graph_latency.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: graph_latency.png")


def plot_recall_graph(results, output_dir):
    """Plot Recall@K comparison."""
    data, get_order = prepare_data(results)
    index_types = ["flat", "ivf", "hnsw"]
    colors = {"pgvector": "#e74c3c", "FAISS": "#3498db"}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(index_types))
    width = 0.35
    
    pg_rec = [data["pgvector"][k]["recall"] for k in index_types]
    fa_rec = [data["FAISS"][k]["recall"] for k in index_types]
    
    ax.bar(x - width/2, pg_rec, width, label="pgvector", color=colors["pgvector"])
    ax.bar(x + width/2, fa_rec, width, label="FAISS", color=colors["FAISS"])
    ax.set_ylabel("Recall@5")
    ax.set_title("Recall@5 Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(["Flat", "IVF", "HNSW"])
    ax.legend()
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(output_dir / "graph_recall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: graph_recall.png")


def plot_precision_graph(results, output_dir):
    """Plot Precision@K comparison."""
    data, get_order = prepare_data(results)
    index_types = ["flat", "ivf", "hnsw"]
    colors = {"pgvector": "#e74c3c", "FAISS": "#3498db"}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(index_types))
    width = 0.35
    
    pg_prec = [data["pgvector"][k]["precision"] for k in index_types]
    fa_prec = [data["FAISS"][k]["precision"] for k in index_types]
    
    ax.bar(x - width/2, pg_prec, width, label="pgvector", color=colors["pgvector"])
    ax.bar(x + width/2, fa_prec, width, label="FAISS", color=colors["FAISS"])
    ax.set_ylabel("Precision@5")
    ax.set_title("Precision@5 Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(["Flat", "IVF", "HNSW"])
    ax.legend()
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(output_dir / "graph_precision.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: graph_precision.png")


def plot_f1_graph(results, output_dir):
    """Plot F1@K comparison."""
    data, get_order = prepare_data(results)
    index_types = ["flat", "ivf", "hnsw"]
    colors = {"pgvector": "#e74c3c", "FAISS": "#3498db"}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(index_types))
    width = 0.35
    
    pg_f1 = [data["pgvector"][k]["f1"] for k in index_types]
    fa_f1 = [data["FAISS"][k]["f1"] for k in index_types]
    
    ax.bar(x - width/2, pg_f1, width, label="pgvector", color=colors["pgvector"])
    ax.bar(x + width/2, fa_f1, width, label="FAISS", color=colors["FAISS"])
    ax.set_ylabel("F1@5")
    ax.set_title("F1@5 Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(["Flat", "IVF", "HNSW"])
    ax.legend()
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(output_dir / "graph_f1.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: graph_f1.png")


def plot_speedup_graph(results, output_dir):
    """Plot speedup comparison."""
    data, get_order = prepare_data(results)
    index_types = ["flat", "ivf", "hnsw"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    speedups = []
    for idx in index_types:
        pg = data["pgvector"][idx]["latency_ms"]
        fa = data["FAISS"][idx]["latency_ms"]
        speedups.append(pg / fa if fa > 0 else 0)
    
    bars = ax.bar(index_types, speedups, color=["#2ecc71", "#9b59b6", "#f39c12"], edgecolor="black")
    ax.set_ylabel("Speedup (pgvector/FAISS)")
    ax.set_title("FAISS Speedup vs pgvector")
    ax.bar_label(bars, fmt="%.0fx", fontsize=12, fontweight="bold")
    ax.axhline(y=1, color="red", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(output_dir / "graph_speedup.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: graph_speedup.png")


def plot_summary_table(results, output_dir):
    """Plot summary table."""
    data, get_order = prepare_data(results)
    index_types = ["flat", "ivf", "hnsw"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    
    table_data = [["Index", "Engine", "Latency", "Recall", "Precision", "F1", "Speedup"]]
    baseline = data["pgvector"]["flat"]["latency_ms"]
    
    for idx in index_types:
        for eng in ["pgvector", "FAISS"]:
            r = data[eng][idx]
            if eng == "FAISS":
                spd = int(baseline / r["latency_ms"])
                spd_str = f"{spd}x"
            else:
                spd_str = "-"
            table_data.append([
                idx.upper(), eng, f"{r['latency_ms']:.2f}ms", 
                f"{r['recall']:.0%}", f"{r['precision']:.0%}", 
                f"{r['f1']:.0%}", spd_str
            ])
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                    loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor("#34495e")
        table[(0, i)].set_text_props(color="white", fontweight="bold")
    ax.set_title("Performance Summary", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(output_dir / "graph_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: graph_summary.png")


def plot_comparison_chart(results, output_dir):
    """Plot all graphs in separate files."""
    plot_latency_graph(results, output_dir)
    plot_recall_graph(results, output_dir)
    plot_precision_graph(results, output_dir)
    plot_f1_graph(results, output_dir)
    plot_speedup_graph(results, output_dir)
    plot_summary_table(results, output_dir)


def plot_latency_detailed(results, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, idx_type in zip(axes, ["flat", "ivf", "hnsw"]):
        pg_result = next(r for r in results if r["engine"].lower() == "pgvector" and r["index_type"] == idx_type)
        fa_result = next(r for r in results if r["engine"].lower() == "faiss" and r["index_type"] == idx_type)
        
        times = [pg_result["times"], fa_result["times"]]
        labels = ["pgvector", "FAISS"]
        colors = ["#e74c3c", "#3498db"]
        
        bp = ax.boxplot(times, labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(f"{idx_type.upper()} Index")
        ax.set_ylabel("Latency (ms)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")
        
        pg_avg = pg_result["latency_ms"]
        fa_avg = fa_result["latency_ms"]
        speedup = pg_avg / fa_avg if fa_avg > 0 else 0
        ax.text(0.5, 0.95, f"FAISS: {speedup:.0f}x faster", transform=ax.transAxes,
                ha="center", fontsize=10, fontweight="bold", color="green")
    
    plt.suptitle("Latency Distribution by Index Type", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "latency_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: latency_boxplot.png")


def main():
    print("=" * 70)
    print("COMPREHENSIVE INDEX COMPARISON: pgvector vs FAISS")
    print("=" * 70)
    
    print("\n[1/4] Loading model...")
    model = SentenceTransformer(EMBEDDING_CONFIG.model_name)
    query_embs = model.encode(QUERIES)
    
    print("[2/4] Loading cached embeddings...")
    embeddings, ids = load_sample_data(100000)
    print(f"  Using {len(embeddings)} embeddings, {embeddings.shape[1]}D")
    
    print("[3/4] Connecting to database...")
    db = PGVectorDB()
    print(f"  Documents in DB: {db.count()}")
    
    all_run_results = []  # Store results from all runs
    
    print(f"\n[4/4] Running benchmark {NUM_RUNS} times...")
    for run_num in range(1, NUM_RUNS + 1):
        print(f"\n{'='*50}")
        print(f"RUN {run_num}/{NUM_RUNS}")
        print(f"{'='*50}")
        
        results = []
        pgvector_flat_results = []
    
    # Pass 1: Run all flat indexes first to establish ground truth
    print("\n" + "="*50)
    print("PASS 1: Establishing ground truth (Flat indexes)")
    print("="*50)
    
    flat_config = INDEX_CONFIGS["flat"]
    ground_truths = {}
    
    for engine in ["pgvector", "FAISS"]:
        print(f"\n  {engine} Flat (ground truth)...")
        if engine == "pgvector":
            pg_result = benchmark_pgvector(db, query_embs, flat_config)
            gt = pg_result["results"]
            results.append({
                "engine": engine, "index_type": "flat", 
                "latency_ms": pg_result["avg_ms"], 
                "recall": 1.0, "precision": 1.0, "mrr": 1.0, "f1": 1.0,
                "times": pg_result["times"]
            })
        else:
            fa_result = benchmark_faiss(embeddings, ids, query_embs, flat_config)
            gt = fa_result["results"]
            results.append({
                "engine": engine, "index_type": "flat", 
                "latency_ms": fa_result["avg_ms"], 
                "recall": 1.0, "precision": 1.0, "mrr": 1.0, "f1": 1.0,
                "times": fa_result["times"]
            })
        
        ground_truths[engine] = gt
    
    # Pass 2: Run IVF and HNSW, compare to respective engine's flat
    print("\n" + "="*50)
    print("PASS 2: Benchmarking IVF and HNSW")
    print("="*50)
    
    for idx_type in ["ivf", "hnsw"]:
        config = INDEX_CONFIGS[idx_type]
        
        print(f"\n--- {idx_type.upper()} Index ---")
        
        print("\n  pgvector...")
        pg_result = benchmark_pgvector(db, query_embs, config)
        pg_gt = ground_truths["pgvector"]
        pg_recall = calculate_recall(pg_result["results"], pg_gt)
        pg_precision = calculate_precision(pg_result["results"], pg_gt)
        pg_mrr = calculate_mrr(pg_result["results"], pg_gt)
        pg_f1 = calculate_f1(pg_precision, pg_recall)
        
        print("  FAISS...")
        fa_result = benchmark_faiss(embeddings, ids, query_embs, config)
        fa_gt = ground_truths["FAISS"]
        fa_recall = calculate_recall(fa_result["results"], fa_gt)
        fa_precision = calculate_precision(fa_result["results"], fa_gt)
        fa_mrr = calculate_mrr(fa_result["results"], fa_gt)
        fa_f1 = calculate_f1(fa_precision, fa_recall)
        
        results.append({"engine": "pgvector", "index_type": idx_type, "latency_ms": pg_result["avg_ms"], 
                    "recall": pg_recall, "precision": pg_precision, "mrr": pg_mrr, "f1": pg_f1, "times": pg_result["times"]})
        results.append({"engine": "FAISS", "index_type": idx_type, "latency_ms": fa_result["avg_ms"], 
                    "recall": fa_recall, "precision": fa_precision, "mrr": fa_mrr, "f1": fa_f1, "times": fa_result["times"]})
        
        speedup = pg_result["avg_ms"] / fa_result["avg_ms"] if fa_result["avg_ms"] > 0 else 0
        print(f"\n  {idx_type.upper()}: pg {pg_result['avg_ms']:.2f}ms vs FAISS {fa_result['avg_ms']:.4f}ms ({speedup:.0f}x)")
        print(f"         pg: R={pg_recall:.0%} P={pg_precision:.0%} F1={pg_f1:.0%}")
        print(f"         FAISS: R={fa_recall:.0%} P={fa_precision:.0%} F1={fa_f1:.0%}")
        
        all_run_results.append(results)
    
    # Aggregate results from all runs
    if NUM_RUNS > 1:
        results = aggregate_run_results(all_run_results)
        print("\n[Aggregated] Results from all runs (mean ± std)")
    
    print("\n[4/4] Generating graphs...")
    plot_comparison_chart(results, RESULTS_DIR)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": datetime.now().isoformat(),
        "dataset_size": len(embeddings),
        "dimension": EMBEDDING_CONFIG.dimension,
        "num_queries": len(QUERIES),
        "num_runs": NUM_RUNS,
        "results": results,
    }
    fp = RESULTS_DIR / f"index_comparison_{ts}.json"
    with open(fp, "w") as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "=" * 110)
    print("FINAL SUMMARY (Statistical Analysis)")
    print("=" * 110)
    print(f"\n{'Engine':<10} {'Index':<6} {'Latency (ms)':<20} {'Recall':<8} {'Precision':<10} {'F1':<8} {'Speedup'}")
    print("-" * 85)
    
    baseline = next(r["latency_ms"] for r in results if r["engine"].lower() == "pgvector" and r["index_type"] == "flat")
    for r in results:
        speedup = baseline / r["latency_ms"] if r["engine"].lower() == "faiss" else 1.0
        marker = f"{speedup:.0f}x" if r["engine"].lower() == "faiss" else "-"
        # Format with mean ± std
        std_val = r.get("std_ms", 0)
        if std_val > 0:
            lat_str = f"{r['latency_ms']:.2f} ± {std_val:.2f}"
        else:
            lat_str = f"{r['latency_ms']:.2f}"
        print(f"{r['engine']:<10} {r['index_type']:<6} {lat_str:<20} {r['recall']:.2%}    {r['precision']:.2%}    {r['f1']:.2%}    {marker}")
    
    # Print sample size info
    num_queries = len(QUERIES)
    num_docs = len(embeddings)
    print(f"\nSample size: {num_queries} queries, {num_docs} embeddings ({num_queries * num_docs} query-doc pairs)")
    print(f"Results saved to: {fp}")
    print("=" * 110)


if __name__ == "__main__":
    main()
