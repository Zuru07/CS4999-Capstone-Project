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

# Expanded query set for statistical significance
QUERIES = [
    "deep learning neural networks",
    "machine learning optimization",
    "natural language processing",
    "computer vision techniques",
    "reinforcement learning algorithms",
    "transformer attention mechanism",
    "convolutional neural networks",
    "自然言語処理",  # NLP in Japanese
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
]  # 50 queries for statistical significance
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
    
    return {"times": times, "avg_ms": np.mean(times), "results": results}


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
    
    return {"times": times, "avg_ms": np.mean(times), "results": results, "build_time": build_time}


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


def plot_comparison_chart(results, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    engines = ["pgvector", "FAISS"]
    index_types = ["flat", "ivf", "hnsw"]
    colors = {"pgvector": "#e74c3c", "FAISS": "#3498db"}
    
    latencies = {eng: [] for eng in engines}
    recalls = {eng: [] for eng in engines}
    
    for r in results:
        eng = r["engine"]  # Keep original case from results
        idx_type = r["index_type"]
        latencies[eng].append((idx_type, r["latency_ms"]))
        recalls[eng].append((idx_type, r["recall"]))
    
    latencies["pgvector"].sort(key=lambda x: x[0])
    latencies["FAISS"].sort(key=lambda x: x[0])
    recalls["pgvector"].sort(key=lambda x: x[0])
    recalls["FAISS"].sort(key=lambda x: x[0])
    
    if not latencies["FAISS"]:
        print("Warning: No FAISS results found!")
        return
    
    ax1 = axes[0, 0]
    x = np.arange(len(index_types))
    width = 0.35
    pg_lat = [v for _, v in latencies["pgvector"]]
    fa_lat = [v for _, v in latencies["FAISS"]]
    
    bars1 = ax1.bar(x - width/2, pg_lat, width, label="pgvector", color=colors["pgvector"])
    bars2 = ax1.bar(x + width/2, fa_lat, width, label="FAISS", color=colors["FAISS"])
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Search Latency Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Flat", "IVF", "HNSW"])
    ax1.legend()
    ax1.set_yscale("log")
    ax1.bar_label(bars1, fmt="%.1f", fontsize=8)
    ax1.bar_label(bars2, fmt="%.1f", fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")
    
    ax2 = axes[0, 1]
    pg_rec = [v for _, v in recalls["pgvector"]]
    fa_rec = [v for _, v in recalls["FAISS"]]
    bars1 = ax2.bar(x - width/2, pg_rec, width, label="pgvector", color=colors["pgvector"])
    bars2 = ax2.bar(x + width/2, fa_rec, width, label="FAISS", color=colors["FAISS"])
    ax2.set_ylabel("Recall@5")
    ax2.set_title("Recall@5 Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Flat", "IVF", "HNSW"])
    ax2.legend()
    ax2.set_ylim(0, 1.2)
    ax2.bar_label(bars1, fmt="%.2f", fontsize=8)
    ax2.bar_label(bars2, fmt="%.2f", fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")
    
    ax3 = axes[1, 0]
    speedups = []
    for idx_type in index_types:
        pg_lat = next(v for k, v in latencies["pgvector"] if k == idx_type)
        fa_lat = next(v for k, v in latencies["FAISS"] if k == idx_type)
        speedups.append(pg_lat / fa_lat if fa_lat > 0 else 0)
    
    bars = ax3.bar(index_types, speedups, color=["#2ecc71", "#9b59b6", "#f39c12"], edgecolor="black")
    ax3.set_ylabel("Speedup (pgvector/FAISS)")
    ax3.set_title("FAISS Speedup vs pgvector")
    ax3.bar_label(bars, fmt="%.0fx", fontsize=10, fontweight="bold")
    ax3.axhline(y=1, color="red", linestyle="--", alpha=0.5, label="1x (equal)")
    ax3.grid(True, alpha=0.3, axis="y")
    
    ax4 = axes[1, 1]
    ax4.axis("off")
    
    table_data = [["Index Type", "pgvector", "FAISS", "Speedup", "pg Recall", "FA Recall"]]
    for idx_type in index_types:
        pg_lat = next(v for k, v in latencies["pgvector"] if k == idx_type)
        fa_lat = next(v for k, v in latencies["FAISS"] if k == idx_type)
        pg_rec = next(v for k, v in recalls["pgvector"] if k == idx_type)
        fa_rec = next(v for k, v in recalls["FAISS"] if k == idx_type)
        speedup = pg_lat / fa_lat if fa_lat > 0 else 0
        table_data.append([
            idx_type.upper(),
            f"{pg_lat:.2f}ms",
            f"{fa_lat:.4f}ms",
            f"{speedup:.0f}x",
            f"{pg_rec:.2f}",
            f"{fa_rec:.2f}"
        ])
    
    table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    for i in range(6):
        table[(0, i)].set_facecolor("#34495e")
        table[(0, i)].set_text_props(color="white", fontweight="bold")
    ax4.set_title("Performance Summary", fontsize=12, fontweight="bold", pad=20)
    
    plt.suptitle("pgvector vs FAISS: Comprehensive Index Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "index_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: index_comparison.png")


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
    
    results = []
    pgvector_flat_results = []  # Will store pgvector flat as ground truth
    
    results = []
    
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
        print(f"         pg: R={pg_recall:.0%} P={pg_precision:.0%} MRR={pg_mrr:.2} F1={pg_f1:.0%}")
        print(f"         FAISS: R={fa_recall:.0%} P={fa_precision:.0%} MRR={fa_mrr:.2} F1={fa_f1:.0%}")
    
    print("\n[4/4] Generating graphs...")
    plot_comparison_chart(results, RESULTS_DIR)
    plot_latency_detailed(results, RESULTS_DIR)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": datetime.now().isoformat(),
        "dataset_size": len(embeddings),
        "dimension": EMBEDDING_CONFIG.dimension,
        "num_queries": len(QUERIES),
        "results": results,
    }
    fp = RESULTS_DIR / f"index_comparison_{ts}.json"
    with open(fp, "w") as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)
    print(f"\n{'Engine':<10} {'Index':<6} {'Latency':<12} {'Recall':<8} {'Precision':<10} {'MRR':<8} {'F1':<8} {'Speedup'}")
    print("-" * 85)
    
    baseline = next(r["latency_ms"] for r in results if r["engine"].lower() == "pgvector" and r["index_type"] == "flat")
    for r in results:
        speedup = baseline / r["latency_ms"] if r["engine"].lower() == "faiss" else 1.0
        marker = f"{speedup:.0f}x" if r["engine"].lower() == "faiss" else "-"
        print(f"{r['engine']:<10} {r['index_type']:<6} {r['latency_ms']:>10.2f}ms   {r['recall']:.2%}    {r['precision']:.2%}       {r['mrr']:.2f}    {r['f1']:.2%}    {marker}")
    
    print(f"\nResults saved to: {fp}")
    print("=" * 70)


if __name__ == "__main__":
    main()
