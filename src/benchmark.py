"""Fixed benchmark with proper recall calculation."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.pgvector import PGVectorDB
from src.db.faiss_index import FAISSIndex
from src.data.loader import load_sample_data
from src.config import EMBEDDING_CONFIG
from sentence_transformers import SentenceTransformer

RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

QUERIES = [
    "deep learning neural networks",
    "machine learning optimization",
    "natural language processing",
    "computer vision",
    "reinforcement learning",
]


def main():
    print("=" * 60)
    print("BENCHMARKS")
    print("=" * 60)
    
    print("\n1. Loading model...")
    model = SentenceTransformer(EMBEDDING_CONFIG.model_name)
    query_embs = model.encode(QUERIES)
    
    print("\n2. Loading embeddings from cache...")
    embeddings, _ = load_sample_data(100000)
    print(f"   Using {len(embeddings)} embeddings")
    
    print("\n3. Connecting to DB...")
    db = PGVectorDB()
    print(f"   Docs in DB: {db.count()}")
    
    all_results = []
    
    # ============ pgvector ============
    print("\n" + "=" * 40)
    print("pgvector BENCHMARKS")
    print("=" * 40)
    
    # pgvector Flat (ground truth baseline)
    print("\n4. pgvector Flat (baseline)...")
    db.create_indexes("flat", 1)
    times = []
    pg_gt = []
    for emb in query_embs:
        start = time.perf_counter()
        r = db.search(emb.tolist(), limit=5)
        times.append((time.perf_counter() - start) * 1000)
        pg_gt.append([x.id for x in r])
    avg = sum(times) / len(times)
    print(f"   Latency: {avg:.2f}ms, Recall@5: 1.0000 (baseline)")
    all_results.append({
        "engine": "pgvector", "index": "flat", 
        "latency_ms": avg, "recall": 1.0,
        "baseline_ids": pg_gt
    })
    
    # pgvector IVFFlat
    print("\n5. pgvector IVFFlat-100...")
    db.create_indexes("ivfflat", 100)
    times = []
    pred_ids = []
    for emb in query_embs:
        start = time.perf_counter()
        r = db.search(emb.tolist(), limit=5)
        times.append((time.perf_counter() - start) * 1000)
        pred_ids.append([x.id for x in r])
    avg = sum(times) / len(times)
    recall = sum(len(set(p) & set(g)) / 5 for p, g in zip(pred_ids, pg_gt)) / len(pg_gt)
    print(f"   Latency: {avg:.2f}ms, Recall@5: {recall:.4f}")
    all_results.append({"engine": "pgvector", "index": "ivfflat-100", "latency_ms": avg, "recall": recall})
    
    # pgvector HNSW
    print("\n6. pgvector HNSW...")
    db.create_indexes("hnsw", 16, ef_construction=100)
    times = []
    pred_ids = []
    for emb in query_embs:
        start = time.perf_counter()
        r = db.search(emb.tolist(), limit=5)
        times.append((time.perf_counter() - start) * 1000)
        pred_ids.append([x.id for x in r])
    avg = sum(times) / len(times)
    recall = sum(len(set(p) & set(g)) / 5 for p, g in zip(pred_ids, pg_gt)) / len(pg_gt)
    print(f"   Latency: {avg:.2f}ms, Recall@5: {recall:.4f}")
    all_results.append({"engine": "pgvector", "index": "hnsw", "latency_ms": avg, "recall": recall})
    
    # ============ FAISS ============
    print("\n" + "=" * 40)
    print("FAISS BENCHMARKS")
    print("=" * 40)
    
    # FAISS Flat (baseline)
    print("\n7. FAISS Flat (baseline)...")
    idx_flat = FAISSIndex(dimension=EMBEDDING_CONFIG.dimension, index_type="flat")
    idx_flat.build(embeddings, list(range(len(embeddings))))
    times = []
    faiss_gt = []
    for emb in query_embs:
        start = time.perf_counter()
        _, _, r = idx_flat.search(emb, k=5)
        times.append((time.perf_counter() - start) * 1000)
        faiss_gt.append([x["id"] for x in r])
    avg = sum(times) / len(times)
    print(f"   Latency: {avg:.4f}ms, Recall@5: 1.0000 (baseline)")
    all_results.append({
        "engine": "FAISS", "index": "flat", 
        "latency_ms": avg, "recall": 1.0,
        "baseline_ids": faiss_gt
    })
    
    # FAISS IVFFlat
    print("\n8. FAISS IVFFlat-100...")
    idx_ivf = FAISSIndex(dimension=EMBEDDING_CONFIG.dimension, index_type="ivf", nlist=100, nprobe=10)
    idx_ivf.build(embeddings, list(range(len(embeddings))))
    times = []
    pred_ids = []
    for emb in query_embs:
        start = time.perf_counter()
        _, _, r = idx_ivf.search(emb, k=5)
        times.append((time.perf_counter() - start) * 1000)
        pred_ids.append([x["id"] for x in r])
    avg = sum(times) / len(times)
    recall = sum(len(set(p) & set(g)) / 5 for p, g in zip(pred_ids, faiss_gt)) / len(faiss_gt)
    print(f"   Latency: {avg:.4f}ms, Recall@5: {recall:.4f}")
    all_results.append({"engine": "FAISS", "index": "ivf-100", "latency_ms": avg, "recall": recall})
    
    # FAISS HNSW
    print("\n9. FAISS HNSW...")
    start = time.perf_counter()
    idx_hnsw = FAISSIndex(dimension=EMBEDDING_CONFIG.dimension, index_type="hnsw", 
                          hnsw_m=8, hnsw_ef_construction=50)
    idx_hnsw.build(embeddings, list(range(len(embeddings))))
    build_time = time.perf_counter() - start
    print(f"   Build: {build_time:.2f}s")
    
    times = []
    pred_ids = []
    for emb in query_embs:
        start = time.perf_counter()
        _, _, r = idx_hnsw.search(emb, k=5)
        times.append((time.perf_counter() - start) * 1000)
        pred_ids.append([x["id"] for x in r])
    avg = sum(times) / len(times)
    recall = sum(len(set(p) & set(g)) / 5 for p, g in zip(pred_ids, faiss_gt)) / len(faiss_gt)
    print(f"   Latency: {avg:.4f}ms, Recall@5: {recall:.4f}")
    all_results.append({"engine": "FAISS", "index": "hnsw", "latency_ms": avg, "recall": recall})
    
    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset_size": len(embeddings),
        "dimension": EMBEDDING_CONFIG.dimension,
        "model": EMBEDDING_CONFIG.model_name,
        "results": all_results
    }
    fp = RESULTS_DIR / f"benchmark_{ts}.json"
    with open(fp, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"\n{'Engine':<10} {'Index':<15} {'Latency':<12} {'Recall@5':<10} {'Speedup'}")
    print("-" * 60)
    
    pg_baseline = all_results[0]["latency_ms"]
    faiss_baseline = all_results[3]["latency_ms"]
    
    for r in all_results:
        if r["engine"] == "pgvector":
            speedup = pg_baseline / r["latency_ms"]
        else:
            speedup = faiss_baseline / r["latency_ms"]
        print(f"{r['engine']:<10} {r['index']:<15} {r['latency_ms']:>8.2f}ms   {r['recall']:.4f}     {speedup:.0f}x")
    
    print(f"\nSaved to: {fp}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
