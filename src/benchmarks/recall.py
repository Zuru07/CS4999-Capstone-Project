"""Recall@K benchmark for vector search quality."""

import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.pgvector import PGVectorDB
from src.db.faiss_index import FAISSIndex


def compute_recall_at_k(
    retrieved_ids: List[int],
    ground_truth_ids: List[int],
    k: int
) -> float:
    """Calculate Recall@K - fraction of relevant docs retrieved."""
    retrieved_set = set(retrieved_ids[:k])
    gt_set = set(ground_truth_ids)
    if len(gt_set) == 0:
        return 0.0
    return len(retrieved_set & gt_set) / len(gt_set)


def get_ground_truth(
    db: PGVectorDB,
    embedding: np.ndarray,
    k: int = 20
) -> List[int]:
    results = db.search(embedding, limit=k)
    return [r.id for r in results]


def benchmark_recall_by_k(
    queries: List[str],
    pipeline,
    faiss_index: FAISSIndex,
    db: PGVectorDB,
    k_values: List[int] = [1, 5, 10, 20]
) -> Dict:
    results = {k: [] for k in k_values}
    
    for query in queries:
        embedding = pipeline.get_query_embedding(query)
        gt_ids = get_ground_truth(db, embedding, k=max(k_values))
        _, _, faiss_results = faiss_index.search(embedding, k=max(k_values))
        faiss_ids = [r["id"] for r in faiss_results]
        
        for k in k_values:
            recall = compute_recall_at_k(faiss_ids, gt_ids, k)
            results[k].append(recall)
    
    return {
        k: {
            "avg": np.mean(vals),
            "min": np.min(vals),
            "max": np.max(vals),
            "all": vals
        }
        for k, vals in results.items()
    }


def main():
    print("=" * 60)
    print("RECALL@K BENCHMARK")
    print("=" * 60)
    
    queries = [
        "deep learning neural networks",
        "machine learning applications",
        "natural language processing transformers",
        "computer vision convolutional neural networks",
        "reinforcement learning policy gradient"
    ]
    
    from src.rag.generator import RAGPipeline
    
    db = PGVectorDB()
    pipeline = RAGPipeline(db=db)
    
    try:
        faiss_index = FAISSIndex.load("data/cache/faiss_index")
        print("Loaded pre-built FAISS index")
    except FileNotFoundError:
        print("Building sample index...")
        faiss_index = FAISSIndex(dimension=384, index_type="hnsw")
        embeddings, ids = db.get_all_embeddings(limit=5000)
        faiss_index.build(embeddings, ids.tolist())
    
    print("\nBenchmarking Recall@K...")
    results = benchmark_recall_by_k(queries, pipeline, faiss_index, db)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n{'K':<8} {'Avg Recall':<15} {'Min':<12} {'Max':<12}")
    print("-" * 50)
    for k in sorted(results.keys()):
        r = results[k]
        print(f"K={k:<5} {r['avg']:.4f}         {r['min']:.4f}     {r['max']:.4f}")
    
    import json
    from datetime import datetime
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "queries": queries,
        "recall_at_k": results
    }
    
    output_path = Path("data/results/recall_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
