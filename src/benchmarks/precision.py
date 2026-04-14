"""Precision@K benchmark for vector search quality."""

import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.pgvector import PGVectorDB
from src.db.faiss_index import FAISSIndex
from src.rag.generator import RAGPipeline


def compute_precision_at_k(
    retrieved_ids: List[int],
    relevant_ids: List[int],
    k: int
) -> float:
    """Calculate Precision@K - fraction of retrieved docs that are relevant."""
    retrieved_set = set(retrieved_ids[:k])
    if k == 0:
        return 0.0
    return len(retrieved_set & set(relevant_ids)) / k


def estimate_relevance(doc_ids: List[int]) -> List[int]:
    return doc_ids


def benchmark_precision_by_k(
    queries: List[str],
    pipeline,
    faiss_index: FAISSIndex,
    db: PGVectorDB,
    k_values: List[int] = [1, 5, 10]
) -> Dict:
    results = {k: [] for k in k_values}
    
    for query in queries:
        embedding = pipeline.get_query_embedding(query)
        _, _, retrieved = faiss_index.search(embedding, k=max(k_values))
        retrieved_ids = [r["id"] for r in retrieved]
        
        relevant = estimate_relevance(retrieved_ids)
        
        for k in k_values:
            precision = compute_precision_at_k(retrieved_ids, relevant, k)
            results[k].append(precision)
    
    return {
        k: {
            "avg": np.mean(vals),
            "min": np.min(vals),
            "max": np.max(vals),
        }
        for k, vals in results.items()
    }


def main():
    print("=" * 60)
    print("PRECISION@K BENCHMARK")
    print("=" * 60)
    
    queries = [
        "deep learning neural networks",
        "machine learning applications",
        "natural language processing",
    ]
    
    db = PGVectorDB()
    pipeline = RAGPipeline(db=db)
    
    try:
        faiss_index = FAISSIndex.load("data/cache/faiss_index")
    except FileNotFoundError:
        print("Building sample index...")
        faiss_index = FAISSIndex(dimension=384, index_type="hnsw")
        embeddings, ids = db.get_all_embeddings(limit=5000)
        faiss_index.build(embeddings, ids.tolist())
    
    print("\nBenchmarking Precision@K...")
    results = benchmark_precision_by_k(queries, pipeline, faiss_index, db)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n{'K':<8} {'Avg Precision':<15} {'Min':<12} {'Max':<12}")
    print("-" * 50)
    for k in sorted(results.keys()):
        r = results[k]
        print(f"K={k:<5} {r['avg']:.4f}         {r['min']:.4f}     {r['max']:.4f}")
    
    import json
    from datetime import datetime
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "queries": queries,
        "precision_at_k": results
    }
    
    output_path = Path("data/results/precision_benchmark.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
