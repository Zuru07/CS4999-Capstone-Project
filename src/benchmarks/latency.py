"""Latency benchmark for RAG pipeline components."""

import time
import sys
from pathlib import Path
from typing import List, Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.pgvector import PGVectorDB
from src.db.faiss_index import FAISSIndex
from src.rag.generator import RAGPipeline
import numpy as np


def benchmark_embedding_generation(queries: List[str], runs: int = 10) -> Dict[str, float]:
    """Benchmark embedding generation time."""
    print("  Benchmarking embedding generation...")
    pipeline = RAGPipeline()
    times = []
    
    for query in queries:
        query_times = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = pipeline.get_query_embedding(query)
            elapsed = (time.perf_counter() - start) * 1000
            query_times.append(elapsed)
        
        avg_time = sum(query_times) / len(query_times)
        times.append(avg_time)
        print(f"    '{query[:30]}...': {avg_time:.2f}ms")
    
    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "times": times
    }


def benchmark_pgvector_search(queries: List[str], runs: int = 10) -> Dict[str, float]:
    """Benchmark pgvector search time."""
    print("  Benchmarking pgvector search...")
    db = PGVectorDB()
    pipeline = RAGPipeline(db=db)
    times = []
    
    for query in queries:
        query_times = []
        for _ in range(runs):
            embedding = pipeline.get_query_embedding(query)
            start = time.perf_counter()
            _ = db.search(embedding, limit=5)
            elapsed = (time.perf_counter() - start) * 1000
            query_times.append(elapsed)
        
        avg_time = sum(query_times) / len(query_times)
        times.append(avg_time)
        print(f"    '{query[:30]}...': {avg_time:.2f}ms")
    
    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "times": times
    }


def benchmark_faiss_search(queries: List[str], runs: int = 10) -> Dict[str, float]:
    print("  Benchmarking FAISS search...")
    db = PGVectorDB()
    pipeline = RAGPipeline(db=db)
    
    try:
        faiss_index = FAISSIndex.load("data/cache/faiss_index")
        print("    Loaded pre-built FAISS index")
    except FileNotFoundError:
        print("    Building sample FAISS index...")
        faiss_index = FAISSIndex(dimension=384, index_type="hnsw")
        embeddings, ids = db.get_all_embeddings(limit=1000)
        faiss_index.build(embeddings, ids.tolist())
    
    times = []
    
    for query in queries:
        query_times = []
        for _ in range(runs):
            embedding = pipeline.get_query_embedding(query)
            start = time.perf_counter()
            _ = faiss_index.search(embedding, k=5)
            elapsed = (time.perf_counter() - start) * 1000
            query_times.append(elapsed)
        
        avg_time = sum(query_times) / len(query_times)
        times.append(avg_time)
        print(f"    '{query[:30]}...': {avg_time:.2f}ms")
    
    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "times": times
    }


def benchmark_document_fetch(doc_ids: List[int], runs: int = 10) -> Dict[str, float]:
    """Benchmark document fetch time from pgvector."""
    print("  Benchmarking document fetch...")
    db = PGVectorDB()
    times = []
    
    for _ in range(runs):
        start = time.perf_counter()
        for doc_id in doc_ids:
            _ = db.get_document_by_id(doc_id)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    per_doc = avg_time / len(doc_ids)
    print(f"    Fetch {len(doc_ids)} docs: {avg_time:.2f}ms ({per_doc:.2f}ms/doc)")
    
    return {
        "avg_ms": avg_time,
        "min_ms": min(times),
        "max_ms": max(times),
        "per_doc_ms": per_doc,
        "times": times
    }


def main():
    print("=" * 70)
    print("RAG PIPELINE LATENCY BENCHMARK")
    print("=" * 70)
    
    queries = [
        "deep learning neural networks",
        "machine learning applications", 
        "natural language processing",
        "computer vision techniques",
        "reinforcement learning algorithms"
    ]
    
    # Sample document IDs for fetch benchmark
    doc_ids = [44051, 33821, 22415, 65941, 42434, 1576, 2551, 10939, 29889, 29892]
    
    print("\nRunning benchmarks...")
    
    # Run all benchmarks
    embedding_results = benchmark_embedding_generation(queries)
    pgvector_results = benchmark_pgvector_search(queries)
    faiss_results = benchmark_faiss_search(queries)
    fetch_results = benchmark_document_fetch(doc_ids)
    
    print("\n" + "=" * 70)
    print("LATENCY RESULTS")
    print("=" * 70)
    
    print(f"\n{'Component':<35} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
    print("-" * 55)
    print(f"{'Embedding Generation':<35} {embedding_results['avg_ms']:<12.2f} "
          f"{embedding_results['min_ms']:<12.2f} {embedding_results['max_ms']:<12.2f}")
    print(f"{'pgvector Search':<35} {pgvector_results['avg_ms']:<12.2f} "
          f"{pgvector_results['min_ms']:<12.2f} {pgvector_results['max_ms']:<12.2f}")
    print(f"{'FAISS Search':<35} {faiss_results['avg_ms']:<12.2f} "
          f"{faiss_results['min_ms']:<12.2f} {faiss_results['max_ms']:<12.2f}")
    print(f"{'Document Fetch (5 docs)':<35} {fetch_results['avg_ms']:<12.2f} "
          f"{fetch_results['min_ms']:<12.2f} {fetch_results['max_ms']:<12.2f}")
    
    # Calculate speedups
    search_speedup = pgvector_results['avg_ms'] / faiss_results['avg_ms'] if faiss_results['avg_ms'] > 0 else 0
    fetch_speedup = embedding_results['avg_ms'] / faiss_results['avg_ms'] if faiss_results['avg_ms'] > 0 else 0
    
    print(f"\nSpeedups:")
    print(f"  FAISS vs pgvector search: {search_speedup:.1f}x faster")
    print(f"  (Note: Embedding generation is now GPU-accelerated)")
    
    # Save results
    import json
    from datetime import datetime
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "queries": queries,
        "embedding_generation": embedding_results,
        "pgvector_search": pgvector_results,
        "faiss_search": faiss_results,
        "document_fetch": fetch_results,
        "speedups": {
            "faiss_vs_pgvector_search": search_speedup,
        }
    }
    
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "latency_benchmark.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
