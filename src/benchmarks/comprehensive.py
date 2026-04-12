"""Comprehensive benchmark runner - runs all benchmarks and generates report."""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks.latency import (
    benchmark_embedding_generation,
    benchmark_pgvector_search,
    benchmark_faiss_search,
    benchmark_document_fetch
)
from src.benchmarks.recall import benchmark_recall_by_k
from src.benchmarks.precision import benchmark_precision_by_k
from src.db.pgvector import PGVectorDB
from src.db.faiss_index import FAISSIndex
from src.rag.generator import RAGPipeline


def run_all_benchmarks():
    print("=" * 70)
    print("COMPREHENSIVE RAG BENCHMARK SUITE")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    results = {"timestamp": datetime.now().isoformat()}
    queries = [
        "deep learning neural networks",
        "machine learning applications",
        "natural language processing transformers",
        "computer vision convolutional neural networks",
        "reinforcement learning policy gradient"
    ]
    doc_ids = [44051, 33821, 22415, 65941, 42434]
    
    db = PGVectorDB()
    pipeline = RAGPipeline(db=db)
    faiss_index = FAISSIndex.load("data/cache/faiss_index")
    print("\nLoaded FAISS index with", faiss_index.total_vectors, "vectors")
    
    print("\n[1/4] Latency Benchmarks...")
    start = time.perf_counter()
    results["latency"] = {
        "embedding_generation": benchmark_embedding_generation(queries),
        "pgvector_search": benchmark_pgvector_search(queries),
        "faiss_search": benchmark_faiss_search(queries),
        "document_fetch": benchmark_document_fetch(doc_ids)
    }
    print(f"    Latency benchmarks completed in {time.perf_counter() - start:.1f}s")
    
    print("\n[2/4] Recall@K Benchmark...")
    start = time.perf_counter()
    results["recall"] = benchmark_recall_by_k(queries, pipeline, faiss_index, db)
    print(f"    Recall benchmark completed in {time.perf_counter() - start:.1f}s")
    
    print("\n[3/4] Precision@K Benchmark...")
    start = time.perf_counter()
    results["precision"] = benchmark_precision_by_k(queries, pipeline, faiss_index, db)
    print(f"    Precision benchmark completed in {time.perf_counter() - start:.1f}s")
    
    results["summary"] = {
        "faiss_speedup_vs_pgvector": (
            results["latency"]["pgvector_search"]["avg_ms"] / 
            results["latency"]["faiss_search"]["avg_ms"]
        ),
        "total_pipeline_time_ms": (
            results["latency"]["embedding_generation"]["avg_ms"] +
            results["latency"]["faiss_search"]["avg_ms"] +
            results["latency"]["document_fetch"]["avg_ms"]
        )
    }
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    print("\n** Latency Summary **")
    lat = results["latency"]
    print(f"  Embedding Generation: {lat['embedding_generation']['avg_ms']:.2f}ms")
    print(f"  pgvector Search:     {lat['pgvector_search']['avg_ms']:.2f}ms")
    print(f"  FAISS Search:        {lat['faiss_search']['avg_ms']:.2f}ms")
    print(f"  Document Fetch:      {lat['document_fetch']['avg_ms']:.2f}ms")
    print(f"\n  FAISS speedup vs pgvector: {results['summary']['faiss_speedup_vs_pgvector']:.1f}x")
    print(f"  Total pipeline time: {results['summary']['total_pipeline_time_ms']:.2f}ms")
    
    print("\n** Recall@K Summary **")
    for k in sorted(results["recall"].keys()):
        r = results["recall"][k]
        print(f"  K={k}: {r['avg']:.2%} (avg)")
    
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "comprehensive_benchmark.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    run_all_benchmarks()
