"""Metadata filtering benchmark - tests SQL filtering performance."""

import json
import time
from datetime import datetime
from pathlib import Path

from src.db.pgvector import PGVectorDB
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_CONFIG

RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def benchmark_filtered_search(db, query_emb, filter_type, filter_value, iterations=10):
    """Benchmark search with metadata filter."""
    times = []
    
    filters_map = {
        "category": {"category": filter_value},
        "date": {"date_from": "2020-01-01", "date_to": "2024-12-31"},
        "no_filter": None,
    }
    
    filters = filters_map.get(filter_type)
    
    for _ in range(iterations):
        start = time.perf_counter()
        results = db.search(query_emb.tolist(), limit=5, filters=filters)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    return {
        "filter_type": filter_type,
        "filter_value": filter_value,
        "avg_ms": sum(times) / len(times),
        "p50_ms": sorted(times)[len(times)//2],
        "p95_ms": sorted(times)[int(len(times)*0.95)],
        "count": len(results),
    }


def main():
    print("=" * 60)
    print("METADATA FILTERING BENCHMARK")
    print("=" * 60)
    
    print("\n1. Loading model...")
    model = SentenceTransformer(EMBEDDING_CONFIG.model_name)
    
    print("\n2. Connecting to DB...")
    db = PGVectorDB()
    print(f"   Docs in DB: {db.count()}")
    
    print("\n3. Sample documents for filter values...")
    sample = db.search(model.encode("machine learning").tolist(), limit=10)
    print(f"   Sample search returned {len(sample)} results")
    
    print("\n" + "=" * 60)
    print("FILTERED SEARCH BENCHMARKS")
    print("=" * 60)
    
    query_emb = model.encode("deep learning neural networks")
    results = []
    
    # No filter (baseline)
    print("\n4. No filter (baseline)...")
    r = benchmark_filtered_search(db, query_emb, "no_filter", None)
    print(f"   Latency: {r['avg_ms']:.2f}ms, Results: {r['count']}")
    results.append(r)
    
    # Category filter
    print("\n5. Category filter...")
    r = benchmark_filtered_search(db, query_emb, "category", "cs.AI")
    print(f"   Latency: {r['avg_ms']:.2f}ms, Results: {r['count']}")
    results.append(r)
    
    # Date filter
    print("\n6. Date range filter...")
    r = benchmark_filtered_search(db, query_emb, "date", "2020-2024")
    print(f"   Latency: {r['avg_ms']:.2f}ms, Results: {r['count']}")
    results.append(r)
    
    # Filter then vector search
    print("\n" + "=" * 60)
    print("METADATA FILTERING LATENCY TARGET: <20ms")
    print("=" * 60)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Filter Type':<20} {'Avg Latency':<15} {'P95':<15} {'Results'}")
    print("-" * 60)
    
    for r in results:
        target_met = "[PASS]" if r["avg_ms"] < 20 else "[FAIL]"
        print(f"{r['filter_type']:<20} {r['avg_ms']:>8.2f}ms     {r['p95_ms']:>8.2f}ms     {r['count']} {target_met}")
    
    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": datetime.now().isoformat(),
        "dataset_size": db.count(),
        "target_ms": 20,
        "results": results,
    }
    fp = RESULTS_DIR / f"metadata_filter_{ts}.json"
    with open(fp, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to: {fp}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
