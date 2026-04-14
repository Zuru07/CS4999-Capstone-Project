"""Benchmarks for latency and Recall@K across different index types."""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import recall_score

from src.config import EMBEDDING_CONFIG
from src.db.faiss_index import FAISSIndex
from src.db.pgvector import PGVectorDB


@dataclass
class LatencyResult:
    index_type: str
    engine: str
    cold_ms: float
    warm_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_qps: float


@dataclass
class RecallResult:
    index_type: str
    engine: str
    recall_at_k: int
    recall_score: float
    avg_precision: float


@dataclass
class BenchmarkReport:
    timestamp: str
    dataset_size: int
    dimension: int
    num_queries: int
    latency_results: List[LatencyResult] = field(default_factory=list)
    recall_results: List[RecallResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "dataset_size": self.dataset_size,
            "dimension": self.dimension,
            "num_queries": self.num_queries,
            "latency": [
                {
                    "index_type": r.index_type,
                    "engine": r.engine,
                    "cold_ms": r.cold_ms,
                    "warm_ms": r.warm_ms,
                    "p50_ms": r.p50_ms,
                    "p95_ms": r.p95_ms,
                    "p99_ms": r.p99_ms,
                    "throughput_qps": r.throughput_qps,
                }
                for r in self.latency_results
            ],
            "recall": [
                {
                    "index_type": r.index_type,
                    "engine": r.engine,
                    f"recall@{r.recall_at_k}": r.recall_score,
                    "avg_precision": r.avg_precision,
                }
                for r in self.recall_results
            ],
        }


class Benchmarks:
    """Benchmark runner for pgvector and FAISS indexes."""

    INDEX_CONFIGS = {
        "flat": {"pgvector": "flat", "faiss": "flat"},
        "ivf100": {"pgvector": "ivfflat", "faiss": "ivf", "nlist": 100, "nprobe": 10},
        "ivf200": {"pgvector": "ivfflat", "faiss": "ivf", "nlist": 200, "nprobe": 20},
        "hnsw": {"pgvector": "hnsw", "faiss": "hnsw"},
    }

    def __init__(
        self,
        db: PGVectorDB,
        embeddings: np.ndarray,
        ids: List[int],
        queries: np.ndarray,
        query_ids: List[int],
        output_dir: str = "results",
    ):
        self.db = db
        self.embeddings = embeddings
        self.ids = ids
        self.queries = queries
        self.query_ids = query_ids
        self.output_dir = Path("data/results")
        self.output_dir.mkdir(exist_ok=True)

    def _compute_percentiles(self, times: List[float]) -> Tuple[float, float, float]:
        """Compute P50, P95, P99 latencies."""
        sorted_times = sorted(times)
        n = len(sorted_times)
        p50_idx = int(n * 0.50)
        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)
        return (
            sorted_times[p50_idx] * 1000,
            sorted_times[p95_idx] * 1000,
            sorted_times[p99_idx] * 1000,
        )

    def _compute_recall(
        self,
        results: List[List[int]],
        ground_truth: List[List[int]],
        k: int,
    ) -> Tuple[float, float]:
        """Compute Recall@K and average precision."""
        recalls = []
        precisions = []

        for result_ids, gt_ids in zip(results, ground_truth):
            result_set = set(result_ids[:k])
            gt_set = set(gt_ids)

            if len(gt_set) == 0:
                continue

            intersection = result_set & gt_set
            recall = len(intersection) / len(gt_set)
            precision = len(intersection) / k if k > 0 else 0

            recalls.append(recall)
            precisions.append(precision)

        avg_recall = np.mean(recalls) if recalls else 0.0
        avg_precision = np.mean(precisions) if precisions else 0.0

        return avg_recall, avg_precision

    def benchmark_pgvector_latency(
        self,
        index_type: str = "ivfflat",
        nlist: int = 100,
        warmup: int = 5,
    ) -> LatencyResult:
        """Benchmark pgvector search latency."""
        pg_index_type = self.INDEX_CONFIGS.get(index_type, {}).get("pgvector", index_type)

        self.db.create_indexes(pg_index_type, nlist)

        cold_times = []
        warm_times = []

        for i, query_emb in enumerate(self.queries):
            start = time.perf_counter()
            self.db.search(query_emb.tolist(), limit=5)
            end = time.perf_counter()
            elapsed = end - start

            if i < warmup:
                cold_times.append(elapsed)
            else:
                warm_times.append(elapsed)

        total_time = sum(warm_times)
        throughput = len(warm_times) / total_time if total_time > 0 else 0

        return LatencyResult(
            index_type=index_type,
            engine="pgvector",
            cold_ms=np.mean(cold_times) * 1000,
            warm_ms=np.mean(warm_times) * 1000,
            p50_ms=self._compute_percentiles(warm_times)[0],
            p95_ms=self._compute_percentiles(warm_times)[1],
            p99_ms=self._compute_percentiles(warm_times)[2],
            throughput_qps=throughput,
        )

    def benchmark_faiss_latency(
        self,
        index_type: str = "ivf",
        nlist: int = 100,
        nprobe: int = 10,
        warmup: int = 5,
    ) -> LatencyResult:
        """Benchmark FAISS search latency."""
        faiss_index = FAISSIndex(
            dimension=EMBEDDING_CONFIG.dimension,
            index_type=index_type,
            nlist=nlist,
            nprobe=nprobe,
        )
        faiss_index.build(self.embeddings, self.ids)

        cold_times = []
        warm_times = []

        for i, query in enumerate(self.queries):
            start = time.perf_counter()
            faiss_index.search(query, k=5)
            end = time.perf_counter()
            elapsed = end - start

            if i < warmup:
                cold_times.append(elapsed)
            else:
                warm_times.append(elapsed)

        total_time = sum(warm_times)
        throughput = len(warm_times) / total_time if total_time > 0 else 0

        return LatencyResult(
            index_type=index_type,
            engine="faiss",
            cold_ms=np.mean(cold_times) * 1000,
            warm_ms=np.mean(warm_times) * 1000,
            p50_ms=self._compute_percentiles(warm_times)[0],
            p95_ms=self._compute_percentiles(warm_times)[1],
            p99_ms=self._compute_percentiles(warm_times)[2],
            throughput_qps=throughput,
        )

    def benchmark_pgvector_recall(
        self,
        index_type: str = "ivfflat",
        nlist: int = 100,
        k: int = 5,
    ) -> RecallResult:
        """Benchmark pgvector Recall@K against ground truth."""
        self.db.create_indexes(index_type, nlist)

        ground_truth = []
        for qid in self.query_ids:
            gt = self.db.search(
                self.queries[self.query_ids.index(qid)].tolist(),
                limit=k * 10,
            )
            ground_truth.append([r.id for r in gt])

        results = []
        for query_emb in self.queries:
            res = self.db.search(query_emb.tolist(), limit=k)
            results.append([r.id for r in res])

        recall, precision = self._compute_recall(results, ground_truth, k)

        return RecallResult(
            index_type=index_type,
            engine="pgvector",
            recall_at_k=k,
            recall_score=recall,
            avg_precision=precision,
        )

    def benchmark_faiss_recall(
        self,
        index_type: str = "ivf",
        nlist: int = 100,
        nprobe: int = 10,
        k: int = 5,
    ) -> RecallResult:
        """Benchmark FAISS Recall@K against ground truth."""
        faiss_index = FAISSIndex(
            dimension=EMBEDDING_CONFIG.dimension,
            index_type=index_type,
            nlist=nlist,
            nprobe=nprobe,
        )
        faiss_index.build(self.embeddings, self.ids)

        ground_truth = []
        for qid in self.query_ids:
            gt = self.db.search(
                self.queries[self.query_ids.index(qid)].tolist(),
                limit=k * 10,
            )
            ground_truth.append([r.id for r in gt])

        results = []
        for query in self.queries:
            _, _, res = faiss_index.search(query, k=k)
            results.append([r["id"] for r in res])

        recall, precision = self._compute_recall(results, ground_truth, k)

        return RecallResult(
            index_type=index_type,
            engine="faiss",
            recall_at_k=k,
            recall_score=recall,
            avg_precision=precision,
        )

    def run_all_latency_benchmarks(self) -> List[LatencyResult]:
        """Run latency benchmarks for all index types."""
        results = []

        results.append(self.benchmark_pgvector_latency("flat"))
        print(f"pgvector flat: {results[-1].warm_ms:.2f}ms")

        for nlist in [100, 200]:
            results.append(self.benchmark_pgvector_latency("ivfflat", nlist))
            print(f"pgvector ivfflat-{nlist}: {results[-1].warm_ms:.2f}ms")

        results.append(self.benchmark_pgvector_latency("hnsw"))
        print(f"pgvector hnsw: {results[-1].warm_ms:.2f}ms")

        results.append(self.benchmark_faiss_latency("flat"))
        print(f"FAISS flat: {results[-1].warm_ms:.2f}ms")

        for nlist, nprobe in [(100, 10), (200, 20)]:
            results.append(self.benchmark_faiss_latency("ivf", nlist, nprobe))
            print(f"FAISS ivf-{nlist}: {results[-1].warm_ms:.2f}ms")

        results.append(self.benchmark_faiss_latency("hnsw"))
        print(f"FAISS hnsw: {results[-1].warm_ms:.2f}ms")

        return results

    def run_all_recall_benchmarks(self, k: int = 5) -> List[RecallResult]:
        """Run Recall@K benchmarks for all index types."""
        results = []

        for nlist in [100, 200]:
            results.append(self.benchmark_pgvector_recall("ivfflat", nlist, k))
            print(f"pgvector ivfflat-{nlist} Recall@{k}: {results[-1].recall_score:.4f}")

            results.append(self.benchmark_faiss_recall("ivf", nlist, nlist, k))
            print(f"FAISS ivf-{nlist} Recall@{k}: {results[-1].recall_score:.4f}")

        results.append(self.benchmark_pgvector_recall("hnsw", k=k))
        print(f"pgvector hnsw Recall@{k}: {results[-1].recall_score:.4f}")

        results.append(self.benchmark_faiss_recall("hnsw", k=k))
        print(f"FAISS hnsw Recall@{k}: {results[-1].recall_score:.4f}")

        return results

    def run_full_benchmark(
        self,
        num_queries: int = 50,
        recall_k: int = 5,
    ) -> BenchmarkReport:
        """Run complete benchmark suite."""
        query_indices = np.random.choice(len(self.embeddings), num_queries, replace=False)
        self.queries = self.embeddings[query_indices]
        self.query_ids = [self.ids[i] for i in query_indices]

        report = BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            dataset_size=len(self.embeddings),
            dimension=EMBEDDING_CONFIG.dimension,
            num_queries=num_queries,
        )

        print("\n=== Latency Benchmarks ===")
        report.latency_results = self.run_all_latency_benchmarks()

        print("\n=== Recall@K Benchmarks ===")
        report.recall_results = self.run_all_recall_benchmarks(recall_k)

        output_path = self.output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nResults saved to {output_path}")

        return report


def main():
    """Run benchmarks with sample data."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.data.loader import load_sample_data

    print("Loading sample data...")
    embeddings, ids = load_sample_data(10000)

    queries = embeddings[:50]
    query_ids = ids[:50]

    db = PGVectorDB()

    benchmarks = Benchmarks(
        db=db,
        embeddings=embeddings,
        ids=ids,
        queries=queries,
        query_ids=query_ids,
    )

    report = benchmarks.run_full_benchmark(num_queries=50, recall_k=5)

    print("\n=== Summary ===")
    print(f"Dataset: {report.dataset_size} vectors, {report.dimension}D")
    print(f"Queries: {report.num_queries}")

    print("\nLatency (warm):")
    for r in sorted(report.latency_results, key=lambda x: x.warm_ms):
        print(f"  {r.engine:10} {r.index_type:10} {r.warm_ms:8.2f}ms  P95: {r.p95_ms:.2f}ms")


if __name__ == "__main__":
    main()
