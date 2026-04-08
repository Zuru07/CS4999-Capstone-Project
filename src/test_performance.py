"""Test script for RAG pipeline performance."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.generator import RAGPipeline
from src.db.pgvector import PGVectorDB
from src.db.faiss_index import FAISSIndex
from src.config import EMBEDDING_CONFIG


def test_pgvector_search(db: PGVectorDB, num_queries: int = 10):
    """Test pgvector search latency."""
    print("\n--- pgvector Search Latency ---")

    queries = [
        "What is deep learning?",
        "Machine learning optimization",
        "Neural network architectures",
        "Natural language processing",
        "Computer vision techniques",
        "Reinforcement learning algorithms",
        "Transformer models attention",
        "Graph neural networks",
        "Federated learning privacy",
        "Quantum machine learning",
    ][:num_queries]

    pipeline = RAGPipeline(db=db)

    latencies = []
    for query in queries:
        start = time.perf_counter()
        results = pipeline.retrieve(query, limit=5, use_hybrid=False)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
        print(f"  Query: '{query[:40]}...' -> {len(results)} results, {latencies[-1]:.2f}ms")

    avg_latency = sum(latencies) / len(latencies)
    print(f"\nAverage latency: {avg_latency:.2f}ms")
    return avg_latency


def test_faiss_search(embeddings, ids, num_queries: int = 10):
    """Test FAISS search latency."""
    print("\n--- FAISS Search Latency ---")

    queries = [
        "What is deep learning?",
        "Machine learning optimization",
        "Neural network architectures",
        "Natural language processing",
        "Computer vision techniques",
        "Reinforcement learning algorithms",
        "Transformer models attention",
        "Graph neural networks",
        "Federated learning privacy",
        "Quantum machine learning",
    ][:num_queries]

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_CONFIG.model_name)
    query_embeddings = model.encode(queries)

    print("Building FAISS index...")
    faiss_index = FAISSIndex(
        dimension=EMBEDDING_CONFIG.dimension,
        index_type="hnsw",
    )
    faiss_index.build(embeddings, ids)

    latencies = []
    for i, (query, query_emb) in enumerate(zip(queries, query_embeddings)):
        start = time.perf_counter()
        distances, indices, results = faiss_index.search(query_emb, k=5)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
        print(f"  Query {i+1}: {len(results)} results, {latencies[-1]:.4f}ms")

    avg_latency = sum(latencies) / len(latencies)
    print(f"\nAverage latency: {avg_latency:.4f}ms")
    return avg_latency


def test_rag_query(pipeline: RAGPipeline):
    """Test full RAG query (no streaming for timing)."""
    print("\n--- Full RAG Query (non-streaming) ---")

    query = "What are the main applications of deep learning in computer vision?"

    start = time.perf_counter()
    response = pipeline.query(query, limit=3, stream=False)
    end = time.perf_counter()

    print(f"\nQuery: {query}")
    print(f"Response: {response[:200]}...")
    print(f"Total time: {(end - start) * 1000:.2f}ms")


def main():
    """Run all performance tests."""
    print("=" * 60)
    print("RAG PIPELINE PERFORMANCE TEST")
    print("=" * 60)

    db = PGVectorDB()
    doc_count = db.count()
    print(f"\nDatabase contains {doc_count} documents")

    if doc_count == 0:
        print("\nERROR: Database is empty. Run `python -m src.setup` first.")
        return

    print("\n1. Testing pgvector search...")
    test_pgvector_search(db, num_queries=5)

    print("\n2. Testing FAISS search...")
    from src.data.loader import load_sample_data
    embeddings, ids = load_sample_data(10000)
    test_faiss_search(embeddings, ids, num_queries=5)

    print("\n3. Testing full RAG query...")
    pipeline = RAGPipeline(db=db)
    test_rag_query(pipeline)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
