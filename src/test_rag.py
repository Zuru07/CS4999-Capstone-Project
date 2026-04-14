"""Quick test script - run this to test the RAG pipeline manually."""

from src.rag.generator import RAGPipeline

print("=" * 50)
print("RAG PIPELINE TEST")
print("=" * 50)

pipeline = RAGPipeline()

query = input("\nEnter your question: ")

print("\nSearching...")
response = pipeline.query(query, limit=3, stream=True)

print("\n" + "=" * 50)
