"""Setup script to load data, build indexes, and test the pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_sample_data, generate_embeddings
from src.db.pgvector import PGVectorDB


def setup_database(num_samples: int = 1000):
    """Load data and populate the database."""
    print("=" * 60)
    print("RAG PIPELINE SETUP")
    print("=" * 60)

    print("\n1. Loading/generating embeddings...")
    embeddings, ids = load_sample_data(num_samples=num_samples)
    print(f"   Loaded {len(embeddings)} embeddings, dimension: {embeddings.shape[1]}")

    print("\n2. Connecting to database...")
    db = PGVectorDB()

    try:
        count = db.count()
        print(f"   Current document count: {count}")
    except Exception as e:
        print(f"   Note: {e}")

    print("\n3. Creating table...")
    db.create_table(dimension=embeddings.shape[1])
    print("   Table created successfully")

    print("\n4. Dropping existing data...")
    db.drop_table()
    db.create_table(dimension=embeddings.shape[1])

    print("\n5. Inserting documents...")
    from src.data.loader import load_raw_documents
    documents = load_raw_documents(num_samples)

    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_docs = [d["content"] for d in documents[i:i+batch_size]]
        batch_embs = embeddings[i:i+batch_size]
        batch_authors = ["arXiv"] * len(batch_docs)
        batch_categories = ["cs.AI"] * len(batch_docs)
        
        db.insert_batch(
            documents=batch_docs,
            embeddings=batch_embs,
            authors=batch_authors,
            categories=batch_categories,
        )
        print(f"   Inserted {min(i+batch_size, len(documents))}/{len(documents)} documents")

    print(f"\n6. Total documents in database: {db.count()}")

    print("\n7. Building indexes...")
    for index_type, nlist in [("flat", 1), ("ivfflat", 100), ("hnsw", 16)]:
        print(f"   Building {index_type} index...")
        db.create_indexes(index_type, nlist)
    print("   Indexes built successfully")

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Setup RAG pipeline database")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    args = parser.parse_args()

    setup_database(args.samples)
