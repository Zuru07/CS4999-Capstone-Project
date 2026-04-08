"""Setup script - run this ONCE to populate the database."""

from src.data.loader import load_sample_data, load_raw_documents
from src.db.pgvector import PGVectorDB

def main():
    print("=" * 50)
    print("DATABASE SETUP")
    print("=" * 50)
    
    num_samples = int(input("How many documents? (default: 1000): ") or "1000")

    print("\n1. Loading/generating embeddings...")
    embeddings, ids = load_sample_data(num_samples)
    print(f"   Done: {len(embeddings)} docs, {embeddings.shape[1]}D")
    
    print("\n2. Setting up database...")
    db = PGVectorDB()
    
    print("   Dropping old table...")
    db.drop_table()
    
    print("   Creating new table...")
    db.create_table(dimension=embeddings.shape[1])
    
    print("   Loading raw documents...")
    documents = load_raw_documents(num_samples)
    
    print("   Inserting documents...")
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        db.insert_batch(
            documents=[d["content"] for d in batch],
            embeddings=embeddings[i:i+batch_size],
            authors=["arXiv"] * len(batch),
            categories=["cs.AI"] * len(batch),
        )
        print(f"   {min(i+batch_size, len(documents))}/{len(documents)}")
    
    print("\n3. Building indexes...")
    db.create_indexes("flat", 1)
    db.create_indexes("ivfflat", 100)
    db.create_indexes("hnsw", 16)
    
    print(f"\n4. Total documents: {db.count()}")
    print("\n" + "=" * 50)
    print("SETUP COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    main()
