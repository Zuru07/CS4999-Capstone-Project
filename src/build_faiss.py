"""Build and save FAISS index."""

import numpy as np
from pathlib import Path

from src.db.pgvector import PGVectorDB
from src.db.faiss_index import FAISSIndex

def main():
    print("Building FAISS index...")
    
    db = PGVectorDB()
    print(f"Loaded {db.count()} documents")
    
    print("Fetching embeddings and IDs...")
    embeddings, ids = db.get_all_embeddings(limit=100000)
    print(f"Got {len(ids)} embeddings, shape: {embeddings.shape}")
    
    print("Building FAISS HNSW index...")
    index = FAISSIndex(
        dimension=384,
        index_type="hnsw",
        hnsw_m=16,
        hnsw_ef_construction=200,
    )
    
    index.build(embeddings, ids)
    print(f"Index built with {index.total_vectors} vectors")
    
    save_path = Path("data/cache/faiss_index")
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to {save_path}...")
    index.save(str(save_path))
    print("Done!")

if __name__ == "__main__":
    main()
