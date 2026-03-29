import numpy as np
import faiss
import time
from sentence_transformers import SentenceTransformer

# Load embeddings
embeddings = np.load("../dataset/embeddings.npy").astype("float32")

dimension = embeddings.shape[1]

# Build FAISS Flat index
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

model = SentenceTransformer('all-MiniLM-L6-v2')
query = "Deep learning approaches for medical image segmentation"
query_vector = model.encode([query]).astype("float32")

start = time.time()
distances, indices = index.search(query_vector, 5)
end = time.time()

print("FAISS Flat Latency:", end - start)
print(indices)