from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# =========================
# LOAD ARXIV DATASET
# =========================

print("Loading arXiv dataset...")

dataset = load_dataset("ccdv/arxiv-summarization", split="train")

# Extract first 10,000 abstracts
abstracts = []

for i in range(10000):
    abstracts.append(dataset[i]["abstract"])

print("Loaded 10,000 abstracts.")

df = pd.DataFrame({"content": abstracts})

# =========================
# GENERATE EMBEDDINGS
# =========================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")

embeddings = model.encode(
    df["content"].tolist(),
    show_progress_bar=True
)

print("Embeddings shape:", embeddings.shape)

# =========================
# SAVE TO DISK
# =========================

os.makedirs("../dataset", exist_ok=True)

df.to_csv("../dataset/documents.csv", index=False)
np.save("../dataset/embeddings.npy", embeddings)

print("Saved arXiv dataset + embeddings.")