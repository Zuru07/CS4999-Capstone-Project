"""
data_prep.py
Generates embeddings for a given HuggingFace dataset and sample size.
Usage: python data_prep.py --dataset ag_news --size 50000
"""
import argparse
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ag_news")
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--size", type=int, default=10000)
parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
parser.add_argument("--outdir", type=str, default="data")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
out_path = os.path.join(args.outdir, f"{args.dataset}_{args.size}.parquet")

print(f"Loading dataset {args.dataset} split={args.split} ...")
ds = load_dataset(args.dataset, split=args.split)
if args.size > len(ds):
    raise ValueError(f"Requested size {args.size} > dataset size {len(ds)}")

print("Detecting text column...")
text_col = None
for col in ds.column_names:
    if "text" in col.lower() or "sentence" in col.lower() or "review" in col.lower():
        text_col = col
        break
if text_col is None:
    raise ValueError("Could not find text-like column in dataset")

print("Using text column:", text_col)
sub = ds.select(range(args.size))

print("Loading model:", args.model)
model = SentenceTransformer(args.model)

batch_size = 512
emb_list = []
texts = []
labels = []
ids = []
text_lens = []
timestamps = []

# Convert the selected dataset to a list of dicts once
dataset_list = []
sub_dict = sub.to_dict()
for i in range(len(sub_dict[text_col])):
    dataset_list.append({k: sub_dict[k][i] for k in sub_dict.keys()})

# Now dataset_list is a list of dicts — safe to slice
for i in tqdm(range(0, args.size, batch_size), desc="Embedding"):
    batch_list = dataset_list[i:i+batch_size]

    batch_texts = []
    batch_labels = []

    for item in batch_list:
        t = item[text_col]
        if isinstance(t, dict):
            t = t.get("text") or t.get("sentence") or t.get("review") or ""
        batch_texts.append(str(t))
        batch_labels.append(item.get("label", None))

    emb = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
    emb_list.append(emb)
    texts.extend(batch_texts)
    labels.extend(batch_labels)

    start_id = i + 1
    for j, t in enumerate(batch_texts):
        ids.append(start_id + j)
        text_lens.append(len(t))
        timestamps.append(datetime.utcnow())

embeddings = np.vstack(emb_list)
df = pd.DataFrame({
    "id": ids,
    "text": texts,
    "label": labels,
    "text_len": text_lens,
    "ts": timestamps,
    "embedding": list(map(lambda row: row.tolist(), embeddings))
})
df.to_parquet(out_path, index=False)
print("✅ Wrote parquet:", out_path)
