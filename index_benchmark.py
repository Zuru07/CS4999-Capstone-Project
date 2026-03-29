"""
index_benchmark.py
Usage:
  python index_benchmark.py --table hf_docs_50k
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

parser = argparse.ArgumentParser()
parser.add_argument("--table", required=True)
parser.add_argument("--results", default="results/index_benchmark_results.csv")
args = parser.parse_args()

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

if not all([DB_USER, DB_PASSWORD, DB_NAME]):
    raise ValueError("Missing DB credentials in .env")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

TABLE = args.table
os.makedirs(os.path.dirname(args.results), exist_ok=True)

# detect dim
with engine.connect() as conn:
    dim_row = conn.execute(text(f"SELECT vector_dims(embedding) FROM {TABLE} LIMIT 1;")).fetchone()
    if not dim_row:
        raise ValueError(f"No rows in table {TABLE} or vector column not found")
    dim = dim_row[0]

print(f"Embedding dimension detected = {dim}")

# create random query vector of correct dim
query_vec = np.random.rand(dim).tolist()

indexes = {
    f"{TABLE}_ivfflat": f"CREATE INDEX {TABLE}_ivfflat ON {TABLE} USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);",
    f"{TABLE}_hnsw": f"CREATE INDEX {TABLE}_hnsw ON {TABLE} USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 64);"
}

results = []
with engine.connect() as conn:
    # drop indexes
    for idx_name in indexes.keys():
        conn.execute(text(f"DROP INDEX IF EXISTS {idx_name};"))
    conn.commit()

    for idx_name, create_sql in indexes.items():
        print("Building", idx_name)
        t0 = time.time()
        conn.execute(text(create_sql))
        conn.commit()
        build_time = time.time() - t0
        print(f"Built {idx_name} in {build_time:.2f}s")

        # EXPLAIN ANALYZE
        vec_literal = "[" + ",".join(map(str, query_vec)) + "]"
        explain_sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT id FROM {TABLE} ORDER BY embedding <-> '{vec_literal}' LIMIT 10;"
        row = conn.execute(text(explain_sql)).fetchone()
        plan_json = row[0][0]
        plan_time = plan_json.get("Planning Time", 0.0)
        exec_time = plan_json.get("Execution Time", 0.0)

        results.append({
            "table": TABLE,
            "index": idx_name,
            "build_time_s": round(build_time,3),
            "plan_ms": round(plan_time,3),
            "exec_ms": round(exec_time,3)
        })

# save results: append if exists
df_new = pd.DataFrame(results)
if os.path.exists(args.results):
    df_old = pd.read_csv(args.results)
    df = pd.concat([df_old, df_new], ignore_index=True)
else:
    df = df_new
df.to_csv(args.results, index=False)
print("Saved results to", args.results)
print(df_new)
