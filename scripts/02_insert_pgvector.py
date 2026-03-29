import psycopg2
import pandas as pd
import numpy as np
from psycopg2.extras import execute_values

# Load saved data
df = pd.read_csv("../dataset/documents.csv")
embeddings = np.load("../dataset/embeddings.npy")

# Connect to Docker Postgres (PORT 5433)
conn = psycopg2.connect(
    dbname="",
    user="",
    password="",
    host="",
    port=""
)

cur = conn.cursor()

print("Connected to PostgreSQL.")

# Prepare rows
data = [
    (df.iloc[i]["content"], embeddings[i].tolist())
    for i in range(len(df))
]

print("Inserting rows...")

execute_values(
    cur,
    "INSERT INTO documents (content, embedding) VALUES %s",
    data
)

conn.commit()
cur.close()
conn.close()

print("Insertion complete.")