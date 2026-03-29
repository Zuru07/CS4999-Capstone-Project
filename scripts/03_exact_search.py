import psycopg2
from sentence_transformers import SentenceTransformer
import time

model = SentenceTransformer('all-MiniLM-L6-v2')

conn = psycopg2.connect(
    dbname="",
    user="",
    password="",
    host="",
    port=""
)

cur = conn.cursor()

query = "Deep learning approaches for medical image segmentation"
query_embedding = model.encode([query])[0].tolist()

start = time.time()

cur.execute("""
    SELECT id, content
    FROM documents
    ORDER BY embedding <-> %s::vector
    LIMIT 5;
""", (query_embedding,))

results = cur.fetchall()

end = time.time()

print("Exact Search Latency:", end - start)
print(results)

cur.close()
conn.close()
