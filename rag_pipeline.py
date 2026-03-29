import os
import requests
import psycopg2
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables just like seed_db.py
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

# --- Configuration ---
# Set this to the exact table name you passed to --table in seed_db.py
TABLE_NAME = "hf_docs_50k" 
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.2"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print(f"Loading embedding model ({EMBED_MODEL_NAME})...")
embedding_model = SentenceTransformer(EMBED_MODEL_NAME)

def get_query_embedding(query):
    """Embeds the query using the same model that prepared the data."""
    # Convert to list so psycopg2 can format it for pgvector
    return embedding_model.encode(query, convert_to_numpy=True).tolist()

def hybrid_search(query, query_embedding, limit=5):
    """Hybrid Search matching your specific HFDocs schema."""
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )
    cur = conn.cursor()
    
    # We format the query vector for pgvector
    vec_literal = "[" + ",".join(map(str, query_embedding)) + "]"
    
    # RRF SQL Query adapted for your 'text' column
    sql = f"""
    WITH semantic_search AS (
        SELECT id, text,
               RANK() OVER (ORDER BY embedding <-> '{vec_literal}'::vector) AS rank
        FROM {TABLE_NAME}
        ORDER BY embedding <-> '{vec_literal}'::vector
        LIMIT 20
    ),
    keyword_search AS (
        SELECT id, text,
               RANK() OVER (ORDER BY ts_rank_cd(to_tsvector('english', text), plainto_tsquery('english', %s)) DESC) AS rank
        FROM {TABLE_NAME}
        WHERE to_tsvector('english', text) @@ plainto_tsquery('english', %s)
        LIMIT 20
    )
    SELECT 
        COALESCE(s.id, k.id) AS id,
        COALESCE(s.text, k.text) AS text,
        (COALESCE(1.0 / (60 + s.rank), 0.0) + COALESCE(1.0 / (60 + k.rank), 0.0)) AS rrf_score
    FROM semantic_search s
    FULL OUTER JOIN keyword_search k ON s.id = k.id
    ORDER BY rrf_score DESC
    LIMIT %s;
    """
    
    cur.execute(sql, (query, query, limit))
    results = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return [{"id": r[0], "text": r[1], "score": r[2]} for r in results]

def ask_llama(query, retrieved_chunks):
    """Constructs the prompt and queries the local Ollama Llama 3.2 model."""
    context_text = "\n\n".join([f"Document {chunk['id']}: {chunk['text']}" for chunk in retrieved_chunks])
    
    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the provided context documents.
If the answer is not contained in the context, say "I don't have enough information to answer that."

Context:
{context_text}

Question: {query}
Answer:"""

    print("\nQuerying Llama 3.2...\n")
    
    response = requests.post(OLLAMA_URL, json={
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": True
    }, stream=True)

    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            print(chunk.get("response", ""), end="", flush=True)
    print("\n")

if __name__ == "__main__":
    # Test your pipeline
    user_query = "What is the main topic of the news articles regarding space?"
    
    print("1. Embedding query...")
    q_emb = get_query_embedding(user_query)
    
    print("2. Searching database...")
    top_docs = hybrid_search(user_query, q_emb, limit=3)
    
    if top_docs:
        ask_llama(user_query, top_docs)
    else:
        print("No documents found.")