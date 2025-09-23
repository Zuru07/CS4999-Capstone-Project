import os
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Env Variables Config
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT")

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
TABLE_NAME = 'wikipedia_qa'


# Database Connection 
def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        register_vector(conn)
        return conn
    except psycopg2.OperationalError as e:
        print(f"Could not connect to the database: {e}")
        return None


# Similarity Search 
def find_similar_questions(query_text, conn, top_k=5):
    """Finds the most similar questions to a given query text."""
    if not conn:
        print("No database connection.")
        return []

    print(f"Loading sentence transformer model '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Generating embedding for query: '{query_text}'")
    query_embedding = model.encode(query_text)

    print(f"Searching for top {top_k} similar questions...")
    results = []
    with conn.cursor() as cur:
        # The <=> operator calculates the cosine distance (1 - cosine similarity). We use cosine similarity as a start
        cur.execute(
            f"""
            SELECT question, answer, 1 - (embedding <=> %s) AS similarity
            FROM {TABLE_NAME}
            ORDER BY embedding <=> %s
            LIMIT %s;
            """,
            (query_embedding, query_embedding, top_k)
        )
        results = cur.fetchall()

    return results


# Main Exec
if __name__ == "__main__":
    search_query = input("Enter your query: ")

    connection = get_db_connection()
    if connection:
        similar_items = find_similar_questions(search_query, connection)

        print("\n--- Search Results ---")
        print(f"Query: {search_query}\n")

        if similar_items:
            for i, (question, answer, similarity) in enumerate(similar_items):
                print(f"{i+1}. Question: {question}")
                print(f"   Answer: {answer[:100]}...") # Print a snippet
                print(f"   Similarity Score: {similarity:.4f}\n")
        else:
            print("No similar questions found.")

        connection.close()
        print("Database connection closed.")