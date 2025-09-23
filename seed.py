import os
from dotenv import load_dotenv
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector

# Load environment variables from .env file
load_dotenv()

# Env variables config
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT")

# Model for generating embeddings
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
TABLE_NAME = 'wikipedia_corpus'


# Database Connection and Setup
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
        print("Successfully connected to the database.")
        return conn
    except psycopg2.OperationalError as e:
        print(f"Could not connect to the database: {e}")
        exit()


def setup_database(conn):
    """Sets up the pgvector extension and creates the table for storing the corpus."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        print("pgvector extension is enabled.")

        register_vector(conn)

        model = SentenceTransformer(EMBEDDING_MODEL)
        embedding_dim = model.get_sentence_embedding_dimension()

        # FIX: Use 'doc_id' (INTEGER) and 'content' to match the dataset's 'id' and 'passage'
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id SERIAL PRIMARY KEY,
            doc_id INTEGER,
            content TEXT NOT NULL,
            embedding vector({embedding_dim})
        );
        """)
        conn.commit()
        print(f"Table '{TABLE_NAME}' is ready.")


# Data Loading and Embedding Generation
def load_and_embed_corpus():
    """Loads the text-corpus and generates embeddings for its content."""
    print("Loading dataset 'rag-datasets/rag-mini-wikipedia' (text-corpus)...")
    ds = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus", split="passages")
    print("Dataset loaded successfully.")

    print(f"Loading sentence transformer model '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded successfully.")

    # FIX: Use the correct keys 'id' and 'passage' from the dataset
    doc_ids = [item['id'] for item in ds]
    contents = [item['passage'] for item in ds]

    print("Generating embeddings for text corpus... This may take a while.")
    embeddings = model.encode(contents, show_progress_bar=True)
    print("Embeddings generated.")

    return doc_ids, contents, embeddings


# Data Insertion
def insert_data(conn, doc_ids, contents, embeddings):
    """Inserts the doc_ids, content, and embeddings into the database."""
    print("Inserting data into the database...")
    with conn.cursor() as cur:
        # FIX: Loop with doc_ids and insert into the 'doc_id' column
        for i, (doc_id, content, embedding) in enumerate(zip(doc_ids, contents, embeddings)):
            cur.execute(
                f"INSERT INTO {TABLE_NAME} (doc_id, content, embedding) VALUES (%s, %s, %s)",
                (doc_id, content, embedding)
            )
            if (i + 1) % 100 == 0:
                print(f"   Inserted {i+1}/{len(contents)} records.")
    conn.commit()
    print("Data insertion complete.")


# Main Exec
if __name__ == "__main__":
    connection = get_db_connection()
    if connection:
        setup_database(connection)
        doc_ids_data, contents_data, embeddings_data = load_and_embed_corpus()
        insert_data(connection, doc_ids_data, contents_data, embeddings_data)
        connection.close()
        print("Process finished and database connection closed.")