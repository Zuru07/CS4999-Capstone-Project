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

# Model for generating embeddings - Minimal transformer for embedding
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
TABLE_NAME = 'wikipedia_qa'


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
    """Sets up the pgvector extension and creates the table for storing data."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        print("pgvector extension is enabled.")

        register_vector(conn)

        model = SentenceTransformer(EMBEDDING_MODEL)
        embedding_dim = model.get_sentence_embedding_dimension()

        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id SERIAL PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT,
            embedding vector({embedding_dim})
        );
        """)
        conn.commit()
        print(f"Table '{TABLE_NAME}' is ready.")


# Data Loading and Embedding Generation 
def load_and_embed_data():
    """Loads the dataset and generates embeddings for the questions."""
    print("Loading dataset 'rag-datasets/rag-mini-wikipedia'...")
    ds = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer", split="train")
    print("Dataset loaded successfully.")

    print(f"Loading sentence transformer model '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded successfully.")

    questions = [item['question'] for item in ds]
    answers = [item['answer'] for item in ds]

    print("Generating embeddings for questions... This may take a moment.")
    embeddings = model.encode(questions, show_progress_bar=True)
    print("Embeddings generated.")

    return questions, answers, embeddings


# Data Insertion
def insert_data(conn, questions, answers, embeddings):
    """Inserts the questions, answers, and their embeddings into the database."""
    print("Inserting data into the database...")
    with conn.cursor() as cur:
        for i, (question, answer, embedding) in enumerate(zip(questions, answers, embeddings)):
            cur.execute(
                f"INSERT INTO {TABLE_NAME} (question, answer, embedding) VALUES (%s, %s, %s)",
                (question, answer, embedding)
            )
            if (i + 1) % 100 == 0:
                print(f"   Inserted {i+1}/{len(questions)} records.")
    conn.commit()
    print("Data insertion complete.")


# Main Exec
if __name__ == "__main__":
    connection = get_db_connection()
    if connection:
        setup_database(connection)
        questions_data, answers_data, embeddings_data = load_and_embed_data()
        insert_data(connection, questions_data, answers_data, embeddings_data)
        connection.close()
        print("Process finished and database connection closed.")
