# Local RAG Pipeline (Llama 3.2 + pgvector)

This repository contains the data preparation, vector database indexing, and Retrieval-Augmented Generation (RAG) pipeline using a local installation of Ollama.

## Prerequisites
1. **PostgreSQL + pgvector:** You must have PostgreSQL installed with the `pgvector` extension enabled.
2. **Ollama:** Download and install Ollama from [ollama.com](https://ollama.com).
3. **Pull the Model:** Open your terminal and run `ollama pull llama3.2` to download the 3B parameter model. Make sure Ollama is running in the background.

## Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

2. **Environment Variables:**
    Copy the .env.example file, rename it to .env, and fill in your local PostgreSQL credentials.

3. **Seed the database:**
    ```bash
    python seed_db.py --parquet data/ag_news_50000.parquet --table hf_docs_50k --dim 384

4. **Index Benchmarking:**
    This one recheck once and se
    ```bash
    python index_benchmark.py --table hf_docs_50k

5. **Run the pipeline:**
    ```bash
    python rag_pipeline.py

