# Fine-tuning Vector Databases for RAG Systems

A high-performance Retrieval-Augmented Generation (RAG) pipeline using PostgreSQL (pgvector) + FAISS for optimized vector search.

## Overview

- **Dataset**: ML-ArXiv-Papers (10K-100K abstracts)
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **Vector Stores**: pgvector (PostgreSQL) + FAISS (GPU-accelerated)
- **Index Types**: Flat, IVFFlat, HNSW
- **LLM**: Ollama (llama3.2)

## Prerequisites

1. **Python 3.10+**
2. **PostgreSQL 15+** with pgvector extension
3. **Ollama** (optional, for LLM generation)

### Install PostgreSQL + pgvector

**Docker:**
```bash
docker run -d \
  --name pgvector \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=secret \
  -e POSTGRES_DB=rag_db \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

**Enable pgvector extension:**
```sql
CREATE EXTENSION vector;
```

### Install Ollama (optional)

```bash
ollama pull llama3.2
ollama serve
```

## Installation

```bash
# Clone and enter directory
cd capstone-phase2

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create `.env` file:

```env
DB_USER=postgres
DB_PASSWORD=secret
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_db
OLLAMA_URL=http://localhost:11434/api/generate
LLM_MODEL=llama3.2
```

## Usage

### 1. Setup Database

Load arXiv papers and build indexes:

```bash
python -m src.setup_db
```

### 2. Ask Questions (RAG)

```bash
python -m src.test_rag
```

Enter a question when prompted. Requires Ollama running.

### 3. Run Benchmarks

```bash
# Vector search benchmarks (pgvector vs FAISS)
python -m src.benchmark

# Metadata filtering benchmarks
python -m src.benchmark_metadata
```

Compares pgvector vs FAISS search latencies and metadata filtering performance.

## Project Structure

```
src/
├── config.py           # Configuration
├── exceptions.py        # Custom exceptions
├── setup_db.py         # Database setup
├── test_rag.py         # Interactive RAG query
├── benchmark.py        # Latency benchmarks
├── data/
│   └── loader.py       # Dataset loading
├── db/
│   ├── pgvector.py     # PostgreSQL wrapper
│   └── faiss_index.py  # FAISS wrapper
├── retrieval/
│   └── benchmarks.py   # Performance benchmarks
└── rag/
    └── generator.py    # RAG pipeline
```

## Performance

| Engine | Index | Latency | Recall@5 | Speedup |
|--------|-------|---------|----------|---------|
| pgvector | Flat | 299ms | 100% | 1x |
| pgvector | IVFFlat-100 | 81ms | 68% | 4x |
| pgvector | HNSW | 269ms | 100% | 1x |
| FAISS | Flat | 1.1ms | 100% | 264x |
| FAISS | IVFFlat-100 | 0.12ms | 100% | 2490x |
| FAISS | HNSW | 0.32ms | 84% | 942x |

**Key Finding:** FAISS HNSW achieves **942x speedup** over pgvector with **84% recall retention**.

Full benchmark report: `data/results/BENCHMARK_REPORT.md`

## Requirements

See `requirements.txt` for full dependency list.
