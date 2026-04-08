"""Configuration and environment variables."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class DBConfig:
    user: str
    password: str
    host: str
    port: str
    name: str


@dataclass
class EmbeddingConfig:
    model_name: str
    dimension: int = 384
    batch_size: int = 32


@dataclass
class IndexConfig:
    index_type: str
    ivf_lists: int = 100
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200


DB_CONFIG = DBConfig(
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD", "postgres"),
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", "5432"),
    name=os.getenv("DB_NAME", "rag_db"),
)

EMBEDDING_CONFIG = EmbeddingConfig(
    model_name=os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
    dimension=384,
    batch_size=128,  # Larger batch for GPU
)

INDEX_CONFIG = IndexConfig(
    index_type=os.getenv("INDEX_TYPE", "ivfflat"),
    ivf_lists=int(os.getenv("IVF_LISTS", "100")),
    hnsw_m=int(os.getenv("HNSW_M", "16")),
    hnsw_ef_construction=int(os.getenv("HNSW_EFCONSTRUCTION", "200")),
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")

DATASET_NAME = "ccdv/arxiv-summarization"
DEFAULT_TOP_K = 5
METADATA_FILTER_LATENCY_TARGET_MS = 20
