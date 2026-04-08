"""Custom exceptions for the RAG pipeline."""


class RAGPipelineError(Exception):
    """Base exception for RAG pipeline errors."""
    pass


class VectorSearchError(RAGPipelineError):
    """Raised when vector search operations fail."""
    pass


class EmbeddingError(RAGPipelineError):
    """Raised when embedding generation fails."""
    pass


class DatabaseConnectionError(RAGPipelineError):
    """Raised when database connection fails."""
    pass


class IndexBuildError(RAGPipelineError):
    """Raised when index building fails."""
    pass


class BenchmarkError(RAGPipelineError):
    """Raised when benchmarking operations fail."""
    pass


class LLMGenerationError(RAGPipelineError):
    """Raised when LLM generation fails."""
    pass
