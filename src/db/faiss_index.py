"""FAISS index wrapper supporting multiple index types."""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from src.config import EMBEDDING_CONFIG
from src.exceptions import IndexBuildError


class FAISSIndex:
    """Wrapper for FAISS index operations supporting multiple index types.

    Supports:
    - IndexFlatL2 (exact search)
    - IndexIVFFlat (inverted file with flat quantization)
    - IndexHNSWFlat (hierarchical navigable small world)
    """

    def __init__(
        self,
        dimension: int = EMBEDDING_CONFIG.dimension,
        index_type: str = "flat",
        metric: str = "l2",
        nlist: int = 100,
        nprobe: int = 10,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 50,
        use_gpu: bool = True,
    ):
        """Initialize FAISS index wrapper.

        Args:
            dimension: Embedding vector dimension.
            index_type: Type of index ("flat", "ivf", "hnsw").
            metric: Distance metric ("l2" or "ip" for inner product).
            nlist: Number of clusters for IVF index.
            nprobe: Number of clusters to search for IVF.
            hnsw_m: HNSW connections per layer.
            hnsw_ef_construction: HNSW construction parameter.
            hnsw_ef_search: HNSW search parameter.
            use_gpu: Use GPU for index operations if available.
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.use_gpu = use_gpu
        
        self._gpu_resources = None
        self._index: Optional[faiss.Index] = None
        self._id_map: Dict[int, int] = {}
        self._reverse_map: Dict[int, int] = {}
        self._next_id: int = 0
        
        if use_gpu and hasattr(faiss, "StandardGpuResources"):
            try:
                self._gpu_resources = faiss.StandardGpuResources()
                print("Using GPU for FAISS index")
            except Exception:
                self._gpu_resources = None
                print("GPU not available, using CPU")

    def _create_index(self) -> faiss.Index:
        """Create the appropriate FAISS index based on configuration."""
        if self.metric == "l2":
            metric_index = faiss.METRIC_L2
        else:
            metric_index = faiss.METRIC_INNER_PRODUCT

        if self.index_type == "flat":
            return faiss.IndexFlatL2(self.dimension)

        elif self.index_type == "ivf":
            base_index = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(base_index, self.dimension, self.nlist)
            index.nprobe = self.nprobe
            return index

        elif self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.dimension, self.hnsw_m)
            index.hnsw.efConstruction = self.hnsw_ef_construction
            index.hnsw.efSearch = self.hnsw_ef_search
            return index

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

    def build(
        self,
        embeddings: np.ndarray,
        ids: Optional[List[int]] = None,
        train_embeddings: Optional[np.ndarray] = None,
    ) -> None:
        """Build the FAISS index from embeddings.

        Args:
            embeddings: Numpy array of shape (n_vectors, dimension).
            ids: Optional list of document IDs (defaults to 0..n-1).
            train_embeddings: Training data for IVF indices.
        """
        if embeddings.shape[1] != self.dimension:
            raise IndexBuildError(
                f"Embedding dimension {embeddings.shape[1]} != expected {self.dimension}"
            )

        embeddings = embeddings.astype("float32")

        if ids is None:
            ids = list(range(len(embeddings)))

        self._index = self._create_index()

        if self.index_type == "ivf":
            if train_embeddings is None:
                train_size = min(100000, len(embeddings))
                train_embeddings = embeddings[:train_size]
            self._index.train(train_embeddings.astype("float32"))

        embeddings = embeddings.astype("float32")
        start_idx = self._index.ntotal
        self._index.add(embeddings)

        for i, doc_id in enumerate(ids):
            internal_idx = start_idx + i
            self._id_map[internal_idx] = doc_id
            self._reverse_map[doc_id] = internal_idx

        self._next_id = len(ids)

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        ef_search: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """Search for k nearest neighbors.

        Args:
            query: Query vector(s), shape (n_queries, dimension) or (dimension,).
            k: Number of neighbors to return.
            ef_search: Override HNSW search parameter.

        Returns:
            Tuple of (distances, internal_indices, results_dicts)
        """
        if self._index is None:
            raise IndexBuildError("Index not built. Call build() first.")

        if isinstance(query, list):
            query = np.array(query)

        original_shape = query.shape
        query = query.reshape(-1, self.dimension).astype("float32")

        if ef_search is not None and self.index_type == "hnsw":
            self._index.hnsw.efSearch = ef_search

        distances, indices = self._index.search(query, k)

        results = []
        for q_idx, (dists, idxs) in enumerate(zip(distances, indices)):
            for dist, idx in zip(dists, idxs):
                if idx >= 0:
                    results.append({
                        "id": self._id_map[idx],
                        "distance": float(dist),
                        "internal_index": int(idx),
                    })

        return distances, indices, results

    def search_by_id(
        self,
        query_id: int,
        k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """Search using an existing vector by its ID."""
        if query_id not in self._reverse_map:
            raise IndexBuildError(f"ID {query_id} not found in index")

        internal_idx = self._reverse_map[query_id]
        query = self._index.reconstruct(int(internal_idx))
        return self.search(query, k)

    def set_ef_search(self, ef: int) -> None:
        """Set HNSW search parameter for trade-off between speed/recall."""
        if self._index is None:
            raise IndexBuildError("Index not built. Call build() first.")
        if self.index_type == "hnsw":
            self._index.hnsw.efSearch = ef
            self.hnsw_ef_search = ef

    def set_nprobe(self, nprobe: int) -> None:
        """Set IVF search parameter (number of clusters to search)."""
        if self._index is None:
            raise IndexBuildError("Index not built. Call build() first.")
        if self.index_type == "ivf":
            self._index.nprobe = nprobe
            self.nprobe = nprobe

    def save(self, path: str, with_id_map: bool = True) -> None:
        """Save index and optionally ID map to disk.

        Args:
            path: Directory path for saving.
            with_id_map: Whether to save the ID mapping.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._index is None:
            raise IndexBuildError("No index to save.")

        faiss.write_index(self._index, str(path / "index.faiss"))

        if with_id_map:
            with open(path / "id_map.pkl", "wb") as f:
                pickle.dump({
                    "id_map": self._id_map,
                    "reverse_map": self._reverse_map,
                    "next_id": self._next_id,
                    "config": {
                        "dimension": self.dimension,
                        "index_type": self.index_type,
                        "metric": self.metric,
                        "nlist": self.nlist,
                        "nprobe": self.nprobe,
                        "hnsw_m": self.hnsw_m,
                        "hnsw_ef_construction": self.hnsw_ef_construction,
                        "hnsw_ef_search": self.hnsw_ef_search,
                    }
                }, f)

    @classmethod
    def load(cls, path: str) -> "FAISSIndex":
        """Load index and ID map from disk.

        Args:
            path: Directory path containing saved index.

        Returns:
            Loaded FAISSIndex instance.
        """
        path = Path(path)

        with open(path / "id_map.pkl", "rb") as f:
            data = pickle.load(f)

        config = data["config"]
        instance = cls(
            dimension=config["dimension"],
            index_type=config["index_type"],
            metric=config["metric"],
            nlist=config["nlist"],
            nprobe=config["nprobe"],
            hnsw_m=config["hnsw_m"],
            hnsw_ef_construction=config["hnsw_ef_construction"],
            hnsw_ef_search=config["hnsw_ef_search"],
        )

        instance._index = faiss.read_index(str(path / "index.faiss"))
        instance._id_map = data["id_map"]
        instance._reverse_map = data["reverse_map"]
        instance._next_id = data["next_id"]

        return instance

    @property
    def total_vectors(self) -> int:
        """Return total number of vectors in the index."""
        return self._index.ntotal if self._index else 0

    def __len__(self) -> int:
        return self.total_vectors
