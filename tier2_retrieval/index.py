"""
Generic HNSW Vector Indexing Engine for Tier 2 CPU Pattern Matching.
Provides holistic nearest-neighbor search over symbolic key-space without domain hardcoding.
Includes robust NumPy fallback if hnswlib is unavailable.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
import numpy as np

try:
    import hnswlib
    HAS_HNSWLIB = True
except ImportError:
    HAS_HNSWLIB = False

logger = logging.getLogger(__name__)


class GenericHNSWIndex:
    """
    High-performance, generic HNSW index wrapper using hnswlib (with pure NumPy fallback).
    Operates on CPU DRAM for Tier 2 sparse pattern retrieval.
    """

    def __init__(
        self,
        dim: int = 256,
        space: str = "cosine",
        max_elements: int = 100000,
        ef_construction: int = 200,
        M: int = 16,
        random_seed: int = 42,
    ) -> None:
        self.dim = dim
        self.space = space
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.M = M
        self.has_hnswlib = HAS_HNSWLIB

        if self.has_hnswlib:
            self.index = hnswlib.Index(space=space, dim=dim)
            self.index.init_index(
                max_elements=max_elements,
                ef_construction=ef_construction,
                M=M,
                random_seed=random_seed,
            )
            self.index.set_ef(50)
        else:
            logger.warning("hnswlib not found; using pure NumPy vector search fallback.")
            self._keys_list: List[np.ndarray] = []
            self._ids_list: List[int] = []

        self._id_to_metadata: Dict[int, Dict[str, Any]] = {}

    def set_query_ef(self, ef: int) -> None:
        """Adjust search-time accuracy vs latency trade-off."""
        if self.has_hnswlib:
            self.index.set_ef(ef)

    def add_items(
        self,
        keys: np.ndarray,
        ids: Union[List[int], np.ndarray],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add key vectors and optional metadata to the index."""
        keys = np.asarray(keys, dtype=np.float32)
        if keys.ndim == 1:
            keys = keys.reshape(1, -1)
            ids = [ids[0]] if isinstance(ids, (list, np.ndarray)) else [ids]

        if keys.shape[1] != self.dim:
            raise ValueError(
                f"Key dimension mismatch: expected {self.dim}, got {keys.shape[1]}"
            )

        ids = np.asarray(ids, dtype=np.int64)

        if self.space == "cosine":
            norms = np.linalg.norm(keys, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            keys = keys / norms

        if self.has_hnswlib:
            curr_count = self.index.get_current_count()
            if curr_count + keys.shape[0] > self.max_elements:
                new_capacity = max(self.max_elements * 2, curr_count + keys.shape[0] + 10000)
                self.index.resize_index(new_capacity)
                self.max_elements = new_capacity
            self.index.add_items(keys, ids)
        else:
            for k, item_id in zip(keys, ids):
                self._keys_list.append(k)
                self._ids_list.append(int(item_id))

        if metadata:
            for idx, item_id in enumerate(ids):
                self._id_to_metadata[int(item_id)] = metadata[idx]

    def query(
        self, query_vectors: np.ndarray, top_m: int = 8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform top-m nearest-neighbor search."""
        query_vectors = np.asarray(query_vectors, dtype=np.float32)
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)

        if self.space == "cosine":
            norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            query_vectors = query_vectors / norms

        if self.has_hnswlib:
            labels, distances = self.index.knn_query(query_vectors, k=top_m)
            return labels, distances

        # Pure NumPy Fallback
        if not self._keys_list:
            return np.zeros((query_vectors.shape[0], top_m), dtype=np.int64), np.zeros((query_vectors.shape[0], top_m), dtype=np.float32)

        all_keys = np.vstack(self._keys_list)
        all_ids = np.array(self._ids_list, dtype=np.int64)
        n_queries = query_vectors.shape[0]

        labels_list = []
        dists_list = []

        for i in range(n_queries):
            q = query_vectors[i]
            if self.space == "cosine":
                sims = np.dot(all_keys, q)
                dists = 1.0 - sims
            else:
                dists = np.linalg.norm(all_keys - q, axis=1)

            top_idx = np.argsort(dists)[:top_m]
            labels_list.append(all_ids[top_idx])
            dists_list.append(dists[top_idx])

        return np.vstack(labels_list), np.vstack(dists_list)

    def get_metadata(self, item_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve stored metadata for a given template ID."""
        return self._id_to_metadata.get(item_id, None)

    def count(self) -> int:
        """Return total number of indexed templates."""
        if self.has_hnswlib:
            return self.index.get_current_count()
        return len(self._keys_list)
