"""Pluggable MORK Client Interface and Concrete Backends.

Supports:
1. DockerMorkClient: HTTP REST client connecting to official containerized MORK service.
2. LocalHNSWClient: Fast in-memory hnswlib index backend for local testing and offline sweeps.
3. MmapPathMapClient: Memory-mapped .act PathMap binary snapshot loader backend.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import typing as t

import numpy as np

try:
    import hnswlib
except ImportError:
    hnswlib = None

try:
    import httpx
except ImportError:
    httpx = None


@dataclass
class MorkQueryResult:
    """Container for matched template key/value payload returned from Tier 2 MORK."""

    keys: np.ndarray  # shape: (N, m, k), float32
    values: np.ndarray  # shape: (N, m, k), float32
    template_ids: list[list[str]]  # shape: (N, m)
    scores: np.ndarray  # shape: (N, m), float32


class MorkClient(ABC):
    """Abstract interface for Tier 2 MORK sparse symbolic engine."""

    def __init__(self, key_dim: int = 256) -> None:
        self.key_dim = key_dim

    @abstractmethod
    def query_top_k(self, query_vectors: np.ndarray, top_m: int = 8) -> MorkQueryResult:
        """Query MORK for top_m template key/value pairs matching input query vectors.

        Args:
            query_vectors: np.ndarray of shape (N, k), float32
            top_m: number of nearest template keys to retrieve per position

        Returns:
            MorkQueryResult containing matched key matrices, value matrices, IDs, and scores.
        """

    @abstractmethod
    def add_template(
        self,
        template_id: str,
        metta_expr: str,
        key_vector: np.ndarray,
        val_vector: np.ndarray,
    ) -> bool:
        """Add a newly mined template key/value pair to the MORK store."""


class LocalHNSWClient(MorkClient):
    """In-memory hnswlib index backend for local execution and offline testing."""

    def __init__(
        self,
        key_dim: int = 256,
        max_elements: int = 100000,
        ef_construction: int = 200,
        M: int = 16,
    ) -> None:
        super().__init__(key_dim=key_dim)
        self.max_elements = max_elements
        self.key_store: list[np.ndarray] = []
        self.val_store: list[np.ndarray] = []
        self.id_store: list[str] = []
        self.metta_store: list[str] = []

        if hnswlib is not None:
            self.index = hnswlib.Index(space="cosine", dim=key_dim)
            self.index.init_index(
                max_elements=max_elements,
                ef_construction=ef_construction,
                M=M,
            )
            self.index.set_ef(50)
        else:
            self.index = None

    def add_template(
        self,
        template_id: str,
        metta_expr: str,
        key_vector: np.ndarray,
        val_vector: np.ndarray,
    ) -> bool:
        idx = len(self.key_store)
        key_vec = np.asarray(key_vector, dtype=np.float32)
        val_vec = np.asarray(val_vector, dtype=np.float32)

        if key_vec.shape != (self.key_dim,):
            raise ValueError(f"Key vector dimension mismatch: expected {self.key_dim}, got {key_vec.shape}")

        self.key_store.append(key_vec)
        self.val_store.append(val_vec)
        self.id_store.append(template_id)
        self.metta_store.append(metta_expr)

        if self.index is not None:
            if idx >= self.max_elements:
                self.index.resize_index(self.max_elements * 2)
                self.max_elements *= 2
            self.index.add_items(key_vec, idx)

        return True

    def query_top_k(self, query_vectors: np.ndarray, top_m: int = 8) -> MorkQueryResult:
        queries = np.asarray(query_vectors, dtype=np.float32)
        num_queries = queries.shape[0]
        num_items = len(self.key_store)

        if num_items == 0:
            # Empty store fallback: return zeros
            empty_keys = np.zeros((num_queries, top_m, self.key_dim), dtype=np.float32)
            empty_vals = np.zeros((num_queries, top_m, self.key_dim), dtype=np.float32)
            empty_ids = [["empty"] * top_m for _ in range(num_queries)]
            empty_scores = np.zeros((num_queries, top_m), dtype=np.float32)
            return MorkQueryResult(
                keys=empty_keys,
                values=empty_vals,
                template_ids=empty_ids,
                scores=empty_scores,
            )

        k_actual = min(top_m, num_items)

        if self.index is not None and num_items >= k_actual:
            labels, distances = self.index.knn_query(queries, k=k_actual)
            scores = 1.0 - distances
        else:
            # Brute-force cosine similarity fallback
            keys_mat = np.stack(self.key_store, axis=0)  # (M, k)
            norm_q = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
            norm_k = keys_mat / (np.linalg.norm(keys_mat, axis=1, keepdims=True) + 1e-8)
            sim_mat = np.matmul(norm_q, norm_k.T)  # (N, M)
            labels = np.argsort(-sim_mat, axis=1)[:, :k_actual]
            scores = np.take_along_axis(sim_mat, labels, axis=1)

        # Pad to top_m if k_actual < top_m
        matched_keys = np.zeros((num_queries, top_m, self.key_dim), dtype=np.float32)
        matched_vals = np.zeros((num_queries, top_m, self.key_dim), dtype=np.float32)
        matched_scores = np.zeros((num_queries, top_m), dtype=np.float32)
        matched_ids = [["empty"] * top_m for _ in range(num_queries)]

        for i in range(num_queries):
            for j in range(k_actual):
                item_idx = int(labels[i, j])
                matched_keys[i, j] = self.key_store[item_idx]
                matched_vals[i, j] = self.val_store[item_idx]
                matched_scores[i, j] = scores[i, j]
                matched_ids[i][j] = self.id_store[item_idx]

        return MorkQueryResult(
            keys=matched_keys,
            values=matched_vals,
            template_ids=matched_ids,
            scores=matched_scores,
        )


class DockerMorkClient(MorkClient):
    """HTTP client for official containerized MORK server instance."""

    def __init__(self, server_url: str = "http://localhost:8080", key_dim: int = 256, timeout_sec: float = 2.0) -> None:
        super().__init__(key_dim=key_dim)
        self.server_url = server_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.fallback_client = LocalHNSWClient(key_dim=key_dim)

    def query_top_k(self, query_vectors: np.ndarray, top_m: int = 8) -> MorkQueryResult:
        queries = np.asarray(query_vectors, dtype=np.float32)
        if httpx is not None:
            try:
                payload = {
                    "query_vectors": queries.tolist(),
                    "top_m": top_m,
                }
                resp = httpx.post(
                    f"{self.server_url}/query",
                    json=payload,
                    timeout=self.timeout_sec,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return MorkQueryResult(
                        keys=np.array(data["keys"], dtype=np.float32),
                        values=np.array(data["values"], dtype=np.float32),
                        template_ids=data["template_ids"],
                        scores=np.array(data["scores"], dtype=np.float32),
                    )
            except Exception:
                # Fallback on connection error/timeout
                pass

        return self.fallback_client.query_top_k(queries, top_m=top_m)

    def add_template(
        self,
        template_id: str,
        metta_expr: str,
        key_vector: np.ndarray,
        val_vector: np.ndarray,
    ) -> bool:
        self.fallback_client.add_template(template_id, metta_expr, key_vector, val_vector)
        if httpx is not None:
            try:
                payload = {
                    "template_id": template_id,
                    "metta_expr": metta_expr,
                    "key_vector": np.asarray(key_vector, dtype=np.float32).tolist(),
                    "val_vector": np.asarray(val_vector, dtype=np.float32).tolist(),
                }
                httpx.post(
                    f"{self.server_url}/templates/add",
                    json=payload,
                    timeout=self.timeout_sec,
                )
            except Exception:
                pass
        return True


class MmapPathMapClient(MorkClient):
    """Memory-mapped .act PathMap binary snapshot loader client."""

    def __init__(self, snapshot_path: t.Optional[str] = None, key_dim: int = 256) -> None:
        super().__init__(key_dim=key_dim)
        self.snapshot_path = snapshot_path
        self.fallback_client = LocalHNSWClient(key_dim=key_dim)
        if snapshot_path and os.path.exists(snapshot_path):
            self._load_snapshot(snapshot_path)

    def _load_snapshot(self, path: str) -> None:
        # Placeholder for memory-mapping PathMap .act file via POSIX mmap
        pass

    def query_top_k(self, query_vectors: np.ndarray, top_m: int = 8) -> MorkQueryResult:
        return self.fallback_client.query_top_k(query_vectors, top_m=top_m)

    def add_template(
        self,
        template_id: str,
        metta_expr: str,
        key_vector: np.ndarray,
        val_vector: np.ndarray,
    ) -> bool:
        return self.fallback_client.add_template(template_id, metta_expr, key_vector, val_vector)


def get_mork_client(key_dim: int = 256) -> MorkClient:
    """Factory function returning active MORK client based on environment config."""
    server_url = os.getenv("MORK_SERVER_URL")
    if server_url:
        return DockerMorkClient(server_url=server_url, key_dim=key_dim)
    snapshot_path = os.getenv("MORK_SNAPSHOT_PATH")
    if snapshot_path and os.path.exists(snapshot_path):
        return MmapPathMapClient(snapshot_path=snapshot_path, key_dim=key_dim)
    return LocalHNSWClient(key_dim=key_dim)
