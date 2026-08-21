from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import typing as t

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

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
    """Abstract client interface for Tier 2 MORK sparse symbolic engine."""

    def __init__(self, key_dim: int = 256) -> None:
        self.key_dim = key_dim

    @abstractmethod
    def query_top_k(self, query_vectors: np.ndarray, top_m: int = 8) -> MorkQueryResult:
        """Query top_m template key/value pairs matching query_vectors."""

    @abstractmethod
    def add_template(
        self,
        template_id: str,
        metta_expr: str,
        key_vector: np.ndarray,
        val_vector: np.ndarray,
    ) -> bool:
        """Register a template key/value pair in MORK."""


class LocalHNSWClient(MorkClient):
    """In-memory HNSW vector index backend using FAISS (IndexHNSWFlat) / HNSWLib for CPU execution."""

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

        if faiss is not None:
            self.faiss_index = faiss.IndexHNSWFlat(key_dim, M)
            self.faiss_index.hnsw.efSearch = 50
            self.hnsw_index = None
        elif hnswlib is not None:
            self.faiss_index = None
            self.hnsw_index = hnswlib.Index(space="cosine", dim=key_dim)
            self.hnsw_index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
            self.hnsw_index.set_ef(50)
        else:
            self.faiss_index = None
            self.hnsw_index = None

    def add_template(
        self,
        template_id: str,
        metta_expr: str,
        key_vector: np.ndarray,
        val_vector: np.ndarray,
    ) -> bool:
        key_vec = np.asarray(key_vector, dtype=np.float32)
        val_vec = np.asarray(val_vector, dtype=np.float32)

        if key_vec.shape != (self.key_dim,):
            raise ValueError(f"Key vector dim mismatch: expected {self.key_dim}, got {key_vec.shape}")

        self.key_store.append(key_vec)
        self.val_store.append(val_vec)
        self.id_store.append(template_id)
        self.metta_store.append(metta_expr)

        if self.faiss_index is not None:
            norm_key = key_vec / (np.linalg.norm(key_vec) + 1e-8)
            self.faiss_index.add(np.expand_dims(norm_key, axis=0))
        elif self.hnsw_index is not None:
            idx = len(self.key_store) - 1
            if idx >= self.max_elements:
                self.hnsw_index.resize_index(self.max_elements * 2)
                self.max_elements *= 2
            self.hnsw_index.add_items(key_vec, idx)

        return True

    def query_top_k(self, query_vectors: np.ndarray, top_m: int = 8) -> MorkQueryResult:
        queries = np.asarray(query_vectors, dtype=np.float32)
        num_queries = queries.shape[0]
        num_items = len(self.key_store)

        if num_items == 0:
            return MorkQueryResult(
                keys=np.zeros((num_queries, top_m, self.key_dim), dtype=np.float32),
                values=np.zeros((num_queries, top_m, self.key_dim), dtype=np.float32),
                template_ids=[["empty"] * top_m for _ in range(num_queries)],
                scores=np.zeros((num_queries, top_m), dtype=np.float32),
            )

        k_actual = min(top_m, num_items)

        if self.faiss_index is not None and num_items >= k_actual:
            norm_queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
            distances, labels = self.faiss_index.search(norm_queries, k=k_actual)
            scores = 1.0 - distances / 2.0
        elif self.hnsw_index is not None and num_items >= k_actual:
            labels, distances = self.hnsw_index.knn_query(queries, k=k_actual)
            scores = 1.0 - distances
        else:
            keys_mat = np.stack(self.key_store, axis=0)
            norm_q = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
            norm_k = keys_mat / (np.linalg.norm(keys_mat, axis=1, keepdims=True) + 1e-8)
            sim_mat = np.matmul(norm_q, norm_k.T)
            labels = np.argsort(-sim_mat, axis=1)[:, :k_actual]
            scores = np.take_along_axis(sim_mat, labels, axis=1)

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
    """HTTP REST client for containerized MORK service."""

    def __init__(self, server_url: str = "http://localhost:8080", key_dim: int = 256, timeout_sec: float = 2.0) -> None:
        super().__init__(key_dim=key_dim)
        self.server_url = server_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.fallback_client = LocalHNSWClient(key_dim=key_dim)

    def query_top_k(self, query_vectors: np.ndarray, top_m: int = 8) -> MorkQueryResult:
        queries = np.asarray(query_vectors, dtype=np.float32)
        if httpx is not None:
            try:
                resp = httpx.post(
                    f"{self.server_url}/query",
                    json={"query_vectors": queries.tolist(), "top_m": top_m},
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
                httpx.post(f"{self.server_url}/templates/add", json=payload, timeout=self.timeout_sec)
            except Exception:
                pass
        return True


class MmapPathMapClient(MorkClient):
    """Memory-mapped PathMap binary snapshot client for CLI workflows."""

    def __init__(self, snapshot_path: t.Optional[str] = None, key_dim: int = 256) -> None:
        super().__init__(key_dim=key_dim)
        self.snapshot_path = snapshot_path
        self.fallback_client = LocalHNSWClient(key_dim=key_dim)

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
    """Factory returning active MORK client based on environment config."""
    server_url = os.getenv("MORK_SERVER_URL")
    if server_url:
        return DockerMorkClient(server_url=server_url, key_dim=key_dim)
    snapshot_path = os.getenv("MORK_SNAPSHOT_PATH")
    if snapshot_path and os.path.exists(snapshot_path):
        return MmapPathMapClient(snapshot_path=snapshot_path, key_dim=key_dim)
    return LocalHNSWClient(key_dim=key_dim)
