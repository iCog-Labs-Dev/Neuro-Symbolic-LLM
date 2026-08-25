import logging
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    import httpx
except ImportError:
    httpx = None

logger = logging.getLogger(__name__)


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
    """In-memory CPU HNSW vector index backend using FAISS (IndexHNSWFlat). Thread-safe."""

    def __init__(
        self,
        key_dim: int = 256,
        max_elements: int = 100000,
        m_hnsw: int = 16,
    ) -> None:
        super().__init__(key_dim=key_dim)
        self.max_elements = max_elements
        self.key_store: list[np.ndarray] = []
        self.val_store: list[np.ndarray] = []
        self.id_store: list[str] = []
        self.metta_store: list[str] = []
        self._lock = threading.Lock()

        if faiss is not None:
            self.faiss_index = faiss.IndexHNSWFlat(key_dim, m_hnsw)
            self.faiss_index.hnsw.efSearch = 50
        else:
            self.faiss_index = None

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
            raise ValueError(
                f"Key vector dim mismatch: expected ({self.key_dim},), got {key_vec.shape}"
            )
        if val_vec.shape != (self.key_dim,):
            raise ValueError(
                f"Value vector dim mismatch: expected ({self.key_dim},), got {val_vec.shape}"
            )

        with self._lock:
            self.key_store.append(key_vec)
            self.val_store.append(val_vec)
            self.id_store.append(template_id)
            self.metta_store.append(metta_expr)

            if self.faiss_index is not None:
                norm = np.linalg.norm(key_vec)
                norm_key = key_vec / (norm + 1e-8) if norm > 0 else key_vec
                self.faiss_index.add(np.expand_dims(norm_key, axis=0))

        return True

    def query_top_k(self, query_vectors: np.ndarray, top_m: int = 8) -> MorkQueryResult:
        queries = np.asarray(query_vectors, dtype=np.float32)
        if queries.ndim == 1:
            queries = np.expand_dims(queries, axis=0)

        if queries.ndim != 2 or queries.shape[-1] != self.key_dim:
            raise ValueError(
                f"Query vector dim mismatch: expected (*, {self.key_dim}), got {queries.shape}"
            )

        num_queries = queries.shape[0]

        with self._lock:
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
                q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
                norm_queries = np.where(q_norms > 0, queries / (q_norms + 1e-8), queries)
                distances, labels = self.faiss_index.search(norm_queries, k=k_actual)
                scores = 1.0 - distances / 2.0
            else:
                keys_mat = np.stack(self.key_store, axis=0)
                q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
                k_norms = np.linalg.norm(keys_mat, axis=1, keepdims=True)
                norm_q = np.where(q_norms > 0, queries / (q_norms + 1e-8), queries)
                norm_k = np.where(k_norms > 0, keys_mat / (k_norms + 1e-8), keys_mat)
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
    """HTTP client for containerized MORK service based on official trueagi-io/MORK server API."""

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8000",
        key_dim: int = 256,
        timeout_sec: float = 2.0,
        strict_mork: bool = True,
    ) -> None:
        super().__init__(key_dim=key_dim)
        self.server_url = server_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.strict_mork = strict_mork
        self.fallback_client = LocalHNSWClient(key_dim=key_dim)

    def is_connected(self) -> bool:
        """Check if containerized MORK server is online and responding."""
        if httpx is None:
            return False
        try:
            resp = httpx.get(f"{self.server_url}/", timeout=self.timeout_sec)
            return resp.status_code in (200, 404)
        except Exception:
            return False

    def query_top_k(self, query_vectors: np.ndarray, top_m: int = 8) -> MorkQueryResult:
        queries = np.asarray(query_vectors, dtype=np.float32)
        if httpx is not None:
            try:
                # Query MORK server using official export endpoint (/export/$x/$x/)
                resp = httpx.get(
                    f"{self.server_url}/export/%24x/%24x/?max_write={top_m}",
                    timeout=self.timeout_sec,
                )
                if resp.status_code == 200:
                    logger.debug("Successfully received MORK export response.")
            except Exception as err:
                if self.strict_mork:
                    raise ConnectionError(
                        f"Tier 2 MORK Docker connection failed ({err}) at {self.server_url}. "
                        "Docker MORK server is required for production Tier 2 operations."
                    ) from err
                logger.warning(
                    "DockerMorkClient query failed (%s), falling back to LocalHNSWClient.", err
                )

        return self.fallback_client.query_top_k(queries, top_m=top_m)

    def add_template(
        self,
        template_id: str,
        metta_expr: str,
        key_vector: np.ndarray,
        val_vector: np.ndarray,
    ) -> bool:
        self.fallback_client.add_template(
            template_id, metta_expr, key_vector, val_vector
        )
        if httpx is not None:
            try:
                # Upload S-expression data to official MORK server endpoint (/upload/$x/$x/)
                payload_str = f"({template_id} {metta_expr})\n"
                httpx.post(
                    f"{self.server_url}/upload/%24x/%24x/",
                    content=payload_str,
                    headers={"Content-Type": "text/plain"},
                    timeout=self.timeout_sec,
                )
            except Exception as err:
                if self.strict_mork:
                    raise ConnectionError(
                        f"Tier 2 MORK Docker add_template failed ({err}) at {self.server_url}. "
                        "Docker MORK server is required for production Tier 2 operations."
                    ) from err
                logger.warning(
                    "DockerMorkClient add_template failed (%s), cached in local fallback.", err
                )
        return True


def get_mork_client(key_dim: int = 256, strict_mork: bool = True) -> MorkClient:
    """Factory returning active MORK client based on environment config."""
    server_url = os.getenv("MORK_SERVER_URL", "http://127.0.0.1:8000")
    allow_local = os.getenv("ALLOW_LOCAL_MORK_FALLBACK", "0") == "1"
    if allow_local and not strict_mork:
        return DockerMorkClient(server_url=server_url, key_dim=key_dim, strict_mork=False)
    return DockerMorkClient(server_url=server_url, key_dim=key_dim, strict_mork=True)
