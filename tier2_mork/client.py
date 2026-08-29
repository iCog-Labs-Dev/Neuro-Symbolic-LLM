"""Tier 2 MORK client and local FAISS index wrapper."""

from __future__ import annotations

import os
import socket
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from urllib.parse import quote, urlparse

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    import httpx
except ImportError:
    httpx = None

DEFAULT_MORK_SERVER_URL = "http://127.0.0.1:8000"
SEXPR_PATTERN = "$x"
SEXPR_TEMPLATE = "$x"


@dataclass
class MorkQueryResult:
    """Top-m keys/values/ids/scores from the vector index."""

    keys: np.ndarray
    values: np.ndarray
    template_ids: list[list[str]]
    scores: np.ndarray


class MorkClient(ABC):
    """Interface for registering templates and retrieving top-m key/value pairs."""

    def __init__(self, key_dim: int = 256) -> None:
        self.key_dim = key_dim

    @abstractmethod
    def query_top_k(self, query_vectors: np.ndarray, top_m: int = 8) -> MorkQueryResult:
        """Return top_m template key/value pairs matching query_vectors."""

    @abstractmethod
    def add_template(
        self,
        template_id: str,
        metta_expr: str,
        key_vector: np.ndarray,
        val_vector: np.ndarray,
    ) -> bool:
        """Register a template key/value pair in the vector index."""


class _LocalFAISSIndex:
    """In-memory cosine similarity search via FAISS with NumPy fallback.

    This is an internal component used exclusively inside DockerMorkClient
    for cosine ANN on the same CPU node. It is NOT a standalone MorkClient
    and must not be used outside of DockerMorkClient.
    """

    def __init__(self, key_dim: int = 256) -> None:
        self.key_dim = key_dim
        self.key_store: list[np.ndarray] = []
        self.val_store: list[np.ndarray] = []
        self.id_store: list[str] = []
        self.metta_store: list[str] = []
        self._lock = threading.Lock()
        self._faiss_index = None
        if faiss is not None:
            self._faiss_index = faiss.IndexFlatIP(key_dim)

    @property
    def index_backend(self) -> str:
        return "faiss" if self._faiss_index is not None else "numpy"

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
            if self._faiss_index is not None:
                norm = np.linalg.norm(key_vec)
                normalized = key_vec / (norm + 1e-8) if norm > 0 else key_vec
                self._faiss_index.add(np.expand_dims(normalized, axis=0))
            self.key_store.append(key_vec)
            self.val_store.append(val_vec)
            self.id_store.append(template_id)
            self.metta_store.append(metta_expr)

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

            if self._faiss_index is not None:
                q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
                norm_q = np.where(q_norms > 0, queries / (q_norms + 1e-8), queries)
                scores, labels = self._faiss_index.search(norm_q, k_actual)
                scores = np.atleast_2d(np.asarray(scores))
                labels = np.atleast_2d(np.asarray(labels))
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


def _floats_sexpr(vec: np.ndarray) -> str:
    return " ".join(f"{float(x):.8g}" for x in np.asarray(vec, dtype=np.float32).tolist())


def _vector_sexpr(label: str, vec: np.ndarray) -> str:
    values = np.asarray(vec, dtype=np.float32).tolist()
    chunk_size = 16
    if len(values) <= chunk_size:
        return f"({label} {_floats_sexpr(vec)})"

    chunks = [
        "(chunk " + " ".join(f"{float(x):.8g}" for x in values[i : i + chunk_size]) + ")"
        for i in range(0, len(values), chunk_size)
    ]
    return f"({label} {' '.join(chunks)})"


def template_record_sexpr(
    template_id: str, metta_expr: str, key_vector: np.ndarray, val_vector: np.ndarray
) -> str:
    """One S-expr: id, MeTTa, key floats, value floats."""
    return (
        f"(record {template_id} {metta_expr} "
        f"{_vector_sexpr('key', key_vector)} {_vector_sexpr('val', val_vector)})\n"
    )


class DockerMorkClient(MorkClient):
    """Upload Atomese records to MORK and keep a local FAISS key index."""

    def __init__(
        self,
        server_url: str = DEFAULT_MORK_SERVER_URL,
        key_dim: int = 256,
        timeout_sec: float = 10.0,
    ) -> None:
        super().__init__(key_dim=key_dim)
        self.server_url = server_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.vector_index = _LocalFAISSIndex(key_dim=key_dim)
        self._mork_checked = False

    @property
    def index_backend(self) -> str:
        return self.vector_index.index_backend

    def _upload_url(self) -> str:
        pattern = quote(SEXPR_PATTERN, safe="")
        template = quote(SEXPR_TEMPLATE, safe="")
        return f"{self.server_url}/upload/{pattern}/{template}/"

    def is_connected(self) -> bool:
        parsed_url = urlparse(self.server_url)
        if parsed_url.scheme not in ("http", "https") or not parsed_url.hostname:
            return False
        port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
        try:
            with socket.create_connection(
                (parsed_url.hostname, port), timeout=self.timeout_sec
            ):
                return True
        except Exception:
            return False

    def _require_mork(self, operation: str) -> None:
        """Verify MORK server is reachable. Always enforced — no bypass."""
        if httpx is None:
            raise ConnectionError(
                f"httpx is required for MORK {operation} at {self.server_url}. "
                f"Install it: pip install httpx"
            )
        if not self._mork_checked and not self.is_connected():
            raise ConnectionError(
                f"MORK server unreachable at {self.server_url} during {operation}. "
                f"Start it with: docker compose up -d --build"
            )
        self._mork_checked = True

    def _upload_record(self, payload: str) -> None:
        if httpx is None:
            raise ConnectionError(f"httpx required to upload to {self.server_url}.")
        try:
            resp = httpx.post(
                self._upload_url(),
                content=payload,
                headers={"Content-Type": "text/plain"},
                timeout=self.timeout_sec,
            )
        except Exception as err:
            raise ConnectionError(
                f"MORK upload failed at {self.server_url}: {err}"
            ) from err
        if resp.status_code >= 400:
            raise ConnectionError(
                f"MORK upload HTTP {resp.status_code} at {self._upload_url()}."
            )

    def query_top_k(self, query_vectors: np.ndarray, top_m: int = 8) -> MorkQueryResult:
        self._require_mork("query_top_k")
        return self.vector_index.query_top_k(
            np.asarray(query_vectors, dtype=np.float32), top_m=top_m
        )

    def add_template(
        self,
        template_id: str,
        metta_expr: str,
        key_vector: np.ndarray,
        val_vector: np.ndarray,
    ) -> bool:
        key_vec = np.asarray(key_vector, dtype=np.float32)
        val_vec = np.asarray(val_vector, dtype=np.float32)
        payload = template_record_sexpr(template_id, metta_expr, key_vec, val_vec)
        self._require_mork("add_template")
        self._upload_record(payload)
        self.vector_index.add_template(template_id, metta_expr, key_vec, val_vec)
        return True


def get_mork_client(key_dim: int = 256) -> MorkClient:
    """Create the Docker-backed MORK client used by Tier 2."""
    server_url = os.getenv("MORK_SERVER_URL", DEFAULT_MORK_SERVER_URL)
    return DockerMorkClient(server_url=server_url, key_dim=key_dim)
