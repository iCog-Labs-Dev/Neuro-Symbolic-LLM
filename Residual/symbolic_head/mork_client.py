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


@dataclass(frozen=True)
class TemplateRecord:
    """Authoritative template data used to rebuild a local retrieval index."""

    template_id: str
    metta_expr: str
    key_vector: np.ndarray
    val_vector: np.ndarray


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

    def __init__(
        self,
        key_dim: int = 256,
        backend: str = "flat",
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 50,
    ) -> None:
        self.key_dim = key_dim
        if backend not in {"flat", "hnsw"}:
            raise ValueError("backend must be 'flat' or 'hnsw'")
        self.key_store: list[np.ndarray] = []
        self.val_store: list[np.ndarray] = []
        self.id_store: list[str] = []
        self.metta_store: list[str] = []
        self._lock = threading.Lock()
        self._faiss_index = None
        self._backend = backend
        self._hnsw_ef_search = hnsw_ef_search
        if faiss is not None:
            if backend == "hnsw":
                self._faiss_index = faiss.IndexHNSWFlat(
                    key_dim, hnsw_m, faiss.METRIC_INNER_PRODUCT
                )
                self._faiss_index.hnsw.efConstruction = hnsw_ef_construction
                self._faiss_index.hnsw.efSearch = hnsw_ef_search
            else:
                self._faiss_index = faiss.IndexFlatIP(key_dim)

    @property
    def index_backend(self) -> str:
        if self._faiss_index is None:
            return "numpy"
        return self._backend

    def clear(self) -> None:
        with self._lock:
            self.key_store.clear()
            self.val_store.clear()
            self.id_store.clear()
            self.metta_store.clear()
            if self._faiss_index is not None:
                self._faiss_index.reset()

    def rebuild(self, records: list[TemplateRecord]) -> None:
        self.clear()
        for record in records:
            self.add_template(
                record.template_id,
                record.metta_expr,
                record.key_vector,
                record.val_vector,
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


def _split_top_level_expressions(text: str) -> list[str]:
    expressions: list[str] = []
    depth = 0
    start = None
    for index, char in enumerate(text):
        if char == "(":
            if depth == 0:
                start = index
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("MORK export contains an unmatched closing parenthesis")
            if depth == 0 and start is not None:
                expressions.append(text[start : index + 1])
                start = None
    if depth != 0:
        raise ValueError("MORK export contains an unclosed expression")
    return expressions


def _parse_sexpr(expression: str) -> list[object]:
    tokens = expression.replace("(", " ( ").replace(")", " ) ").split()
    position = 0

    def parse_list() -> list[object]:
        nonlocal position
        if position >= len(tokens) or tokens[position] != "(":
            raise ValueError("Expected an opening parenthesis")
        position += 1
        values: list[object] = []
        while position < len(tokens) and tokens[position] != ")":
            if tokens[position] == "(":
                values.append(parse_list())
            else:
                values.append(tokens[position])
                position += 1
        if position >= len(tokens):
            raise ValueError("Unclosed S-expression")
        position += 1
        return values

    parsed = parse_list()
    if position != len(tokens):
        raise ValueError("Unexpected tokens after S-expression")
    return parsed


def _flatten_vector(node: object) -> np.ndarray:
    if not isinstance(node, list) or not node or node[0] not in {"key", "val"}:
        raise ValueError("Expected a key or val vector")
    values: list[float] = []
    for item in node[1:]:
        if isinstance(item, list):
            if not item or item[0] != "chunk":
                raise ValueError("Unexpected vector group in MORK record")
            values.extend(float(value) for value in item[1:])
        else:
            values.append(float(item))
    return np.asarray(values, dtype=np.float32)


def _render_sexpr(node: object) -> str:
    if isinstance(node, list):
        return "(" + " ".join(_render_sexpr(item) for item in node) + ")"
    return str(node)


def _parse_export_records(text: str, key_dim: int) -> list[TemplateRecord]:
    records: list[TemplateRecord] = []
    for expression in _split_top_level_expressions(text):
        parsed = _parse_sexpr(expression)
        if len(parsed) != 5 or parsed[0] != "record":
            continue
        template_id, metta_expr = parsed[1], _render_sexpr(parsed[2])
        if not isinstance(template_id, str):
            raise ValueError("MORK record id must be a symbol")
        key_vector = _flatten_vector(parsed[3])
        val_vector = _flatten_vector(parsed[4])
        if key_vector.shape != (key_dim,) or val_vector.shape != (key_dim,):
            raise ValueError(
                f"MORK record {template_id!r} has invalid vector dimensions"
            )
        records.append(
            TemplateRecord(
                template_id=template_id,
                metta_expr=metta_expr,
                key_vector=key_vector,
                val_vector=val_vector,
            )
        )
    return records


class DockerMorkClient(MorkClient):
    """Upload Atomese records to MORK and keep a local FAISS key index."""

    def __init__(
        self,
        server_url: str = DEFAULT_MORK_SERVER_URL,
        key_dim: int = 256,
        timeout_sec: float = 10.0,
        index_backend: str | None = None,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 50,
    ) -> None:
        super().__init__(key_dim=key_dim)
        self.server_url = server_url.rstrip("/")
        self.timeout_sec = timeout_sec
        selected_backend = index_backend or os.getenv("MORK_INDEX_BACKEND", "flat")
        self.vector_index = _LocalFAISSIndex(
            key_dim=key_dim,
            backend=selected_backend,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef_search=hnsw_ef_search,
        )
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

    def _export_url(self) -> str:
        pattern = quote(SEXPR_PATTERN, safe="")
        template = quote(SEXPR_TEMPLATE, safe="")
        return f"{self.server_url}/export/{pattern}/{template}/"

    def _export_records(self) -> list[TemplateRecord]:
        if httpx is None:
            raise ConnectionError(f"httpx required to export from {self.server_url}.")
        try:
            response = httpx.get(
                self._export_url(),
                params={"max_write": "0"},
                timeout=self.timeout_sec,
            )
        except Exception as err:
            raise ConnectionError(
                f"MORK export failed at {self.server_url}: {err}"
            ) from err
        if response.status_code >= 400:
            raise ConnectionError(
                f"MORK export HTTP {response.status_code} at {self._export_url()}."
            )
        return _parse_export_records(response.text, self.key_dim)

    def rebuild_vector_index_from_mork(self) -> int:
        """Rebuild the derived local index from the current MORK export."""
        self._require_mork("rebuild_vector_index_from_mork")
        records = self._export_records()
        self.vector_index.rebuild(records)
        return len(records)

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
