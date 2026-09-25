"""Tier 2 MORK client and local FAISS index wrapper."""

from __future__ import annotations

import os
import socket
import threading
from abc import ABC, abstractmethod
from urllib.parse import quote, urlparse

import numpy as np

from Residual.symbolic_head.contracts.template_record import (
    MorkQueryResult,
    TemplateRecord,
    validate_template_record,
)
from Residual.symbolic_head.retrieval import FaissTemplateIndex, load_retrieval_config

try:
    import httpx
except ImportError:
    httpx = None

DEFAULT_MORK_SERVER_URL = "http://127.0.0.1:8000"
SEXPR_PATTERN = "$x"
SEXPR_TEMPLATE = "$x"


class MorkClient(ABC):
    """Interface for registering templates and retrieving top-m key/value pairs."""

    def __init__(self, key_dim: int = 256) -> None:
        self.key_dim = key_dim

    @abstractmethod
    def query_top_k(
        self, query_vectors: np.ndarray, top_m: int | None = None
    ) -> MorkQueryResult:
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


def _floats_sexpr(vec: np.ndarray) -> str:
    return " ".join(
        f"{float(x):.8g}" for x in np.asarray(vec, dtype=np.float32).tolist()
    )


def _vector_sexpr(label: str, vec: np.ndarray) -> str:
    values = np.asarray(vec, dtype=np.float32).tolist()
    chunk_size = 16
    if len(values) <= chunk_size:
        return f"({label} {_floats_sexpr(vec)})"

    chunks = [
        "(chunk "
        + " ".join(f"{float(x):.8g}" for x in values[i : i + chunk_size])
        + ")"
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
                raise ValueError(
                    "MORK export contains an unmatched closing parenthesis"
                )
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
        key_dim: int | None = None,
        timeout_sec: float = 10.0,
        index_backend: str | None = None,
        hnsw_m: int | None = None,
        hnsw_ef_construction: int | None = None,
        hnsw_ef_search: int | None = None,
    ) -> None:
        config = load_retrieval_config()
        effective_key_dim = key_dim if key_dim is not None else config.key_dim
        super().__init__(key_dim=effective_key_dim)
        self.server_url = server_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.default_top_m = config.top_m
        self.vector_index = FaissTemplateIndex(
            key_dim=effective_key_dim,
            backend=index_backend if index_backend is not None else config.backend,
            hnsw_m=hnsw_m if hnsw_m is not None else config.hnsw_m,
            hnsw_ef_construction=(
                hnsw_ef_construction
                if hnsw_ef_construction is not None
                else config.hnsw_ef_construction
            ),
            hnsw_ef_search=(
                hnsw_ef_search if hnsw_ef_search is not None else config.hnsw_ef_search
            ),
        )
        self._mork_checked = False
        self._mutation_lock = threading.Lock()

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
        with self._mutation_lock:
            self._require_mork("rebuild_vector_index_from_mork")
            records = self._export_records()
            self.vector_index.rebuild(records)
        return len(records)

    def query_top_k(
        self, query_vectors: np.ndarray, top_m: int | None = None
    ) -> MorkQueryResult:
        self._require_mork("query_top_k")
        return self.vector_index.query_top_k(
            np.asarray(query_vectors, dtype=np.float32),
            top_m=top_m if top_m is not None else self.default_top_m,
        )

    def add_template(
        self,
        template_id: str,
        metta_expr: str,
        key_vector: np.ndarray,
        val_vector: np.ndarray,
    ) -> bool:
        template_id, metta_expr, key_vec, val_vec = validate_template_record(
            template_id, metta_expr, key_vector, val_vector, self.key_dim
        )
        with self._mutation_lock:
            if self.vector_index.contains_template(template_id):
                raise ValueError(f"Duplicate template id: {template_id}")
            payload = template_record_sexpr(template_id, metta_expr, key_vec, val_vec)
            self._require_mork("add_template")
            self._upload_record(payload)
            self.vector_index.add_template(template_id, metta_expr, key_vec, val_vec)
        return True


def get_mork_client(key_dim: int | None = None) -> MorkClient:
    """Create the Docker-backed MORK client used by Tier 2."""
    server_url = os.getenv("MORK_SERVER_URL", DEFAULT_MORK_SERVER_URL)
    return DockerMorkClient(server_url=server_url, key_dim=key_dim)
