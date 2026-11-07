"""Rebuildable CPU FAISS index derived from authoritative MORK records."""

import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from residual.symbolic_head.contracts.template_record import (
    MorkQueryResult,
    TemplateRecord,
    validate_template_record,
)

faiss: Any
try:
    import faiss
except ImportError:
    faiss = None


class FaissTemplateIndex:
    """Cosine top-m search; HNSW is primary and Flat is an exact reference."""

    def __init__(
        self,
        key_dim: int = 256,
        backend: str = "hnsw",
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 50,
    ) -> None:
        if faiss is None:
            raise RuntimeError(
                "FAISS is required for symbolic template retrieval; install faiss-cpu"
            )
        if backend not in {"flat", "hnsw"}:
            raise ValueError("backend must be 'flat' or 'hnsw'")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (key_dim, hnsw_m, hnsw_ef_construction, hnsw_ef_search)
        ):
            raise ValueError(
                "Index dimension and HNSW parameters must be positive integers"
            )
        self.key_dim = key_dim
        self._backend = backend
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._hnsw_ef_search = hnsw_ef_search
        self._lock = threading.Lock()
        self.key_store: list[np.ndarray] = []
        self.val_store: list[np.ndarray] = []
        self.id_store: list[str] = []
        self.metta_store: list[str] = []
        self._id_set: set[str] = set()
        self._gather_arrays: tuple[np.ndarray, np.ndarray] | None = None
        self._generation = 0
        self._native_listeners: dict[int, Callable[[], None]] = {}
        self._next_listener = 0
        self._faiss_index = self._new_index()

    def _invalidate_native(self) -> None:
        """Invalidate compiled snapshots before a supported index mutation."""
        for invalidate in tuple(self._native_listeners.values()):
            invalidate()

    def native_snapshot(
        self,
    ) -> tuple[int, tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
        """Export an atomic CPU snapshot, including the existing FAISS graph."""
        with self._lock:
            if not self.id_store:
                raise ValueError("Native retrieval requires a nonempty bank")
            return (
                self._generation,
                tuple(self.id_store),
                np.ascontiguousarray(faiss.serialize_index(self._faiss_index)),
                np.ascontiguousarray(np.stack(self.key_store), dtype=np.float32),
                np.ascontiguousarray(np.stack(self.val_store), dtype=np.float32),
            )

    def watch_native_generation(
        self, generation: int, invalidate: Callable[[], None]
    ) -> int:
        """Register only if the exported generation is still current."""
        with self._lock:
            if generation != self._generation:
                raise RuntimeError("Native bank generation changed during setup")
            self._next_listener += 1
            self._native_listeners[self._next_listener] = invalidate
            return self._next_listener

    def unwatch_native_generation(self, token: int) -> None:
        with self._lock:
            self._native_listeners.pop(token, None)

    def _new_index(self):
        if self._backend == "hnsw":
            index = faiss.IndexHNSWFlat(
                self.key_dim, self._hnsw_m, faiss.METRIC_INNER_PRODUCT
            )
            index.hnsw.efConstruction = self._hnsw_ef_construction
            index.hnsw.efSearch = self._hnsw_ef_search
            return index
        return faiss.IndexFlatIP(self.key_dim)

    @property
    def index_backend(self) -> str:
        return self._backend

    def clear(self) -> None:
        self.rebuild([])

    def rebuild(self, records: list[TemplateRecord]) -> None:
        """Prepare a complete replacement before exposing it to concurrent queries."""
        keys: list[np.ndarray] = []
        values: list[np.ndarray] = []
        ids: list[str] = []
        expressions: list[str] = []
        seen: set[str] = set()
        for record in records:
            template_id, expression, key, value = validate_template_record(
                record.template_id,
                record.metta_expr,
                record.key_vector,
                record.val_vector,
                self.key_dim,
            )
            if template_id in seen:
                raise ValueError(f"Duplicate template id: {template_id}")
            seen.add(template_id)
            ids.append(template_id)
            expressions.append(expression)
            keys.append(key.copy())
            values.append(value.copy())

        replacement = self._new_index()
        if keys:
            normalized = np.ascontiguousarray(
                np.stack([key / np.linalg.norm(key) for key in keys]),
                dtype=np.float32,
            )
            replacement.add(normalized)

        with self._lock:
            self._invalidate_native()
            self._faiss_index = replacement
            self.key_store = keys
            self.val_store = values
            self.id_store = ids
            self.metta_store = expressions
            self._id_set = seen
            self._gather_arrays = None
            self._generation += 1

    def add_template(
        self,
        template_id: str,
        metta_expr: str,
        key_vector: np.ndarray,
        val_vector: np.ndarray,
    ) -> bool:
        template_id, metta_expr, key, value = validate_template_record(
            template_id, metta_expr, key_vector, val_vector, self.key_dim
        )
        normalized = np.ascontiguousarray(
            np.expand_dims(key / np.linalg.norm(key), axis=0), dtype=np.float32
        )
        with self._lock:
            if template_id in self._id_set:
                raise ValueError(f"Duplicate template id: {template_id}")
            self._invalidate_native()
            self._faiss_index.add(normalized)
            self.key_store.append(key.copy())
            self.val_store.append(value.copy())
            self.id_store.append(template_id)
            self.metta_store.append(metta_expr)
            self._id_set.add(template_id)
            self._gather_arrays = None
            self._generation += 1
        return True

    def numeric_binding(self) -> tuple[int, tuple[str, ...]]:
        """Bind an immutable ID ordering; supported mutations invalidate the generation."""
        with self._lock:
            if not self.id_store:
                raise ValueError("Native retrieval requires a nonempty bank")
            if self._gather_arrays is None:
                self._gather_arrays = (
                    np.stack(self.key_store),
                    np.stack(self.val_store),
                )
            return self._generation, tuple(self.id_store)

    def query_top_k_into(
        self,
        query_vectors: np.ndarray,
        top_m: int,
        packed: np.ndarray,
        *,
        expected_generation: int,
    ) -> float:
        """Fill contiguous float32 K/V, scores and bit-preserved int32 indices.

        Intended for reusable pinned host buffers. Return seconds spent on CPU
        validation/normalization, FAISS search and K/V gathering; metadata packing
        remains outside that interval. No IDs, vectors or labels are synthesized.
        A stale generation fails instead of reinterpreting a compiled ID registry.
        """
        if isinstance(top_m, bool) or not isinstance(top_m, int) or top_m <= 0:
            raise ValueError("top_m must be a positive integer")
        queries = np.asarray(query_vectors, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.key_dim or not len(queries):
            raise ValueError("Queries must have nonempty [count, key_dim] shape")
        with self._lock:
            if self._generation != expected_generation:
                raise RuntimeError(
                    "Native bank generation changed; create a new retriever"
                )
            count, matches = len(queries), min(top_m, len(self.id_store))
            vectors = count * matches * self.key_dim
            metadata = count * matches
            if (
                matches == 0
                or packed.dtype != np.float32
                or packed.shape != (2 * vectors + 2 * metadata,)
                or not packed.flags.c_contiguous
                or not packed.flags.writeable
            ):
                raise ValueError(
                    "Packed result must be writable contiguous float32 with the exact bank shape"
                )
            if np.shares_memory(queries, packed):
                raise ValueError("Query and output buffers must not overlap")
            if self._gather_arrays is None:
                self._gather_arrays = (
                    np.stack(self.key_store),
                    np.stack(self.val_store),
                )
            keys, values = self._gather_arrays
            selected_keys = packed[:vectors].reshape(count, matches, self.key_dim)
            selected_values = packed[vectors : 2 * vectors].reshape(
                count, matches, self.key_dim
            )
            started = time.perf_counter_ns()
            if not np.isfinite(queries).all():
                raise ValueError("Query vectors must contain only finite values")
            norms = np.linalg.norm(queries, axis=1, keepdims=True)
            normalized = np.ascontiguousarray(
                np.where(norms > 0, queries / (norms + 1e-8), queries),
                dtype=np.float32,
            )
            scores, labels = self._faiss_index.search(normalized, matches)
            if np.any((labels < 0) | (labels >= len(self.id_store))):
                raise RuntimeError("FAISS returned an invalid template label")
            np.take(keys, labels, axis=0, out=selected_keys, mode="clip")
            np.take(values, labels, axis=0, out=selected_values, mode="clip")
            elapsed = (time.perf_counter_ns() - started) / 1e9
            packed[2 * vectors : 2 * vectors + metadata] = scores.reshape(-1)
            packed.view(np.int32)[2 * vectors + metadata :] = labels.reshape(-1)
            return elapsed

    def contains_template(self, template_id: str) -> bool:
        with self._lock:
            return template_id.strip() in self._id_set

    def query_top_k(self, query_vectors: np.ndarray, top_m: int = 8) -> MorkQueryResult:
        if top_m <= 0:
            raise ValueError("top_m must be greater than zero")
        queries = np.asarray(query_vectors, dtype=np.float32)
        if queries.ndim == 1:
            queries = np.expand_dims(queries, axis=0)
        if queries.ndim != 2 or queries.shape[-1] != self.key_dim:
            raise ValueError(
                f"Query vector dim mismatch: expected (*, {self.key_dim}), got {queries.shape}"
            )
        if not np.isfinite(queries).all():
            raise ValueError("Query vectors must contain only finite values")

        num_queries = queries.shape[0]
        with self._lock:
            num_items = len(self.key_store)
            if num_items == 0:
                return MorkQueryResult(
                    keys=np.empty((num_queries, 0, self.key_dim), dtype=np.float32),
                    values=np.empty((num_queries, 0, self.key_dim), dtype=np.float32),
                    template_ids=[[] for _ in range(num_queries)],
                    scores=np.empty((num_queries, 0), dtype=np.float32),
                )

            k_actual = min(top_m, num_items)
            q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
            norm_q = np.ascontiguousarray(
                np.where(q_norms > 0, queries / (q_norms + 1e-8), queries),
                dtype=np.float32,
            )
            scores, labels = self._faiss_index.search(norm_q, k_actual)
            scores = np.atleast_2d(np.asarray(scores))
            labels = np.atleast_2d(np.asarray(labels))

            matched_keys: np.ndarray
            matched_vals: np.ndarray
            matched_scores: np.ndarray
            matched_ids: list[list[str]]
            if num_queries == 1:
                matched_keys = np.empty((1, k_actual, self.key_dim), dtype=np.float32)
                matched_vals = np.empty_like(matched_keys)
                matched_scores = np.empty((1, k_actual), dtype=np.float32)
                matched_ids = [[]]
                for j in range(k_actual):
                    item_idx = int(labels[0, j])
                    if item_idx < 0 or item_idx >= num_items:
                        raise RuntimeError("FAISS returned an invalid template label")
                    matched_keys[0, j] = self.key_store[item_idx]
                    matched_vals[0, j] = self.val_store[item_idx]
                    matched_scores[0, j] = scores[0, j]
                    matched_ids[0].append(self.id_store[item_idx])
            else:
                if np.any((labels < 0) | (labels >= num_items)):
                    raise RuntimeError("FAISS returned an invalid template label")
                if self._gather_arrays is None:
                    self._gather_arrays = (
                        np.stack(self.key_store),
                        np.stack(self.val_store),
                    )
                keys, values = self._gather_arrays
                matched_keys = keys[labels]
                matched_vals = values[labels]
                matched_scores = np.asarray(scores, dtype=np.float32).copy()
                matched_ids = [
                    [self.id_store[int(index)] for index in row] for row in labels
                ]

        return MorkQueryResult(
            keys=matched_keys,
            values=matched_vals,
            template_ids=matched_ids,
            scores=matched_scores,
        )
