"""Rebuildable CPU FAISS index derived from authoritative MORK records."""

import threading

import numpy as np

from Residual.symbolic_head.contracts.template_record import (
    MorkQueryResult,
    TemplateRecord,
    validate_template_record,
)

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
        self._faiss_index = self._new_index()

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
            keys.append(key)
            values.append(value)

        replacement = self._new_index()
        if keys:
            normalized = np.ascontiguousarray(
                np.stack([key / np.linalg.norm(key) for key in keys]),
                dtype=np.float32,
            )
            replacement.add(normalized)

        with self._lock:
            self._faiss_index = replacement
            self.key_store = keys
            self.val_store = values
            self.id_store = ids
            self.metta_store = expressions
            self._id_set = seen

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
            self._faiss_index.add(normalized)
            self.key_store.append(key)
            self.val_store.append(value)
            self.id_store.append(template_id)
            self.metta_store.append(metta_expr)
            self._id_set.add(template_id)
        return True

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

            matched_keys = np.empty(
                (num_queries, k_actual, self.key_dim), dtype=np.float32
            )
            matched_vals = np.empty(
                (num_queries, k_actual, self.key_dim), dtype=np.float32
            )
            matched_scores = np.empty((num_queries, k_actual), dtype=np.float32)
            matched_ids: list[list[str]] = [[] for _ in range(num_queries)]
            for i in range(num_queries):
                for j in range(k_actual):
                    item_idx = int(labels[i, j])
                    if item_idx < 0 or item_idx >= num_items:
                        raise RuntimeError("FAISS returned an invalid template label")
                    matched_keys[i, j] = self.key_store[item_idx]
                    matched_vals[i, j] = self.val_store[item_idx]
                    matched_scores[i, j] = scores[i, j]
                    matched_ids[i].append(self.id_store[item_idx])

        return MorkQueryResult(
            keys=matched_keys,
            values=matched_vals,
            template_ids=matched_ids,
            scores=matched_scores,
        )
