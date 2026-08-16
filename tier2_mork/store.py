"""
MORK Template Hypergraph Store for Tier 2 Sparse Symbolic Engine.
Hosts Atomese/MeTTa templates, key/value embeddings (p_j, v_j) in CPU DRAM.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np

from tier2_mork.index import GenericHNSWIndex

logger = logging.getLogger(__name__)


@dataclass
class TemplateRecord:
    """Represents a single Atomese/MeTTa hypergraph template in MORK."""

    template_id: int
    metta_ast: str
    key_embedding: np.ndarray  # p_j in R^k
    value_embedding: np.ndarray  # v_j in R^k
    category: str = "general"
    fire_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MORKTemplateStore:
    """
    CPU DRAM MORK Store combining HNSW index search with template record storage.
    """

    def __init__(
        self,
        dim: int = 256,
        space: str = "cosine",
        max_capacity: int = 100000,
        ef_construction: int = 200,
        M: int = 16,
    ) -> None:
        self.dim = dim
        self.hnsw_index = GenericHNSWIndex(
            dim=dim,
            space=space,
            max_elements=max_capacity,
            ef_construction=ef_construction,
            M=M,
        )
        self.records: Dict[int, TemplateRecord] = {}

    def insert_template(self, record: TemplateRecord) -> None:
        """Store a single template record and update HNSW index."""
        record.key_embedding = np.asarray(record.key_embedding, dtype=np.float32)
        record.value_embedding = np.asarray(record.value_embedding, dtype=np.float32)

        if record.key_embedding.shape[0] != self.dim:
            raise ValueError(
                f"Key dimension mismatch: expected {self.dim}, got {record.key_embedding.shape[0]}"
            )

        self.records[record.template_id] = record
        self.hnsw_index.add_items(
            keys=record.key_embedding,
            ids=[record.template_id],
            metadata=[{"metta_ast": record.metta_ast, "category": record.category}],
        )

    def insert_batch(self, records: List[TemplateRecord]) -> None:
        """Batch insert templates into store and HNSW index."""
        if not records:
            return

        keys = np.vstack([r.key_embedding for r in records]).astype(np.float32)
        ids = [r.template_id for r in records]
        metadata = [{"metta_ast": r.metta_ast, "category": r.category} for r in records]

        for r in records:
            self.records[r.template_id] = r

        self.hnsw_index.add_items(keys=keys, ids=ids, metadata=metadata)

    def retrieve_top_m(
        self, query_vector: np.ndarray, top_m: int = 8
    ) -> Tuple[List[TemplateRecord], np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform fast top-m template retrieval for a given query vector q_sym.

        :param query_vector: Projected query vector q_sym in R^k.
        :param top_m: Number of matching templates to return.
        :return: Tuple of (matched_records, distances, key_matrix, value_matrix).
                 key_matrix has shape (top_m, k), value_matrix has shape (top_m, k).
        """
        labels, distances = self.hnsw_index.query(query_vector, top_m=top_m)
        matched_ids = labels[0]
        matched_distances = distances[0]

        matched_records = []
        matched_keys = []
        matched_values = []

        for tid in matched_ids:
            record = self.records.get(int(tid))
            if record is not None:
                record.fire_count += 1
                matched_records.append(record)
                matched_keys.append(record.key_embedding)
                matched_values.append(record.value_embedding)

        key_matrix = np.vstack(matched_keys).astype(np.float32) if matched_keys else np.zeros((0, self.dim), dtype=np.float32)
        value_matrix = np.vstack(matched_values).astype(np.float32) if matched_values else np.zeros((0, self.dim), dtype=np.float32)

        return matched_records, matched_distances, key_matrix, value_matrix

    def size(self) -> int:
        """Return total template count in store."""
        return len(self.records)
