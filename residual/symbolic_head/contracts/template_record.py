"""Template data and retrieval result contracts for the Tier 2 CPU index."""

from dataclasses import dataclass

import numpy as np


@dataclass
class MorkQueryResult:
    """Real matches only; empty or short results never contain synthetic rows."""

    keys: np.ndarray
    values: np.ndarray
    template_ids: list[list[str]]
    scores: np.ndarray


@dataclass(frozen=True)
class TemplateRecord:
    """Authoritative MORK record used to rebuild a derived retrieval index."""

    template_id: str
    metta_expr: str
    key_vector: np.ndarray
    val_vector: np.ndarray


def validate_template_record(
    template_id: str,
    metta_expr: str,
    key_vector: np.ndarray,
    val_vector: np.ndarray,
    key_dim: int,
) -> tuple[str, str, np.ndarray, np.ndarray]:
    """Validate and convert a template before persistence or indexing."""
    template_id = template_id.strip()
    metta_expr = metta_expr.strip()
    key_vec = np.asarray(key_vector, dtype=np.float32)
    val_vec = np.asarray(val_vector, dtype=np.float32)

    if not template_id:
        raise ValueError("Template id cannot be empty")
    if not metta_expr:
        raise ValueError("MeTTa expression cannot be empty")
    if key_vec.shape != (key_dim,):
        raise ValueError(
            f"Key vector dim mismatch: expected ({key_dim},), got {key_vec.shape}"
        )
    if val_vec.shape != (key_dim,):
        raise ValueError(
            f"Value vector dim mismatch: expected ({key_dim},), got {val_vec.shape}"
        )
    if not np.isfinite(key_vec).all() or not np.isfinite(val_vec).all():
        raise ValueError("Template vectors must contain only finite values")
    if np.linalg.norm(key_vec) == 0:
        raise ValueError("Template key vector must have non-zero norm")
    return template_id, metta_expr, key_vec, val_vec
