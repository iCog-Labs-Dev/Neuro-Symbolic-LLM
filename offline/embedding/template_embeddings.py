"""Initialize and persist template-associated key/value vectors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .context_builder import TemplateContexts

DEFAULT_SYMBOLIC_DIM = 256
DEFAULT_PROJECTION_SEED = 42


class TextEncoder(Protocol):
    model_name: str
    pooling: str
    layer_idx: int

    def encode(self, texts: tuple[str, ...]) -> np.ndarray: ...


def create_random_projection(
    hidden_dim: int,
    symbolic_dim: int = DEFAULT_SYMBOLIC_DIM,
    seed: int = DEFAULT_PROJECTION_SEED,
) -> np.ndarray:
    """Create a deterministic random projection with orthonormal rows."""
    if hidden_dim < 1 or symbolic_dim < 1:
        raise ValueError("projection dimensions must be positive")

    if symbolic_dim > hidden_dim:
        raise ValueError("symbolic_dim cannot exceed hidden_dim")

    rng = np.random.default_rng(seed)

    random_matrix = rng.standard_normal((hidden_dim, symbolic_dim))

    q, _ = np.linalg.qr(random_matrix, mode="reduced")

    return q.T.astype(np.float32)


def initialize_template_kv(
    template: TemplateContexts,
    encoder: TextEncoder,
    symbolic_dim: int = DEFAULT_SYMBOLIC_DIM,
    projection_seed: int = DEFAULT_PROJECTION_SEED,
) -> dict:
    """Average context embeddings and project into symbolic K/V space."""
    embeddings = np.asarray(
        encoder.encode(template.contexts),
        dtype=np.float32,
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(template.contexts):
        raise ValueError("encoder must return [num_contexts, hidden_dim]")

    if embeddings.shape[1] == 0 or not np.isfinite(embeddings).all():
        raise ValueError("encoder returned invalid embeddings")

    average = embeddings.mean(axis=0)
    hidden_dim = int(average.shape[0])

    projection = create_random_projection(
        hidden_dim=hidden_dim,
        symbolic_dim=symbolic_dim,
        seed=projection_seed,
    )

    projected = projection @ average

    if not np.isfinite(projected).all():
        raise ValueError("projected embedding contains nonfinite values")

    vector = projected.astype(np.float32).tolist()

    projection_sha256 = hashlib.sha256(projection.tobytes()).hexdigest()

    return {
        "pattern_id": template.pattern_id,
        "key": vector,
        "value": vector.copy(),
        "num_contexts": len(template.contexts),
        "embedding_dim": len(vector),
        "encoder_hidden_dim": hidden_dim,
        "encoder_model": encoder.model_name,
        "pooling": encoder.pooling,
        "intercept_layer": encoder.layer_idx,
        "encoder_backend": "frozenllm.FrozenSubstrate",
        "projection_method": "random_orthonormal",
        "projection_seed": projection_seed,
        "projection_shape": [symbolic_dim, hidden_dim],
        "projection_sha256": projection_sha256,
        "initialization": "random_projected_mean_context_embeddings_for_key_and_value",
    }


def save_template_embeddings(
    records: list[dict],
    output_path: str | Path,
) -> None:
    """Write one template-associated K/V record per JSONL line."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, allow_nan=False) + "\n")
