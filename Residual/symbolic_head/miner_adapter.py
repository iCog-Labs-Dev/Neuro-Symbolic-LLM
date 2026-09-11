"""Adapter between hybrid-miner payloads and the Tier 2 store contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from Residual.symbolic_head.mork_client import MorkClient, TemplateRecord


class MinerTemplatePayload(Protocol):
    template_id: str
    sexpr: str
    key: Sequence[float]
    value: Sequence[float]


@dataclass(frozen=True)
class PublishedTemplate:
    """Validated payload ready for MORK publication and local indexing."""

    record: TemplateRecord


def adapt_miner_payload(
    payload: MinerTemplatePayload,
    *,
    key_dim: int,
) -> PublishedTemplate:
    """Validate a hybrid-miner payload without importing its package."""
    if not payload.template_id.strip():
        raise ValueError("template_id cannot be empty")
    if not payload.sexpr.strip():
        raise ValueError("sexpr cannot be empty")

    key_vector = np.asarray(payload.key, dtype=np.float32)
    val_vector = np.asarray(payload.value, dtype=np.float32)
    if key_vector.shape != (key_dim,) or val_vector.shape != (key_dim,):
        raise ValueError(f"Miner vectors must both have shape ({key_dim},)")
    if not np.isfinite(key_vector).all() or not np.isfinite(val_vector).all():
        raise ValueError("Miner vectors must contain only finite values")

    return PublishedTemplate(
        record=TemplateRecord(
            template_id=payload.template_id,
            metta_expr=payload.sexpr,
            key_vector=key_vector,
            val_vector=val_vector,
        )
    )


def publish_miner_payload(
    client: MorkClient,
    payload: MinerTemplatePayload,
    *,
    key_dim: int,
) -> PublishedTemplate:
    """Publish a validated miner payload through the Tier 2 client."""
    published = adapt_miner_payload(payload, key_dim=key_dim)
    record = published.record
    client.add_template(
        record.template_id,
        record.metta_expr,
        record.key_vector,
        record.val_vector,
    )
    return published