"""JAX device memory monitoring with a configurable headroom safety rule."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax


@dataclass(frozen=True)
class MemoryStatus:
    available: bool
    total_bytes: int | None = None
    allocated_bytes: int | None = None
    available_bytes: int | None = None
    headroom_ratio: float | None = None
    device: str = "unknown"
    platform: str = "unknown"
    diagnostic: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def bytes_in_use(self) -> int | None:
        return self.allocated_bytes


def get_memory_status(device: jax.Device | None = None) -> MemoryStatus:
    device = device or jax.devices()[0]
    platform = jax.default_backend()
    stats_fn = getattr(device, "memory_stats", None)
    if stats_fn is None:
        return MemoryStatus(
            available=False,
            device=str(device),
            platform=platform,
            diagnostic=(
                "Per-device memory statistics are not available on this "
                f"platform (backend={platform!r})."
            ),
        )
    try:
        stats = stats_fn()
    except Exception as exc:  # pragma: no cover - platform dependent
        return MemoryStatus(
            available=False,
            device=str(device),
            platform=platform,
            diagnostic=f"memory_stats() raised: {exc}",
        )
    if not stats:
        return MemoryStatus(
            available=False,
            device=str(device),
            platform=platform,
            diagnostic=(
                f"memory_stats() returned no data on backend={platform!r}; "
                "device memory monitoring unsupported."
            ),
            raw=stats,
        )

    total = int(stats.get("bytes_limit") or 0)
    allocated = int(stats.get("bytes_in_use") or 0)
    available_bytes = max(0, total - allocated)
    headroom = (available_bytes / total) if total else None

    return MemoryStatus(
        available=True,
        total_bytes=total,
        allocated_bytes=allocated,
        available_bytes=available_bytes,
        headroom_ratio=headroom,
        device=str(device),
        platform=platform,
        diagnostic="ok",
        raw=stats,
    )


def compute_memory_headroom(status: MemoryStatus) -> float | None:
    if not status.available:
        return None
    if status.total_bytes is None or status.available_bytes is None:
        return None
    if status.total_bytes == 0:
        return None
    return status.available_bytes / status.total_bytes


def check_memory_headroom(status: MemoryStatus, min_headroom: float = 0.5) -> list[str]:
    headroom = compute_memory_headroom(status)
    if headroom is None:
        return [
            "WARNING: cannot verify device memory headroom "
            f"({status.diagnostic or 'unsupported platform'})."
        ]
    if headroom < min_headroom:
        return [
            f"WARNING: GPU memory headroom is below {min_headroom:.0%} "
            f"(headroom={headroom:.1%}). "
            "Consider reducing batch size or sequence length."
        ]
    return []


def maybe_reduce_batch_size(
    status: MemoryStatus,
    batch_size: int,
    min_headroom: float = 0.5,
    auto_reduce: bool = False,
) -> tuple[int, bool, list[str]]:
    warnings = check_memory_headroom(status, min_headroom)
    if not warnings:
        return batch_size, False, []

    headroom = compute_memory_headroom(status)
    if headroom is None:
        # Memory stats are unavailable: we cannot verify the rule, so the
        # configuration is left untouched and only the diagnostic warning is
        # reported.
        return batch_size, False, warnings

    if not auto_reduce:
        return batch_size, False, warnings

    reduced = batch_size
    while reduced > 1 and headroom < min_headroom:
        reduced //= 2
        # Conservative linear rescaling estimate; exact measurement happens
        # after re-execution via get_memory_status().
        headroom = min(1.0, headroom * (batch_size / max(1, reduced)))

    if reduced != batch_size:
        warnings.append(
            f"NOTICE: batch size automatically reduced from {batch_size} to "
            f"{reduced} to satisfy the {min_headroom:.0%} headroom rule."
        )
    return reduced, reduced != batch_size, warnings
