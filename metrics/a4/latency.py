"""Paired A4 step, bridge and CUDA-copy timing statistics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import fmean
from typing import Any


def _seconds(row: dict[str, Any], name: str, *, positive: bool = False) -> float:
    value = row[name]
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
        )
    return float(value)


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(
    times: list[float], steps: list[float], prefix: str
) -> dict[str, Any]:
    fraction = 100 * fmean(times) / fmean(steps)
    return {
        f"{prefix}_mean_ms": 1000 * fmean(times),
        f"{prefix}_p95_ms": 1000 * _p95(times),
        f"{prefix}_ratio_of_means_percent": fraction,
        f"{prefix}_paired_ratio_p95_percent": _p95(
            [100 * time / step for time, step in zip(times, steps, strict=True)]
        ),
        f"{prefix}_mean_below_five_percent": fraction < 5,
    }


def summarize_latency(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize paired samples; bridge replay is not added model runtime."""
    if not rows:
        raise ValueError("At least one paired latency sample is required")
    steps, bridges, searches = [], [], []
    for row in rows:
        steps.append(_seconds(row, "step_seconds", positive=True))
        bridges.append(_seconds(row, "full_round_trip_seconds"))
        searches.append(_seconds(row, "cpu_index_seconds"))
        if searches[-1] > bridges[-1]:
            raise ValueError("CPU search cannot exceed the enclosing bridge interval")
    communication = [
        bridge - cpu for bridge, cpu in zip(bridges, searches, strict=True)
    ]
    return {
        "samples": len(rows),
        "step_mean_ms": 1000 * fmean(steps),
        "cpu_index_mean_ms": 1000 * fmean(searches),
        **_distribution(bridges, steps, "full_round_trip"),
        **_distribution(communication, steps, "communication"),
        "communication_definition": "Full bridge replay minus CPU normalization/search/gather; retains dispatch, metadata, synchronization and transfers",
        "denominator": "Paired synchronized complete A4 step, not a no-head baseline",
    }


def summarize_transfers(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize CUDA-event copy intervals with uninstrumented denominators."""
    if not rows:
        raise ValueError("At least one CUDA transfer sample is required")
    steps, d2h, h2d = [], [], []
    for row in rows:
        steps.append(_seconds(row, "unprofiled_step_seconds", positive=True))
        d2h.append(_seconds(row, "d2h_gpu_seconds"))
        h2d.append(_seconds(row, "h2d_gpu_seconds"))
        if d2h[-1] + h2d[-1] > _seconds(row, "profiled_model_seconds", positive=True):
            raise ValueError("CUDA copy intervals exceed the profiled model interval")
    copies = [left + right for left, right in zip(d2h, h2d, strict=True)]
    return {
        "samples": len(rows),
        "gpu_to_cpu_mean_ms": 1000 * fmean(d2h),
        "cpu_to_gpu_mean_ms": 1000 * fmean(h2d),
        **_distribution(copies, steps, "transfer_only"),
        "denominator": "Batch-matched uninstrumented A4 model steps from a separate pass",
        "excludes": "CPU search and work/waits outside the CUDA-event brackets",
        "limitations": "GPU-stream intervals, not physical PCIe bus measurements; can include scheduling gaps",
        "does_not_decide_bridge_overhead_gate": True,
    }


def interval_breakdown(
    seconds: float, counters: dict[str, float], return_mode: str = "sync"
) -> dict[str, float]:
    """Partition native host-wall phases without dropping outstanding work."""
    if return_mode not in {"sync", "event"}:
        raise ValueError("Unsupported native return mode")
    parts = {
        "gpu_to_cpu_enqueue_and_wait": counters["d2h_seconds"],
        "cpu_normalization_search_gather": counters["cpu_index_seconds"],
        "cpu_thread_setup_and_metadata": counters["host_search_seconds"]
        - counters["cpu_index_seconds"],
        "cpu_to_gpu_enqueue_only"
        if return_mode == "event"
        else "cpu_to_gpu_enqueue_and_wait": counters["h2d_seconds"],
        "outside_native_interval": seconds
        - counters["d2h_seconds"]
        - counters["host_search_seconds"]
        - counters["h2d_seconds"],
    }
    if not all(math.isfinite(value) and value >= 0 for value in parts.values()):
        raise ValueError("Invalid native timing decomposition")
    return parts
