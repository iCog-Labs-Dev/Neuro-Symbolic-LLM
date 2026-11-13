"""Pure paired A4 step, bridge, and CUDA-copy timing statistics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import fmean


def _checked_rows(rows: object, *, name: str) -> tuple[Mapping[str, object], ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError(f"{name} must be a sequence of mappings")
    if not rows:
        raise ValueError(f"At least one {name} is required")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"Each {name} must be a mapping")
    return tuple(rows)  # type: ignore[return-value]


def _checked_seconds(value: object, name: str, *, positive: bool = False) -> float:
    bound = "positive" if positive else "nonnegative"
    message = f"{name} must be a finite {bound} number of seconds"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(message)
    try:
        seconds = float(value)
    except OverflowError:
        raise ValueError(message) from None
    if not math.isfinite(seconds) or seconds < 0 or (positive and seconds == 0):
        raise ValueError(message)
    return seconds


def _seconds(row: Mapping[str, object], name: str, *, positive: bool = False) -> float:
    if name not in row:
        raise ValueError(f"Timing sample is missing required field {name!r}")
    return _checked_seconds(row[name], name, positive=positive)


def _p95(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("A percentile requires at least one value")
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(
    times: list[float], steps: list[float], prefix: str
) -> dict[str, object]:
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


def summarize_latency(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize paired step and bridge samples.

    Full bridge replay is not added model runtime. Ratio-of-means and the p95 of
    paired ratios are intentionally reported separately because they answer
    different questions.
    """
    records = _checked_rows(rows, name="paired latency sample")
    steps: list[float] = []
    bridges: list[float] = []
    searches: list[float] = []
    for row in records:
        steps.append(_seconds(row, "step_seconds", positive=True))
        bridges.append(_seconds(row, "full_round_trip_seconds"))
        searches.append(_seconds(row, "cpu_index_seconds"))
        if searches[-1] > bridges[-1]:
            raise ValueError("CPU search cannot exceed the enclosing bridge interval")

    communication = [
        bridge - cpu for bridge, cpu in zip(bridges, searches, strict=True)
    ]
    return {
        "samples": len(records),
        "step_mean_ms": 1000 * fmean(steps),
        "cpu_index_mean_ms": 1000 * fmean(searches),
        **_distribution(bridges, steps, "full_round_trip"),
        **_distribution(communication, steps, "communication"),
        "communication_definition": "Full bridge replay minus CPU normalization/search/gather; retains dispatch, metadata, synchronization and transfers",
        "denominator": "Paired synchronized complete A4 step, not a no-head baseline",
    }


def summarize_transfers(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize CUDA-event copy intervals with uninstrumented denominators."""
    records = _checked_rows(rows, name="CUDA transfer sample")
    steps: list[float] = []
    device_to_host: list[float] = []
    host_to_device: list[float] = []
    for row in records:
        steps.append(_seconds(row, "unprofiled_step_seconds", positive=True))
        device_to_host.append(_seconds(row, "d2h_gpu_seconds"))
        host_to_device.append(_seconds(row, "h2d_gpu_seconds"))
        profiled = _seconds(row, "profiled_model_seconds", positive=True)
        if device_to_host[-1] + host_to_device[-1] > profiled:
            raise ValueError("CUDA copy intervals exceed the profiled model interval")

    copies = [
        left + right for left, right in zip(device_to_host, host_to_device, strict=True)
    ]
    return {
        "samples": len(records),
        "gpu_to_cpu_mean_ms": 1000 * fmean(device_to_host),
        "cpu_to_gpu_mean_ms": 1000 * fmean(host_to_device),
        **_distribution(copies, steps, "transfer_only"),
        "denominator": "Batch-matched uninstrumented A4 model steps from a separate pass",
        "excludes": "CPU search and work/waits outside the CUDA-event brackets",
        "limitations": "GPU-stream intervals, not physical PCIe bus measurements; can include scheduling gaps",
        "does_not_decide_bridge_overhead_gate": True,
    }


def interval_breakdown(
    seconds: float, counters: Mapping[str, object], return_mode: str = "sync"
) -> dict[str, float]:
    """Partition native host-wall phases without dropping outstanding work."""
    if not isinstance(return_mode, str) or return_mode not in {"sync", "event"}:
        raise ValueError("Unsupported native return mode")
    duration = _checked_seconds(seconds, "seconds")
    if not isinstance(counters, Mapping):
        raise ValueError("counters must be a mapping")

    device_to_host = _seconds(counters, "d2h_seconds")
    cpu_index = _seconds(counters, "cpu_index_seconds")
    host_search = _seconds(counters, "host_search_seconds")
    host_to_device = _seconds(counters, "h2d_seconds")
    parts = {
        "gpu_to_cpu_enqueue_and_wait": device_to_host,
        "cpu_normalization_search_gather": cpu_index,
        "cpu_thread_setup_and_metadata": host_search - cpu_index,
        (
            "cpu_to_gpu_enqueue_only"
            if return_mode == "event"
            else "cpu_to_gpu_enqueue_and_wait"
        ): host_to_device,
        "outside_native_interval": (
            duration - device_to_host - host_search - host_to_device
        ),
    }
    if not all(math.isfinite(value) and value >= 0 for value in parts.values()):
        raise ValueError("Invalid native timing decomposition")
    return parts


__all__ = ["interval_breakdown", "summarize_latency", "summarize_transfers"]
