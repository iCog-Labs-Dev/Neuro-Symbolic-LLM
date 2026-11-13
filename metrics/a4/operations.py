"""Pure operational metrics for future A4 consolidation and template banks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from statistics import fmean
from typing import Any


def _records(value: object, name: str) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of mappings")
    if any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"each {name} record must be a mapping")
    return tuple(value)  # type: ignore[return-value]


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _optional_timestamp(value: object, name: str) -> float | None:
    return None if value is None else _finite_number(value, name)


def _optional_bool(value: object, name: str) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean or None")
    return value


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _linear_slope(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    if len(x_values) < 2:
        return None
    x_mean = fmean(x_values)
    y_mean = fmean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0:
        return None
    return (
        sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x_values, y_values, strict=True)
        )
        / denominator
    )


def consolidation_queue_metrics(
    events: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    """Summarize asynchronous consolidation job lifecycles and queue depth.

    Every event has ``enqueue``, optional ``start`` and ``finish`` timestamps,
    optional boolean ``accepted`` and ``injected`` decisions, and a nonnegative
    integer queue ``depth`` sampled at enqueue time.  ``None`` represents a
    lifecycle stage or decision that has not happened yet.  An injected result
    must have passed the acceptance gate.
    """
    rows = _records(events, "event")
    if not rows:
        return {
            "status": "unavailable",
            "reason": "no consolidation events",
            "event_count": 0,
        }

    normalized = []
    required = ("enqueue", "start", "finish", "injected", "accepted", "depth")
    for index, row in enumerate(rows):
        missing = [name for name in required if name not in row]
        if missing:
            raise ValueError(
                f"event {index} is missing required fields: {', '.join(missing)}"
            )
        enqueue = _finite_number(row["enqueue"], f"event {index} enqueue")
        start = _optional_timestamp(row["start"], f"event {index} start")
        finish = _optional_timestamp(row["finish"], f"event {index} finish")
        injected = _optional_bool(row["injected"], f"event {index} injected")
        accepted = _optional_bool(row["accepted"], f"event {index} accepted")
        depth = _nonnegative_integer(row["depth"], f"event {index} depth")
        if start is not None and start < enqueue:
            raise ValueError("event start cannot precede enqueue")
        if finish is not None and start is None:
            raise ValueError("event finish requires a start timestamp")
        if finish is not None and finish < start:
            raise ValueError("event finish cannot precede start")
        if injected is True and accepted is not True:
            raise ValueError("an injected event must be accepted")
        normalized.append(
            {
                "enqueue": enqueue,
                "start": start,
                "finish": finish,
                "injected": injected,
                "accepted": accepted,
                "depth": depth,
            }
        )

    started = [row for row in normalized if row["start"] is not None]
    finished = [row for row in normalized if row["finish"] is not None]
    waiting = [row for row in normalized if row["start"] is None]
    in_progress = [
        row for row in normalized if row["start"] is not None and row["finish"] is None
    ]
    waits = [float(row["start"] - row["enqueue"]) for row in started]
    processing = [float(row["finish"] - row["start"]) for row in finished]
    end_to_end = [float(row["finish"] - row["enqueue"]) for row in finished]

    acceptance_decisions = [
        row["accepted"] for row in normalized if row["accepted"] is not None
    ]
    injection_decisions = [
        row["injected"] for row in normalized if row["injected"] is not None
    ]
    accepted_count = sum(value is True for value in acceptance_decisions)
    injected_count = sum(value is True for value in injection_decisions)

    enqueue_times = [float(row["enqueue"]) for row in normalized]
    depths = [int(row["depth"]) for row in normalized]
    depth_slope = _linear_slope(enqueue_times, depths)
    latest_index = max(
        range(len(normalized)), key=lambda index: (enqueue_times[index], index)
    )
    observation_start = min(enqueue_times)
    observation_end = max(
        float(
            row["finish"]
            if row["finish"] is not None
            else row["start"] if row["start"] is not None else row["enqueue"]
        )
        for row in normalized
    )
    observation_seconds = observation_end - observation_start
    return {
        "status": "ok",
        "reason": None,
        "event_count": len(normalized),
        "started_count": len(started),
        "finished_count": len(finished),
        "waiting_count": len(waiting),
        "in_progress_count": len(in_progress),
        "pending_count": len(normalized) - len(finished),
        "completion_rate": len(finished) / len(normalized),
        "acceptance_decision_count": len(acceptance_decisions),
        "accepted_count": accepted_count,
        "acceptance_rate": (
            accepted_count / len(acceptance_decisions) if acceptance_decisions else None
        ),
        "injection_decision_count": len(injection_decisions),
        "injected_count": injected_count,
        "injection_rate": (
            injected_count / len(injection_decisions) if injection_decisions else None
        ),
        "injection_rate_per_finished_event": (
            injected_count / len(finished) if finished else None
        ),
        "wait_mean_seconds": fmean(waits) if waits else None,
        "wait_p95_seconds": _percentile(waits, 0.95),
        "processing_mean_seconds": fmean(processing) if processing else None,
        "processing_p95_seconds": _percentile(processing, 0.95),
        "end_to_end_mean_seconds": fmean(end_to_end) if end_to_end else None,
        "end_to_end_p95_seconds": _percentile(end_to_end, 0.95),
        "queue_depth_latest": depths[latest_index],
        "queue_depth_mean": fmean(depths),
        "queue_depth_max": max(depths),
        "queue_depth_slope_per_second": depth_slope,
        "net_drain_rate_per_second": -depth_slope if depth_slope is not None else None,
        "depth_trend_status": "ok" if depth_slope is not None else "unavailable",
        "observation_seconds": observation_seconds,
        "finish_throughput_per_second": (
            len(finished) / observation_seconds if observation_seconds > 0 else None
        ),
        "injection_throughput_per_second": (
            injected_count / observation_seconds if observation_seconds > 0 else None
        ),
    }


def bank_growth_stability(
    snapshots: Sequence[Mapping[str, object]], *, tail_window: int = 5
) -> dict[str, Any]:
    """Measure template-bank growth and exact stabilization over recent snapshots.

    Snapshots must be ordered by strictly increasing ``timestamp`` and ``step``
    and contain a nonnegative integer ``count``.  The effective tail is the last
    ``min(tail_window, len(snapshots))`` records.  It is stable only when every
    count in that tail is equal.
    """
    if (
        isinstance(tail_window, bool)
        or not isinstance(tail_window, Integral)
        or tail_window < 2
    ):
        raise ValueError("tail_window must be an integer of at least two snapshots")
    tail_window = int(tail_window)
    rows = _records(snapshots, "snapshot")
    if not rows:
        return {
            "status": "unavailable",
            "reason": "no bank snapshots",
            "snapshot_count": 0,
            "tail_window_requested": tail_window,
        }

    timestamps = []
    steps = []
    counts = []
    for index, row in enumerate(rows):
        missing = [name for name in ("timestamp", "step", "count") if name not in row]
        if missing:
            raise ValueError(
                f"snapshot {index} is missing required fields: {', '.join(missing)}"
            )
        timestamps.append(
            _finite_number(row["timestamp"], f"snapshot {index} timestamp")
        )
        steps.append(_nonnegative_integer(row["step"], f"snapshot {index} step"))
        counts.append(_nonnegative_integer(row["count"], f"snapshot {index} count"))
    if any(
        right <= left for left, right in zip(timestamps, timestamps[1:], strict=False)
    ):
        raise ValueError("snapshot timestamps must be strictly increasing")
    if any(right <= left for left, right in zip(steps, steps[1:], strict=False)):
        raise ValueError("snapshot steps must be strictly increasing")

    used = min(tail_window, len(rows))
    base = {
        "snapshot_count": len(rows),
        "tail_window_requested": tail_window,
        "tail_window_used": used,
        "tail_window_complete": len(rows) >= tail_window,
        "latest_timestamp": timestamps[-1],
        "latest_step": steps[-1],
        "latest_count": counts[-1],
        "minimum_count": min(counts),
        "maximum_count": max(counts),
    }
    if len(rows) == 1:
        return {
            "status": "unavailable",
            "reason": "at least two bank snapshots are required",
            **base,
            "stable": None,
        }

    deltas = [right - left for left, right in zip(counts, counts[1:], strict=False)]
    tail_timestamps = timestamps[-used:]
    tail_steps = steps[-used:]
    tail_counts = counts[-used:]
    tail_deltas = [
        right - left for left, right in zip(tail_counts, tail_counts[1:], strict=False)
    ]
    duration = timestamps[-1] - timestamps[0]
    step_span = steps[-1] - steps[0]
    tail_duration = tail_timestamps[-1] - tail_timestamps[0]
    tail_step_span = tail_steps[-1] - tail_steps[0]
    tail_growth = tail_counts[-1] - tail_counts[0]
    return {
        "status": "ok",
        "reason": None,
        **base,
        "total_growth": counts[-1] - counts[0],
        "gross_additions": sum(max(delta, 0) for delta in deltas),
        "gross_removals": sum(max(-delta, 0) for delta in deltas),
        "overall_growth_per_second": (counts[-1] - counts[0]) / duration,
        "overall_growth_per_step": (counts[-1] - counts[0]) / step_span,
        "overall_count_slope_per_second": _linear_slope(timestamps, counts),
        "tail_growth": tail_growth,
        "tail_gross_additions": sum(max(delta, 0) for delta in tail_deltas),
        "tail_gross_removals": sum(max(-delta, 0) for delta in tail_deltas),
        "tail_growth_per_second": tail_growth / tail_duration,
        "tail_growth_per_step": tail_growth / tail_step_span,
        "tail_count_slope_per_second": _linear_slope(tail_timestamps, tail_counts),
        "tail_unchanged_interval_fraction": sum(delta == 0 for delta in tail_deltas)
        / len(tail_deltas),
        "stable": all(delta == 0 for delta in tail_deltas),
        "stability_definition": "no template-count change within the effective tail window",
    }


__all__ = ["bank_growth_stability", "consolidation_queue_metrics"]
