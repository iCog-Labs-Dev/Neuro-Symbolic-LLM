"""Pure calculation tests for future A4 operational metrics."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import metrics.a4 as a4
import metrics.a4.latency as latency
import metrics.a4.operations as operations
import metrics.a4.representation as representation
import metrics.a4.retrieval as retrieval
from metrics.a4.operations import bank_growth_stability, consolidation_queue_metrics


def test_operations_module_has_no_framework_or_stage_imports() -> None:
    tree = ast.parse(Path(operations.__file__).read_text(encoding="utf-8"))
    imports = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    imports.update(
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert imports <= {
        "__future__",
        "collections",
        "math",
        "numbers",
        "statistics",
        "typing",
    }


def test_a4_package_exports_every_public_metric_api() -> None:
    expected = {
        *latency.__all__,
        *operations.__all__,
        *representation.__all__,
        *retrieval.__all__,
    }
    assert set(a4.__all__) == expected
    assert all(getattr(a4, name) is not None for name in expected)


def test_consolidation_queue_metrics_tracks_partial_lifecycles() -> None:
    result = consolidation_queue_metrics(
        [
            {
                "enqueue": 0,
                "start": 1,
                "finish": 3,
                "accepted": True,
                "injected": True,
                "depth": 4,
            },
            {
                "enqueue": 2,
                "start": 4,
                "finish": 8,
                "accepted": True,
                "injected": False,
                "depth": 3,
            },
            {
                "enqueue": 4,
                "start": None,
                "finish": None,
                "accepted": None,
                "injected": None,
                "depth": 2,
            },
        ]
    )
    assert result["status"] == "ok"
    assert result["event_count"] == 3
    assert result["finished_count"] == 2
    assert result["waiting_count"] == 1
    assert result["pending_count"] == 1
    assert result["completion_rate"] == pytest.approx(2 / 3)
    assert result["acceptance_rate"] == pytest.approx(1)
    assert result["injection_rate"] == pytest.approx(0.5)
    assert result["wait_mean_seconds"] == pytest.approx(1.5)
    assert result["wait_p95_seconds"] == pytest.approx(1.95)
    assert result["processing_mean_seconds"] == pytest.approx(3)
    assert result["processing_p95_seconds"] == pytest.approx(3.9)
    assert result["end_to_end_mean_seconds"] == pytest.approx(4.5)
    assert result["queue_depth_latest"] == 2
    assert result["queue_depth_slope_per_second"] == pytest.approx(-0.5)
    assert result["net_drain_rate_per_second"] == pytest.approx(0.5)
    assert result["observation_seconds"] == pytest.approx(8)


def test_consolidation_queue_metrics_reports_no_observations() -> None:
    result = consolidation_queue_metrics([])
    assert result == {
        "status": "unavailable",
        "reason": "no consolidation events",
        "event_count": 0,
    }


@pytest.mark.parametrize(
    "event",
    [
        {},
        {
            "enqueue": 2,
            "start": 1,
            "finish": None,
            "accepted": None,
            "injected": None,
            "depth": 0,
        },
        {
            "enqueue": 1,
            "start": None,
            "finish": 2,
            "accepted": True,
            "injected": False,
            "depth": 0,
        },
        {
            "enqueue": 1,
            "start": 2,
            "finish": 3,
            "accepted": False,
            "injected": True,
            "depth": 0,
        },
        {
            "enqueue": 1,
            "start": 2,
            "finish": 3,
            "accepted": 1,
            "injected": False,
            "depth": 0,
        },
        {
            "enqueue": 1,
            "start": 2,
            "finish": 3,
            "accepted": True,
            "injected": False,
            "depth": True,
        },
    ],
)
def test_consolidation_queue_metrics_strictly_validates_events(event) -> None:
    with pytest.raises(ValueError):
        consolidation_queue_metrics([event])


def test_bank_growth_stability_detects_a_flat_tail() -> None:
    result = bank_growth_stability(
        [
            {"timestamp": 0, "step": 0, "count": 10},
            {"timestamp": 1, "step": 10, "count": 12},
            {"timestamp": 2, "step": 20, "count": 12},
            {"timestamp": 3, "step": 30, "count": 12},
        ],
        tail_window=3,
    )
    assert result["status"] == "ok"
    assert result["total_growth"] == 2
    assert result["overall_growth_per_second"] == pytest.approx(2 / 3)
    assert result["overall_growth_per_step"] == pytest.approx(2 / 30)
    assert result["tail_window_used"] == 3
    assert result["tail_growth"] == 0
    assert result["tail_unchanged_interval_fraction"] == pytest.approx(1)
    assert result["stable"] is True


def test_bank_growth_stability_retains_additions_and_removals() -> None:
    result = bank_growth_stability(
        [
            {"timestamp": 0, "step": 0, "count": 0},
            {"timestamp": 2, "step": 2, "count": 2},
            {"timestamp": 4, "step": 4, "count": 1},
        ],
        tail_window=2,
    )
    assert result["gross_additions"] == 2
    assert result["gross_removals"] == 1
    assert result["tail_growth"] == -1
    assert result["tail_growth_per_second"] == pytest.approx(-0.5)
    assert result["stable"] is False


def test_bank_growth_stability_reports_insufficient_history() -> None:
    result = bank_growth_stability([{"timestamp": 1, "step": 0, "count": 4}])
    assert result["status"] == "unavailable"
    assert result["latest_count"] == 4
    assert result["stable"] is None
    assert bank_growth_stability([])["status"] == "unavailable"


@pytest.mark.parametrize(
    ("snapshots", "tail_window"),
    [
        ([{"timestamp": 0, "step": 0}], 2),
        (
            [
                {"timestamp": 1, "step": 0, "count": 1},
                {"timestamp": 1, "step": 1, "count": 2},
            ],
            2,
        ),
        (
            [
                {"timestamp": 1, "step": 1, "count": 1},
                {"timestamp": 2, "step": 0, "count": 2},
            ],
            2,
        ),
        ([{"timestamp": 1, "step": 0, "count": -1}], 2),
        ([], 1),
        ([], True),
    ],
)
def test_bank_growth_stability_strictly_validates_snapshots(
    snapshots, tail_window
) -> None:
    with pytest.raises(ValueError):
        bank_growth_stability(snapshots, tail_window=tail_window)
