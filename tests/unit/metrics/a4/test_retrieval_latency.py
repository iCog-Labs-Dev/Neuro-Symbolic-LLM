"""Pure unit tests for A4 retrieval, event, and latency metrics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from metrics.a4.latency import (
    interval_breakdown,
    summarize_latency,
    summarize_transfers,
)
from metrics.a4.retrieval import (
    RetrievalMetrics,
    retrieval_activation_rate,
    retrieval_metrics,
    template_coverage,
    tier3_trigger_rate,
)


def test_retrieval_metrics_preserve_ranked_macro_and_micro_formulas() -> None:
    result = retrieval_metrics(
        [["a", "x"], ["x", "b"], ["x", "y"]],
        [["a"], ["b"], ["c"]],
    )

    assert isinstance(result, RetrievalMetrics)
    assert result.query_count == 3
    assert result.retrieved_count == 6
    assert result.relevant_count == 3
    assert result.true_positive_count == 2
    assert result.macro_precision == pytest.approx(1 / 3)
    assert result.macro_recall == pytest.approx(2 / 3)
    assert result.micro_precision == pytest.approx(1 / 3)
    assert result.micro_recall == pytest.approx(2 / 3)
    assert result.hit_rate == pytest.approx(2 / 3)
    assert result.mean_reciprocal_rank == pytest.approx(0.5)


def test_retrieval_metrics_support_multiple_labels_and_empty_returns() -> None:
    result = retrieval_metrics([["a", "b"], [], ["c"]], [["a", "z"], ["q"], ["c"]])

    assert result.macro_precision == pytest.approx(0.5)
    assert result.macro_recall == pytest.approx(0.5)
    assert result.micro_precision == pytest.approx(2 / 3)
    assert result.micro_recall == pytest.approx(0.5)
    assert result.mean_reciprocal_rank == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("predictions", "targets"),
    [
        ([], []),
        ([["a"]], []),
        ([["a"]], [[]]),
        ([["a", "a"]], [["a"]]),
        ([["a"]], [["a", "a"]]),
        ("a", [["a"]]),
        ([["a"]], "a"),
        ([[""]], [["a"]]),
        ([["a"]], [[3]]),
        ([{"a"}], [["a"]]),
    ],
)
def test_retrieval_metrics_reject_ambiguous_or_invalid_accounting(
    predictions: object, targets: object
) -> None:
    with pytest.raises(ValueError):
        retrieval_metrics(predictions, targets)  # type: ignore[arg-type]


def test_retrieval_metrics_are_immutable() -> None:
    result = retrieval_metrics([["a"]], [["a"]])
    with pytest.raises(FrozenInstanceError):
        result.query_count = 2  # type: ignore[misc]


def test_explicit_retrieval_activation_rate() -> None:
    result = retrieval_activation_rate(
        [
            {"retrieval_activated": True, "layer": 4},
            {"retrieval_activated": False, "layer": 4},
            {"retrieval_activated": True, "layer": 8},
        ]
    )

    assert result.event_count == 3
    assert result.activated_count == 2
    assert result.activation_rate == pytest.approx(2 / 3)


def test_explicit_low_confidence_and_tier3_rates_remain_distinct() -> None:
    result = tier3_trigger_rate(
        [
            {"low_confidence": True, "tier3_triggered": True},
            {"low_confidence": True, "tier3_triggered": False},
            {"low_confidence": False, "tier3_triggered": False},
        ]
    )

    assert result.event_count == 3
    assert result.low_confidence_count == 2
    assert result.tier3_trigger_count == 1
    assert result.low_confidence_rate == pytest.approx(2 / 3)
    assert result.tier3_trigger_rate == pytest.approx(1 / 3)
    assert result.trigger_rate_given_low_confidence == pytest.approx(0.5)


def test_tier3_conditional_rate_is_absent_without_low_confidence_events() -> None:
    result = tier3_trigger_rate([{"low_confidence": False, "tier3_triggered": False}])
    assert result.trigger_rate_given_low_confidence is None


@pytest.mark.parametrize(
    "events",
    [
        [],
        "event",
        [1],
        [{}],
        [{"retrieval_activated": 1}],
    ],
)
def test_activation_rate_strictly_validates_event_records(events: object) -> None:
    with pytest.raises(ValueError):
        retrieval_activation_rate(events)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "events",
    [
        [{"low_confidence": False, "tier3_triggered": True}],
        [{"low_confidence": 0, "tier3_triggered": False}],
        [{"low_confidence": True}],
    ],
)
def test_tier3_rate_rejects_inconsistent_event_records(events: object) -> None:
    with pytest.raises(ValueError):
        tier3_trigger_rate(events)  # type: ignore[arg-type]


def test_template_coverage_uses_unique_ids_and_library_denominator() -> None:
    result = template_coverage(
        [
            {"retrieved_template_ids": ["b", "a"]},
            {"retrieved_template_ids": ["b"]},
            {"retrieved_template_ids": []},
        ],
        ["a", "b", "c"],
    )

    assert result.event_count == 3
    assert result.template_count == 3
    assert result.covered_template_count == 2
    assert result.coverage == pytest.approx(2 / 3)
    assert result.covered_template_ids == ("a", "b")
    assert result.uncovered_template_ids == ("c",)


def test_template_coverage_allows_no_events_but_requires_a_library() -> None:
    result = template_coverage([], ["a"])
    assert result.coverage == 0
    assert result.uncovered_template_ids == ("a",)

    with pytest.raises(ValueError, match="at least one"):
        template_coverage([], [])


@pytest.mark.parametrize(
    ("events", "library"),
    [
        ([{}], ["a"]),
        ([{"retrieved_template_ids": ["missing"]}], ["a"]),
        ([{"retrieved_template_ids": ["a", "a"]}], ["a"]),
        ([{"retrieved_template_ids": "a"}], ["a"]),
        ([{"retrieved_template_ids": []}], ["a", "a"]),
    ],
)
def test_template_coverage_rejects_invalid_snapshots(
    events: object, library: object
) -> None:
    with pytest.raises(ValueError):
        template_coverage(events, library)  # type: ignore[arg-type]


def test_bridge_and_transfer_formulas_keep_boundaries_separate() -> None:
    bridge = summarize_latency(
        [
            {
                "step_seconds": 0.01,
                "full_round_trip_seconds": 0.004,
                "cpu_index_seconds": 0.003,
            }
        ]
    )
    copies = summarize_transfers(
        [
            {
                "unprofiled_step_seconds": 0.01,
                "profiled_model_seconds": 0.02,
                "d2h_gpu_seconds": 0.00005,
                "h2d_gpu_seconds": 0.00035,
            }
        ]
    )

    assert bridge["communication_ratio_of_means_percent"] == pytest.approx(10)
    assert bridge["full_round_trip_ratio_of_means_percent"] == pytest.approx(40)
    assert bridge["communication_mean_below_five_percent"] is False
    assert copies["transfer_only_ratio_of_means_percent"] == pytest.approx(4)
    assert copies["transfer_only_mean_below_five_percent"] is True
    assert copies["does_not_decide_bridge_overhead_gate"] is True


def test_ratio_of_means_is_not_mean_or_p95_of_paired_ratios() -> None:
    result = summarize_latency(
        [
            {
                "step_seconds": 1,
                "full_round_trip_seconds": 0.1,
                "cpu_index_seconds": 0,
            },
            {
                "step_seconds": 9,
                "full_round_trip_seconds": 0.1,
                "cpu_index_seconds": 0,
            },
        ]
    )

    assert result["communication_ratio_of_means_percent"] == pytest.approx(2)
    assert result["communication_paired_ratio_p95_percent"] == pytest.approx(
        9.5555555556
    )
    assert result["communication_p95_ms"] == pytest.approx(100)


def test_exactly_five_percent_does_not_pass_strict_gate() -> None:
    result = summarize_latency(
        [
            {
                "step_seconds": 1,
                "full_round_trip_seconds": 0.05,
                "cpu_index_seconds": 0,
            }
        ]
    )
    assert result["communication_mean_below_five_percent"] is False


@pytest.mark.parametrize("summarize", [summarize_latency, summarize_transfers])
def test_latency_summaries_reject_empty_or_nonrecord_samples(summarize: object) -> None:
    with pytest.raises(ValueError):
        summarize([])  # type: ignore[operator]
    with pytest.raises(ValueError):
        summarize([1])  # type: ignore[operator]


@pytest.mark.parametrize(
    "value", [0, -1, float("nan"), float("inf"), True, "1", 10**400]
)
def test_latency_rejects_invalid_step_denominators(value: object) -> None:
    with pytest.raises(ValueError):
        summarize_latency(
            [
                {
                    "step_seconds": value,
                    "full_round_trip_seconds": 0.004,
                    "cpu_index_seconds": 0.003,
                }
            ]
        )


def test_latency_rejects_missing_fields_and_negative_remainders() -> None:
    with pytest.raises(ValueError, match="missing"):
        summarize_latency([{"step_seconds": 1, "full_round_trip_seconds": 0.1}])
    with pytest.raises(ValueError, match="CPU search"):
        summarize_latency(
            [
                {
                    "step_seconds": 1,
                    "full_round_trip_seconds": 0.1,
                    "cpu_index_seconds": 0.2,
                }
            ]
        )


def test_transfer_intervals_cannot_exceed_profiled_model_time() -> None:
    with pytest.raises(ValueError, match="copy intervals"):
        summarize_transfers(
            [
                {
                    "unprofiled_step_seconds": 1,
                    "profiled_model_seconds": 0.01,
                    "d2h_gpu_seconds": 0.01,
                    "h2d_gpu_seconds": 0.01,
                }
            ]
        )


def test_interval_breakdown_partitions_sync_and_event_returns() -> None:
    counters = {
        "d2h_seconds": 0.1,
        "cpu_index_seconds": 0.2,
        "host_search_seconds": 0.3,
        "h2d_seconds": 0.1,
    }

    sync = interval_breakdown(0.6, counters)
    event = interval_breakdown(0.6, counters, return_mode="event")

    assert sync == {
        "gpu_to_cpu_enqueue_and_wait": pytest.approx(0.1),
        "cpu_normalization_search_gather": pytest.approx(0.2),
        "cpu_thread_setup_and_metadata": pytest.approx(0.1),
        "cpu_to_gpu_enqueue_and_wait": pytest.approx(0.1),
        "outside_native_interval": pytest.approx(0.1),
    }
    assert "cpu_to_gpu_enqueue_only" in event
    assert "cpu_to_gpu_enqueue_and_wait" not in event


@pytest.mark.parametrize(
    ("seconds", "counters", "mode"),
    [
        (-1, {}, "sync"),
        (1, {}, "sync"),
        (
            1,
            {
                "d2h_seconds": 0,
                "cpu_index_seconds": 0.2,
                "host_search_seconds": 0.1,
                "h2d_seconds": 0,
            },
            "sync",
        ),
        (
            0.1,
            {
                "d2h_seconds": 0.1,
                "cpu_index_seconds": 0,
                "host_search_seconds": 0,
                "h2d_seconds": 0.1,
            },
            "sync",
        ),
        (
            1,
            {
                "d2h_seconds": 0,
                "cpu_index_seconds": 0,
                "host_search_seconds": 0,
                "h2d_seconds": 0,
            },
            "async",
        ),
        (
            1,
            {
                "d2h_seconds": 0,
                "cpu_index_seconds": 0,
                "host_search_seconds": 0,
                "h2d_seconds": 0,
            },
            [],
        ),
    ],
)
def test_interval_breakdown_rejects_invalid_decompositions(
    seconds: object, counters: object, mode: object
) -> None:
    with pytest.raises(ValueError):
        interval_breakdown(  # type: ignore[arg-type]
            seconds, counters, return_mode=mode
        )
