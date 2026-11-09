"""A4 latency boundaries and labelled retrieval calculations."""

import pytest

from metrics.a4.latency import summarize_latency, summarize_transfers
from metrics.a4.retrieval import retrieval_metrics


def test_bridge_and_transfer_only_boundaries_remain_separate():
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
    assert not bridge["communication_mean_below_five_percent"]
    assert copies["transfer_only_ratio_of_means_percent"] == pytest.approx(4)
    assert copies["transfer_only_mean_below_five_percent"]
    assert copies["does_not_decide_bridge_overhead_gate"]


def test_ratio_of_means_is_not_mean_of_paired_ratios():
    result = summarize_latency(
        [
            {"step_seconds": 1, "full_round_trip_seconds": 0.1, "cpu_index_seconds": 0},
            {"step_seconds": 9, "full_round_trip_seconds": 0.1, "cpu_index_seconds": 0},
        ]
    )
    assert result["communication_ratio_of_means_percent"] == pytest.approx(2)
    assert result["communication_paired_ratio_p95_percent"] == pytest.approx(
        9.5555555556
    )


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_denominators_are_rejected(value):
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


def test_negative_remainder_is_rejected():
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


def test_transfer_intervals_cannot_exceed_profiled_wall_time():
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


@pytest.mark.parametrize("summarize", [summarize_latency, summarize_transfers])
def test_empty_samples_are_rejected(summarize):
    with pytest.raises(ValueError):
        summarize([])


def test_five_percent_is_not_strictly_below_five_percent():
    result = summarize_latency(
        [{"step_seconds": 1, "full_round_trip_seconds": 0.05, "cpu_index_seconds": 0}]
    )
    assert not result["communication_mean_below_five_percent"]


def test_single_positive_ranks_and_misses():
    result = retrieval_metrics(
        [["a", "x"], ["x", "b"], ["x", "y"]], [["a"], ["b"], ["c"]]
    )
    assert result.query_count == 3
    assert result.true_positive_count == 2
    assert result.retrieved_count == 6
    assert result.macro_precision == pytest.approx(1 / 3)
    assert result.macro_recall == pytest.approx(2 / 3)
    assert result.hit_rate == pytest.approx(2 / 3)
    assert result.mean_reciprocal_rank == pytest.approx(0.5)


def test_multiple_relevant_ids_and_variable_return_counts():
    result = retrieval_metrics([["a", "b"], ["c"]], [["a", "z"], ["c"]])
    assert result.macro_precision == pytest.approx(0.75)
    assert result.macro_recall == pytest.approx(0.75)
    assert result.micro_precision == pytest.approx(2 / 3)
    assert result.micro_recall == pytest.approx(2 / 3)


def test_empty_return_is_counted_as_a_miss():
    result = retrieval_metrics([[]], [["a"]])
    assert result.query_count == 1
    assert result.retrieved_count == 0
    assert result.macro_precision == result.micro_precision == 0
    assert result.macro_recall == result.mean_reciprocal_rank == 0


@pytest.mark.parametrize(
    ("predictions", "targets"),
    [
        ([], []),
        ([["a"]], []),
        ([["a"]], [[]]),
        ([["a", "a"]], [["a"]]),
        ([["a"]], [["a", "a"]]),
        (["a"], [["a"]]),
        ([["a"]], ["a"]),
        ([[""]], [["a"]]),
        ([["a"]], [[3]]),
    ],
)
def test_rejects_ambiguous_or_invalid_accounting(predictions, targets):
    with pytest.raises(ValueError):
        retrieval_metrics(predictions, targets)
