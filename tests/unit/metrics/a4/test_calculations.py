"""A4 latency, retrieval and representation metric calculations."""

import numpy as np
import pytest

from metrics.a4.latency import summarize_latency, summarize_transfers
from metrics.a4.representation import (
    capacity_sensitivity,
    cosine_geometry,
    factor_organization,
    key_log_probabilities,
    query_silhouette,
    select_factors,
)
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


def test_fracture_overlap_and_zero_effects():
    result = factor_organization([[1, 0], [0, 2], [1, 1], [0, 0]])
    assert result["fracture"] == [0.5, 0.5, 1.0, None]
    assert result["valid_factor_indices"] == [0, 1, 2]
    np.testing.assert_allclose(
        result["overlap_matrix"], [[1, 0, 0.5], [0, 1, 0.5], [0.5, 0.5, 1]]
    )
    assert result["mean_off_diagonal_overlap"] == pytest.approx(1 / 3)
    empty = factor_organization([[0, 0]])
    assert empty["overlap_matrix"] == []
    assert empty["mean_off_diagonal_overlap"] is None


@pytest.mark.parametrize("value", [[[-1, 1]], [[float("nan"), 0]], [], [1, 2]])
def test_invalid_effects_are_rejected(value):
    with pytest.raises(ValueError):
        factor_organization(value)


def test_cosine_geometry_excludes_self_and_duplicate_pair_counts():
    result = cosine_geometry([[1, 0], [2, 0], [0, 3]], block_size=1)
    assert result["pairs"] == 3
    assert result["mean_cosine"] == pytest.approx(1 / 3)
    assert result["mean_squared_cosine"] == pytest.approx(1 / 3)
    assert result["norm_max"] == 3
    assert cosine_geometry([[1, 0]])["mean_cosine"] is None
    with pytest.raises(ValueError, match="zero"):
        cosine_geometry([[0, 0]])


def test_key_probability_uses_raw_dot_and_is_numerically_stable():
    result = key_log_probabilities([[1000, 0]], [[1, 0], [2, 0]])
    np.testing.assert_allclose(result, [[-1000, 0]])
    with pytest.raises(ValueError, match="dimensions"):
        key_log_probabilities([[1]], [[1, 2]])


def test_silhouette_and_multilabel_memberships():
    queries = [[0, 0], [0, 0], [10, 0], [10, 0]]
    result = query_silhouette(queries, [["a"], ["a"], ["b"], ["b"]])
    assert result["silhouette"] == 1
    result = query_silhouette([[0], [0]], [["a", "b"], ["a", "b"]])
    assert result["memberships"] == 4
    assert result["silhouette"] == 0
    assert query_silhouette([[0], [1]], [["a"], ["b"]])["silhouette"] is None
    with pytest.raises(ValueError, match="Unique"):
        query_silhouette([[0]], [["a", "a"]])


def test_factors_require_cross_document_support_and_ignore_model_scores():
    probes = [
        {"corpus_id": "c", "document_id": "one", "relevant_ids": ["a", "b"]},
        {"corpus_id": "c", "document_id": "one", "relevant_ids": ["a", "b"]},
        {"corpus_id": "c", "document_id": "two", "relevant_ids": ["a", "c"]},
    ]
    names, support = select_factors(probes)
    assert names == ["a"] and support == {"a": [0, 1, 2]}


def test_capacity_controls_use_exact_cosine_and_preserve_full_dimension():
    vectors = np.eye(64)[:3]
    result = capacity_sensitivity(
        vectors[:2], vectors, ["a", "b", "c"], [["a"], ["b"]], np.eye(64)
    )
    full = next(row for row in result["controls"] if row["dimensions"] == 64)
    assert full["scores"][0]["all_probe_precision"] == 1
    assert full["available_reference_pair_fraction"] == 1
    assert result["ann_exact_id_agreement_at_8"] is None
    with pytest.raises(ValueError, match="reference"):
        capacity_sensitivity(
            vectors[:2], vectors, ["a", "b", "c"], [["missing"], ["b"]], np.eye(64)
        )
