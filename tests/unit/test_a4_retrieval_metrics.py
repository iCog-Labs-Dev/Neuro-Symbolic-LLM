"""Retrieval accounting on labelled ranked ID lists."""

import pytest

from metrics.a4.retrieval import retrieval_metrics


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
