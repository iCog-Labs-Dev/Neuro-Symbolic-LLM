"""Tests for metrics/symbolic.py (A4 symbolic-head metrics)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metrics.symbolic import (
    OracleRetriever,
    RandomRetriever,
    SymbolicRetriever,
    alignment_infonce,
    alignment_loss,
    evaluate_retriever,
    f1_at_m,
    hit_at_m,
    latency_summary,
    precision_at_m,
    recall_at_m,
    reciprocal_rank,
    retrieval_contribution,
    retrieval_report,
)

RELEVANT = {"MotherOf(Alice,Bob)", "ParentOf(Alice,Bob)"}
RETRIEVED = [
    "MotherOf(Alice,Bob)",
    "ParentOf(Alice,Bob)",
    "SiblingOf(Alice,Bob)",
    "EmployerOf(Alice,Bob)",
]


def test_worked_example_precision_and_recall() -> None:
    assert precision_at_m(RETRIEVED, RELEVANT, 4) == pytest.approx(0.5)
    assert recall_at_m(RETRIEVED, RELEVANT, 4) == pytest.approx(1.0)
    assert precision_at_m(RETRIEVED, RELEVANT, 1) == pytest.approx(1.0)
    assert recall_at_m(RETRIEVED, RELEVANT, 1) == pytest.approx(0.5)


def test_f1_and_hit_and_mrr() -> None:
    assert f1_at_m(RETRIEVED, RELEVANT, 4) == pytest.approx(2 * 0.5 * 1.0 / 1.5)
    assert hit_at_m(RETRIEVED, RELEVANT, 1) == 1.0
    assert hit_at_m(["x", "y"], RELEVANT, 2) == 0.0
    assert reciprocal_rank(["x", "MotherOf(Alice,Bob)"], RELEVANT) == pytest.approx(0.5)
    assert reciprocal_rank(["x", "y"], RELEVANT) == 0.0


def test_undefined_values_are_none_not_nan() -> None:
    assert precision_at_m([], RELEVANT, 4) == 0.0
    assert recall_at_m(RETRIEVED, set(), 4) is None
    assert f1_at_m([], RELEVANT, 4) == 0.0
    assert f1_at_m(RETRIEVED, set(), 4) is None
    assert hit_at_m(RETRIEVED, set(), 4) is None
    assert reciprocal_rank(RETRIEVED, set()) is None


def test_precision_divides_by_items_returned_not_by_m() -> None:
    assert precision_at_m(["a", "b"], {"a", "b"}, 8) == pytest.approx(1.0)


def test_invalid_m_raises() -> None:
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            precision_at_m(RETRIEVED, RELEVANT, bad)


def test_retrieval_report_skips_undefined_and_counts_them() -> None:
    report = retrieval_report(
        [RETRIEVED, ["x"], RETRIEVED],
        [RELEVANT, set(), RELEVANT],
        cutoffs=(1, 4),
    )
    assert report["n_examples"] == 3
    assert report["precision"]["@4"] == pytest.approx((0.5 + 0.0 + 0.5) / 3)
    assert report["recall"]["@4"] == pytest.approx(1.0)
    assert report["skipped"]["recall"]["@4"] == 1
    assert report["coverage"]["@4"] == pytest.approx(1.0)
    assert all(
        v is None or math.isfinite(v)
        for metric in ("precision", "recall", "f1", "hit_rate", "coverage")
        for v in report[metric].values()
    )


def test_empty_retrievals_score_zero_and_report_coverage() -> None:
    report = retrieval_report(
        [RETRIEVED, [], ["x"]],
        [RELEVANT, RELEVANT, RELEVANT],
        cutoffs=(1, 4),
    )
    assert report["n_examples"] == 3
    assert report["precision"]["@4"] == pytest.approx((0.5 + 0.0 + 0.0) / 3)
    assert report["skipped"]["precision"]["@4"] == 0
    assert report["coverage"]["@4"] == pytest.approx(2 / 3)


def test_retrieval_report_validates_inputs() -> None:
    with pytest.raises(ValueError):
        retrieval_report([RETRIEVED], [RELEVANT, RELEVANT])
    with pytest.raises(ValueError):
        retrieval_report([], [])


def test_alignment_loss() -> None:
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert alignment_loss(q, q) == pytest.approx(0.0, abs=1e-9)
    assert alignment_loss(q, np.array([[0.0, 1.0], [1.0, 0.0]])) == pytest.approx(1.0)
    assert alignment_loss(q, -q) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        alignment_loss(q, q[:1])
    with pytest.raises(ValueError):
        alignment_loss(q[0], q[0])


def test_infonce_is_stricter_than_cosine() -> None:
    rng = np.random.default_rng(0)
    keys = rng.normal(size=(8, 16))
    good = keys + 0.01 * rng.normal(size=keys.shape)
    collapsed = np.tile(
        keys.mean(axis=0, keepdims=True), (8, 1)
    )
    assert alignment_infonce(good, keys) < alignment_infonce(collapsed, keys)
    assert alignment_infonce(good, keys) < 0.05
    with pytest.raises(ValueError):
        alignment_infonce(good, keys, temperature=0.0)


def test_latency_summary() -> None:
    out = latency_summary([0.1, 0.2, 0.3, 0.4, 1.0])
    assert out["n"] == 5
    assert out["mean_s"] == pytest.approx(0.4)
    assert out["p50_s"] == pytest.approx(0.3)
    p95, p50 = out["p95_s"], out["p50_s"]
    assert isinstance(p95, float) and isinstance(p50, float)
    assert p95 > p50
    assert latency_summary([])["mean_s"] is None
    with pytest.raises(ValueError):
        latency_summary([0.1, float("nan")])
    with pytest.raises(ValueError):
        latency_summary([-0.1])


def test_retrieval_contribution_signs() -> None:
    out = retrieval_contribution(
        with_retrieval={"loss": 3.0, "token_accuracy": 0.40, "kl": 0.10},
        without_retrieval={"loss": 3.2, "token_accuracy": 0.35, "kl": 0.05},
    )
    assert out["loss"]["improvement"] == pytest.approx(0.2)
    assert out["token_accuracy"]["improvement"] == pytest.approx(0.05)
    assert out["kl"]["improvement"] == pytest.approx(-0.05)
    assert "f1" not in out


def test_retrieval_contribution_override_and_nonfinite() -> None:
    out = retrieval_contribution(
        {"score": 1.0, "bad": float("nan")},
        {"score": 2.0, "bad": 1.0},
        higher_is_better={"score": False},
    )
    assert out["score"]["improvement"] == pytest.approx(1.0)
    assert "bad" not in out


def test_oracle_and_random_retrievers_bracket_the_metrics() -> None:
    library = [f"T{i}" for i in range(40)]
    queries = list(range(20))
    relevant = {q: {f"T{q}", f"T{(q + 1) % 40}"} for q in queries}
    sets = [relevant[q] for q in queries]

    oracle = OracleRetriever(relevant, library)
    random_r = RandomRetriever(library, seed=0)
    assert isinstance(oracle, SymbolicRetriever)

    best = evaluate_retriever(oracle, queries, sets, cutoffs=(1, 4, 8))
    floor = evaluate_retriever(random_r, queries, sets, cutoffs=(1, 4, 8))
    assert best["recall"]["@4"] == pytest.approx(1.0)
    assert best["mrr"] == pytest.approx(1.0)
    assert floor["recall"]["@4"] < best["recall"]["@4"]
    assert floor["precision"]["@4"] < best["precision"]["@4"]
    assert best["latency"]["n"] == 20
    assert best["latency"]["mean_s"] >= 0.0


def test_random_retriever_is_deterministic_in_seed() -> None:
    library = [f"T{i}" for i in range(30)]
    a = RandomRetriever(library, seed=3).retrieve("q", 5)
    b = RandomRetriever(library, seed=3).retrieve("q", 5)
    assert a == b
    with pytest.raises(ValueError):
        RandomRetriever([])


def test_evaluate_retriever_validates_inputs() -> None:
    oracle = OracleRetriever({}, ["a"])
    with pytest.raises(ValueError):
        evaluate_retriever(oracle, [1, 2], [{"a"}])
    with pytest.raises(ValueError):
        evaluate_retriever(oracle, [], [])
