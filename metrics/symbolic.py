"""Symbolic-head (A4) evaluation metrics."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Collection, Hashable, Mapping, Sequence
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np

DEFAULT_CUTOFFS: tuple[int, ...] = (1, 4, 8, 16)

def _check_m(m: int) -> None:
    if not isinstance(m, int) or isinstance(m, bool) or m < 1:
        raise ValueError(f"m must be an integer >= 1, got {m!r}")


def _top(retrieved: Sequence[Hashable], m: int) -> list[Hashable]:
    _check_m(m)
    return list(retrieved)[:m]


def precision_at_m(
    retrieved: Sequence[Hashable], relevant: Collection[Hashable], m: int
) -> float:
    top = _top(retrieved, m)
    if not top:
        return 0.0
    rel = set(relevant)
    return sum(1 for item in top if item in rel) / len(top)


def recall_at_m(
    retrieved: Sequence[Hashable], relevant: Collection[Hashable], m: int
) -> float | None:
    top = _top(retrieved, m)
    rel = set(relevant)
    if not rel:
        return None
    return len(rel.intersection(top)) / len(rel)


def f1_at_m(
    retrieved: Sequence[Hashable], relevant: Collection[Hashable], m: int
) -> float | None:
    p = precision_at_m(retrieved, relevant, m)
    r = recall_at_m(retrieved, relevant, m)
    if p is None or r is None:
        return None
    if p + r == 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)


def hit_at_m(
    retrieved: Sequence[Hashable], relevant: Collection[Hashable], m: int
) -> float | None:
    top = _top(retrieved, m)
    rel = set(relevant)
    if not rel:
        return None
    return 1.0 if rel.intersection(top) else 0.0


def reciprocal_rank(
    retrieved: Sequence[Hashable], relevant: Collection[Hashable]
) -> float | None:
    rel = set(relevant)
    if not rel:
        return None
    for rank, item in enumerate(retrieved, start=1):
        if item in rel:
            return 1.0 / rank
    return 0.0


def _mean_skipping_none(values: Sequence[float | None]) -> tuple[float | None, int]:
    kept = [v for v in values if v is not None]
    skipped = len(values) - len(kept)
    if not kept:
        return None, skipped
    return float(sum(kept) / len(kept)), skipped


def retrieval_report(
    retrieved_lists: Sequence[Sequence[Hashable]],
    relevant_sets: Sequence[Collection[Hashable]],
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> dict[str, Any]:
    if len(retrieved_lists) != len(relevant_sets):
        raise ValueError(
            f"{len(retrieved_lists)} retrieved lists vs {len(relevant_sets)} relevant sets"
        )
    if not retrieved_lists:
        raise ValueError("no examples to evaluate")
    cuts = sorted(set(cutoffs))
    for m in cuts:
        _check_m(m)

    report: dict[str, Any] = {
        "n_examples": len(retrieved_lists),
        "cutoffs": cuts,
        "precision": {},
        "recall": {},
        "f1": {},
        "hit_rate": {},
        "coverage": {},
        "skipped": {"precision": {}, "recall": {}, "f1": {}, "hit_rate": {}, "mrr": 0},
    }
    functions = {
        "precision": precision_at_m,
        "recall": recall_at_m,
        "f1": f1_at_m,
        "hit_rate": hit_at_m,
    }
    for name, fn in functions.items():
        for m in cuts:
            values = [fn(r, rel, m) for r, rel in zip(retrieved_lists, relevant_sets, strict=True)]
            mean, skipped = _mean_skipping_none(values)
            report[name][f"@{m}"] = mean
            report["skipped"][name][f"@{m}"] = skipped
    for m in cuts:
        cov = sum(1 for r in retrieved_lists if len(_top(r, m)) > 0) / len(retrieved_lists)
        report["coverage"][f"@{m}"] = float(cov)
    mrr_values = [reciprocal_rank(r, rel) for r, rel in zip(retrieved_lists, relevant_sets, strict=True)]
    mrr, skipped = _mean_skipping_none(mrr_values)
    report["mrr"] = mrr
    report["skipped"]["mrr"] = skipped
    return report


def _as_2d(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D [N, D], got shape {arr.shape}")
    return cast(np.ndarray, arr)


def alignment_loss(queries: Any, keys: Any, eps: float = 1e-12) -> float:
    q = _as_2d(queries, "queries")
    k = _as_2d(keys, "keys")
    if q.shape != k.shape:
        raise ValueError(f"queries {q.shape} and keys {k.shape} must have the same shape")
    cos = np.sum(q * k, axis=1) / (
        np.maximum(np.linalg.norm(q, axis=1) * np.linalg.norm(k, axis=1), eps)
    )
    return float(np.mean(1.0 - cos))


def alignment_infonce(
    queries: Any, keys: Any, temperature: float = 0.1, eps: float = 1e-12
) -> float:
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature!r}")
    q = _as_2d(queries, "queries")
    k = _as_2d(keys, "keys")
    if q.shape != k.shape:
        raise ValueError(f"queries {q.shape} and keys {k.shape} must have the same shape")
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), eps)
    k = k / np.maximum(np.linalg.norm(k, axis=1, keepdims=True), eps)
    logits = (q @ k.T) / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
    return float(-np.mean(np.diag(log_probs)))


def latency_summary(seconds: Sequence[float]) -> dict[str, float | int | None]:
    """Mean, median and 95th percentile of retrieval latencies, in seconds."""
    values = [float(s) for s in seconds]
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("latencies must be finite and non-negative")
    if not values:
        return {"n": 0, "mean_s": None, "p50_s": None, "p95_s": None}
    return {
        "n": len(values),
        "mean_s": float(np.mean(values)),
        "p50_s": float(np.percentile(values, 50)),
        "p95_s": float(np.percentile(values, 95)),
    }


def retrieval_contribution(
    with_retrieval: Mapping[str, float],
    without_retrieval: Mapping[str, float],
    higher_is_better: Mapping[str, bool] | None = None,
) -> dict[str, dict[str, float | bool]]:
    lower_is_better_names = ("loss", "ppl", "perplexity", "kl", "ece")
    flags = dict(higher_is_better or {})
    out: dict[str, dict[str, float | bool]] = {}
    for name in sorted(set(with_retrieval) & set(without_retrieval)):
        a, b = float(with_retrieval[name]), float(without_retrieval[name])
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        hib = flags.get(name, not any(tag in name.lower() for tag in lower_is_better_names))
        out[name] = {
            "with_retrieval": a,
            "without_retrieval": b,
            "improvement": (a - b) if hib else (b - a),
            "higher_is_better": hib,
        }
    return out


@runtime_checkable
class SymbolicRetriever(Protocol):
    def retrieve(self, query: Any, m: int) -> list[Hashable]: ...


class OracleRetriever:
    def __init__(self, relevant_by_query: Mapping[Hashable, Collection[Hashable]],
                 library: Sequence[Hashable]):
        self._relevant = {k: list(v) for k, v in relevant_by_query.items()}
        self._library = list(library)

    def retrieve(self, query: Hashable, m: int) -> list[Hashable]:
        _check_m(m)
        found = list(self._relevant.get(query, []))
        padding = [t for t in self._library if t not in set(found)]
        return (found + padding)[:m]


class RandomRetriever:
    def __init__(self, library: Sequence[Hashable], seed: int = 0):
        if not library:
            raise ValueError("library must not be empty")
        self._library = list(library)
        self._rng = random.Random(seed)

    def retrieve(self, query: Any, m: int) -> list[Hashable]:
        _check_m(m)
        return self._rng.sample(self._library, min(m, len(self._library)))


def evaluate_retriever(
    retriever: SymbolicRetriever,
    queries: Sequence[Any],
    relevant_sets: Sequence[Collection[Hashable]],
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> dict[str, Any]:
    if len(queries) != len(relevant_sets):
        raise ValueError(f"{len(queries)} queries vs {len(relevant_sets)} relevant sets")
    if not queries:
        raise ValueError("no queries to evaluate")
    cuts = sorted(set(cutoffs))
    max_m = cuts[-1]
    retrieved: list[list[Hashable]] = []
    timings: list[float] = []
    for query in queries:
        start = time.perf_counter()
        result = retriever.retrieve(query, max_m)
        timings.append(time.perf_counter() - start)
        retrieved.append(list(result))
    report = retrieval_report(retrieved, relevant_sets, cuts)
    report["latency"] = latency_summary(timings)
    return report


__all__ = [
    "DEFAULT_CUTOFFS",
    "OracleRetriever",
    "RandomRetriever",
    "SymbolicRetriever",
    "alignment_infonce",
    "alignment_loss",
    "evaluate_retriever",
    "f1_at_m",
    "hit_at_m",
    "latency_summary",
    "precision_at_m",
    "recall_at_m",
    "reciprocal_rank",
    "retrieval_contribution",
    "retrieval_report",
]
