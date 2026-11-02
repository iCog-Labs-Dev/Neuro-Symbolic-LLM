"""ID-based retrieval quality on explicitly labelled probes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalMetrics:
    query_count: int
    retrieved_count: int
    relevant_count: int
    true_positive_count: int
    macro_precision: float
    macro_recall: float
    micro_precision: float
    micro_recall: float
    hit_rate: float
    mean_reciprocal_rank: float


def _checked_ids(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, str | bytes):
        raise ValueError(f"{name} must be a sequence of IDs, not one string")
    ids = tuple(values)
    if any(not isinstance(item, str) or not item.strip() for item in ids):
        raise ValueError(f"{name} contains an empty or non-string ID")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} contains duplicate IDs")
    return ids


def retrieval_metrics(
    retrieved_ids: Sequence[Sequence[str]], relevant_ids: Sequence[Sequence[str]]
) -> RetrievalMetrics:
    """Score ranked retrieval against independent, nonempty relevance labels.

    Empty retrievals score zero; empty relevance sets are rejected because an
    unlabelled probe must not be treated as a known negative. Precision uses
    the actual returned count, not an inferred top-m. Ground truth may contain
    multiple relevant IDs. Duplicate IDs are rejected rather than counted twice.

    Semantic template-firing labels measure semantic retrieval quality. Exact
    nearest-neighbor IDs instead measure index agreement, not semantic accuracy.
    Callers must retain label origin and the number of unlabelled probes.
    """
    if (
        isinstance(retrieved_ids, str | bytes)
        or isinstance(relevant_ids, str | bytes)
        or len(retrieved_ids) == 0
        or len(retrieved_ids) != len(relevant_ids)
    ):
        raise ValueError("Nonempty retrieval and relevance batches must align")
    precisions = []
    recalls = []
    reciprocal_ranks = []
    returned = relevant_total = true_positives = hit_queries = 0
    for prediction, target in zip(retrieved_ids, relevant_ids, strict=True):
        ranked = _checked_ids(prediction, name="Retrieved IDs")
        truth = _checked_ids(target, name="Relevant IDs")
        if not truth:
            raise ValueError("Each scored probe needs at least one relevance label")
        target_set = set(truth)
        hits = len(set(ranked) & target_set)
        returned += len(ranked)
        relevant_total += len(truth)
        true_positives += hits
        hit_queries += hits > 0
        precisions.append(hits / len(ranked) if ranked else 0.0)
        recalls.append(hits / len(truth))
        reciprocal_ranks.append(
            next(
                (
                    1.0 / rank
                    for rank, item in enumerate(ranked, 1)
                    if item in target_set
                ),
                0.0,
            )
        )
    count = len(retrieved_ids)
    return RetrievalMetrics(
        query_count=count,
        retrieved_count=returned,
        relevant_count=relevant_total,
        true_positive_count=true_positives,
        macro_precision=sum(precisions) / count,
        macro_recall=sum(recalls) / count,
        micro_precision=true_positives / returned if returned else 0.0,
        micro_recall=true_positives / relevant_total,
        hit_rate=hit_queries / count,
        mean_reciprocal_rank=sum(reciprocal_ranks) / count,
    )
