"""Pure A4 retrieval-quality and symbolic-head event metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalMetrics:
    """Aggregate ranked-retrieval quality over labelled probes."""

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


@dataclass(frozen=True)
class RetrievalActivationMetrics:
    """Rate at which explicitly recorded retrieval opportunities activated."""

    event_count: int
    activated_count: int
    activation_rate: float


@dataclass(frozen=True)
class Tier3TriggerMetrics:
    """Low-confidence and Tier-3 trigger rates over retrieval events."""

    event_count: int
    low_confidence_count: int
    tier3_trigger_count: int
    low_confidence_rate: float
    tier3_trigger_rate: float
    trigger_rate_given_low_confidence: float | None


@dataclass(frozen=True)
class TemplateCoverageMetrics:
    """Unique retrieved-template coverage of an explicit template library."""

    event_count: int
    template_count: int
    covered_template_count: int
    coverage: float
    covered_template_ids: tuple[str, ...]
    uncovered_template_ids: tuple[str, ...]


def _checked_sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return value


def _checked_ids(values: object, *, name: str) -> tuple[str, ...]:
    sequence = _checked_sequence(values, name=name)
    ids = tuple(sequence)
    if any(not isinstance(item, str) or not item.strip() for item in ids):
        raise ValueError(f"{name} contains an empty or non-string ID")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} contains duplicate IDs")
    return ids


def _checked_events(
    events: object, *, allow_empty: bool = False
) -> tuple[Mapping[str, object], ...]:
    sequence = _checked_sequence(events, name="events")
    if not sequence and not allow_empty:
        raise ValueError("At least one event record is required")
    if any(not isinstance(event, Mapping) for event in sequence):
        raise ValueError("Each event must be a mapping")
    return tuple(sequence)  # type: ignore[return-value]


def _event_bool(event: Mapping[str, object], name: str) -> bool:
    if name not in event:
        raise ValueError(f"Event is missing required field {name!r}")
    value = event[name]
    if not isinstance(value, bool):
        raise ValueError(f"Event field {name!r} must be a boolean")
    return value


def retrieval_metrics(
    retrieved_ids: Sequence[Sequence[str]], relevant_ids: Sequence[Sequence[str]]
) -> RetrievalMetrics:
    """Score ranked retrieval against independent, nonempty relevance labels.

    Empty retrievals score zero. Empty relevance sets are rejected because an
    unlabelled probe is not a known negative. Precision uses the actual returned
    count, ground truth may contain multiple IDs, and duplicate IDs are rejected.

    Semantic firing labels measure semantic retrieval quality. Exact-neighbor
    labels instead measure index agreement; callers must preserve that provenance.
    """
    retrieved = _checked_sequence(retrieved_ids, name="retrieved_ids")
    relevant = _checked_sequence(relevant_ids, name="relevant_ids")
    if not retrieved or len(retrieved) != len(relevant):
        raise ValueError("Nonempty retrieval and relevance batches must align")

    precisions: list[float] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    returned = relevant_total = true_positives = hit_queries = 0
    for prediction, target in zip(retrieved, relevant, strict=True):
        ranked = _checked_ids(prediction, name="Retrieved IDs")
        truth = _checked_ids(target, name="Relevant IDs")
        if not truth:
            raise ValueError("Each scored probe needs at least one relevance label")

        target_set = set(truth)
        hits = len(set(ranked) & target_set)
        returned += len(ranked)
        relevant_total += len(truth)
        true_positives += hits
        hit_queries += int(hits > 0)
        precisions.append(hits / len(ranked) if ranked else 0.0)
        recalls.append(hits / len(truth))
        reciprocal_ranks.append(
            next(
                (
                    1.0 / rank
                    for rank, item in enumerate(ranked, start=1)
                    if item in target_set
                ),
                0.0,
            )
        )

    count = len(retrieved)
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


def retrieval_activation_rate(
    events: Sequence[Mapping[str, object]],
) -> RetrievalActivationMetrics:
    """Summarize explicit ``retrieval_activated`` event decisions.

    The decision is consumed as logged rather than inferred from returned IDs or
    attention values, so future runtimes can retain their own activation policy.
    """
    records = _checked_events(events)
    activated = sum(_event_bool(event, "retrieval_activated") for event in records)
    return RetrievalActivationMetrics(
        event_count=len(records),
        activated_count=activated,
        activation_rate=activated / len(records),
    )


def tier3_trigger_rate(
    events: Sequence[Mapping[str, object]],
) -> Tier3TriggerMetrics:
    """Summarize explicit low-confidence and Tier-3 trigger decisions.

    No confidence or surprise threshold is reconstructed. A Tier-3 trigger must
    be marked low-confidence; low-confidence events may remain untriggered when
    another runtime condition, such as high surprise, is absent.
    """
    records = _checked_events(events)
    low_confidence = 0
    triggered = 0
    for event in records:
        is_low_confidence = _event_bool(event, "low_confidence")
        is_triggered = _event_bool(event, "tier3_triggered")
        if is_triggered and not is_low_confidence:
            raise ValueError("A Tier-3 trigger must be marked low-confidence")
        low_confidence += int(is_low_confidence)
        triggered += int(is_triggered)

    count = len(records)
    return Tier3TriggerMetrics(
        event_count=count,
        low_confidence_count=low_confidence,
        tier3_trigger_count=triggered,
        low_confidence_rate=low_confidence / count,
        tier3_trigger_rate=triggered / count,
        trigger_rate_given_low_confidence=(
            triggered / low_confidence if low_confidence else None
        ),
    )


def template_coverage(
    events: Sequence[Mapping[str, object]], template_ids: Sequence[str]
) -> TemplateCoverageMetrics:
    """Measure unique library coverage from explicit retrieval event records.

    Each event supplies ``retrieved_template_ids``. The denominator is the
    separately supplied, nonempty library snapshot; unknown and duplicate IDs are
    rejected so the result cannot silently mix snapshots or count one template
    twice. An empty event sequence has zero coverage.
    """
    library = _checked_ids(template_ids, name="template_ids")
    if not library:
        raise ValueError("template_ids must contain at least one library template")
    records = _checked_events(events, allow_empty=True)
    known = set(library)
    covered: set[str] = set()
    for event in records:
        if "retrieved_template_ids" not in event:
            raise ValueError("Event is missing required field 'retrieved_template_ids'")
        retrieved = _checked_ids(
            event["retrieved_template_ids"], name="retrieved_template_ids"
        )
        unknown = set(retrieved) - known
        if unknown:
            raise ValueError(
                "Retrieved template IDs are absent from the library snapshot: "
                + ", ".join(sorted(unknown))
            )
        covered.update(retrieved)

    covered_ids = tuple(
        template_id for template_id in library if template_id in covered
    )
    uncovered_ids = tuple(
        template_id for template_id in library if template_id not in covered
    )
    return TemplateCoverageMetrics(
        event_count=len(records),
        template_count=len(library),
        covered_template_count=len(covered_ids),
        coverage=len(covered_ids) / len(library),
        covered_template_ids=covered_ids,
        uncovered_template_ids=uncovered_ids,
    )


__all__ = [
    "RetrievalActivationMetrics",
    "RetrievalMetrics",
    "retrieval_activation_rate",
    "retrieval_metrics",
    "TemplateCoverageMetrics",
    "template_coverage",
    "Tier3TriggerMetrics",
    "tier3_trigger_rate",
]
