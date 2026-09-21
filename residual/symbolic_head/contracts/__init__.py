"""Stable data contracts shared across Tier 2 boundaries."""

from residual.symbolic_head.contracts.pattern_record import (
    PatternProvenance,
    PatternRecord,
    PatternRecordError,
    SemanticLevel,
    SourceSpan,
    validate_q1_pattern_record,
)

__all__ = [
    "PatternProvenance",
    "PatternRecord",
    "PatternRecordError",
    "SemanticLevel",
    "SourceSpan",
    "validate_q1_pattern_record",
]
