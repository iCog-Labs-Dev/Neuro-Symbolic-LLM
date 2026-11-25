"""Miner-owned pattern contracts and backward-compatible A4 aliases."""

from hybrid_miner.contracts import (
    PatternProvenance,
    PatternRecord,
    PatternRecordError,
    validate_extraction_pattern_record,
)
from hybrid_miner.contracts import (
    PatternSourceSpan as SourceSpan,
)
from hybrid_miner.contracts.pattern_record import SemanticLevel

validate_q1_pattern_record = validate_extraction_pattern_record

__all__ = [
    "PatternProvenance",
    "PatternRecord",
    "PatternRecordError",
    "SemanticLevel",
    "SourceSpan",
    "validate_extraction_pattern_record",
    "validate_q1_pattern_record",
]
