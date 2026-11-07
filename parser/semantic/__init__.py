"""Public semantic-parser API; optional dependencies load on first use."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from parser.semantic.backends import (
        AnthropicBackend,
        CallableBackend,
        GeminiBackend,
        ModelBackend,
        OllamaBackend,
        OpenAIBackend,
        OpenAICompatibleBackend,
        TransformersBackend,
    )
    from parser.semantic.dataset import (
        DistillationRecord,
        RejectedRecord,
        SemanticDatasetBuilder,
    )
    from parser.semantic.model_config import (
        ModelProfile,
        build_distilled_semantic_parser,
        build_reference_semantic_parser,
        create_backend,
        default_model_config_path,
        load_model_profile,
    )
    from parser.semantic.normalization import (
        SemanticNormalizationError,
        normalize_extraction_payload,
        normalize_symbol,
    )
    from parser.semantic.schema import (
        SemanticArgument,
        SemanticAssertion,
        SemanticEntity,
        SemanticExtraction,
        SemanticParseProvenance,
        SemanticParseResult,
        SemanticRepair,
        SemanticRule,
        SemanticRuleVariable,
        SourceDocument,
        SourceQuote,
    )
    from parser.semantic.semantic_parser import (
        PARSER_VERSION,
        DistilledSemanticParser,
        ModelGenerationError,
        ReferenceSemanticParser,
        SemanticParseError,
        SemanticParserConfig,
    )

_EXPORTS = {
    **dict.fromkeys(
        (
            "AnthropicBackend",
            "CallableBackend",
            "GeminiBackend",
            "ModelBackend",
            "OllamaBackend",
            "OpenAIBackend",
            "OpenAICompatibleBackend",
            "TransformersBackend",
        ),
        "parser.semantic.backends",
    ),
    **dict.fromkeys(
        ("DistillationRecord", "RejectedRecord", "SemanticDatasetBuilder"),
        "parser.semantic.dataset",
    ),
    **dict.fromkeys(
        (
            "ModelProfile",
            "build_distilled_semantic_parser",
            "build_reference_semantic_parser",
            "create_backend",
            "default_model_config_path",
            "load_model_profile",
        ),
        "parser.semantic.model_config",
    ),
    **dict.fromkeys(
        (
            "SemanticNormalizationError",
            "normalize_extraction_payload",
            "normalize_symbol",
        ),
        "parser.semantic.normalization",
    ),
    **dict.fromkeys(
        (
            "SemanticArgument",
            "SemanticAssertion",
            "SemanticEntity",
            "SemanticExtraction",
            "SemanticParseProvenance",
            "SemanticParseResult",
            "SemanticRepair",
            "SemanticRule",
            "SemanticRuleVariable",
            "SourceDocument",
            "SourceQuote",
        ),
        "parser.semantic.schema",
    ),
    **dict.fromkeys(
        (
            "PARSER_VERSION",
            "DistilledSemanticParser",
            "ModelGenerationError",
            "ReferenceSemanticParser",
            "SemanticParseError",
            "SemanticParserConfig",
        ),
        "parser.semantic.semantic_parser",
    ),
}
__all__ = [
    "AnthropicBackend",
    "CallableBackend",
    "DistillationRecord",
    "DistilledSemanticParser",
    "GeminiBackend",
    "ModelBackend",
    "ModelGenerationError",
    "ModelProfile",
    "OllamaBackend",
    "OpenAIBackend",
    "OpenAICompatibleBackend",
    "PARSER_VERSION",
    "ReferenceSemanticParser",
    "RejectedRecord",
    "SemanticArgument",
    "SemanticAssertion",
    "SemanticDatasetBuilder",
    "SemanticEntity",
    "SemanticExtraction",
    "SemanticNormalizationError",
    "SemanticParseError",
    "SemanticParseProvenance",
    "SemanticParseResult",
    "SemanticParserConfig",
    "SemanticRepair",
    "SemanticRule",
    "SemanticRuleVariable",
    "SourceDocument",
    "SourceQuote",
    "TransformersBackend",
    "build_distilled_semantic_parser",
    "build_reference_semantic_parser",
    "create_backend",
    "default_model_config_path",
    "load_model_profile",
    "normalize_extraction_payload",
    "normalize_symbol",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)

    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
