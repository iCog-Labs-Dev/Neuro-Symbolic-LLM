"""Public API for reference and distilled semantic parsing."""

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
from parser.semantic.semantic_parser import (
    ALLOWED_PREDICATES,
    DistilledSemanticParser,
    ModelGenerationError,
    ReferenceSemanticParser,
    SemanticParseError,
    SemanticParserConfig,
)

__all__ = [
    "ALLOWED_PREDICATES",
    "AnthropicBackend",
    "CallableBackend",
    "DistillationRecord",
    "DistilledSemanticParser",
    "GeminiBackend",
    "ModelBackend",
    "ModelProfile",
    "ModelGenerationError",
    "OllamaBackend",
    "OpenAIBackend",
    "OpenAICompatibleBackend",
    "RejectedRecord",
    "SemanticParseError",
    "SemanticParserConfig",
    "SemanticDatasetBuilder",
    "ReferenceSemanticParser",
    "TransformersBackend",
    "build_reference_semantic_parser",
    "build_distilled_semantic_parser",
    "create_backend",
    "default_model_config_path",
    "load_model_profile",
]
