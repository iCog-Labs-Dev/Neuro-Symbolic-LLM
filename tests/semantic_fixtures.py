"""Synthetic contract fixtures, not extraction-quality ground truth."""

import json

from parser.ontology.catalog import load_ontology_catalog
from parser.semantic.backends import CallableBackend
from parser.semantic.catalog_student import make_student_pair
from parser.semantic.schema import (
    SEMANTIC_CONTRACT_VERSION,
    SemanticParseResult,
    SourceDocument,
)
from parser.semantic.semantic_parser import (
    ReferenceSemanticParser,
    SemanticParserConfig,
)
from parser.student.student_prompt import build_student_prompt_spec


def target(text):
    """Return a valid model extraction that marks the whole text as one event."""
    return {
        "entities": [
            {"id": "e1", "entity_type": "Event", "mention": {"text": text}},
        ],
        "assertions": [
            {
                "predicate": "Event",
                "arguments": [{"entity_id": "e1", "role": "event_id"}],
                "factuality": "asserted",
                "confidence": 0.95,
                "evidence": [{"text": text}],
            }
        ],
        "rules": [],
    }


def result(text="Mira left.", *, catalog=None) -> SemanticParseResult:
    """Parse text with a fixed general-extraction response."""
    parser = ReferenceSemanticParser(
        backend=CallableBackend(
            lambda **_: json.dumps(target(text)),
            provider_name="fixture",
        ),
        config=SemanticParserConfig(
            model_name="fixture",
            enable_routing=False,
        ),
        catalog=catalog,
    )
    return parser.parse_document(
        SourceDocument(
            document_id="fixture",
            corpus_id="fixtures",
            source_text=text,
        )
    )


def pair(text="Mira left.", *, catalog=None):
    """Return the student pair the export writes for text."""
    return make_student_pair(result(text, catalog=catalog))


def write_checkpoint_metadata(path):
    """Write metadata for a compatible combined-prompt student checkpoint."""
    catalog = load_ontology_catalog()

    prompt = build_student_prompt_spec(
        "Checkpoint.",
        catalog=catalog,
    )

    (path / "training_metadata.json").write_text(
        json.dumps(
            {
                "target_contract_version": SEMANTIC_CONTRACT_VERSION,
                "prompt_format": "catalog-combined-compact-v1",
                "prompt_version": prompt.version,
                "ontology_versions": list(catalog.versions),
            }
        ),
        encoding="utf-8",
    )
