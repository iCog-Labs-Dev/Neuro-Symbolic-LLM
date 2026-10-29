"""Synthetic contract fixtures, not extraction-quality ground truth."""

import json

from parser.ontology.catalog import load_ontology_catalog
from parser.semantic.backends import CallableBackend
from parser.semantic.catalog_prompt import build_catalog_document_prompt
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
    """Parse ``text`` with a fixed model answer, as the export would."""
    parser = ReferenceSemanticParser(
        backend=CallableBackend(
            lambda **_: json.dumps(target(text)), provider_name="fixture"
        ),
        config=SemanticParserConfig(model_name="fixture"),
        catalog=catalog,
    )
    return parser.parse_document(
        SourceDocument(document_id="fixture", corpus_id="fixtures", source_text=text)
    )


def pair(text="Mira left.", *, catalog=None):
    """Return the student pair the export writes for ``text``."""
    return make_student_pair(result(text, catalog=catalog))


def write_checkpoint_metadata(path):
    """Write the training metadata a compatible student checkpoint carries."""
    catalog = load_ontology_catalog()
    (path / "training_metadata.json").write_text(
        json.dumps(
            {
                "target_contract_version": SEMANTIC_CONTRACT_VERSION,
                "prompt_version": build_catalog_document_prompt(
                    "Checkpoint.", catalog=catalog
                ).version,
                "ontology_versions": list(catalog.versions),
            }
        ),
        encoding="utf-8",
    )
