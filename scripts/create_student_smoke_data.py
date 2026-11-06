"""Generate small schema-validated student training data."""

import json
from pathlib import Path

from parser.ontology.catalog import load_manifest_catalog
from parser.semantic.catalog_prompt import build_catalog_document_prompt
from parser.semantic.catalog_student import (
    make_student_pair,
    validate_catalog_student_pair,
)
from parser.semantic.schema import (
    SemanticAssertion,
    SemanticExtraction,
    SemanticParseProvenance,
    SemanticParseResult,
    SourceDocument,
)

OUTPUT = Path("data/student_smoke")
OUTPUT.mkdir(parents=True, exist_ok=True)

catalog, _ = load_manifest_catalog(
    Path("parser/ontology/bundles/semantic-extraction-v1.yaml")
)

# Minimal synthetic examples for pipeline verification.
# They are not teacher-generated annotations.
samples = [
    ("Rain falls.", "fall", "Rain falls."),
    ("Birds sing.", "sing", "Birds sing."),
    ("The baby sleeps.", "sleep", "The baby sleeps."),
    ("The door opens.", "open", "The door opens."),
    ("The engine starts.", "start", "The engine starts."),
    ("The bell rings.", "ring", "The bell rings."),
    ("The candle burns.", "burn", "The candle burns."),
    ("The river flows.", "flow", "The river flows."),
]


def create_pair(index, source_text, relation, evidence):
    document = SourceDocument(
        document_id=f"student-smoke-{index:03d}",
        corpus_id="student-smoke",
        source_text=source_text,
    )

    prompt = build_catalog_document_prompt(
        source_text,
        catalog=catalog,
    )

    extraction = SemanticExtraction(
        entities=[],
        assertions=[
            SemanticAssertion(
                predicate="Evaluation",
                relation=relation,
                arguments=[],
                fallback=True,
                confidence=1.0,
                evidence=[{"text": evidence, "occurrence": 0}],
            )
        ],
        rules=[],
    )

    result = SemanticParseResult(
        document=document,
        provenance=SemanticParseProvenance(
            provider="synthetic",
            model="student-smoke-fixture",
            parser_version="student-smoke-v1",
            prompt_version=prompt.version,
            ontology_versions=list(catalog.versions),
        ),
        extraction=extraction,
    )

    pair = make_student_pair(result)

    # Reject incompatible fixtures before starting training.
    validate_catalog_student_pair(pair, catalog=catalog)
    return pair


pairs = [create_pair(i, *sample) for i, sample in enumerate(samples, start=1)]

for name, subset in (
    ("train.jsonl", pairs[:6]),
    ("val.jsonl", pairs[6:]),
):
    path = OUTPUT / name
    with path.open("w", encoding="utf-8") as file:
        for pair in subset:
            file.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Created {path}: {len(subset)} examples")

print("All smoke examples passed schema and catalog validation.")
