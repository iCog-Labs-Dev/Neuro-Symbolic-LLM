"""Integration tests for the export of parse results and the miner hand-off."""

# ruff: noqa: E402

import json
from pathlib import Path

import pytest

pytest.importorskip("hybrid_miner.contracts.semantic_document")

from hybrid_miner.contracts.semantic_document import SemanticDocument

from parser.ontology.catalog import load_ontology_catalog
from parser.semantic import catalog_document_cli, cli, model_config
from parser.semantic.backends import CallableBackend
from parser.semantic.catalog_document_export import export_catalog_documents
from parser.semantic.catalog_student import validate_catalog_student_pair
from parser.semantic.schema import SEMANTIC_CONTRACT_VERSION, SemanticParseResult
from parser.semantic.semantic_parser import (
    ModelGenerationError,
    ReferenceSemanticParser,
    SemanticParserConfig,
)
from symbolic_pipeline.document_export import export_semantic_documents
from symbolic_pipeline.miner_bundle import load_catalog_bundle
from tests.semantic_fixtures import target


def _source_line(
    document_id: str, source_text: str, corpus_id: str = "corpus-1"
) -> str:
    return json.dumps(
        {
            "document_id": document_id,
            "corpus_id": corpus_id,
            "source_text": source_text,
        },
        ensure_ascii=False,
    )


def _source_text(prompt: str) -> str:
    return prompt.rsplit("<source_text>\n", 1)[1].split("\n</source_text>", 1)[0]


def _answer(*, prompt: str, model: str) -> str:
    """Answer every prompt with a valid extraction of its own source text."""
    return json.dumps(target(_source_text(prompt)))


def _parser(generate=_answer, catalog=None, model_name="teacher-model"):
    return ReferenceSemanticParser(
        backend=CallableBackend(generate, provider_name="test-provider"),
        config=SemanticParserConfig(model_name=model_name),
        catalog=catalog,
    )


def test_exports_parse_results_and_auditable_rejections(tmp_path: Path) -> None:
    input_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "parsed.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    student_pairs_path = tmp_path / "student-pairs.jsonl"
    source_text = "  Mira left. 🙂  "
    input_path.write_text(
        "\n".join(
            [
                _source_line("doc-1", source_text),
                "not-json",
                _source_line("doc-1", "A duplicate."),
                _source_line("bad id", "An invalid identifier."),
                _source_line("doc-2", "Sol arrived."),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[str] = []

    def generate(*, prompt: str, model: str) -> str:
        calls.append(prompt)
        return _answer(prompt=prompt, model=model)

    result = export_catalog_documents(
        _parser(generate, catalog=load_ontology_catalog(modules=("upper",))),
        input_path=input_path,
        output_path=output_path,
        rejected_path=rejected_path,
        student_pairs_path=student_pairs_path,
    )

    assert (result.accepted, result.rejected) == (2, 3)
    assert len(calls) == 2
    accepted_lines = output_path.read_text(encoding="utf-8").splitlines()
    results = [SemanticParseResult.model_validate_json(line) for line in accepted_lines]
    assert [item.document.document_id for item in results] == ["doc-1", "doc-2"]
    assert results[0].document.source_text == source_text
    assert results[0].provenance.ontology_versions == ["upper-v1"]
    assert accepted_lines == [item.model_dump_json() for item in results]
    rejections = [
        json.loads(line)
        for line in rejected_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [rejection["line_number"] for rejection in rejections] == [2, 3, 4]
    assert "Duplicate source document" in rejections[1]["error"]
    assert rejections[0]["raw_line"] == "not-json"
    pairs = [
        json.loads(line)
        for line in student_pairs_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [pair["document_id"] for pair in pairs] == ["doc-1", "doc-2"]
    assert pairs[0]["contract_version"] == SEMANTIC_CONTRACT_VERSION
    assert json.loads(pairs[0]["label"]) == results[0].extraction.model_dump(
        mode="json"
    )
    assert pairs[0]["parser"] == results[0].provenance.model_dump(mode="json")


def test_rejects_colliding_paths_and_existing_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "source.jsonl"
    input_path.write_text(_source_line("doc-1", "Mira left.") + "\n", encoding="utf-8")
    parser = _parser()
    rejected_path = tmp_path / "rejected.jsonl"

    with pytest.raises(ValueError, match="must all differ"):
        export_catalog_documents(
            parser,
            input_path=input_path,
            output_path=input_path,
            rejected_path=rejected_path,
        )
    assert input_path.read_text(encoding="utf-8").startswith('{"document_id"')

    output_path = tmp_path / "parsed.jsonl"
    output_path.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Output already exists"):
        export_catalog_documents(
            parser,
            input_path=input_path,
            output_path=output_path,
            rejected_path=rejected_path,
        )
    assert output_path.read_text(encoding="utf-8") == "existing\n"


def test_backend_failure_stops_export_without_partial_output(tmp_path: Path) -> None:
    input_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "parsed.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    input_path.write_text(
        _source_line("doc-1", "Mira left.")
        + "\n"
        + _source_line("doc-2", "Sol arrived.")
        + "\n",
        encoding="utf-8",
    )

    def generate(*, prompt: str, model: str) -> str:
        if "Sol arrived." in prompt:
            raise ConnectionError("model server unavailable")
        return _answer(prompt=prompt, model=model)

    with pytest.raises(ModelGenerationError, match="parser failed"):
        export_catalog_documents(
            _parser(generate),
            input_path=input_path,
            output_path=output_path,
            rejected_path=rejected_path,
        )

    assert not output_path.exists()
    assert not rejected_path.exists()
    assert output_path.with_name(output_path.name + ".partial").exists()
    assert output_path.with_name(output_path.name + ".checkpoint.json").exists()
    assert input_path.read_text(encoding="utf-8").count("\n") == 2


def test_cli_uses_selected_catalog_and_reports_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "parsed.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    input_path.write_text(
        _source_line("doc-1", "Mira left.") + "\n" + "not-json\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        model_config, "create_backend", lambda profile: CallableBackend(_answer)
    )

    exit_code = catalog_document_cli.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--rejected-output",
            str(rejected_path),
            "--ontology-module",
            "upper",
        ]
    )

    assert exit_code == 2
    result = SemanticParseResult.model_validate_json(
        output_path.read_text(encoding="utf-8").strip()
    )
    assert result.provenance.ontology_versions == ["upper-v1"]
    assert len(rejected_path.read_text(encoding="utf-8").splitlines()) == 1


def test_cli_exports_results_with_the_selected_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(_source_line("doc-1", "Mira left.") + "\n", encoding="utf-8")
    selected = []

    def backend(profile):
        selected.append(profile.name)
        return CallableBackend(_answer)

    monkeypatch.setattr(model_config, "create_backend", backend)
    status = cli.main(
        [
            "--profile",
            "ollama_catalog",
            "export-documents",
            "--input",
            str(source),
            "--output",
            str(tmp_path / "results.jsonl"),
            "--rejected-output",
            str(tmp_path / "rejected.jsonl"),
            "--student-pairs-output",
            str(tmp_path / "pairs.jsonl"),
        ]
    )
    assert status == 0
    assert selected == ["ollama_catalog"]
    assert (tmp_path / "pairs.jsonl").read_text(encoding="utf-8")


def test_miner_export_includes_the_same_bundle_used_to_validate_documents(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(_source_line("doc-1", "Mira left.") + "\n", encoding="utf-8")
    catalog, bundle = load_catalog_bundle()
    export_catalog_documents(
        _parser(catalog=catalog),
        input_path=source,
        output_path=tmp_path / "results.jsonl",
        rejected_path=tmp_path / "rejected.jsonl",
    )
    result = export_semantic_documents(
        tmp_path / "results.jsonl",
        tmp_path / "documents.jsonl",
        tmp_path / "documents.rejected.jsonl",
        bundle=bundle,
        bundle_path=tmp_path / "bundle.json",
    )
    assert result.bundle_path is not None
    assert (
        result.bundle_path.read_text(encoding="utf-8") == bundle.canonical_json() + "\n"
    )


def test_build_dataset_label_matches_export_and_student_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(_source_line("doc-1", "Mira left.") + "\n", encoding="utf-8")
    draft = {
        "entities": [
            {"id": "Departure", "entity_type": " Event ", "mention": {"text": "left"}}
        ],
        "assertions": [
            {
                "predicate": " Event ",
                "arguments": [{"entity_id": "Departure", "role": "Event ID"}],
                "factuality": " Asserted ",
                "confidence": 0.9,
                "evidence": [{"text": "left"}],
            }
        ],
    }
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return json.dumps(draft)

    monkeypatch.setattr(
        model_config, "create_backend", lambda profile: CallableBackend(generate)
    )
    pairs = tmp_path / "pairs.jsonl"
    results = tmp_path / "results.jsonl"
    status = cli.main(
        [
            "build-dataset",
            "--target",
            SEMANTIC_CONTRACT_VERSION,
            "--input",
            str(source),
            "--output",
            str(results),
            "--student-pairs-output",
            str(pairs),
            "--rejected-output",
            str(tmp_path / "rejected.jsonl"),
            "--ontology-module",
            "upper",
        ]
    )
    assert status == 0
    assert len(calls) == 1
    pair = json.loads(pairs.read_text(encoding="utf-8"))
    _, label, rebuilt = validate_catalog_student_pair(
        pair, catalog=load_ontology_catalog(modules=("upper",))
    )
    assert rebuilt.model_dump_json() + "\n" == results.read_text(encoding="utf-8")
    assert json.loads(label) == json.loads(pair["label"])
    assertion = json.loads(label)["assertions"][0]
    assert assertion["predicate"] == "Event"
    assert assertion["factuality"] == "asserted"
    assert assertion["arguments"] == [{"entity_id": "departure", "role": "event_id"}]


def test_dataset_keeps_complete_results_with_a_filtered_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = target("Mira left.")
    payload["assertions"][0]["factuality"] = "speculative"
    monkeypatch.setattr(
        model_config,
        "create_backend",
        lambda _: CallableBackend(lambda **__: json.dumps(payload)),
    )
    (tmp_path / "source.jsonl").write_text(
        _source_line("one", "Mira left.") + "\n", encoding="utf-8"
    )
    assert (
        cli.main(
            [
                "build-dataset",
                "--input",
                str(tmp_path / "source.jsonl"),
                "--output",
                str(tmp_path / "results.jsonl"),
                "--rejected-output",
                str(tmp_path / "rejected.jsonl"),
                "--asserted-only",
            ]
        )
        == 0
    )
    result = SemanticParseResult.model_validate_json(
        (tmp_path / "results.jsonl").read_text(encoding="utf-8")
    )
    assert result.extraction.assertions[0].factuality == "speculative"
    metadata = json.loads(
        (tmp_path / "results.metadata.jsonl").read_text(encoding="utf-8")
    )
    assert metadata["metta"] == []
    assert metadata["metta_asserted_only"] is True


def test_resume_preserves_work_and_does_not_repeat_completed_calls(tmp_path):
    calls = []
    available = False

    def generate(*, prompt, model):
        text = _source_text(prompt)
        calls.append(text)
        if text == "Second." and not available:
            raise ConnectionError("offline")
        return json.dumps(target(text))

    parser = _parser(generate, model_name="resume-review")
    source = tmp_path / "source.txt"
    source.write_text("First.\nSecond.\n", encoding="utf-8")
    options = {
        "input_path": source,
        "output_path": tmp_path / "results.jsonl",
        "rejected_path": tmp_path / "rejected.jsonl",
        "student_pairs_path": tmp_path / "pairs.jsonl",
    }
    with pytest.raises(ModelGenerationError):
        export_catalog_documents(parser, **options)
    checkpoint = tmp_path / "results.jsonl.checkpoint.json"
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["accepted"] == 1
    assert not options["output_path"].exists()
    available = True
    result = export_catalog_documents(parser, **options, resume=True)
    assert result.accepted == 2
    assert calls == ["First.", "Second.", "Second."]
    assert not checkpoint.exists()
    assert len(options["output_path"].read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.parametrize("changed", ["source", "artifact"])
def test_resume_rejects_changed_inputs_before_generation(tmp_path, changed):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        raise ConnectionError("offline")

    parser = _parser(generate, model_name="resume-review")
    source = tmp_path / "source.txt"
    source.write_text("First.\n", encoding="utf-8")
    options = {
        "input_path": source,
        "output_path": tmp_path / "results.jsonl",
        "rejected_path": tmp_path / "rejected.jsonl",
    }
    with pytest.raises(ModelGenerationError):
        export_catalog_documents(parser, **options)
    if changed == "source":
        source.write_text("Changed.\n", encoding="utf-8")
    else:
        partial = tmp_path / "results.jsonl.partial"
        partial.write_text("corrupt", encoding="utf-8")
        state_path = tmp_path / "results.jsonl.checkpoint.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["files"]["documents"]["size"] = 7
        state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint"):
        export_catalog_documents(parser, **options, resume=True)
    assert len(calls) == 1


def test_exported_results_convert_to_valid_miner_documents(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        _source_line("doc-1", "Mira left.")
        + "\n"
        + _source_line("doc-2", "Sol sang.")
        + "\n",
        encoding="utf-8",
    )
    catalog, bundle = load_catalog_bundle()
    export_catalog_documents(
        _parser(catalog=catalog),
        input_path=source,
        output_path=tmp_path / "results.jsonl",
        rejected_path=tmp_path / "rejected.jsonl",
    )

    converted = export_semantic_documents(
        tmp_path / "results.jsonl",
        tmp_path / "documents.jsonl",
        tmp_path / "documents.rejected.jsonl",
        bundle=bundle,
    )

    assert (converted.accepted, converted.rejected) == (2, 0)
    documents = [
        SemanticDocument.model_validate_json(line)
        for line in (tmp_path / "documents.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [document.document_id for document in documents] == ["doc-1", "doc-2"]
    assert documents[0].assertions[0].predicate == "Event"
    assert documents[0].parser.ontology_versions == catalog.versions
