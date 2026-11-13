"""Prepare causal, source-disjoint probes from admitted reference documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.prepare_alignment_labels import (
    FACTUALITY_POLICY,
    hypothetical_assertion_only,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def read_documents(paths: list[Path]) -> list[dict[str, Any]]:
    documents = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            identity = row["corpus_id"], row["document_id"]
            if identity in documents:
                raise ValueError("Duplicate document identity")
            if (
                not isinstance(row["source_text"], str)
                or not row["source_text"].strip()
            ):
                raise ValueError("Documents require nonempty source text")
            documents[identity] = row
    return list(documents.values())


def held_out_documents(documents, excluded, bank_document_ids):
    """Exclude complete documents by identity and exact source-text digest."""
    identities = {(row["corpus_id"], row["document_id"]) for row in excluded}
    hashes = {text_digest(row["source_text"]) for row in excluded}
    seen = set()
    result = []
    for row in documents:
        identity = row["corpus_id"], row["document_id"]
        sha = text_digest(row["source_text"])
        if (
            identity in identities
            or row["document_id"] in bank_document_ids
            or sha in hashes
        ):
            continue
        if sha in seen:
            raise ValueError("Held-out documents contain duplicate source text")
        seen.add(sha)
        result.append(row)
    if not result:
        raise ValueError("No source-disjoint documents remain")
    return result


def possible_clauses(clauses, facts):
    """Reject only clauses whose constants/arity cannot match any ground fact."""
    return all(
        any(
            len(clause) == len(fact)
            and all(
                token.startswith("$") or token == value
                for token, value in zip(clause, fact, strict=True)
            )
            for fact in facts
        )
        for clause in clauses
    )


def causal_probes(document, tokenizer, bindings, seq_len):
    """Label all complete bindings visible at each annotated sentence endpoint."""
    text = document["source_text"]
    encoded = tokenizer(text, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    ends = {
        span["end"]
        for row in [*document["assertions"], *document.get("rules", [])]
        for span in row["source_spans"]
    }
    resolved = []
    for template_id, binding in bindings:
        spans = [
            span
            for evidence in binding["evidence"]
            for span in evidence["source_spans"]
        ]
        if not spans:
            raise ValueError("A reference binding lacks source evidence")
        if any(
            not (0 <= s["start"] < s["end"] <= len(text))
            or text[s["start"] : s["end"]] != s["text"]
            for s in spans
        ):
            raise ValueError("Reference evidence differs from source text")
        if hypothetical_assertion_only(binding):
            continue
        resolved.append(
            (template_id, min(s["start"] for s in spans), max(s["end"] for s in spans))
        )
    probes = []
    seen = set()
    for end in sorted(ends):
        # Do not include a token straddling into text beyond the evidence endpoint.
        visible = [i for i, (a, b) in enumerate(offsets) if b > a and b <= end]
        if not visible:
            continue
        last = max(visible)
        start = max(0, last - seq_len + 1)
        if last < 1 or (start, last) in seen:
            continue
        seen.add((start, last))
        left, right = offsets[start][0], offsets[last][1]
        relevant = sorted({name for name, a, b in resolved if left <= a and b <= right})
        probes.append(
            {
                "corpus_id": document["corpus_id"],
                "document_id": document["document_id"],
                "source_text_sha256": text_digest(text),
                "window_start": start,
                "window_end": last + 1,
                "token_index": last - start,
                "input_ids": ids[start : last + 1],
                "relevant_ids": relevant,
            }
        )
    return probes


def prepare(config_path: Path) -> Path:
    import inspect

    import yaml
    from hybrid_miner.emission.emit_petta import emit_petta
    from hybrid_miner.ingestion.petta_gate import gate_emitted_facts
    from hybrid_miner.mining import exact_matches as exact
    from hybrid_miner.ontology import OntologyBundle
    from transformers import AutoTokenizer

    from symbolic_memory.bank import load_bank

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    required = {
        "documents",
        "exclude_sources",
        "bundle",
        "patterns",
        "embeddings",
        "model",
        "seq_len",
        "petta_root",
        "output",
        "split_policy",
    }
    if (
        not isinstance(cfg, dict)
        or not required.issubset(cfg)
        or set(cfg) - required - {"swipl"}
    ):
        raise ValueError(f"Probe configuration requires {sorted(required)}")
    base = config_path.resolve().parent

    def paths(key):
        if not isinstance(cfg[key], list) or not cfg[key]:
            raise ValueError(f"{key} must list explicit source files")
        return [(base / p).resolve() for p in cfg[key]]

    document_paths, exclusions = paths("documents"), paths("exclude_sources")
    bundle_path, patterns, embeddings, petta, output = (
        (base / cfg[k]).resolve()
        for k in ("bundle", "patterns", "embeddings", "petta_root", "output")
    )
    seq_len = cfg["seq_len"]
    if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len < 2:
        raise ValueError("seq_len must be an integer >= 2")
    if not isinstance(cfg["split_policy"], str) or not cfg["split_policy"].strip():
        raise ValueError("Describe the split origin explicitly")
    swipl = cfg.get("swipl", "swipl")
    if not isinstance(swipl, str) or not shutil.which(swipl):
        raise ValueError("Configure an existing SWI-Prolog executable with swipl")
    bank = load_bank(patterns, embeddings)
    bundle = OntologyBundle.model_validate_json(bundle_path.read_text())
    documents = held_out_documents(
        read_documents(document_paths),
        [row for path in exclusions for row in read_documents([path])],
        {name for row in bank.patterns for name in row.provenance.source_document_ids},
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], use_fast=True)
    output.mkdir(parents=True, exist_ok=False)
    status = output / "status.json"
    status.write_text(json.dumps({"status": "running"}))
    try:
        parsed = {}
        for index, pattern in enumerate(bank.patterns):
            qualified = pattern.expensive_trace["qualified_pattern"]
            identity = json.dumps(
                ["pattern-id-v1", bundle.bundle_id, bundle.bundle_version, qualified],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if pattern.id != "pattern-" + hashlib.sha256(identity.encode()).hexdigest():
                raise ValueError("Qualified oracle pattern differs from the bank ID")
            clauses, variables = exact._pattern_clauses(qualified, bundle)
            parsed[index] = qualified, clauses, variables
        probes = []
        counts: Counter[str] = Counter()
        with (output / "bindings.jsonl").open("x", encoding="utf-8") as oracle:
            for index, row in enumerate(documents):
                folder = output / f"document-{index:04d}"
                folder.mkdir()
                source = folder / "documents.jsonl"
                source.write_text(json.dumps(row) + "\n", encoding="utf-8")
                emission = emit_petta(
                    bundle=bundle,
                    input_path=source,
                    atoms_path=folder / "facts.metta",
                    metadata_path=folder / "metadata.metta",
                    provenance_path=folder / "provenance.jsonl",
                )
                admission = folder / "accepted.admission.json"
                gate = gate_emitted_facts(
                    bundle=bundle,
                    documents_path=source,
                    atoms_path=emission.atoms_path,
                    metadata_path=emission.metadata_path,
                    provenance_path=emission.provenance_path,
                    petta_root=petta,
                    accepted_path=folder / "accepted.metta",
                    rejected_path=folder / "rejected.jsonl",
                    admission_path=admission,
                    swipl=swipl,
                    timeout_seconds=120,
                )
                if gate.rejected_documents or gate.accepted_documents != 1:
                    raise ValueError("Held-out document did not pass PeTTa admission")
                facts = exact._facts(gate.atoms_path)
                reference = exact._documents(source, bundle)
                evidence = exact._evidence_index(
                    emission.provenance_path, facts, reference, bundle
                )
                flat_facts = [exact.parse_expression(fact) for fact in facts]
                queries = {
                    i: (qualified, variables)
                    for i, (qualified, clauses, variables) in parsed.items()
                    if possible_clauses(clauses, flat_facts)
                }
                matched = (
                    exact._query_bindings(
                        queries,
                        atoms_path=gate.atoms_path,
                        petta_root=petta,
                        swipl=swipl,
                        timeout_seconds=120,
                    )
                    if queries
                    else {}
                )
                linked = []
                for i, values_list in matched.items():
                    if len(set(values_list)) != len(values_list):
                        raise ValueError("Duplicate exact bindings")
                    for values in values_list:
                        try:
                            binding = exact._linked_binding(
                                parsed[i][1],
                                parsed[i][2],
                                values,
                                facts=facts,
                                evidence=evidence,
                                documents=reference,
                            )
                        except exact._ReusedGroundFact:
                            counts["reused_fact_bindings_excluded"] += 1
                            continue
                        if any(
                            evidence.get("factuality") is None
                            for evidence in binding["evidence"]
                        ):
                            counts["unclassified_factuality_bindings"] += 1
                        if hypothetical_assertion_only(binding):
                            counts["hypothetical_only_bindings_excluded"] += 1
                            continue
                        name = bank.patterns[i].id
                        linked.append((name, binding))
                        oracle.write(
                            json.dumps(
                                {"template_id": name, "binding": binding},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                selected = causal_probes(row, tokenizer, linked, seq_len)
                probes.extend(selected)
                counts.update(
                    documents=1,
                    bank_pattern_checks=len(parsed),
                    petta_pattern_queries=len(queries),
                    constant_or_arity_rejections=len(parsed) - len(queries),
                    exact_bindings=len(linked),
                    probes=len(selected),
                    positive_probes=sum(bool(p["relevant_ids"]) for p in selected),
                )
                print(
                    f"Reference document {index + 1}/{len(documents)}: {len(selected)} causal probes",
                    flush=True,
                )
        if not counts["positive_probes"]:
            raise ValueError("No held-out probe has a bank-template firing")
        payload = {
            "contract_version": "a4-retrieval-probes-v1",
            "split": "validation",
            "split_policy": cfg["split_policy"],
            "seq_len": seq_len,
            "model": cfg["model"],
            "bank_snapshot": bank.input_snapshot_sha256,
            "label_kind": "reference_graph_firing",
            "oracle": "All bank patterns checked against admitted annotated graphs; exact PeTTa bindings; distinct ground facts; visible evidence only",
            "semantics": "Structural template firing, not executable rule entailment or an LLM parser accuracy measurement",
            "review_status": "Dataset-reference transcription; independent semantic audit not asserted",
            "factuality_policy": FACTUALITY_POLICY,
            "preparation_source_sha256": digest(Path(__file__)),
            "matcher_source_sha256": digest(Path(inspect.getfile(exact))),
            "inputs": [
                {"path": str(p), "sha256": digest(p)}
                for p in [
                    config_path,
                    bundle_path,
                    patterns,
                    embeddings,
                    *document_paths,
                    *exclusions,
                ]
            ],
            "bindings_sha256": digest(output / "bindings.jsonl"),
            "coverage": dict(counts),
            "documents": [
                {
                    "corpus_id": r["corpus_id"],
                    "document_id": r["document_id"],
                    "source_text": r["source_text"],
                    "text_sha256": text_digest(r["source_text"]),
                }
                for r in documents
            ],
            "probes": probes,
        }
        manifest: Path = output / "probes.json"
        manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        status.write_text(json.dumps({"status": "completed", "coverage": dict(counts)}))
        return manifest
    except BaseException as error:
        status.write_text(json.dumps({"status": "failed", "error": str(error)}))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    print(prepare(parser.parse_args().config))


if __name__ == "__main__":
    main()
