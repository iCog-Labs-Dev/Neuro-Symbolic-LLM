"""Prepare source-linked, right-padded occurrence/template training pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

FACTUALITY_POLICY = (
    "Exclude hypothetical-only assertion bindings; retain explicit rule-structure "
    "bindings; retain and count unclassified factuality"
)


def hypothetical_assertion_only(binding: Mapping[str, Any]) -> bool:
    """Do not treat a hypothetical assertion as an independently asserted fact."""
    evidence = binding["evidence"]
    return (
        bool(evidence)
        and not any(row.get("kind") == "rule" for row in evidence)
        and all(row.get("factuality") == "hypothetical" for row in evidence)
    )


def qualified_pattern_ids(patterns: Any) -> dict[str, str]:
    """Reuse IDs of exact qualified bank patterns without recomputing miner IDs."""
    identifiers: dict[str, str] = {}
    for pattern in patterns:
        trace = pattern.expensive_trace
        qualified = (
            trace.get("qualified_pattern") if isinstance(trace, Mapping) else None
        )
        if not isinstance(qualified, str) or not qualified.strip():
            raise ValueError("Bank patterns require their recorded qualified pattern")
        if qualified in identifiers and identifiers[qualified] != pattern.id:
            raise ValueError("A qualified bank pattern has conflicting IDs")
        identifiers[qualified] = pattern.id
    return identifiers


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    for name in ("sources", "documents", "bundle", "patterns", "embeddings", "output"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--exact", type=Path, nargs="+")
    parser.add_argument("--model")
    parser.add_argument(
        "--group-positives",
        action="store_true",
        default=None,
        help="Balance query locations across documents and retain every known visible positive",
    )
    for name in (
        "seq-len",
        "target-examples",
        "minimum-examples",
        "seed",
        "max-rows",
        "max-blocks",
    ):
        parser.add_argument(f"--{name}", type=int)
    args = parser.parse_args(argv)
    settings: dict[str, Any] = {}
    base = Path.cwd()
    if args.config:
        import yaml

        base = args.config.resolve().parent
        settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        permitted = {
            "training_config",
            "sources",
            "documents",
            "exact",
            "bundle",
            "patterns",
            "embeddings",
            "output",
            "model",
            "group_positives",
            "seq_len",
            "target_examples",
            "minimum_examples",
            "seed",
            "max_rows",
            "max_blocks",
        }
        if not isinstance(settings, dict) or set(settings) - permitted:
            parser.error("Invalid preparation configuration fields")
        if "training_config" in settings:
            training_path = (base / settings.pop("training_config")).resolve()
            training = yaml.safe_load(training_path.read_text(encoding="utf-8"))[
                "training"
            ]
            if "seq_len" in settings and settings["seq_len"] != training["seq_len"]:
                parser.error("Preparation and training sequence lengths differ")
            if args.seq_len is not None and args.seq_len != training["seq_len"]:
                parser.error("Preparation and training sequence lengths differ")
            settings.setdefault("seq_len", training["seq_len"])
            settings.setdefault("model", training["model"])
            settings.setdefault("seed", training.get("seed", 0))
    for name, value in settings.items():
        if getattr(args, name) is None:
            if name == "exact":
                value = [
                    (base / item).resolve()
                    for item in ([value] if isinstance(value, str) else value)
                ]
            elif name in {
                "sources",
                "documents",
                "bundle",
                "patterns",
                "embeddings",
                "output",
            }:
                value = (base / value).resolve()
            setattr(args, name, value)
    for name, default in (
        ("seq_len", 128),
        ("target_examples", 10000),
        ("minimum_examples", 5000),
        ("seed", 0),
        ("max_rows", 20000),
        ("max_blocks", 50000),
    ):
        if getattr(args, name) is None:
            setattr(args, name, default)
    args.sources = args.sources or args.documents
    if args.group_positives is None:
        args.group_positives = False
    if not isinstance(args.group_positives, bool):
        parser.error("group_positives must be boolean")
    for name in (
        "sources",
        "documents",
        "exact",
        "bundle",
        "patterns",
        "embeddings",
        "output",
        "model",
    ):
        if not getattr(args, name):
            parser.error(f"Missing --{name} or preparation configuration field")
    for name in (
        "seq_len",
        "target_examples",
        "minimum_examples",
        "max_rows",
        "max_blocks",
    ):
        value = getattr(args, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (2 if name == "seq_len" else 1)
        ):
            parser.error(f"Invalid {name}")
    if not args.minimum_examples <= args.target_examples <= args.max_blocks:
        parser.error("Require minimum_examples <= target_examples <= max_blocks")
    if isinstance(args.seed, bool) or not isinstance(args.seed, int):
        parser.error("seed must be an integer")
    if args.output.suffix != ".json":
        parser.error("output must be a fresh .json manifest path")
    return args


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = row["corpus_id"], row["document_id"]
            if (
                identity in rows
                or not isinstance(row["source_text"], str)
                or not row["source_text"].strip()
            ):
                raise ValueError("Sources require unique identities and nonempty text")
            rows[identity] = row
    if not rows:
        raise ValueError("No source documents")
    return rows


def exact_paths(inputs: list[Path]) -> list[Path]:
    """Accept exact files or completed-shard directories without a merge step."""
    paths = set()
    for path in inputs:
        if path.is_file():
            paths.add(path.resolve())
        elif path.is_dir():
            for result in sorted(path.glob("documents-*/result.json")):
                if (
                    json.loads(result.read_text(encoding="utf-8"))["status"]
                    == "completed"
                ):
                    exact = result.parent / "exact.jsonl"
                    if not exact.is_file():
                        raise ValueError(
                            f"Completed shard lacks exact matches: {exact}"
                        )
                    paths.add(exact.resolve())
        else:
            raise FileNotFoundError(path)
    if not paths:
        raise ValueError("No completed exact-match files")
    return sorted(paths)


def collect_examples(
    sources: Mapping[tuple[str, str], Mapping[str, Any]],
    tokenizer: Any,
    paths: list[Path],
    template_ids: set[str],
    pattern_id: Callable[[dict[str, Any]], str | None],
    seq_len: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], Any], dict[str, int]]:
    """Deduplicate by template, source, window and causal evidence endpoint."""
    encoded = {
        identity: tokenizer(row["source_text"], return_offsets_mapping=True)
        for identity, row in sources.items()
    }
    examples = {}
    counts: Counter[str] = Counter()
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                match = json.loads(line)
                template = pattern_id(match)
                if template not in template_ids:
                    continue
                for binding in match["exact_bindings"]:
                    identity = binding["corpus_id"], binding["document_id"]
                    if identity not in sources:
                        continue
                    text = sources[identity]["source_text"]
                    spans = [
                        span
                        for evidence in binding["evidence"]
                        for span in evidence["source_spans"]
                    ]
                    counts["source_bindings"] += 1
                    if not spans:
                        counts["without_source_spans"] += 1
                        continue
                    if any(
                        not isinstance(span["start"], int)
                        or isinstance(span["start"], bool)
                        or not isinstance(span["end"], int)
                        or isinstance(span["end"], bool)
                        or not (0 <= span["start"] < span["end"] <= len(text))
                        or text[span["start"] : span["end"]] != span["text"]
                        for span in spans
                    ):
                        raise ValueError("Exact evidence does not match source text")
                    if any(
                        row.get("factuality") is None for row in binding["evidence"]
                    ):
                        counts["unclassified_factuality_bindings"] += 1
                    if hypothetical_assertion_only(binding):
                        counts["hypothetical_only_bindings_excluded"] += 1
                        continue
                    left = min(span["start"] for span in spans)
                    right = max(span["end"] for span in spans)
                    tokens = encoded[identity]["input_ids"]
                    covered = [
                        i
                        for i, (start, end) in enumerate(
                            encoded[identity]["offset_mapping"]
                        )
                        if end > left and start < right
                    ]
                    if not covered:
                        counts["without_tokens"] += 1
                        continue
                    first, last = min(covered), max(covered)
                    start = max(0, last - seq_len + 1)
                    end = min(len(tokens), start + seq_len)
                    if first < start:
                        counts["evidence_wider_than_window"] += 1
                        continue
                    if end - start < 2:
                        counts["without_next_token_target"] += 1
                        continue
                    key = (template, identity, start, last - start)
                    if key in examples:
                        counts["duplicate_position_template_pairs"] += 1
                        continue
                    unique_spans = sorted(
                        {(span["start"], span["end"], span["text"]) for span in spans}
                    )
                    examples[key] = {
                        "corpus_id": identity[0],
                        "document_id": identity[1],
                        "template_id": template,
                        "token_index": last - start,
                        "window_start": start,
                        "window_end": end,
                        "source_spans": [
                            {"start": a, "end": b, "text": t}
                            for a, b, t in unique_spans
                        ],
                    }
    return list(examples.values()), encoded, dict(counts)


def select_examples(
    examples: list[dict[str, Any]],
    target: int,
    minimum: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Round-robin templates, without replacing or manufacturing occurrences."""
    if len(examples) < minimum:
        raise ValueError(
            f"Only {len(examples)} eligible unique pairs; minimum is {minimum}"
        )
    rng = random.Random(seed)
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        groups[example["template_id"]].append(example)
    names = sorted(groups)
    rng.shuffle(names)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    active = names
    while active and len(selected) < target:
        next_round = []
        for name in active:
            selected.append(groups[name].pop())
            if groups[name]:
                next_round.append(name)
            if len(selected) == target:
                break
        active = next_round
    return selected


def group_positive_queries(examples, encoded):
    """Collect all audited positives visible at each existing query endpoint.

    No future evidence, cross-document merging, new matches or artificial text
    variations are introduced. Absence from these supplied audits is not a
    certificate of semantic non-relevance.
    """
    by_document = defaultdict(list)
    for row in examples:
        identity = row["corpus_id"], row["document_id"]
        spans = row["source_spans"]
        left, right = min(s["start"] for s in spans), max(s["end"] for s in spans)
        covered = [
            i
            for i, (a, b) in enumerate(encoded[identity]["offset_mapping"])
            if b > left and a < right
        ]
        if not covered:
            raise ValueError("Positive evidence does not cover tokens")
        by_document[identity].append((min(covered), max(covered), row))
    result = []
    for identity in sorted(by_document):
        occurrences = by_document[identity]
        anchors = {
            (row["window_start"], row["window_end"], row["token_index"])
            for _, _, row in occurrences
        }
        for start, end, position in sorted(anchors):
            positives = {}
            for first, last, row in occurrences:
                if first >= start and last <= start + position:
                    previous = positives.get(row["template_id"])
                    if previous is None or last > previous[0]:
                        positives[row["template_id"]] = (last, row)
            if not positives:
                raise ValueError("A query must retain its source-derived positive")
            result.append(
                {
                    "corpus_id": identity[0],
                    "document_id": identity[1],
                    "window_start": start,
                    "window_end": end,
                    "token_index": position,
                    "positive_labels": [
                        {
                            "template_id": name,
                            "source_spans": positives[name][1]["source_spans"],
                        }
                        for name in sorted(positives)
                    ],
                }
            )
    return result


def select_query_examples(queries, target, minimum, seed):
    """Round-robin documents, then windows, then query locations; no replacement."""
    if len(queries) < minimum:
        raise ValueError(
            f"Only {len(queries)} eligible query locations; minimum is {minimum}"
        )
    rng = random.Random(seed)
    documents = defaultdict(lambda: defaultdict(list))
    for row in queries:
        documents[row["corpus_id"], row["document_id"]][
            row["window_start"], row["window_end"]
        ].append(row)
    ordered = {}
    for identity in sorted(documents):
        windows = documents[identity]
        names = sorted(windows)
        rng.shuffle(names)
        for rows in windows.values():
            rng.shuffle(rows)
        ordered[identity] = []
        while names:
            for name in names:
                ordered[identity].append(windows[name].pop())
            names = [name for name in names if windows[name]]
        ordered[identity].reverse()
    names = list(ordered)
    rng.shuffle(names)
    selected = []
    while names and len(selected) < target:
        for name in names:
            selected.append(ordered[name].pop())
            if len(selected) == target:
                break
        names = [name for name in names if ordered[name]]
    return selected


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    from hybrid_miner.ontology import OntologyBundle
    from transformers import AutoTokenizer

    from symbolic_memory.bank import load_bank

    arrays_path = args.output.with_suffix(".npz")
    if args.output.exists() or arrays_path.exists():
        raise FileExistsError("Choose fresh manifest and array output names")
    sources, documents = document_rows(args.sources), document_rows(args.documents)
    if len(sources) > args.max_rows:
        raise ValueError(
            "Source count exceeds max_rows; supply an explicit source subset"
        )
    if any(
        identity not in documents
        or row["source_text"] != documents[identity]["source_text"]
        for identity, row in sources.items()
    ):
        raise ValueError("Source texts and semantic documents differ")
    bank = load_bank(args.patterns, args.embeddings)
    lookup = {record.template_id: index for index, record in enumerate(bank.records)}
    qualified_ids = qualified_pattern_ids(bank.patterns)
    bundle = OntologyBundle.model_validate_json(args.bundle.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    pad = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad is None:
        raise ValueError("Tokenizer requires a pad or EOS token")
    paths = exact_paths(args.exact)

    def identify(match):
        if (match["bundle_id"], match["bundle_version"]) != (
            bundle.bundle_id,
            bundle.bundle_version,
        ):
            raise ValueError("Exact-match bundle identity differs")
        return qualified_ids.get(match["pattern"])

    eligible, encoded, counts = collect_examples(
        sources, tokenizer, paths, set(lookup), identify, args.seq_len
    )
    grouped = getattr(args, "group_positives", False)
    candidates = group_positive_queries(eligible, encoded) if grouped else eligible
    labels = (select_query_examples if grouped else select_examples)(
        candidates, args.target_examples, args.minimum_examples, args.seed
    )
    ids, masks, targets = [], [], []

    def positive_rows(row):
        return row["positive_labels"] if grouped else [row]

    max_positives = max(len(positive_rows(row)) for row in labels)
    for label in labels:
        source = encoded[label["corpus_id"], label["document_id"]]["input_ids"]
        tokens = source[label["window_start"] : label["window_end"]]
        ids.append(tokens + [pad] * (args.seq_len - len(tokens)))
        masks.append([True] * len(tokens) + [False] * (args.seq_len - len(tokens)))
        positives = [lookup[row["template_id"]] for row in positive_rows(label)]
        targets.append(
            [label["token_index"], *positives] + [-1] * (max_positives - len(positives))
        )
    coverage = {
        **counts,
        "eligible_pairs": len(eligible),
        "selected_pairs": len(labels),
        "target_examples": args.target_examples,
        "minimum_examples": args.minimum_examples,
        "target_met": len(labels) >= args.target_examples,
        "eligible_templates": len({row["template_id"] for row in eligible}),
        "selected_templates": len(
            {p["template_id"] for row in labels for p in positive_rows(row)}
        ),
        "eligible_query_locations": len(candidates) if grouped else None,
        "selected_positive_memberships": sum(len(positive_rows(row)) for row in labels),
        "maximum_positives_per_query": max_positives,
        "selection_unit": "query" if grouped else "pair",
        "selected_source_documents": len(
            {(row["corpus_id"], row["document_id"]) for row in labels}
        ),
        "distinct_token_contexts": len(
            {
                tuple(
                    encoded[row["corpus_id"], row["document_id"]]["input_ids"][
                        row["window_start"] : row["window_end"]
                    ]
                )
                for row in labels
            }
        ),
        "source_document_count": len(sources),
        "distinct_visible_query_prefixes": len(
            {
                tuple(
                    encoded[row["corpus_id"], row["document_id"]]["input_ids"][
                        row["window_start"] : row["window_start"]
                        + row["token_index"]
                        + 1
                    ]
                )
                for row in labels
            }
        ),
        "hypothetical_only_bindings_excluded": counts.get(
            "hypothetical_only_bindings_excluded", 0
        ),
        "unclassified_factuality_bindings": counts.get(
            "unclassified_factuality_bindings", 0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with arrays_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            input_ids=np.asarray(ids, dtype=np.int32),
            token_mask=np.asarray(masks, dtype=bool),
            targets=np.asarray(targets, dtype=np.int32),
        )
    payload = {
        "contract_version": (
            "alignment-query-sets-v1" if grouped else "alignment-occurrences-v2"
        ),
        "preparation_source_sha256": file_digest(Path(__file__).resolve()),
        "seq_len": args.seq_len,
        "model": args.model,
        "seed": args.seed,
        "bank_snapshot_sha256": bank.input_snapshot_sha256,
        "arrays_file": arrays_path.name,
        "arrays_sha256": file_digest(arrays_path),
        "source_sha256": {
            name: file_digest(getattr(args, name))
            for name in ("sources", "documents", "bundle", "patterns", "embeddings")
        },
        "exact_inputs": [
            {"path": str(path), "sha256": file_digest(path)} for path in paths
        ],
        "bank_source_text_sha256": sorted(
            {
                hashlib.sha256(row["source_text"].encode()).hexdigest()
                for row in documents.values()
            }
        ),
        "selection_policy": (
            "query-level known positives; seeded document/window round-robin; no replacement"
            if grouped
            else "unique template/source/window/endpoint pairs; seeded template round-robin; no replacement"
        ),
        "positive_set_scope": "All supplied audited matches whose complete evidence is visible at the query; not independent semantic completeness certification",
        "factuality_policy": FACTUALITY_POLICY,
        "source_semantic_correctness": "Source-span identity checked; semantic correctness requires independent audit",
        "coverage": coverage,
        "labels": labels,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return {**coverage, "output": str(args.output), "arrays": str(arrays_path)}


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(prepare(parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
