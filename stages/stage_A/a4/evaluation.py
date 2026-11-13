"""Held-out reference-template retrieval and matched frozen-A3 evaluation."""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

from metrics.a4.retrieval import retrieval_metrics
from stages.stage_A.a4.replay import digest, read_json


def validate_probes(payload, replay, tokenizer):
    """Check bank, causal token windows, relevance IDs and source separation."""
    if (
        payload.get("contract_version") != "a4-retrieval-probes-v1"
        or payload.get("split") != "validation"
    ):
        raise ValueError("Expected a validation reference-probe manifest")
    if payload.get("label_kind") != "reference_graph_firing" or not payload.get(
        "review_status"
    ):
        raise ValueError("Reference label origin and review status are required")
    if (
        payload.get("bank_snapshot") != replay.bank.input_snapshot_sha256
        or payload.get("seq_len") != replay.config.training.seq_len
        or payload.get("model") != replay.config.training.model
    ):
        raise ValueError("Probe bank, sequence length or tokenizer model differs")
    for artifact in payload["inputs"]:
        if digest(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError("Probe preparation input changed")
    training = read_json(replay.arguments.alignment_labels)
    forbidden_text = set(training.get("bank_source_text_sha256", []))
    training_ids = set()
    for line in replay.arguments.train_sources.read_text(
        encoding="utf-8-sig"
    ).splitlines():
        if line.strip():
            row = json.loads(line)
            training_ids.add(row["document_id"])
            forbidden_text.add(hashlib.sha256(row["source_text"].encode()).hexdigest())
    forbidden_ids = {
        name
        for pattern in replay.bank.patterns
        for name in pattern.provenance.source_document_ids
    }
    forbidden_ids.update(training_ids)
    documents, encoded = {}, {}
    for row in payload["documents"]:
        identity = row["corpus_id"], row["document_id"]
        sha = hashlib.sha256(row["source_text"].encode()).hexdigest()
        if (
            sha != row["text_sha256"]
            or sha in forbidden_text
            or row["document_id"] in forbidden_ids
        ):
            raise ValueError("Probe document overlaps training/bank sources or changed")
        if identity in documents:
            raise ValueError("Duplicate probe document")
        documents[identity] = row
        encoded[identity] = tokenizer(row["source_text"])["input_ids"]
    known = {record.template_id for record in replay.bank.records}
    seen = set()
    probes = payload["probes"]
    if not probes:
        raise ValueError("No retrieval probes")
    for row in probes:
        identity = row["corpus_id"], row["document_id"]
        source = documents[identity]
        start, end = row["window_start"], row["window_end"]
        if any(
            isinstance(v, bool) or not isinstance(v, int)
            for v in (start, end, row["token_index"])
        ) or not (0 <= start < end <= len(encoded[identity])):
            raise ValueError("Invalid probe window")
        tokens = encoded[identity][start:end]
        if (
            not (2 <= len(tokens) <= replay.config.training.seq_len)
            or row["input_ids"] != tokens
            or row["token_index"] != len(tokens) - 1
            or row["source_text_sha256"] != source["text_sha256"]
        ):
            raise ValueError("Probe tokens, endpoint or source identity differ")
        relevant = row["relevant_ids"]
        if (
            not isinstance(relevant, list)
            or len(set(relevant)) != len(relevant)
            or not set(relevant).issubset(known)
        ):
            raise ValueError("Unknown or duplicate relevance labels")
        key = (*identity, start, end)
        if key in seen:
            raise ValueError("Duplicate probe position")
        seen.add(key)
    return probes


def score_rankings(predictions, relevance, cutoffs):
    """Retain known-negative probes; report positive-only recall explicitly."""
    if len(predictions) != len(relevance) or not predictions:
        raise ValueError("Predictions and reference probes must align")
    positive = [i for i, truth in enumerate(relevance) if truth]
    negative = [i for i, truth in enumerate(relevance) if not truth]
    if not positive:
        raise ValueError("No positive reference probes")
    reports = []
    for k in cutoffs:
        if (
            isinstance(k, bool)
            or not isinstance(k, int)
            or k < 1
            or any(len(row) < k for row in predictions)
        ):
            raise ValueError("Requested cutoff exceeds measured retrieval depth")
        scores = asdict(
            retrieval_metrics(
                [predictions[i][:k] for i in positive], [relevance[i] for i in positive]
            )
        )
        total_returned = sum(len(row[:k]) for row in predictions)
        reports.append(
            {
                "top_k": k,
                "positive_probes": len(positive),
                "known_negative_probes": len(negative),
                "positive_only": scores,
                "all_probe_micro_precision": scores["true_positive_count"]
                / total_returned,
                "negative_probes_with_returns": sum(
                    bool(predictions[i][:k]) for i in negative
                ),
            }
        )
    return reports


def score_support_prior(patterns, relevance, cutoffs):
    """Rank by original bank-source support without using held-out labels."""
    ranked = []
    for pattern in patterns:
        bindings = pattern.expensive_trace["binding_types"]
        if not isinstance(bindings, list):
            raise ValueError("Support prior requires recorded qualifying bindings")
        ranked.append((len(bindings), pattern.id))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    ids = [name for _, name in ranked[: max(cutoffs)]]
    return {
        "condition": "bank_support_prior",
        "origin": "Original bank-source qualifying-binding counts; deterministic ID tie break; no held-out label fitting",
        "top_ids": ids,
        "top_support_counts": [count for count, _ in ranked[: max(cutoffs)]],
        "scores": score_rankings([ids for _ in relevance], relevance, cutoffs),
    }


def measure_retrieval(
    run: Path,
    probes_path: Path,
    *,
    output: Path | None = None,
    native_library: Path | None = None,
    allow_source_change: bool = False,
    substrate: Any = None,
) -> Path:
    from stages.stage_A.a4.benchmark import require_idle_gpu

    require_idle_gpu()
    run, probes_path = run.resolve(strict=True), probes_path.resolve(strict=True)
    payload = read_json(probes_path)
    if digest(probes_path.parent / "bindings.jsonl") != payload["bindings_sha256"]:
        raise ValueError("Exact reference bindings changed")
    output = output or run / "evaluation" / time.strftime(
        "retrieval-%Y%m%d-%H%M%S", time.gmtime()
    )
    if output.exists():
        raise FileExistsError("Choose a fresh retrieval output directory")

    import jax
    import jax.numpy as jnp
    import numpy as np

    from stages.results import RunWriter
    from stages.stage_A.a4.replay import checkpoint_replay

    with ExitStack() as runtime:
        replay = runtime.enter_context(
            checkpoint_replay(
                run,
                native_library=native_library,
                allow_source_change=allow_source_change,
                substrate=substrate,
            )
        )
        tokenizer = replay.model.continuous.sub.tokenizer
        probes = validate_probes(payload, replay, tokenizer)
        batch_size, seq_len = (
            replay.config.training.batch_size,
            replay.config.training.seq_len,
        )
        pad = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        if pad is None:
            raise ValueError("Tokenizer requires a padding or EOS token")
        layers = replay.model.symbolic.layers
        retrieve = jax.jit(replay.retriever)

        @jax.jit
        def forward(symbolic, ids, mask, positions):
            continuous = replay.model.continuous
            clean, _ = continuous.clean_pass(ids)
            latents = continuous.predict_latents(replay.frozen, clean)
            result, states = replay.model.forward_with_states_at_latents(
                replay.frozen, symbolic, ids, latents
            )
            labels = jnp.where(mask, ids, -100)
            losses = jax.vmap(
                lambda logits, targets: continuous.sub.compute_loss(
                    logits[None], targets[None], losses=("CE",)
                )
            )(result.logits, labels)
            queries = {
                layer: replay.model.symbolic.project(
                    symbolic[f"layer_{layer}"], states[layer]
                )[jnp.arange(batch_size), positions]
                for layer in layers
            }
            return losses, queries

        initial = replay.model.symbolic.init_params(replay.config.training.seed)
        conditions = (("initial", initial), ("trained", replay.params["symbolic"]))
        cutoffs = [k for k in (1, 4, 8, 16) if k <= replay.config.top_m]
        with RunWriter(output.parent, output.name) as writer:
            writer.write_config(
                {
                    "run": str(run),
                    "probes": str(probes_path),
                    "native_library": str(native_library),
                    "allow_source_change": allow_source_change,
                }
            )
            writer.write_json("status.json", {"stage": "A4", "status": "running"})
            try:
                reports, raw = [], []
                for name, params in conditions:
                    rankings: dict[int, list[list[str]]] = {
                        layer: [] for layer in layers
                    }
                    total_loss, total_tokens = 0.0, 0
                    print(
                        f"Evaluating {name} head: {len(probes)} reference probes.",
                        flush=True,
                    )
                    for start in range(0, len(probes), batch_size):
                        selected = probes[start : start + batch_size]
                        count = len(selected)
                        padded = selected + [selected[-1]] * (batch_size - count)
                        ids = jnp.asarray(
                            [
                                p["input_ids"] + [pad] * (seq_len - len(p["input_ids"]))
                                for p in padded
                            ],
                            dtype=jnp.int32,
                        )
                        mask = jnp.asarray(
                            [
                                [True] * len(p["input_ids"])
                                + [False] * (seq_len - len(p["input_ids"]))
                                for p in padded
                            ]
                        )
                        positions = jnp.asarray(
                            [p["token_index"] for p in padded], dtype=jnp.int32
                        )
                        losses, queries = jax.block_until_ready(
                            forward(params, ids, mask, positions)
                        )
                        for i, p in enumerate(selected):
                            tokens = len(p["input_ids"]) - 1
                            total_loss += float(losses[i]) * tokens
                            total_tokens += tokens
                        for layer in layers:
                            matches = jax.device_get(retrieve(queries[layer]))
                            indices = np.asarray(matches.template_indices)
                            if (
                                indices.shape != (batch_size, replay.config.top_m)
                                or np.any(indices < 0)
                                or np.any(indices >= len(replay.bank.records))
                            ):
                                raise ValueError(
                                    "Native retrieval returned invalid template indices"
                                )
                            for i, p in enumerate(selected):
                                names = [
                                    replay.bank.records[int(index)].template_id
                                    for index in indices[i]
                                ]
                                rankings[layer].append(names)
                                raw.append(
                                    {
                                        "condition": name,
                                        "layer": layer,
                                        "probe": start + i,
                                        "document_id": p["document_id"],
                                        "retrieved_ids": names,
                                        "relevant_ids": p["relevant_ids"],
                                    }
                                )
                    relevance = [p["relevant_ids"] for p in probes]
                    report = {
                        "condition": name,
                        "causal_window_ce": total_loss / total_tokens,
                        "next_token_targets": total_tokens,
                        "layers": [
                            {
                                "layer": layer,
                                "scores": score_rankings(
                                    rankings[layer], relevance, cutoffs
                                ),
                            }
                            for layer in layers
                        ],
                    }
                    reports.append(report)
                    writer.log("retrieval_evaluation", **report)
                summary = {
                    "stage": "A4",
                    "status": "completed",
                    "contract_version": "a4-retrieval-evaluation-v1",
                    "run": str(run),
                    "checkpoint_sha256": replay.checkpoint_sha256,
                    "probes_sha256": digest(probes_path),
                    "bank_snapshot": replay.bank.input_snapshot_sha256,
                    "bank_records": len(replay.bank.records),
                    "retrieval_backend": replay.retriever.info,
                    "split_policy": payload["split_policy"],
                    "review_status": payload["review_status"],
                    "label_semantics": payload["semantics"],
                    "coverage": payload["coverage"],
                    "changed_training_sources": replay.changed_sources,
                    "measurement_source_sha256": digest(Path(__file__)),
                    "runtime_library_sha256": digest(
                        native_library or replay.arguments.native_library
                    ),
                    "frozen_weights_verified_unchanged": True,
                    "training_updates_persisted": False,
                    "cutoffs": cutoffs,
                    "reports": reports,
                    "support_prior": score_support_prior(
                        replay.bank.patterns,
                        [p["relevant_ids"] for p in probes],
                        cutoffs,
                    ),
                    "predictions": raw,
                    "scope": "Fixed-checkpoint deployment queries on source-disjoint reference-graph probes; initial U_out=0 is the matched frozen-A3 CE control; not official test performance or independent semantic-audit certification",
                }
                runtime.close()
                writer.write_json("result.json", summary)
                writer.write_json("status.json", {"stage": "A4", "status": "completed"})
            except BaseException as error:
                writer.write_json(
                    "status.json",
                    {"stage": "A4", "status": "failed", "error": str(error)},
                )
                raise
    print(f"Saved: {output / 'result.json'}", flush=True)
    return output
