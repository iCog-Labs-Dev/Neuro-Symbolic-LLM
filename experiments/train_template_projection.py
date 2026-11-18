"""Train a frozen-feature bank projection and compare held-out cosine retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from offline.embedding.context_builder import load_pattern_contexts
from offline.embedding.projection_training import (
    cosine_scores,
    fit_projection,
    retrieval_report,
)
from offline.embedding.template_embeddings import create_random_projection
from scripts.prepare_alignment_labels import qualified_pattern_ids
from scripts.prepare_retrieval_probes import causal_probes, text_digest


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def labels(rows, lookup):
    result = np.zeros((len(rows), len(lookup)), dtype=bool)
    for index, row in enumerate(rows):
        for name in row["relevant_ids"]:
            result[index, lookup[name]] = True
    return result


def prepare_training_rows(documents, manifest, patterns, tokenizer, seq_len):
    """Retain known positives; use foreign document constants as safe negatives."""
    qualified = qualified_pattern_ids(patterns)
    lookup = {p.id: index for index, p in enumerate(patterns)}
    bindings = defaultdict(list)
    for artifact in manifest["exact_inputs"]:
        path = Path(artifact["path"])
        if digest(path) != artifact["sha256"]:
            raise ValueError("Training exact-match artifact changed")
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            match = json.loads(line)
            template_id = qualified.get(match["pattern"])
            if template_id in lookup:
                for binding in match["exact_bindings"]:
                    bindings[binding["corpus_id"], binding["document_id"]].append(
                        (template_id, binding)
                    )
    document_ids = {row["document_id"] for row in documents}
    owners = [
        {
            name
            for name in document_ids
            if f":{name}:" in p.expensive_trace["qualified_pattern"]
        }
        for p in patterns
    ]
    rows, candidates = [], []
    for document in documents:
        current = document["document_id"]
        negatives = np.array([bool(names - {current}) for names in owners])
        for row in causal_probes(
            document, tokenizer, bindings[document["corpus_id"], current], seq_len
        ):
            if not row["relevant_ids"] or not negatives.any():
                continue
            positive = labels([row], lookup)[0]
            if np.any(positive & negatives):
                raise ValueError(
                    "Positive label contradicts a document-specific constant"
                )
            rows.append(row)
            candidates.append(negatives | positive)
    if not rows:
        raise ValueError("No source-linked training examples with explicit negatives")
    return rows, np.stack(candidates)


def encode_queries(model_name, rows, context_texts, context_vectors, seq_len):
    """Cache frozen GPT-2 final-block means; no gradients or A3 modifications."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires an idle CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = (
        AutoModel.from_pretrained(
            model_name, dtype=torch.float32, attn_implementation="eager"
        )
        .eval()
        .to("cuda")
    )
    model.requires_grad_(False)
    if getattr(model.config, "model_type", None) != "gpt2":
        raise ValueError(
            "Projection replay currently supports GPT-2 final-block features"
        )
    captured = []
    hook = model.h[-1].register_forward_hook(
        lambda module, args, output: captured.append(
            output[0] if isinstance(output, tuple) else output
        )
    )
    unique = list(dict.fromkeys(tuple(row["input_ids"]) for row in rows))
    # Verify the encoder against the producer's raw cache before training P.
    checks = [tuple(tokenizer(text)["input_ids"]) for text in context_texts[:2]]
    sequences = checks + unique
    encoded = []
    try:
        with torch.inference_mode():
            for offset in range(0, len(sequences), 16):
                group = sequences[offset : offset + 16]
                if any(not 0 < len(ids) <= seq_len for ids in group):
                    raise ValueError("Query exceeds the configured causal token window")
                ids = np.full((len(group), seq_len), tokenizer.eos_token_id, np.int64)
                mask = np.zeros_like(ids)
                for index, tokens in enumerate(group):
                    ids[index, : len(tokens)] = tokens
                    mask[index, : len(tokens)] = 1
                captured.clear()
                token_mask = torch.as_tensor(mask, device="cuda")
                model(
                    input_ids=torch.as_tensor(ids, device="cuda"),
                    attention_mask=token_mask,
                    use_cache=False,
                )
                hidden = captured[-1]
                pooled = (hidden * token_mask[..., None]).sum(dim=1) / token_mask.sum(
                    dim=1
                )[:, None]
                encoded.extend(pooled.cpu().numpy())
    finally:
        hook.remove()
        del model
        captured.clear()
        torch.cuda.empty_cache()
    encoded = np.asarray(encoded, dtype=np.float32)
    np.testing.assert_allclose(encoded[:2], context_vectors[:2], rtol=2e-4, atol=0.002)
    cache = dict(zip(unique, encoded[2:], strict=True))
    return np.stack([cache[tuple(row["input_ids"])] for row in rows])


def run(config_path):
    from importlib.metadata import version

    from transformers import AutoTokenizer

    from stages.stage_A.a4.benchmark import require_idle_gpu
    from symbolic_memory.bank import load_bank

    cfg = yaml.safe_load(config_path.read_text())
    required = {
        "patterns",
        "embeddings",
        "documents",
        "alignment",
        "probes",
        "context_texts",
        "context_vectors",
        "output",
    }
    if (
        not isinstance(cfg, dict)
        or required - cfg.keys()
        or cfg.keys() - required - {"epochs", "batch_size", "learning_rate"}
    ):
        raise ValueError(f"Projection config requires paths {sorted(required)}")
    paths = {
        name: (config_path.resolve().parent / cfg[name]).resolve() for name in required
    }
    if paths["output"].exists():
        raise FileExistsError("Choose a fresh projection output directory")
    bank = load_bank(paths["patterns"], paths["embeddings"])
    original = [
        json.loads(line)
        for line in paths["embeddings"].read_text().splitlines()
        if line.strip()
    ]
    manifest, probes = read_json(paths["alignment"]), read_json(paths["probes"])
    if (
        manifest["bank_snapshot_sha256"] != bank.input_snapshot_sha256
        or probes["bank_snapshot"] != bank.input_snapshot_sha256
    ):
        raise ValueError("Projection labels belong to a different bank")
    if digest(paths["documents"]) != manifest["source_sha256"]["documents"]:
        raise ValueError("Training source documents changed")
    documents = [
        json.loads(line)
        for line in paths["documents"].read_text().splitlines()
        if line.strip()
    ]
    train_ids = {row["document_id"] for row in documents}
    train_text = {text_digest(row["source_text"]) for row in documents}
    if any(
        row["document_id"] in train_ids or text_digest(row["source_text"]) in train_text
        for row in probes["documents"]
    ):
        raise ValueError("Projection validation overlaps training sources")
    if (
        probes["split"] != "validation"
        or probes["label_kind"] != "reference_graph_firing"
    ):
        raise ValueError("Independent reference validation probes are required")
    model_name, seq_len = probes["model"], probes["seq_len"]
    first = original[0]
    if (
        first["encoder_model"] != model_name
        or first["projection_method"] != "random_orthonormal"
        or first["intercept_layer"] != 11
    ):
        raise ValueError("Expected the recorded GPT-2 final-block random baseline")
    initial = create_random_projection(
        first["encoder_hidden_dim"], first["embedding_dim"], first["projection_seed"]
    )
    if hashlib.sha256(initial.tobytes()).hexdigest() != first["projection_sha256"]:
        raise ValueError("Baseline projection identity differs")
    context_texts = read_json(paths["context_texts"])
    context_vectors = np.load(paths["context_vectors"], allow_pickle=False)
    if context_vectors.shape != (len(context_texts), initial.shape[1]) or len(
        set(context_texts)
    ) != len(context_texts):
        raise ValueError("Cached contexts and feature vectors must align uniquely")
    raw = dict(zip(context_texts, context_vectors, strict=True))
    templates = load_pattern_contexts(paths["patterns"], paths["documents"])
    if [row.pattern_id for row in templates] != [
        row.template_id for row in bank.records
    ]:
        raise ValueError("Template order changed")
    features = np.stack(
        [
            np.stack([raw[text] for text in row.contexts]).mean(axis=0)
            for row in templates
        ]
    )
    baseline_vectors = features @ initial.T
    np.testing.assert_allclose(
        baseline_vectors,
        np.stack([row.key_vector for row in bank.records]),
        rtol=1e-5,
        atol=2e-4,
    )
    if any(not np.array_equal(row.key_vector, row.val_vector) for row in bank.records):
        raise ValueError("This controlled experiment requires equal initial K/V")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_rows, allowed = prepare_training_rows(
        documents, manifest, bank.patterns, tokenizer, seq_len
    )
    validation_rows = probes["probes"]
    lookup = {row.template_id: i for i, row in enumerate(bank.records)}
    positive = labels(train_rows, lookup)
    validation_positive = labels(validation_rows, lookup)
    require_idle_gpu()
    output = paths["output"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "status.json").write_text(json.dumps({"status": "running"}) + "\n")
    try:
        print(
            f"Verified {len(bank.records)} templates; {len(train_rows)} training queries; {len(validation_rows)} validation queries.",
            flush=True,
        )
        queries = encode_queries(
            model_name,
            train_rows + validation_rows,
            context_texts,
            context_vectors,
            seq_len,
        )
        train_queries, validation_queries = (
            queries[: len(train_rows)],
            queries[len(train_rows) :],
        )
        np.savez_compressed(
            output / "features.npz",
            templates=features,
            training=train_queries,
            validation=validation_queries,
            positives=positive,
            allowed=allowed,
            validation_positives=validation_positive,
        )
        before = retrieval_report(
            cosine_scores(initial, validation_queries, features), validation_positive
        )
        (output / "baseline.json").write_text(json.dumps(before, indent=2) + "\n")
        with (
            (output / "training.log").open("x") as log,
            (output / "metrics.jsonl").open("x") as metrics,
        ):

            def report(row):
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                train_loss = (
                    f"{row['training_loss']:.6f}" if "training_loss" in row else "--"
                )
                line = f"Epoch {row['epoch']:2d} | train={train_loss} | validation={row['validation_loss']:.6f}"
                log.write(line + "\n")
                log.flush()
                print(line, flush=True)

            projection, training = fit_projection(
                initial,
                train_queries,
                features,
                positive,
                allowed,
                validation_queries,
                validation_positive,
                epochs=cfg.get("epochs", 20),
                batch_size=cfg.get("batch_size", 16),
                learning_rate=cfg.get("learning_rate", 0.001),
                report=report,
            )
        after = retrieval_report(
            cosine_scores(projection, validation_queries, features), validation_positive
        )
        np.savez(output / "projection.npz", projection=projection, initial=initial)
        projected = features @ projection.T
        projection_hash = hashlib.sha256(projection.tobytes()).hexdigest()
        with (output / "embeddings.jsonl").open("x") as stream:
            for old, vector in zip(original, projected, strict=True):
                stream.write(
                    json.dumps(
                        {
                            **old,
                            "key": vector.tolist(),
                            "value": vector.tolist(),
                            "projection_method": (
                                "supervised_orthonormal"
                                if training["best_epoch"]
                                else first["projection_method"]
                            ),
                            "projection_sha256": projection_hash,
                            "initialization": (
                                "learned_projected_mean_context_embeddings_for_key_and_value"
                                if training["best_epoch"]
                                else first["initialization"]
                            ),
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
        result = {
            "status": "completed",
            "scope": "Frozen-GPT-2 projection-only cosine retrieval; validation-selected; not an A4 checkpoint evaluation or a miner-semantics rebuild",
            "bank_snapshot": bank.input_snapshot_sha256,
            "miner_version": version("hybrid-miner"),
            "inputs": {
                name: {"path": str(path), "sha256": digest(path)}
                for name, path in paths.items()
                if name != "output"
            },
            "config": cfg,
            "training_queries": len(train_rows),
            "validation_queries": len(validation_rows),
            "distinct_raw_template_vectors": len({row.tobytes() for row in features}),
            "negative_policy": "Only foreign-document constants; other unlabelled candidates excluded from training loss",
            "constraint": "Orthonormal rows after each update",
            "random": before,
            "learned": after,
            **training,
        }
        (output / "result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        (output / "status.json").write_text(json.dumps({"status": "completed"}) + "\n")
        for name, report in (("Random", before), ("Learned", after)):
            print(
                f"{name}: precision@8={100*report['micro_precision']:.3f}% recall@8={100*report['micro_recall']:.3f}% | tie precision bounds={100*report['tie_precision_lower']:.3f}-{100*report['tie_precision_upper']:.3f}%"
            )
        print(f"Selected epoch {training['best_epoch']}; saved: {output}", flush=True)
    except BaseException as error:
        (output / "status.json").write_text(
            json.dumps({"status": "failed", "error": str(error)}) + "\n"
        )
        raise
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    run(parser.parse_args().config)
