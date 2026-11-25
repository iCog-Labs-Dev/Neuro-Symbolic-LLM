"""Read-only A4 representation measurements on source-disjoint reference probes."""

from __future__ import annotations

import csv
import math
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from metrics.a4.representation import (
    capacity_sensitivity,
    cosine_geometry,
    factor_organization,
    key_log_probabilities,
    query_silhouette,
    select_factors,
)
from stages.stage_A.a4.evaluation import score_rankings, validate_probes
from stages.stage_A.a4.replay import checkpoint_replay, digest, read_json


def _document_mean(values, indices, probes):
    groups: dict[tuple[str, str], list[float]] = {}
    for i in indices:
        identity = probes[i]["corpus_id"], probes[i]["document_id"]
        groups.setdefault(identity, []).append(float(values[i]))
    return float(np.mean([np.mean(rows) for rows in groups.values()]))


def measure_representation(
    run: Path,
    probes_path: Path,
    *,
    output: Path | None = None,
    native_library: Path | None = None,
    allow_source_change: bool = False,
    maximum_factors: int = 32,
    substrate: Any = None,
) -> Path:
    """Compare initial/trained heads without advancing parameters or optimizer."""
    if (
        isinstance(maximum_factors, bool)
        or not isinstance(maximum_factors, int)
        or maximum_factors < 2
    ):
        raise ValueError("maximum_factors must be an integer of at least two")
    from stages.stage_A.a4.benchmark import require_idle_gpu

    require_idle_gpu()
    run, probes_path = run.resolve(strict=True), probes_path.resolve(strict=True)
    payload = read_json(probes_path)
    if digest(probes_path.parent / "bindings.jsonl") != payload["bindings_sha256"]:
        raise ValueError("Exact reference bindings changed")
    output = output or run / "evaluation" / time.strftime(
        "representation-%Y%m%d-%H%M%S", time.gmtime()
    )
    if output.exists():
        raise FileExistsError("Choose a fresh representation output directory")

    import jax
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    from residual.symbolic_head.alignment import ValuePredictor
    from stages.results import RunWriter
    from stages.stage_A.a4.training import init_params
    from symbolic_memory.native_bridge import NativeJaxRetriever

    with ExitStack() as runtime:
        replay = runtime.enter_context(
            checkpoint_replay(
                run,
                native_library=native_library,
                allow_source_change=allow_source_change,
                substrate=substrate,
            )
        )
        model, config = replay.model, replay.config
        tokenizer = model.continuous.sub.tokenizer
        probes = validate_probes(payload, replay, tokenizer)
        factors, support = select_factors(probes, maximum_factors=maximum_factors)
        identifiers = [record.template_id for record in replay.bank.records]
        lookup = {name: i for i, name in enumerate(identifiers)}
        factor_indices = jnp.asarray(
            [lookup[name] for name in factors], dtype=jnp.int32
        )
        keys = np.stack([record.key_vector for record in replay.bank.records])
        values = np.stack([record.val_vector for record in replay.bank.records])
        bank_keys = jnp.asarray(keys)
        layers = model.symbolic.layers
        components = [
            (layer, channel)
            for layer in model.continuous.layers
            for channel in range(model.continuous.rank)
        ]
        layer_positions = {layer: i for i, layer in enumerate(model.continuous.layers)}
        active = np.ones(
            (len(model.continuous.layers), model.continuous.rank), dtype=np.float32
        )
        batch_size, seq_len = config.training.batch_size, config.training.seq_len
        pad = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        if pad is None:
            raise ValueError("Tokenizer requires a padding or EOS token")
        predictor = ValuePredictor(model.continuous.hidden_size, replay.bank.vector_dim)

        def states_for(symbolic, ids, switches):
            clean, base = model.continuous.clean_pass(ids)
            latents = model.continuous.predict_latents(replay.frozen, clean)
            latents = {
                layer: value * switches[layer_positions[layer]]
                for layer, value in latents.items()
            }
            result, states = model.forward_with_states_at_latents(
                replay.frozen, symbolic, ids, latents
            )
            return base, result, states

        @jax.jit
        def forward(params, ids, mask, positions):
            base, result, states = states_for(
                params["symbolic"], ids, jnp.asarray(active)
            )
            base_logp = jax.nn.log_softmax(base[:, :-1].astype(jnp.float32), axis=-1)
            logp = jax.nn.log_softmax(
                result.logits[:, :-1].astype(jnp.float32), axis=-1
            )
            valid = mask[:, :-1] & mask[:, 1:]
            targets = ids[:, 1:]
            nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
            kl = jnp.sum(jnp.exp(base_logp) * (base_logp - logp), axis=-1)
            accuracy = jnp.argmax(logp, axis=-1) == targets
            summary = jnp.stack(
                (
                    jnp.sum(nll * valid, axis=1),
                    jnp.sum(kl * valid, axis=1),
                    jnp.sum(accuracy * valid, axis=1),
                ),
                axis=1,
            )
            selected = {
                layer: hidden[jnp.arange(batch_size), positions]
                for layer, hidden in states.items()
            }
            queries = {
                layer: model.symbolic.project(
                    params["symbolic"][f"layer_{layer}"], hidden
                )
                for layer, hidden in selected.items()
            }
            predictions = {
                layer: predictor.apply(params["value"][f"layer_{layer}"], hidden)
                for layer, hidden in selected.items()
            }
            return summary, selected, queries, predictions

        @jax.jit
        def ablate_queries(symbolic, ids, positions, switches):
            _, _, states = states_for(symbolic, ids, switches)
            return {
                layer: model.symbolic.project(symbolic[f"layer_{layer}"], hidden)[
                    jnp.arange(batch_size), positions
                ]
                for layer, hidden in states.items()
            }

        @jax.jit
        def factor_losses(queries):
            scores = queries @ bank_keys.T
            return logsumexp(scores, axis=1, keepdims=True) - scores[:, factor_indices]

        def batches():
            for start in range(0, len(probes), batch_size):
                selected = probes[start : start + batch_size]
                padded = selected + [selected[-1]] * (batch_size - len(selected))
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
                yield len(selected), ids, mask, positions

        retrieval_depth = min(16, len(identifiers))
        retriever = runtime.enter_context(
            NativeJaxRetriever.from_mork(
                replay.bank,
                server_url=replay.arguments.mork_url,
                library_path=native_library or replay.arguments.native_library,
                top_m=retrieval_depth,
                max_queries=max(1024, len(probes)),
            )
        )
        retrieve = jax.jit(retriever)
        retrieve_head = jax.jit(replay.retriever)
        initial = init_params(model, config.training.seed)
        reports, arrays, review = [], {}, []
        with RunWriter(output.parent, output.name) as writer:
            writer.write_json("status.json", {"stage": "A4", "status": "running"})
            writer.write_config(
                {
                    "run": str(run),
                    "probes": str(probes_path),
                    "maximum_factors": maximum_factors,
                    "minimum_source_documents_per_factor": 2,
                    "factor_selection": "descending reference support, then template ID; before model measurement",
                    "component": "one frozen-A3 latent channel at one intercepted layer",
                    "ablation": "zero the channel for the entire input window; no parameter updates",
                    "contribution": "document-balanced mean of positive full-bank target-NLL increase; beneficial removals reported separately",
                    "distance": "Euclidean query distance; expanded multilabel memberships",
                }
            )
            try:
                print("Measuring fixed-bank geometry.", flush=True)
                geometry = cosine_geometry(keys)
                for condition, params in (
                    ("initial", initial),
                    ("trained", replay.params),
                ):
                    print(
                        f"Measuring {condition}: {len(probes)} reference windows.",
                        flush=True,
                    )
                    totals = np.zeros(3)
                    query_rows: dict[int, list[np.ndarray]] = {
                        layer: [] for layer in layers
                    }
                    hidden_rows: dict[int, list[np.ndarray]] = {
                        layer: [] for layer in layers
                    }
                    value_rows: dict[int, list[np.ndarray]] = {
                        layer: [] for layer in layers
                    }
                    for count, ids, mask, positions in batches():
                        summary, states, queries, predicted = jax.device_get(
                            forward(params, ids, mask, positions)
                        )
                        totals += summary[:count].sum(axis=0)
                        for layer in layers:
                            query_rows[layer].append(queries[layer][:count])
                            hidden_rows[layer].append(states[layer][:count])
                            value_rows[layer].append(predicted[layer][:count])
                    token_count = sum(len(p["input_ids"]) - 1 for p in probes)
                    report: dict[str, Any] = {
                        "condition": condition,
                        "next_token_targets": token_count,
                        "ce": float(totals[0] / token_count),
                        "ppl": math.exp(float(totals[0] / token_count)),
                        "kl_base_to_model": float(totals[1] / token_count),
                        "next_token_accuracy": float(totals[2] / token_count),
                        "layers": [],
                    }
                    baselines = {}
                    for layer in layers:
                        queries = np.concatenate(query_rows[layer])
                        hidden = np.concatenate(hidden_rows[layer])
                        predicted = np.concatenate(value_rows[layer])
                        matches = jax.device_get(retrieve(jnp.asarray(queries)))
                        head_matches = jax.device_get(
                            retrieve_head(jnp.asarray(queries))
                        )
                        ranks = [
                            [identifiers[int(i)] for i in row]
                            for row in np.asarray(matches.template_indices)
                        ]
                        relevance = [p["relevant_ids"] for p in probes]
                        weights_logits = np.einsum(
                            "bd,bmd->bm", queries, head_matches.keys
                        )
                        weights = np.exp(
                            weights_logits - weights_logits.max(axis=1, keepdims=True)
                        )
                        weights /= weights.sum(axis=1, keepdims=True)
                        context = np.einsum("bm,bmd->bd", weights, head_matches.values)
                        writeback = context @ np.asarray(
                            params["symbolic"][f"layer_{layer}"]["U_out"]
                        )
                        ratios = np.linalg.norm(writeback, axis=1) / np.maximum(
                            np.linalg.norm(hidden, axis=1), np.finfo(np.float32).tiny
                        )
                        losses = np.asarray(
                            jax.device_get(factor_losses(jnp.asarray(queries)))
                        )
                        baselines[layer] = losses
                        selected_memberships = [
                            [name for name in p["relevant_ids"] if name in support]
                            for p in probes
                        ]
                        # Bound expanded-membership memory independently of the bank size.
                        silhouette_indices = [
                            i for i, names in enumerate(selected_memberships) if names
                        ][:256]
                        silhouette = (
                            query_silhouette(
                                queries[silhouette_indices],
                                [selected_memberships[i] for i in silhouette_indices],
                            )
                            if silhouette_indices
                            else {
                                "silhouette": None,
                                "reason": "No cross-document reference factors",
                            }
                        )
                        silhouette["probe_cap"] = 256
                        silhouette["selection"] = (
                            "first eligible probes in manifest order; no score-based selection"
                        )
                        key_errors, value_errors = [], []
                        log_probabilities = key_log_probabilities(queries, keys)
                        for i, probe in enumerate(probes):
                            for name in probe["relevant_ids"]:
                                # Every visible reference positive contributes, not a selected retrieval hit.
                                index = lookup[name]
                                key_errors.append(-float(log_probabilities[i, index]))
                                value_errors.append(
                                    float(np.square(predicted[i] - values[index]).sum())
                                )
                            if condition == "trained":
                                review.append(
                                    {
                                        "probe": i,
                                        "layer": layer,
                                        "document_id": probe["document_id"],
                                        "text": tokenizer.decode(probe["input_ids"]),
                                        "relevant_ids": probe["relevant_ids"],
                                        "attended_templates": [
                                            {
                                                "template_id": identifiers[int(index)],
                                                "metta_expr": replay.bank.records[
                                                    int(index)
                                                ].metta_expr,
                                                "weight": float(weight),
                                            }
                                            for index, weight in zip(
                                                head_matches.template_indices[i],
                                                weights[i],
                                                strict=True,
                                            )
                                        ],
                                    }
                                )
                        layer_report = {
                            "layer": layer,
                            "retrieval": score_rankings(
                                ranks,
                                relevance,
                                [k for k in (1, 4, 8, 16) if k <= retrieval_depth],
                            ),
                            "reference_pair_key_nll": (
                                float(np.mean(key_errors)) if key_errors else None
                            ),
                            "reference_pair_value_squared_error": (
                                float(np.mean(value_errors)) if value_errors else None
                            ),
                            "alignment_reference_pairs": len(key_errors),
                            "query_clusters": silhouette,
                            "writeback_to_hidden_norm_mean": float(ratios.mean()),
                            "writeback_nonzero_fraction": float(
                                np.mean(np.linalg.norm(writeback, axis=1) > 0)
                            ),
                            "writeback_scope": "probe endpoints using the recorded head retrieval depth; not latency or semantic usefulness",
                        }
                        report["layers"].append(layer_report)
                        arrays[f"{condition}_queries_l{layer}"] = queries
                        arrays[f"{condition}_factor_nll_l{layer}"] = losses
                    if factors:
                        for layer in layers:
                            arrays[f"{condition}_effects_l{layer}"] = np.zeros(
                                (len(factors), len(components))
                            )
                            arrays[f"{condition}_beneficial_removal_l{layer}"] = (
                                np.zeros((len(factors), len(components)))
                            )
                        for component, (ablated_layer, channel) in enumerate(
                            components
                        ):
                            switches = active.copy()
                            switches[layer_positions[ablated_layer], channel] = 0
                            collected: dict[int, list[np.ndarray]] = {
                                layer: [] for layer in layers
                            }
                            for count, ids, _, positions in batches():
                                queries = ablate_queries(
                                    params["symbolic"],
                                    ids,
                                    positions,
                                    jnp.asarray(switches),
                                )
                                for layer in layers:
                                    losses = np.asarray(
                                        jax.device_get(factor_losses(queries[layer]))
                                    )[:count]
                                    collected[layer].append(losses)
                            for layer in layers:
                                losses = np.concatenate(collected[layer])
                                difference = losses - baselines[layer]
                                for factor, name in enumerate(factors):
                                    arrays[f"{condition}_effects_l{layer}"][
                                        factor, component
                                    ] = _document_mean(
                                        np.maximum(difference[:, factor], 0),
                                        support[name],
                                        probes,
                                    )
                                    arrays[f"{condition}_beneficial_removal_l{layer}"][
                                        factor, component
                                    ] = _document_mean(
                                        np.maximum(-difference[:, factor], 0),
                                        support[name],
                                        probes,
                                    )
                            print(
                                f"{condition}: channel ablations {component + 1}/{len(components)}",
                                flush=True,
                            )
                        for row in report["layers"]:
                            layer = row["layer"]
                            row["factor_organization"] = factor_organization(
                                arrays[f"{condition}_effects_l{layer}"]
                            )
                    else:
                        report["factor_organization_status"] = (
                            "unavailable: no template has reference positives in two documents"
                        )
                    reports.append(report)
                    writer.log("representation", **report)
                capacity = {}
                for layer in layers:
                    capacity[str(layer)] = capacity_sensitivity(
                        arrays[f"trained_queries_l{layer}"],
                        keys,
                        identifiers,
                        [p["relevant_ids"] for p in probes],
                        np.asarray(
                            replay.params["symbolic"][f"layer_{layer}"]["W_sym"]
                        ),
                        head_rankings=[
                            [item["template_id"] for item in row["attended_templates"]]
                            for row in review
                            if row["layer"] == layer
                        ],
                    )
                np.savez_compressed(writer.dir / "contributions.npz", **arrays)
                writer.write_json(
                    "review.json", {"status": "not_human_reviewed", "probes": review}
                )
                summary = {
                    "contract_version": "a4-representation-v1",
                    "stage": "A4",
                    "status": "completed",
                    "run": str(run),
                    "checkpoint_sha256": replay.checkpoint_sha256,
                    "bank_snapshot": replay.bank.input_snapshot_sha256,
                    "bank_records": len(identifiers),
                    "probes_sha256": digest(probes_path),
                    "probes": len(probes),
                    "label_semantics": payload["label_kind"],
                    "split_policy": payload.get("split_policy"),
                    "factors": [
                        {"template_id": name, "probes": support[name]}
                        for name in factors
                    ],
                    "components": [
                        {"layer": layer, "channel": channel}
                        for layer, channel in components
                    ],
                    "fixed_bank_geometry": geometry,
                    "capacity": capacity,
                    "retrieval_backend": replay.retriever.info,
                    "measurement_native_library_sha256": digest(
                        native_library or replay.arguments.native_library
                    ),
                    "reports": reports,
                    "runtime_training_state_source": "frozen-a3-prediction-v1",
                    "recorded_training_state_source": replay.metadata.get(
                        "training_state_source", "unspecified-legacy"
                    ),
                    "changed_training_sources": replay.changed_sources,
                    "measurement_source_sha256": digest(Path(__file__).resolve()),
                    "frozen_weights_verified_unchanged": True,
                    "checkpoints_written": False,
                    "scope": "Fixed-checkpoint evaluation; reference-window metrics and causal channel diagnostics, not joint A3 training, a human semantic audit, or proof of disentanglement",
                }
                runtime.close()
                writer.write_json("result.json", summary)
                with (writer.dir / "summary.csv").open(
                    "w", newline="", encoding="utf-8-sig"
                ) as stream:
                    table = csv.DictWriter(
                        stream,
                        fieldnames=[
                            "condition",
                            "ce",
                            "ppl",
                            "kl_base_to_model",
                            "next_token_accuracy",
                            "next_token_targets",
                        ],
                    )
                    table.writeheader()
                    table.writerows(
                        {key: report[key] for key in table.fieldnames}
                        for report in reports
                    )
                writer.write_json("status.json", {"stage": "A4", "status": "completed"})
            except BaseException as error:
                writer.write_json(
                    "status.json",
                    {"stage": "A4", "status": "failed", "error": str(error)},
                )
                raise
    print(f"Saved: {output / 'result.json'}", flush=True)
    return output
