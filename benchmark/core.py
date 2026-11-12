"""Small, artifact-driven A1/A2/A3 benchmark harness."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from benchmark.provenance import (
    effective_condition,
    finalize_attempt,
    next_attempt_name,
    sha256_file,
    source_fingerprint,
    validate_complete_attempt,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_CONFIG = ROOT / "configs" / "runtime_datasets.yaml"
BENCHMARK_RUNTIME_SOURCE = "separate"
BENCHMARK_LEARNING_RATE = 1.0e-3
DEFAULT_CONFIGS = {
    "a1": ROOT / "configs" / "stage_A" / "a1_lora_baseline.yaml",
    "a2": ROOT / "configs" / "stage_A" / "a2_pc_inspired.yaml",
    "a3": ROOT / "configs" / "stage_A" / "a3_hybrid_pc.yaml",
}
RUNNERS = {
    "a1": ROOT / "experiments" / "run_stage_a1.py",
    "a2": ROOT / "experiments" / "run_stage_a2.py",
    "a3": ROOT / "experiments" / "run_stage_a3.py",
}
EXPECTED_DOMAINS = ("eurlex", "ledgar", "scotus")
MEASURED_METRICS = (
    "loss_matrix",
    "perplexity",
    "mean_forgetting",
    "optimizer_steps",
    "tokens",
    "mean_step_time",
    "estimated_tokens_per_second",
    "params_unchanged",
    "residual_compressibility",
    "effective_rank",
    "invariant_redundancy",
    "fracture_index",
    "overlap",
    "taylor_delta",
    "synergy",
    "token_accuracy",
    "task_accuracy",
    "task_f1",
    "accuracy_forgetting",
    "expected_calibration_error",
    "paired_seed_delta",
    "mixed_curvature_coupling",
    "off_support_leakage",
    "update_commutator",
    "causal_fingerprint_separability",
    "category_naturality",
    "ablation_fidelity",
    "heldout_transfer",
    "sequential_joint_gap",
    "anchor_kl",
    "symbolic_retrieval",
    "long_range_coherence",
)


@dataclass(frozen=True)
class RunSpec:
    method: str
    seed: int
    protocol: str
    run_id: str
    config: str
    output_dir: str
    run_name: str
    k: int | None = None
    rank: int = 16
    runtime_source: str = BENCHMARK_RUNTIME_SOURCE
    runtime_config: str = str(DEFAULT_RUNTIME_CONFIG)
    task_order: tuple[str, ...] = EXPECTED_DOMAINS
    heldout_tasks: tuple[str, ...] = ()
    a2_arm: str | None = None


def _sanitize_value(value: Any) -> Any:
    """Recursively sanitize data structure by converting non-finite floats to None."""
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _sanitize_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_sanitize_value(v) for v in value]
    return value


def _jsonable(value: Any) -> Any:
    return _sanitize_value(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = _jsonable(value)
    path.write_text(
        json.dumps(clean, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _spearman_corr(x: list[float] | Any, y: list[float] | Any) -> float:
    """Compute Spearman rank correlation between two vectors without external dependencies."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if len(xa) < 2 or np.all(xa == xa[0]) or np.all(ya == ya[0]):
        return 0.0
    rx = np.argsort(np.argsort(xa)).astype(float)
    ry = np.argsort(np.argsort(ya)).astype(float)
    vx = rx - rx.mean()
    vy = ry - ry.mean()
    denom = np.sqrt(np.sum(vx**2) * np.sum(vy**2))
    if denom == 0.0:
        return 0.0
    return float(np.sum(vx * vy) / denom)


def compute_domain_overlap(
    domain_atom_effects: dict[str, list[float] | Any],
    eps: float = 1e-8,
) -> dict[str, float]:
    import jax
    import jax.numpy as jnp

    from metrics.representation import overlap

    domain_names = sorted(domain_atom_effects)
    normalized: dict[str, jax.Array] = {}
    for name in domain_names:
        raw = jnp.maximum(
            jnp.asarray(domain_atom_effects[name], dtype=jnp.float32), 0.0
        )
        total = jnp.maximum(jnp.sum(raw), eps)
        normalized[name] = raw / total

    matrix: dict[str, float] = {}
    for i, d1 in enumerate(domain_names):
        for d2 in domain_names[i + 1 :]:
            matrix[f"{d1}:{d2}"] = float(
                overlap(normalized[d1], normalized[d2], eps=eps)
            )
    return matrix


def compute_semantic_overlap(
    category_atom_effects: dict[str, list[float] | Any],
    eps: float = 1e-8,
) -> dict[str, float]:
    """Calculate overlap between semantic/category factors within a layer."""
    return compute_domain_overlap(category_atom_effects, eps=eps)


def _load_deployed_metrics(
    run: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate the final checkpoint with the common deployed function.

    Heavy JAX/model imports are intentionally local so ``plan`` remains usable
    in lightweight environments.
    """
    import jax
    import jax.numpy as jnp

    from benchmark.evaluation import (
        aggregate_lm_rows,
        evaluate_legal_tasks,
        evaluate_long_range,
        extract_split_features,
    )
    from frozenllm.substrate import FrozenSubstrate, identity_modify
    from metrics.performance import (
        calibration_stats,
        ece_from_stats,
        kl_to_base,
        token_accuracy,
    )
    from metrics.representation import (
        canonicalize_adapter_params,
        effective_rank,
        gradient_conflict,
        invariant_redundancy,
        pairwise_ablation_interaction,
        synergy_proxy,
        taylor_delta,
    )
    from residual import ResidualAdapter, ResidualConfig
    from runtime.data_loader import (
        legal_task_specs,
        load_labeled_split,
        load_runtime_config,
        load_runtime_data,
        load_split_texts,
        make_eval_set,
        tokenize_texts,
    )
    from stages.results import load_params
    from stages.stage_base import TrainingConfig

    config_path = Path(run.get("materialized_config", run["config"]))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    training = TrainingConfig.from_dict(config.get("training") or {})
    checkpoint_dir = Path(run["output_dir"]) / run["run_name"] / "params"
    checkpoint_paths = sorted(checkpoint_dir.glob("task_*.npz"))
    if not checkpoint_paths:
        return {"status": "unavailable", "reason": "no checkpoint found"}
    substrate = FrozenSubstrate(training.model)
    adapter = ResidualAdapter(
        ResidualConfig.from_dict(config["adapter"]), substrate.architecture
    )

    def load_adapter_params(path: Path) -> dict[str, dict[str, jax.Array]]:
        loaded = _filter_adapter_params(load_params(path))
        return {
            layer: {name: jnp.asarray(value) for name, value in values.items()}
            for layer, values in loaded.items()
        }

    checkpoint_rows = []
    for path in checkpoint_paths:
        fields = path.stem.split("_", 2)
        if len(fields) != 3 or not fields[1].isdigit():
            continue
        checkpoint_rows.append(
            {
                "index": int(fields[1]),
                "task": fields[2],
                "path": path,
                "params": load_adapter_params(path),
            }
        )
    checkpoint_rows.sort(key=lambda row: int(row["index"]))
    if not checkpoint_rows:
        return {"status": "unavailable", "reason": "no valid task checkpoint found"}
    params = checkpoint_rows[-1]["params"]
    adapter.validate_params(params)
    runtime_config_path = Path(
        run.get("materialized_runtime_config", DEFAULT_RUNTIME_CONFIG)
    )
    runtime_data = load_runtime_data(
        config_path=runtime_config_path,
        tokenizer=substrate.tokenizer,
        source=run.get("runtime_source", BENCHMARK_RUNTIME_SOURCE),
        sequence_length=training.seq_len,
        batch_size=training.batch_size,
        eval_batches=training.eval_batches,
        default_epochs=training.epochs,
    )
    domains = runtime_data.tasks
    test_sets = runtime_data.get_test_sets(
        training.batch_size,
        training.eval_batches,
    )
    metrics_by_domain: dict[str, dict[str, float]] = {}
    batches_by_domain: dict[str, tuple[jax.Array, ...]] = {}
    latency: list[float] = []

    def deployed_with_params(
        deployed_params: dict[str, dict[str, jax.Array]],
        ids: jax.Array,
        extra_layers: tuple[int, ...] = (),
    ) -> tuple[jax.Array, jax.Array, Any]:
        intercept_layers = tuple(sorted(set(adapter.layers) | set(extra_layers)))
        if run["method"] != "a3":
            base_result = substrate.run_with_interception(
                ids, modify_fn=identity_modify, intercept_layers=adapter.layers
            )
            adapted_result = substrate.run_with_interception(
                ids,
                modify_fn=adapter.modify_fn(deployed_params),
                intercept_layers=intercept_layers,
            )
            return base_result.logits, adapted_result.logits, adapted_result
        clean = substrate.run_with_interception(
            ids, modify_fn=identity_modify, intercept_layers=adapter.layers
        )
        latent = {
            layer: adapter.encode(
                deployed_params[f"layer_{layer}"], clean.hidden_state(layer)
            )
            for layer in adapter.layers
        }

        def modify(hidden: jax.Array, layer: int) -> jax.Array:
            block = deployed_params.get(f"layer_{layer}")
            if block is None:
                return hidden
            return hidden + adapter.scale * (latent[layer] @ block["B"]).astype(
                hidden.dtype
            )

        adapted_result = substrate.run_with_interception(
            ids, modify_fn=modify, intercept_layers=intercept_layers
        )
        return clean.logits, adapted_result.logits, adapted_result

    def deployed(ids: jax.Array) -> tuple[jax.Array, jax.Array, Any]:
        return deployed_with_params(params, ids)

    def loss_for(p: dict[str, dict[str, jax.Array]], ids: jax.Array) -> jax.Array:
        if run["method"] != "a3":
            adapted = substrate.run_with_interception(
                ids, modify_fn=adapter.modify_fn(p), intercept_layers=adapter.layers
            ).logits
        else:
            clean = substrate.run_with_interception(
                ids, modify_fn=identity_modify, intercept_layers=adapter.layers
            )
            latent = {
                layer: adapter.encode(p[f"layer_{layer}"], clean.hidden_state(layer))
                for layer in adapter.layers
            }

            def modify(hidden: jax.Array, layer: int) -> jax.Array:
                block = p.get(f"layer_{layer}")
                if block is None:
                    return hidden
                return hidden + adapter.scale * (latent[layer] @ block["B"]).astype(
                    hidden.dtype
                )

            adapted = substrate.run_with_interception(
                ids, modify_fn=modify, intercept_layers=adapter.layers
            ).logits
        return substrate.compute_loss(adapted, ids, losses=("CE",))

    for domain in domains:
        batches = test_sets[domain.name]
        batches_by_domain[domain.name] = batches
        rows = []
        conf_sums = []
        acc_sums = []
        count_sums = []
        if batches:
            _, warmed, _ = deployed(batches[0])
            jax.block_until_ready(warmed)
        for ids in batches:
            if run["method"] == "a3":
                start = time.perf_counter()
                base, adapted, _ = deployed(ids)
            else:
                base = substrate.run_with_interception(
                    ids,
                    modify_fn=identity_modify,
                    intercept_layers=adapter.layers,
                ).logits
                start = time.perf_counter()
                adapted = substrate.run_with_interception(
                    ids,
                    modify_fn=adapter.modify_fn(params),
                    intercept_layers=adapter.layers,
                ).logits
            jax.block_until_ready(adapted)
            latency.append(time.perf_counter() - start)
            c_sum, a_sum, cnt = calibration_stats(adapted, ids)
            conf_sums.append(c_sum)
            acc_sums.append(a_sum)
            count_sums.append(cnt)
            rows.append(
                (
                    float(substrate.compute_loss(adapted, ids, losses=("CE",))),
                    float(token_accuracy(adapted, ids)),
                    float(kl_to_base(base, adapted)),
                )
            )
        values = np.asarray(rows, dtype=float)
        if conf_sums:
            pooled_conf = jnp.sum(jnp.stack(conf_sums), axis=0)
            pooled_acc = jnp.sum(jnp.stack(acc_sums), axis=0)
            pooled_cnt = jnp.sum(jnp.stack(count_sums), axis=0)
            pooled_ece = float(ece_from_stats(pooled_conf, pooled_acc, pooled_cnt))
        else:
            pooled_ece = 0.0
        metrics_by_domain[domain.name] = {
            "loss": float(values[:, 0].mean()) if len(values) else 0.0,
            "ppl": float(np.exp(values[:, 0].mean())) if len(values) else 0.0,
            "token_accuracy": float(values[:, 1].mean()) if len(values) else 0.0,
            "ece": pooled_ece,
            "kl": float(values[:, 2].mean()) if len(values) else 0.0,
        }

    gradients = {}
    for name, batches in batches_by_domain.items():
        if not batches:
            continue
        gradients[name] = jax.grad(lambda p, batch=batches[0]: loss_for(p, batch))(
            params
        )
    conflict = {}
    names = list(gradients)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            conflict[f"{left}:{right}"] = float(
                gradient_conflict(gradients[left], gradients[right])
            )

    representation: dict[str, Any] = {
        "effective_rank": {},
        "invariant_redundancy": {},
        "fracture_index": {},
        "overlap": {},
        "overlap_matrices": {},
        "semantic_overlap": {},
        "taylor_delta": {},
        "synergy": {},
        "synergy_proxy": {},
        "measured_synergy": {},
        "synergy_validation": {},
        "cross_domain_conflict": conflict,
    }
    for layer_name, block in params.items():
        if not layer_name.startswith("layer_"):
            continue
        a, b = block["A"], block["B"]
        num_atoms = int(b.shape[0])
        representation["effective_rank"][layer_name] = float(effective_rank(a @ b))
        representation["invariant_redundancy"][layer_name] = float(
            invariant_redundancy(a, b)
        )
        layer_domain_effects: dict[str, list[float]] = {d.name: [] for d in domains}
        for atom in range(num_atoms):
            ablated = {key: dict(value_map) for key, value_map in params.items()}
            ablated[layer_name]["B"] = b.at[atom].set(jnp.zeros_like(b[atom]))
            for domain in domains:
                batches = batches_by_domain.get(domain.name, ())
                if batches:
                    full = float(loss_for(params, batches[0]))
                    ablated_loss = float(loss_for(ablated, batches[0]))
                    layer_domain_effects[domain.name].append(ablated_loss - full)
                else:
                    layer_domain_effects[domain.name].append(0.0)

        layer_gradients = [
            gradients[name][layer_name]["B"]
            for name in names
            if layer_name in gradients[name]
        ]
        if layer_gradients:
            stacked_grads = jnp.stack(layer_gradients)
            mean_gradient = jnp.mean(stacked_grads, axis=0)
            fisher = jnp.mean(stacked_grads**2, axis=0)
            atom_taylor = [
                taylor_delta(mean_gradient[i], fisher[i], -b[i])
                for i in range(num_atoms)
            ]
            metric_g = jnp.mean(fisher, axis=0)
        else:
            atom_taylor = [jnp.asarray(0.0)]
            metric_g = jnp.ones(int(b.shape[1]), dtype=jnp.float32)

        representation["taylor_delta"][layer_name] = float(
            jnp.sum(jnp.asarray(atom_taylor))
        )

        proxy_pairs: dict[str, float] = {}
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                delta_i = -b[i]
                delta_j = -b[j]
                proxy_pairs[f"{i}:{j}"] = float(
                    synergy_proxy(delta_i, delta_j, metric_g)
                )

        top_atoms = sorted(
            range(num_atoms),
            key=lambda a: float(
                np.mean([abs(layer_domain_effects[dn][a]) for dn in names])
            ),
            reverse=True,
        )[: min(num_atoms, 8)]

        measured_pairs: dict[str, float] = {}
        if len(top_atoms) >= 2 and names:
            eval_batch = batches_by_domain[names[0]][0]
            loss_full = float(loss_for(params, eval_batch))
            atom_single_losses: dict[int, float] = {}
            for atom_i in top_atoms:
                ab_i = {k: dict(v) for k, v in params.items()}
                ab_i[layer_name]["B"] = b.at[atom_i].set(jnp.zeros_like(b[atom_i]))
                atom_single_losses[atom_i] = float(loss_for(ab_i, eval_batch))

            for idx_i, atom_i in enumerate(top_atoms):
                loss_i = atom_single_losses[atom_i]
                for atom_j in top_atoms[idx_i + 1 :]:
                    loss_j = atom_single_losses[atom_j]
                    ab_ij = {k: dict(v) for k, v in params.items()}
                    ab_ij[layer_name]["B"] = b.at[atom_i].set(0.0).at[atom_j].set(0.0)
                    loss_ij = float(loss_for(ab_ij, eval_batch))
                    s_meas = float(
                        pairwise_ablation_interaction(
                            loss_full, loss_i, loss_j, loss_ij
                        )
                    )
                    measured_pairs[f"{atom_i}:{atom_j}"] = s_meas

        common_pairs = [k for k in measured_pairs if k in proxy_pairs]
        if len(common_pairs) >= 3:
            p_vals = [proxy_pairs[k] for k in common_pairs]
            m_vals = [measured_pairs[k] for k in common_pairs]
            val_corr = _spearman_corr(p_vals, m_vals)
        else:
            val_corr = None

        representation["synergy_proxy"][layer_name] = {
            "mean": float(np.mean(list(proxy_pairs.values()))) if proxy_pairs else 0.0,
            "pairs": proxy_pairs,
        }
        representation["measured_synergy"][layer_name] = {
            "mean": (
                float(np.mean(list(measured_pairs.values()))) if measured_pairs else 0.0
            ),
            "pairs": measured_pairs,
        }
        representation["synergy_validation"][layer_name] = {
            "spearman_correlation": val_corr,
            "num_pairs": len(common_pairs),
        }
        representation["synergy"][layer_name] = (
            float(np.mean(list(proxy_pairs.values()))) if proxy_pairs else 0.0
        )

    legal_metrics: dict[str, Any]
    try:
        runtime_config = load_runtime_config(runtime_config_path)
        evaluation_config = runtime_config.get("evaluation", {})
        max_examples = evaluation_config.get("max_examples")
        labeled = {}
        for task_spec in legal_task_specs(runtime_config):
            labeled[task_spec.name] = {
                split: load_labeled_split(
                    task_spec,
                    split,
                    substrate.tokenizer,
                    sequence_length=int(
                        evaluation_config.get("sequence_length", training.seq_len)
                    ),
                    max_examples=max_examples,
                )
                for split in ("train", "validation", "test")
            }

        activation = str(config.get("adapter", {}).get("activation", "identity"))
        for row in checkpoint_rows:
            row["params"] = canonicalize_adapter_params(
                row["params"], activation=activation
            )
        params = checkpoint_rows[-1]["params"]
        probe_layer = substrate.architecture.num_layers - 1
        if probe_layer in adapter.layers:
            raise ValueError("The legal probe layer must not be an adapted layer")

        def feature_fn(
            feature_params: dict[str, dict[str, jax.Array]], split: Any
        ) -> np.ndarray:
            def batch_features(ids: np.ndarray, positions: np.ndarray) -> np.ndarray:
                _, _, result = deployed_with_params(
                    feature_params,
                    jnp.asarray(ids),
                    (probe_layer,),
                )
                hidden = result.hidden_state(probe_layer)
                selected = hidden[jnp.arange(hidden.shape[0]), jnp.asarray(positions)]
                return np.asarray(selected, dtype=np.float64)

            return extract_split_features(
                split,
                batch_features,
                batch_size=int(
                    evaluation_config.get("batch_size", training.batch_size)
                ),
            )

        components = [
            {
                "name": f"{layer_name}/sv_{atom}",
                "layer": int(layer_name.removeprefix("layer_")),
                "layer_name": layer_name,
                "atom": atom,
            }
            for layer_name, block in params.items()
            for atom in range(int(block["B"].shape[0]))
        ]

        def ablate_fn(
            original: dict[str, dict[str, jax.Array]],
            component: dict[str, Any],
            retained: float,
        ) -> dict[str, dict[str, jax.Array]]:
            changed = {name: dict(block) for name, block in original.items()}
            layer_name = str(component["layer_name"])
            atom = int(component["atom"])
            changed[layer_name]["B"] = (
                changed[layer_name]["B"].at[atom].multiply(retained)
            )
            return changed

        evaluation_output = Path(
            run.get(
                "evaluation_output_dir",
                Path(run["output_dir"]) / run["run_name"] / "derived_evaluation",
            )
        )
        legal_metrics = evaluate_legal_tasks(
            labeled_splits=labeled,
            checkpoints=checkpoint_rows,
            feature_fn=feature_fn,
            components=components,
            ablate_fn=ablate_fn,
            output_dir=evaluation_output / "legal",
            alpha=float(evaluation_config.get("probe_l2", 1.0e-3)),
            bins=int(evaluation_config.get("calibration_bins", 15)),
            min_factor_support=int(evaluation_config.get("min_factor_support", 20)),
        )
        legal_metrics["feature_layer"] = probe_layer
        legal_metrics["feature_pooling"] = "last_non_padding_token"
        semantic = legal_metrics.get("representation", {})
        if semantic.get("status") in {"measured", "ok", "partial"}:
            representation["fracture_index"] = semantic["fracture"]
            representation["overlap"] = semantic["overlap_matrix"]
            representation["overlap_matrices"] = semantic["overlap_matrix"]
            representation["semantic_overlap"] = semantic["overlap_matrix"]
            representation["cross_domain_conflict"] = semantic["gradient_conflict"]
            representation["mixed_curvature_coupling"] = semantic[
                "mixed_curvature_coupling"
            ]
            representation["off_support_leakage"] = semantic["off_support_leakage"]
            representation["causal_fingerprint_separability"] = semantic[
                "causal_fingerprint_separability"
            ]
            representation["ablation_fidelity"] = semantic["ablation_fidelity"]
            representation["category_naturality"] = semantic["category_naturality"]
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        legal_metrics = {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
        }

    def evaluate_lm_batches(
        evaluation_params: dict[str, dict[str, jax.Array]],
        batches: tuple[jax.Array, ...],
    ) -> dict[str, Any]:
        rows = []
        for ids in batches:
            base_logits, adapted_logits, _ = deployed_with_params(
                evaluation_params, ids
            )
            token_count = int(ids.shape[0] * max(ids.shape[1] - 1, 0))
            adapted_loss = float(
                substrate.compute_loss(adapted_logits, ids, losses=("CE",))
            )
            base_loss = float(substrate.compute_loss(base_logits, ids, losses=("CE",)))
            rows.append(
                {
                    "tokens": token_count,
                    "loss_sum": adapted_loss * token_count,
                    "base_loss_sum": base_loss * token_count,
                    "kl_sum": float(kl_to_base(base_logits, adapted_logits))
                    * token_count,
                    "correct": float(token_accuracy(adapted_logits, ids)) * token_count,
                }
            )
        return aggregate_lm_rows(rows)

    runtime_config = load_runtime_config(runtime_config_path)
    evaluation_config = runtime_config.get("evaluation", {})
    anchor_config = evaluation_config.get("anchor")
    anchor_trajectory = []
    if isinstance(anchor_config, dict) and anchor_config.get("texts"):
        anchor_seq_len = min(32, training.seq_len)
        try:
            anchor_blocks = tokenize_texts(
                [str(text) for text in anchor_config["texts"]],
                substrate.tokenizer,
                sequence_length=anchor_seq_len,
                separator="\n\n",
                task_name=str(anchor_config.get("name", "anchor")),
            )
            anchor_batches = make_eval_set(
                anchor_blocks,
                min(training.batch_size, len(anchor_blocks)),
                training.eval_batches,
            )
            for checkpoint in checkpoint_rows:
                anchor_trajectory.append(
                    {
                        "checkpoint": int(checkpoint["index"]),
                        **evaluate_lm_batches(checkpoint["params"], anchor_batches),
                    }
                )
            anchor_kl: dict[str, Any] = {
                "status": "measured",
                "dataset": str(anchor_config.get("name", "anchor")),
                "trajectory": anchor_trajectory,
                "final": anchor_trajectory[-1],
            }
        except (RuntimeError, TypeError, ValueError) as exc:
            anchor_kl = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    else:
        anchor_kl = {
            "status": "not_configured",
            "reason": "anchor dataset not configured",
        }

    heldout_transfer: dict[str, Any] = {}
    for dataset_config in runtime_config.get("datasets", []):
        if dataset_config.get("type") != "heldout":
            continue
        name = str(dataset_config["name"])
        try:
            texts = load_split_texts(dataset_config, dataset_config["test_split"])
            blocks = tokenize_texts(
                texts,
                substrate.tokenizer,
                sequence_length=training.seq_len,
                separator=str(dataset_config.get("separator", "\n\n")),
                task_name=name,
            )
            batches = make_eval_set(
                blocks,
                training.batch_size,
                training.eval_batches,
            )
            trajectory = [
                {
                    "checkpoint": int(checkpoint["index"]),
                    **evaluate_lm_batches(checkpoint["params"], batches),
                }
                for checkpoint in checkpoint_rows
            ]
            final = trajectory[-1]
            heldout_transfer[name] = {
                "status": "measured",
                "trajectory": trajectory,
                "loss_transfer": float(final["base_loss"]) - float(final["loss"]),
            }
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            heldout_transfer[name] = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }

    long_range_config = evaluation_config.get("long_range")
    if isinstance(long_range_config, dict):

        def score_choice(prompt: str, choice: str) -> float:
            prompt_ids = substrate.tokenizer.encode(prompt, add_special_tokens=True)
            candidate = substrate.tokenizer.encode(choice, add_special_tokens=False)
            if not candidate:
                return float("-inf")
            full_ids = prompt_ids + candidate
            max_positions = substrate.architecture.max_position_embeddings
            if max_positions is not None and len(full_ids) > max_positions:
                raise ValueError("long-range probe exceeds model context length")
            ids = jnp.asarray([full_ids], dtype=jnp.int32)
            _, adapted_logits, _ = deployed_with_params(params, ids)
            log_probs = jax.nn.log_softmax(adapted_logits[0], axis=-1)
            start = len(prompt_ids) - 1
            values = [
                log_probs[start + offset, token]
                for offset, token in enumerate(candidate)
            ]
            return float(jnp.mean(jnp.stack(values)))

        try:
            long_range = evaluate_long_range(long_range_config, score_choice)
        except (RuntimeError, TypeError, ValueError) as exc:
            long_range = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
    else:
        long_range = {
            "status": "not_configured",
            "reason": "long-range probes are not configured",
        }
    symbolic_retrieval = {
        "status": "not_applicable",
        "reason": "A1/A2/A3 have no symbolic retrieval head; evaluate A4 separately",
    }
    peak_bytes = None
    for device in jax.devices():
        stats = device.memory_stats()
        if stats:
            peak_bytes = max(peak_bytes or 0, stats.get("peak_bytes_in_use", 0))
    return {
        "status": "measured",
        "domains": metrics_by_domain,
        "representation": representation,
        "legal_tasks": legal_metrics,
        "ops": {
            "inference_latency_mean_s": float(np.mean(latency)) if latency else None,
            "inference_latency_p95_s": (
                float(np.percentile(latency, 95)) if latency else None
            ),
            "peak_memory_bytes": peak_bytes,
        },
        "anchor_kl": anchor_kl,
        "heldout_transfer": heldout_transfer
        or {"status": "not_configured", "reason": "held-out datasets not configured"},
        "joint_gap": {
            "status": "deferred",
            "reason": "computed after matching sequential and joint manifest records",
        },
        "symbolic_retrieval": symbolic_retrieval,
        "long_range_coherence": long_range,
    }


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _config_fingerprint(config: dict[str, Any]) -> str:
    return _sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _filter_adapter_params(params: dict[str, Any]) -> dict[str, Any]:
    """Keep only deployable residual-adapter parameter blocks."""
    return {name: block for name, block in params.items() if name.startswith("layer_")}


def _compressibility_curve(deltas: dict[int, float]) -> dict[int, float]:

    return _compressibility_detail(deltas)["curve"] or {}


def _compressibility_detail(deltas: dict[int, float]) -> dict[str, Any]:
    from metrics.representation import guarded_compressibility

    return guarded_compressibility(deltas)


def _set_seed(config: dict[str, Any], method: str, seed: int) -> None:
    config.setdefault("adapter", {})["seed"] = seed
    config.setdefault("training", {})["seed"] = seed
    if method == "a2":
        config.setdefault("objective", {})["seed"] = seed
        hid = config["objective"].setdefault("hid", {})
        if not isinstance(hid, dict):
            raise ValueError("objective.hid must be a mapping")
        predictor = hid.setdefault("predictor", {})
        if not isinstance(predictor, dict):
            raise ValueError("objective.hid.predictor must be a mapping")
        predictor["seed"] = seed


def _set_epochs(config: dict[str, Any], epochs: int | None) -> None:
    if epochs is None:
        return
    config.setdefault("training", {})["epochs"] = epochs
    for domain in config.get("domains", []):
        if isinstance(domain, dict):
            domain["epochs"] = epochs


def _set_protocol(config: dict[str, Any], protocol: str) -> None:
    if protocol not in {"sequential", "joint", "heldout", "commutator"}:
        raise ValueError(f"Unsupported protocol: {protocol!r}")


def _set_rank(config: dict[str, Any], rank: int) -> None:
    if rank < 1:
        raise ValueError("rank must be >= 1")
    config.setdefault("adapter", {})["rank"] = rank


def _set_shared_training(config: dict[str, Any]) -> None:
    training = config.setdefault("training", {})
    training["learning_rate"] = BENCHMARK_LEARNING_RATE
    training["reset_optimizer"] = False


def _set_method_variant(config: dict[str, Any], spec: RunSpec) -> None:
    if spec.method == "a2" and spec.a2_arm == "full":
        objective = config.setdefault("objective", {})
        objective["cpc"] = {
            **objective.get("cpc", {}),
            "weight": 0.05,
            "positive": "base",
        }
        objective["stab"] = {
            **objective.get("stab", {}),
            "weight": 0.1,
            "w_anch": 1.0,
            "w_smooth": 0.0,
        }
        hid = {**objective.get("hid", {}), "weight": 0.1}
        hid["predictor"] = {**hid.get("predictor", {}), "detach_input": False}
        objective["hid"] = hid
        objective["hid_detach_input"] = False
    if spec.method == "a3" and spec.k is not None:
        config.setdefault("a3_hybrid", {})["infer_steps"] = spec.k


def _validate_domains(
    config: dict[str, Any],
    source: Path,
    runtime_source: str = BENCHMARK_RUNTIME_SOURCE,
) -> None:
    datasets = config.get("datasets")
    names = (
        tuple(
            dataset.get("name")
            for dataset in datasets
            if isinstance(dataset, dict) and dataset.get("type") == runtime_source
        )
        if isinstance(datasets, list)
        else ()
    )
    if names != EXPECTED_DOMAINS:
        raise ValueError(
            f"{source} must define {runtime_source!r} datasets in order "
            f"{list(EXPECTED_DOMAINS)}, "
            f"got {list(names)}"
        )


def expand_specs(
    output_dir: Path,
    *,
    seeds: list[int],
    methods: list[str],
    k_values: list[int],
    rank_values: list[int] | None = None,
    epochs: int | None = None,
    protocol: str = "sequential",
    task_order: list[str] | tuple[str, ...] | None = None,
    heldout_tasks: list[str] | tuple[str, ...] | None = None,
) -> list[RunSpec]:
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    unknown = sorted(set(methods) - set(DEFAULT_CONFIGS))
    if unknown:
        raise ValueError(
            f"Unknown methods: {unknown}; choose from {sorted(DEFAULT_CONFIGS)}"
        )
    if not k_values or any(k < 0 for k in k_values):
        raise ValueError("k_values must contain non-negative integers.")
    if len(set(k_values)) != len(k_values):
        raise ValueError("k_values must be unique")
    ranks = rank_values or [16]
    if not ranks or any(rank < 1 for rank in ranks):
        raise ValueError("rank_values must contain positive integers.")
    if len(set(ranks)) != len(ranks):
        raise ValueError("rank_values must be unique")
    specs: list[RunSpec] = []
    order = tuple(task_order or EXPECTED_DOMAINS)
    heldout = tuple(heldout_tasks or ())
    if set(order) != set(EXPECTED_DOMAINS) or len(order) != len(EXPECTED_DOMAINS):
        raise ValueError(
            f"task_order must contain {list(EXPECTED_DOMAINS)} exactly once"
        )
    if any(name not in EXPECTED_DOMAINS for name in heldout):
        raise ValueError(
            f"heldout_tasks must be selected from {list(EXPECTED_DOMAINS)}"
        )
    for method in methods:
        for seed in seeds:
            ks: list[int | None] = (
                [int(k) for k in k_values] if method == "a3" else [None]
            )
            for rank in ranks:
                for k in ks:
                    suffix = f"-r{rank}" + (f"-k{k}" if k is not None else "")
                    order_tag = "-".join(order)
                    heldout_tag = "" if not heldout else f"-holdout-{'-'.join(heldout)}"
                    run_id = f"{method}-seed{seed}{suffix}-{protocol}-{order_tag}{heldout_tag}"
                    attempts = output_dir / "runs" / run_id / "attempts"
                    specs.append(
                        RunSpec(
                            method=method,
                            seed=seed,
                            protocol=protocol,
                            run_id=run_id,
                            config=str(DEFAULT_CONFIGS[method]),
                            output_dir=str(attempts),
                            run_name="attempt-0001",
                            k=k,
                            rank=rank,
                            runtime_config=str(DEFAULT_RUNTIME_CONFIG),
                            task_order=order,
                            heldout_tasks=heldout,
                            a2_arm="full" if method == "a2" else None,
                        )
                    )
    return specs


def _materialize(spec: RunSpec, root: Path, epochs: int | None) -> Path:
    source = Path(spec.config)
    config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    runtime_source = Path(spec.runtime_config)
    runtime_config = yaml.safe_load(runtime_source.read_text(encoding="utf-8")) or {}
    _validate_domains(runtime_config, runtime_source, spec.runtime_source)
    _set_seed(config, spec.method, spec.seed)
    _set_rank(config, spec.rank)
    _set_shared_training(config)
    _set_epochs(config, epochs)
    _set_protocol(config, spec.protocol)
    _set_method_variant(config, spec)
    target = root / "configs" / f"{spec.run_id}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target


def _materialize_runtime(spec: RunSpec, root: Path) -> Path:
    source = Path(spec.runtime_config)
    config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    config.setdefault("runtime", {})["source"] = spec.runtime_source
    config["runtime"]["training_mode"] = (
        "joint" if spec.protocol == "joint" else "sequential"
    )
    config["runtime"]["task_order"] = list(spec.task_order)
    config["runtime"]["heldout_tasks"] = list(spec.heldout_tasks)
    target = root / "configs" / f"{spec.run_id}.runtime.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target


def _condition(
    spec: RunSpec,
    config: Path,
    runtime_config: Path,
    epochs: int | None,
    source_sha256: str,
) -> dict[str, Any]:
    return effective_condition(
        method=spec.method,
        seed=spec.seed,
        rank=spec.rank,
        k=spec.k,
        protocol=spec.protocol,
        task_order=spec.task_order,
        heldout_tasks=spec.heldout_tasks,
        a2_arm=spec.a2_arm,
        stage_config_sha256=sha256_file(config),
        runtime_config_sha256=sha256_file(runtime_config),
        runtime_source=spec.runtime_source,
        epochs=epochs,
        source_sha256=source_sha256,
    )


def make_manifest(
    output_dir: Path,
    *,
    seeds: list[int],
    methods: list[str],
    k_values: list[int],
    rank_values: list[int] | None = None,
    epochs: int | None = None,
    protocol: str = "sequential",
    protocols: list[str] | tuple[str, ...] | None = None,
    task_order: list[str] | tuple[str, ...] | None = None,
    heldout_tasks: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    selected_protocols = tuple(protocols or (protocol,))
    if len(set(selected_protocols)) != len(selected_protocols):
        raise ValueError("protocols must be unique")
    specs = []
    base_order = tuple(task_order or EXPECTED_DOMAINS)
    for selected in selected_protocols:
        if selected == "commutator":
            orders = (base_order, tuple(reversed(base_order)))
            heldout_variants = ((),)
        elif selected == "heldout" and not heldout_tasks:
            orders = (base_order,)
            heldout_variants = tuple((name,) for name in EXPECTED_DOMAINS)
        else:
            orders = (base_order,)
            heldout_variants = (tuple(heldout_tasks or ()),)
        for order in orders:
            for heldout in heldout_variants:
                specs.extend(
                    expand_specs(
                        output_dir,
                        seeds=seeds,
                        methods=methods,
                        k_values=k_values,
                        rank_values=rank_values,
                        epochs=epochs,
                        protocol=selected,
                        task_order=order,
                        heldout_tasks=heldout,
                    )
                )
    source = source_fingerprint(ROOT)
    run_rows = []
    for spec in specs:
        config = _materialize(spec, output_dir, epochs)
        runtime_config = _materialize_runtime(spec, output_dir)
        condition = _condition(
            spec, config, runtime_config, epochs, str(source["source_sha256"])
        )
        run_rows.append(
            {
                **asdict(spec),
                "status": "pending",
                "materialized_config": str(config),
                "materialized_runtime_config": str(runtime_config),
                **condition,
            }
        )
    manifest = {
        "schema": 2,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repository": str(ROOT),
        "protocols": list(selected_protocols),
        "methods": methods,
        "seeds": seeds,
        "k_values": k_values,
        "rank_values": rank_values or [16],
        "epochs": epochs,
        "domains": list(EXPECTED_DOMAINS),
        "runtime_config": str(DEFAULT_RUNTIME_CONFIG),
        "runtime_source": BENCHMARK_RUNTIME_SOURCE,
        "source": source,
        "measured_metrics": list(MEASURED_METRICS),
        "runs": run_rows,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _update_manifest(path: Path, run_id: str, **fields: Any) -> None:
    manifest = _read_json(path)
    for run in manifest["runs"]:
        if run["run_id"] == run_id:
            run.update(fields)
            break
    else:
        raise KeyError(run_id)
    _write_json(path, manifest)


def _command(
    spec: RunSpec, config: Path, runtime_config: Path | None = None
) -> list[str]:
    runtime_config = runtime_config or Path(spec.runtime_config)
    command = [
        sys.executable,
        str(RUNNERS[spec.method]),
        "--config",
        str(config),
        "--output-dir",
        spec.output_dir,
        "--run-name",
        spec.run_name,
        "--runtime-source",
        spec.runtime_source,
        "--runtime-config",
        str(runtime_config),
        "--training-mode",
        "joint" if spec.protocol == "joint" else "sequential",
        "--task-order",
        *spec.task_order,
    ]
    for name in spec.heldout_tasks:
        command.extend(["--heldout-task", name])
    if spec.method == "a2" and spec.a2_arm is not None:
        command.extend(["--arm", spec.a2_arm])
    return command


def train_manifest(
    output_dir: Path, *, resume: bool = True, epochs: int | None = None
) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    config_root = output_dir
    planned_source_sha = str(manifest.get("source", {}).get("source_sha256", ""))
    for raw in manifest["runs"]:
        current_source = source_fingerprint(ROOT)
        if planned_source_sha and current_source["source_sha256"] != planned_source_sha:
            raise RuntimeError(
                "Repository source changed after planning; create a new manifest before training."
            )
        spec = RunSpec(
            method=raw["method"],
            seed=raw["seed"],
            protocol=raw["protocol"],
            run_id=raw["run_id"],
            config=raw["config"],
            output_dir=raw["output_dir"],
            run_name=raw["run_name"],
            k=raw.get("k"),
            rank=raw.get("rank", 16),
            runtime_source=raw.get("runtime_source", BENCHMARK_RUNTIME_SOURCE),
            runtime_config=raw.get("runtime_config", str(DEFAULT_RUNTIME_CONFIG)),
            task_order=tuple(raw.get("task_order", EXPECTED_DOMAINS)),
            heldout_tasks=tuple(raw.get("heldout_tasks", ())),
            a2_arm=raw.get("a2_arm", "full" if raw["method"] == "a2" else None),
        )
        effective_epochs = epochs if epochs is not None else manifest.get("epochs")
        config = _materialize(
            spec,
            config_root,
            effective_epochs,
        )
        runtime_config = _materialize_runtime(spec, config_root)
        condition = _condition(
            spec,
            config,
            runtime_config,
            effective_epochs,
            str(current_source["source_sha256"]),
        )
        config_sha = sha256_file(config)
        previous_run = Path(raw["run_dir"]) if raw.get("run_dir") else None
        if resume and previous_run is not None:
            reusable, reason = validate_complete_attempt(
                previous_run,
                condition_sha256=condition["condition_sha256"],
                config_sha256=config_sha,
            )
            if reusable:
                _update_manifest(
                    manifest_path,
                    spec.run_id,
                    status="complete",
                    results=str(previous_run / "results.json"),
                    resume_validation="verified",
                )
                continue
            raw["resume_rejection"] = reason

        attempts_root = Path(spec.output_dir)
        attempt_name = next_attempt_name(attempts_root)
        attempt_spec = replace(spec, run_name=attempt_name)
        run_dir = attempts_root / attempt_name
        results = run_dir / "results.json"
        _update_manifest(
            manifest_path,
            spec.run_id,
            status="running",
            materialized_config=str(config),
            materialized_runtime_config=str(runtime_config),
            config_sha=config_sha,
            run_name=attempt_name,
            run_dir=str(run_dir),
            **condition,
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        log_path = output_dir / "logs" / f"{spec.run_id}-{attempt_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                _command(attempt_spec, config, runtime_config),
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        if run_dir.is_dir():
            run_log = run_dir / "logs" / "stdout.log"
            run_log.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(log_path, run_log)
        if completed.returncode != 0:
            internal_log = run_dir / "logs" / "stdout.log"
            _update_manifest(
                manifest_path,
                spec.run_id,
                status="failed",
                returncode=completed.returncode,
                log=str(internal_log if internal_log.is_file() else log_path),
                run_dir=str(run_dir),
            )
            try:
                log_tail = "\n".join(
                    log_path.read_text(encoding="utf-8", errors="replace").splitlines()[
                        -40:
                    ]
                )
            except OSError as exc:
                log_tail = f"Unable to read failure log: {exc}"
            raise RuntimeError(
                f"{spec.run_id} failed with exit code {completed.returncode}; "
                f"see {log_path}\n\nLast log lines:\n{log_tail}"
            )
        if not results.is_file():
            raise RuntimeError(
                f"{spec.run_id} exited successfully but wrote no results.json"
            )
        provenance_dir = run_dir / "provenance"
        provenance_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config, provenance_dir / "materialized_stage.yaml")
        shutil.copy2(runtime_config, provenance_dir / "materialized_runtime.yaml")
        completion_spec = {
            **condition,
            "run_id": spec.run_id,
            "attempt": attempt_name,
            "config_sha256": config_sha,
            "runtime_config_sha256": sha256_file(runtime_config),
            "source": current_source,
        }
        finalize_attempt(run_dir, completion_spec)
        _update_manifest(
            manifest_path,
            spec.run_id,
            status="complete",
            returncode=0,
            results=str(results),
            log=str(run_dir / "logs" / "stdout.log"),
            run_dir=str(run_dir),
            finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )


def _result_record(raw: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    test = raw.get("test", {})
    domains = raw.get("domains", [])
    losses = raw.get("loss", [])
    final_losses = losses[-1] if losses else []
    perplexities = raw.get("ppl", [])
    base_losses = raw.get("base_loss", [])
    final_mean_loss = sum(final_losses) / len(final_losses) if final_losses else None
    base_mean_loss = sum(base_losses) / len(base_losses) if base_losses else None
    if run.get("protocol") == "joint":
        acquisition_losses = list(final_losses)
        acquisition_improvement = {
            str(name): float(base) - float(loss)
            for name, base, loss in zip(
                domains, base_losses, final_losses, strict=False
            )
        }
    else:
        acquisition_losses = [
            losses[index][index]
            for index in range(min(len(domains), len(losses)))
            if index < len(losses[index])
        ]
        acquisition_improvement = (
            {
                str(domains[index]): float(base_losses[index])
                - float(losses[index][index])
                for index in range(min(len(domains), len(base_losses), len(losses)))
                if index < len(losses[index])
            }
            if domains
            else None
        )
    return {
        "run_id": run["run_id"],
        "method": run["method"],
        "seed": run["seed"],
        "k": run.get("k"),
        "rank": run.get("rank", 16),
        "protocol": run.get("protocol", "sequential"),
        "task_order": run.get("task_order", list(EXPECTED_DOMAINS)),
        "heldout_tasks": run.get("heldout_tasks", []),
        "condition_sha256": run.get("condition_sha256"),
        "domains": domains,
        "final_mean_loss": final_mean_loss,
        "domain_loss_improvement": acquisition_improvement,
        "loss_improvement": (
            base_mean_loss - (sum(acquisition_losses) / len(acquisition_losses))
            if base_mean_loss is not None and acquisition_losses
            else None
        ),
        "final_mean_ppl": (
            math.exp(final_mean_loss) if final_mean_loss is not None else None
        ),
        "final_test_mean_loss": (
            sum(float(row["loss"]) for row in test.values()) / len(test)
            if test
            else None
        ),
        "final_test_mean_ppl": (
            math.exp(sum(float(row["loss"]) for row in test.values()) / len(test))
            if test
            else None
        ),
        "loss_matrix": losses,
        "perplexity_matrix": perplexities,
        "mean_forgetting": (
            raw.get("mean_forgetting", [])[-1] if raw.get("mean_forgetting") else None
        ),
        "accuracy_forgetting": (
            raw.get("accuracy_forgetting", [])[-1]
            if raw.get("accuracy_forgetting")
            else None
        ),
        "task_accuracy": (
            float(np.nanmean(np.asarray(raw["task_accuracy"], dtype=float)[-1]))
            if raw.get("task_accuracy")
            and np.isfinite(np.asarray(raw["task_accuracy"], dtype=float)[-1]).any()
            else None
        ),
        "task_f1": (
            float(np.nanmean(np.asarray(raw["task_f1"], dtype=float)[-1]))
            if raw.get("task_f1")
            and np.isfinite(np.asarray(raw["task_f1"], dtype=float)[-1]).any()
            else None
        ),
        "steps": raw.get("steps"),
        "tokens": raw.get("tokens"),
        "mean_step_time": raw.get("mean_step_time"),
        "params_unchanged": raw.get("params_unchanged"),
        "results": run.get("results"),
        "heldout_transfer": {
            name: float(test[name]["base_loss"]) - float(test[name]["loss"])
            for name in run.get("heldout_tasks", [])
            if name in test
        }
        or None,
    }


def evaluate_manifest(output_dir: Path) -> dict[str, Any]:
    manifest = _read_json(output_dir / "manifest.json")
    records = []
    fingerprints = {}
    for run in manifest["runs"]:
        result_path = Path(run.get("results", ""))
        if not result_path.is_file():
            continue
        config_path = Path(run.get("materialized_config", run["config"]))
        config_bytes = config_path.read_bytes()
        fingerprints[run["run_id"]] = _sha256(config_bytes)
        if manifest.get("schema", 1) >= 2:
            run_dir = Path(run.get("run_dir", result_path.parent))
            valid, reason = validate_complete_attempt(
                run_dir,
                condition_sha256=str(run.get("condition_sha256", "")),
                config_sha256=sha256_file(config_path),
            )
            if not valid:
                records.append(
                    {
                        "run_id": run["run_id"],
                        "method": run["method"],
                        "seed": run["seed"],
                        "rank": run.get("rank", 16),
                        "k": run.get("k"),
                        "protocol": run.get("protocol"),
                        "status": "invalid",
                        "reason": reason,
                    }
                )
                continue
        raw_result = _read_json(result_path)
        record = _result_record(raw_result, run)
        meta_path = result_path.parent / "meta.json"
        if meta_path.is_file():
            meta = _read_json(meta_path)
            record["trainable_params"] = meta.get("trainable_params")
            record["trainable_params_by_group"] = meta.get("trainable_params_by_group")
            if isinstance(meta.get("runtime_data"), dict):
                record["runtime_data_sha256"] = _config_fingerprint(
                    meta["runtime_data"]
                )
        try:
            evaluation_run = {
                **run,
                "evaluation_output_dir": str(
                    output_dir / "evaluation_artifacts" / run["run_id"]
                ),
            }
            deployed = _load_deployed_metrics(evaluation_run, raw_result)
            record["deployed_evaluation"] = deployed
            evaluation_dir = Path(evaluation_run["evaluation_output_dir"])
            if evaluation_dir.is_dir():
                record["evaluation_artifacts"] = {
                    path.relative_to(evaluation_dir).as_posix(): sha256_file(path)
                    for path in sorted(evaluation_dir.rglob("*"))
                    if path.is_file()
                }
            legal = deployed.get("legal_tasks", {})
            aggregate = legal.get("aggregate", {})
            if legal.get("status") == "measured":
                record["task_accuracy"] = aggregate.get("task_accuracy")
                record["task_f1"] = aggregate.get("task_macro_f1")
                record["task_primary_score"] = aggregate.get("task_primary_score")
                record["accuracy_forgetting"] = aggregate.get("task_score_forgetting")
        except (ImportError, KeyError, OSError, RuntimeError, ValueError) as exc:
            record["deployed_evaluation"] = {
                "status": "unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        records.append(record)
    by_group: dict[tuple[Any, ...], dict[int, float]] = {}
    by_group_domain: dict[tuple[Any, ...], dict[str, dict[int, float]]] = {}
    for record in records:
        if record.get("status") == "invalid":
            continue
        group_key = (
            record["method"],
            record["seed"],
            record.get("k"),
            record.get("protocol"),
            tuple(record.get("task_order", ())),
            tuple(record.get("heldout_tasks", ())),
        )
        loss = record.get("loss_improvement")
        if loss is not None:
            by_group.setdefault(group_key, {})[record["rank"]] = loss
        for name, value in (record.get("domain_loss_improvement") or {}).items():
            by_group_domain.setdefault(group_key, {}).setdefault(name, {})[
                record["rank"]
            ] = value
    for record in records:
        if record.get("status") == "invalid":
            continue
        group_key = (
            record["method"],
            record["seed"],
            record.get("k"),
            record.get("protocol"),
            tuple(record.get("task_order", ())),
            tuple(record.get("heldout_tasks", ())),
        )
        detail = _compressibility_detail(by_group.get(group_key, {}))
        record["compressibility_status"] = detail["status"]
        record["compressibility_detail"] = {
            k: v for k, v in detail.items() if k != "curve"
        }
        record["residual_compressibility"] = (
            {str(rank): float(v) for rank, v in detail["curve"].items()}
            if detail["curve"]
            else None
        )
        per_domain = {}
        for name, deltas in by_group_domain.get(group_key, {}).items():
            d = _compressibility_detail(deltas)
            per_domain[name] = {
                "status": d["status"],
                "reference_improvement": d["reference_improvement"],
                "curve": (
                    {str(r): float(v) for r, v in d["curve"].items()}
                    if d["curve"]
                    else None
                ),
            }
        record["residual_compressibility_by_domain"] = per_domain or None
    reference = {
        (
            record["seed"],
            record["rank"],
            record.get("protocol"),
            tuple(record.get("task_order", ())),
            tuple(record.get("heldout_tasks", ())),
        ): record["final_mean_loss"]
        for record in records
        if record.get("method") == "a1" and record.get("status") != "invalid"
    }
    for record in records:
        if record.get("status") == "invalid":
            continue
        baseline = reference.get(
            (
                record["seed"],
                record["rank"],
                record.get("protocol"),
                tuple(record.get("task_order", ())),
                tuple(record.get("heldout_tasks", ())),
            )
        )
        record["paired_delta_vs_a1"] = (
            record["final_mean_loss"] - baseline
            if baseline is not None and record["method"] != "a1"
            else None
        )

    joint_reference = {
        (
            record["method"],
            record["seed"],
            record["rank"],
            record.get("k"),
            tuple(record.get("task_order", ())),
        ): record
        for record in records
        if record.get("protocol") == "joint" and record.get("status") != "invalid"
    }
    for record in records:
        if record.get("status") == "invalid":
            continue
        key = (
            record["method"],
            record["seed"],
            record["rank"],
            record.get("k"),
            tuple(record.get("task_order", ())),
        )
        joint_record = joint_reference.get(key)
        budgets_match = bool(
            joint_record
            and record.get("steps") == joint_record.get("steps")
            and record.get("tokens") == joint_record.get("tokens")
        )
        record["sequential_joint_gap"] = (
            record["final_mean_loss"] - joint_record["final_mean_loss"]
            if record.get("protocol") == "sequential"
            and record.get("final_mean_loss") is not None
            and joint_record is not None
            and joint_record.get("final_mean_loss") is not None
            and budgets_match
            else None
        )
        if (
            record.get("protocol") == "sequential"
            and joint_record
            and not budgets_match
        ):
            record["sequential_joint_gap_status"] = "invalid_budget_mismatch"

    commutator_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        if record.get("protocol") != "commutator" or record.get("status") == "invalid":
            continue
        key = (
            record["method"],
            record["seed"],
            record["rank"],
            record.get("k"),
        )
        commutator_groups.setdefault(key, []).append(record)
    for group in commutator_groups.values():
        if len(group) != 2:
            continue
        from stages.results import load_params as load_saved_params

        vectors = []
        for record in group:
            run = next(
                item for item in manifest["runs"] if item["run_id"] == record["run_id"]
            )
            checkpoint_dir = Path(run["output_dir"]) / run["run_name"] / "params"
            paths = sorted(checkpoint_dir.glob("task_*.npz"))
            loaded = (
                _filter_adapter_params(load_saved_params(paths[-1])) if paths else {}
            )
            vectors.append(
                np.concatenate(
                    [
                        (np.asarray(block["A"]) @ np.asarray(block["B"])).ravel()
                        for _, block in sorted(loaded.items())
                    ]
                )
                if loaded
                else np.asarray([])
            )
        if vectors[0].shape != vectors[1].shape or not vectors[0].size:
            continue
        denominator = 0.5 * (np.linalg.norm(vectors[0]) + np.linalg.norm(vectors[1]))
        value = float(np.linalg.norm(vectors[0] - vectors[1]) / max(denominator, 1e-8))
        for record in group:
            record["update_commutator"] = value
    evaluation = {
        "schema": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "domains": manifest.get("domains", list(EXPECTED_DOMAINS)),
        "measured_metrics": manifest.get("measured_metrics", list(MEASURED_METRICS)),
        "config_fingerprints": fingerprints,
        "records": records,
        "completed_runs": sum(row.get("status") != "invalid" for row in records),
        "invalid_runs": sum(row.get("status") == "invalid" for row in records),
        "planned_runs": len(manifest["runs"]),
    }
    _write_json(output_dir / "evaluation.json", evaluation)
    return evaluation


def _numeric_values(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return [float(value)]
    if isinstance(value, dict):
        values: list[float] = []
        for item in value.values():
            values.extend(_numeric_values(item))
        return values
    if isinstance(value, list | tuple):
        values = []
        for item in value:
            values.extend(_numeric_values(item))
        return values
    return []


def _report_metric_value(record: dict[str, Any], metric: str, rank: int) -> list[float]:
    deployed = record.get("deployed_evaluation", {})
    representation = deployed.get("representation", {})
    legal_representation = deployed.get("legal_tasks", {}).get("representation", {})
    if metric == "residual_compressibility":
        curve = record.get("residual_compressibility") or {}
        value = curve.get(str(rank))
    elif metric == "validation_loss":
        value = record.get("final_mean_loss")
    elif metric == "test_loss":
        value = record.get("final_test_mean_loss")
    elif metric == "perplexity":
        value = record.get("final_mean_ppl")
    elif metric == "test_perplexity":
        value = record.get("final_test_mean_ppl")
    elif metric == "forgetting":
        value = record.get("mean_forgetting")
    elif metric == "accuracy_forgetting":
        value = record.get("accuracy_forgetting")
    elif metric == "task_accuracy":
        value = record.get("task_accuracy")
    elif metric == "task_f1":
        value = record.get("task_f1")
    elif metric == "task_primary":
        value = record.get("task_primary_score")
    elif metric == "calibration":
        value = {
            domain: values.get("ece")
            for domain, values in deployed.get("domains", {}).items()
            if isinstance(values, dict)
        }
    elif metric == "gradient_conflict":
        value = legal_representation.get("gradient_conflict", {}).get("mean_conflict")
    elif metric == "factor_overlap":
        value = legal_representation.get("overlap_mean_off_diagonal")
    elif metric == "fracture":
        value = legal_representation.get("fracture_mean")
    elif metric == "mixed_curvature":
        value = legal_representation.get("mixed_curvature_coupling")
    elif metric == "off_support_leakage":
        value = legal_representation.get("off_support_leakage")
    elif metric == "update_commutator":
        value = record.get("update_commutator")
    elif metric == "causal_fingerprint":
        value = legal_representation.get("causal_fingerprint_separability", {}).get(
            "score"
        )
    elif metric == "category_alignment":
        naturality = legal_representation.get("category_naturality", {})
        value = (
            naturality.get("value")
            if naturality.get("status") in {"measured", "measured_proxy"}
            else None
        )
    elif metric == "ablation_fidelity":
        value = legal_representation.get("ablation_fidelity", {}).get("spearman")
    elif metric == "heldout_transfer":
        values = []
        if record.get("heldout_transfer"):
            values.extend(record["heldout_transfer"].values())
        for row in deployed.get("heldout_transfer", {}).values():
            if isinstance(row, dict) and row.get("status") == "measured":
                values.append(row.get("loss_transfer"))
        value = values
    elif metric == "sequential_joint_gap":
        value = record.get("sequential_joint_gap")
    elif metric == "anchor_kl":
        anchor = deployed.get("anchor_kl", {})
        value = (
            anchor.get("final", {}).get("kl")
            if anchor.get("status") == "measured"
            else None
        )
    elif metric == "symbolic_retrieval":
        symbolic = deployed.get("symbolic_retrieval", {})
        value = (
            symbolic.get("f1", {}).get("@4")
            if symbolic.get("status") == "measured"
            else None
        )
    elif metric == "long_range":
        coherence = deployed.get("long_range_coherence", {})
        value = (
            coherence.get("accuracy") if coherence.get("status") == "measured" else None
        )
    elif metric == "trainable_params":
        value = record.get("trainable_params")
    elif metric == "tokens_per_second":
        value = (
            record["tokens"] / (record["steps"] * record["mean_step_time"])
            if record.get("tokens")
            and record.get("steps")
            and record.get("mean_step_time")
            else None
        )
    elif metric == "inference_latency":
        value = deployed.get("ops", {}).get("inference_latency_mean_s")
    elif metric == "peak_memory":
        value = deployed.get("ops", {}).get("peak_memory_bytes")
    elif metric == "taylor_delta":
        value = representation.get("taylor_delta")
    elif metric == "synergy_validation":
        value = representation.get("synergy_validation")
    elif metric == "synergy":
        value = representation.get("synergy")
    else:
        value = None
    return _numeric_values(value)


def _format_config_label(method: str, rank: int | None, k: int | None) -> str:
    m = method.upper()
    parts: list[str] = []
    if rank is not None:
        parts.append(f"r={rank}")
    if k is not None:
        parts.append(f"K={k}")
    if parts:
        return f"{m} ({', '.join(parts)})"
    return m


def _append_metric_table(
    lines: list[str],
    title: str,
    metric_specs: list[tuple[str, str]],
    configs: list[tuple[str, int | None, int | None]],
    by_config: dict[tuple[str, int | None, int | None], list[dict[str, Any]]],
    report_rank: int,
) -> None:
    labels = [_format_config_label(m, r, k) for m, r, k in configs]
    lines.extend(
        [
            "",
            f"## {title}",
            "",
            f"| Metric | {' | '.join(labels)} | Coverage |",
            f"|---|{'---:|' * len(labels)}---|",
        ]
    )
    for label, key in metric_specs:
        cells: list[str] = []
        measured = False
        for cfg in configs:
            values = [
                value
                for record in by_config.get(cfg, [])
                for value in _report_metric_value(record, key, report_rank)
            ]
            if values:
                measured = True
                cells.append(f"{sum(values) / len(values):.6g}")
            else:
                cells.append("N/A")
        coverage = "measured" if measured else "not measured"
        lines.append(f"| {label} | {' | '.join(cells)} | {coverage} |")


def _cell(values: list[float]) -> str:
    if not values:
        return "N/A"
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else None
    return (
        f"{mean:.6g}"
        + (f" +/- {std:.3g}" if std is not None else "")
        + f" (n={len(values)})"
    )


def report_manifest(output_dir: Path, report_rank: int = 4) -> Path:
    evaluation = _read_json(output_dir / "evaluation.json")
    records = [row for row in evaluation["records"] if row.get("status") != "invalid"]
    invalid = [row for row in evaluation["records"] if row.get("status") == "invalid"]
    external_specs = [
        ("Validation loss", "validation_loss", "lower"),
        ("Perplexity", "perplexity", "lower"),
        ("Held-out test loss", "test_loss", "lower"),
        ("Held-out test perplexity", "test_perplexity", "lower"),
        ("Task accuracy", "task_accuracy", "higher"),
        ("Task primary score", "task_primary", "higher"),
        ("Task macro-F1", "task_f1", "higher"),
        ("Token calibration ECE", "calibration", "lower"),
        ("Validation-loss forgetting", "forgetting", "lower"),
        ("Task-score forgetting", "accuracy_forgetting", "lower"),
        ("Held-out transfer", "heldout_transfer", "higher"),
        ("Sequential versus joint gap", "sequential_joint_gap", "lower"),
        ("Anchor KL drift", "anchor_kl", "lower"),
        ("Symbolic retrieval F1@4", "symbolic_retrieval", "higher"),
        ("Long-range coherence accuracy", "long_range", "higher"),
    ]
    internal_specs = [
        ("Residual compressibility", "residual_compressibility", "higher"),
        ("Semantic-factor fracture", "fracture", "context"),
        ("Semantic-factor overlap", "factor_overlap", "lower"),
        ("Semantic-factor gradient conflict", "gradient_conflict", "lower"),
        ("Mixed curvature coupling proxy", "mixed_curvature", "lower"),
        ("Off-support leakage", "off_support_leakage", "lower"),
        ("Update commutator", "update_commutator", "lower"),
        ("Causal fingerprint separability", "causal_fingerprint", "higher"),
        ("Lowercase category naturality proxy", "category_alignment", "lower"),
        ("Quadratic ablation fidelity Spearman", "ablation_fidelity", "higher"),
    ]
    operational_specs = [
        ("Trainable parameters", "trainable_params", "lower"),
        ("Training tokens/second", "tokens_per_second", "higher"),
        ("Inference latency (seconds)", "inference_latency", "lower"),
        ("Peak evaluator memory (bytes)", "peak_memory", "lower"),
    ]
    base_conditions = sorted(
        {
            (
                str(row.get("protocol", "sequential")),
                int(row.get("rank", 16)),
                tuple(row.get("task_order", EXPECTED_DOMAINS)),
                tuple(row.get("heldout_tasks", ())),
            )
            for row in records
        },
        key=lambda item: (item[0], item[1], item[2], item[3]),
    )
    lines = [
        "# Stage A Metrics Report",
        "",
        f"- Completed valid runs: {len(records)}/{evaluation['planned_runs']}",
        f"- Invalid runs excluded: {len(invalid)}",
        f"- Domains: {', '.join(evaluation.get('domains', []))}",
        "- Every numeric comparison below uses seeds present for all three stages.",
        "",
        "## Run Completeness",
        "",
        "| Run | Method | Protocol | Rank | K | Status | Reason |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for row in evaluation["records"]:
        lines.append(
            f"| {row.get('run_id')} | {row.get('method')} | {row.get('protocol', 'N/A')} | "
            f"{row.get('rank', 'N/A')} | {row.get('k', 'N/A')} | "
            f"{row.get('status', 'valid')} | {row.get('reason', '')} |"
        )

    lines.extend(
        [
            "",
            "## A3 K Selection",
            "",
            "| Protocol | Rank | Order | Selected K | Validation loss | Rule |",
            "|---|---:|---|---:|---:|---|",
        ]
    )
    selected_k_by_condition: dict[tuple[Any, ...], int] = {}
    for protocol, rank, order, heldout in base_conditions:
        candidates = {}
        for row in records:
            if (
                row.get("method") == "a3"
                and row.get("protocol") == protocol
                and row.get("rank", 16) == rank
                and tuple(row.get("task_order", ())) == order
                and tuple(row.get("heldout_tasks", ())) == heldout
                and row.get("k") is not None
                and row.get("final_mean_loss") is not None
            ):
                candidates.setdefault(int(row["k"]), []).append(
                    float(row["final_mean_loss"])
                )
        if candidates:
            selected_k = min(
                candidates, key=lambda value: (np.mean(candidates[value]), value)
            )
            selected_k_by_condition[(protocol, rank, order, heldout)] = selected_k
            lines.append(
                f"| {protocol} | {rank} | {' -> '.join(order)} | {selected_k} | "
                f"{np.mean(candidates[selected_k]):.6g} | lowest mean validation loss; "
                "smaller K breaks ties |"
            )

    summary_candidates = []
    for protocol, rank, order, heldout in base_conditions:
        a3_k_values = sorted(
            {
                row.get("k")
                for row in records
                if row.get("method") == "a3"
                and row.get("protocol") == protocol
                and row.get("rank", 16) == rank
                and tuple(row.get("task_order", ())) == order
                and tuple(row.get("heldout_tasks", ())) == heldout
            },
            key=lambda value: -1 if value is None else int(value),
        ) or [None]
        for k in a3_k_values:
            method_rows = {}
            for method in ("a1", "a2", "a3"):
                method_rows[method] = [
                    row
                    for row in records
                    if row.get("method") == method
                    and row.get("protocol") == protocol
                    and row.get("rank", 16) == rank
                    and tuple(row.get("task_order", ())) == order
                    and tuple(row.get("heldout_tasks", ())) == heldout
                    and (method != "a3" or row.get("k") == k)
                ]
            title = (
                f"protocol={protocol}, rank={rank}, A3 K={k}, "
                f"order={' -> '.join(order)}"
            )
            if heldout:
                title += f", heldout={','.join(heldout)}"
            candidate_seeds = (
                set.intersection(
                    *[
                        {int(row["seed"]) for row in method_rows[method]}
                        for method in ("a1", "a2", "a3")
                    ]
                )
                if all(method_rows.values())
                else set()
            )
            fair_seeds = {
                seed
                for seed in candidate_seeds
                if len(
                    {
                        row.get("runtime_data_sha256")
                        for method in ("a1", "a2", "a3")
                        for row in method_rows[method]
                        if int(row["seed"]) == seed
                    }
                )
                == 1
            }
            lines.extend(["", f"## Comparison: {title}", ""])
            for table_title, specs in (
                ("External Metrics", external_specs),
                ("Internal Metrics", internal_specs),
                ("Operational Metrics", operational_specs),
            ):
                lines.extend(
                    [
                        f"### {table_title}",
                        "",
                        "| Metric | A1 | A2 | A3 | A2-A1 paired delta | A3-A1 paired delta | Direction | Matched seeds |",
                        "|---|---:|---:|---:|---:|---:|---|---:|",
                    ]
                )
                for label, metric, direction in specs:
                    by_method_seed: dict[str, dict[int, float]] = {}
                    for method in ("a1", "a2", "a3"):
                        seed_values = {}
                        for row in method_rows[method]:
                            values = _report_metric_value(row, metric, rank)
                            if values:
                                seed_values[int(row["seed"])] = float(np.mean(values))
                        by_method_seed[method] = seed_values
                    metric_seed_sets = [
                        set(by_method_seed[method]) for method in ("a1", "a2", "a3")
                    ]
                    metric_seeds = (
                        set.intersection(*metric_seed_sets)
                        if all(metric_seed_sets)
                        else set()
                    )
                    metric_seeds &= fair_seeds
                    cells = []
                    means = {}
                    for method in ("a1", "a2", "a3"):
                        values = [
                            by_method_seed[method][seed]
                            for seed in sorted(metric_seeds)
                        ]
                        cells.append(_cell(values))
                        means[method] = float(np.mean(values)) if values else None
                    a2_delta = [
                        by_method_seed["a2"][seed] - by_method_seed["a1"][seed]
                        for seed in sorted(metric_seeds)
                    ]
                    a3_delta = [
                        by_method_seed["a3"][seed] - by_method_seed["a1"][seed]
                        for seed in sorted(metric_seeds)
                    ]
                    lines.append(
                        f"| {label} | {' | '.join(cells)} | {_cell(a2_delta)} | "
                        f"{_cell(a3_delta)} | {direction} | {len(metric_seeds)} |"
                    )
                    selected_k = selected_k_by_condition.get(
                        (protocol, rank, order, heldout)
                    )
                    if (
                        metric in {"test_loss", "task_f1", "forgetting"}
                        and metric_seeds
                        and (k == selected_k or selected_k is None)
                    ):
                        summary_candidates.append((title, label, direction, means))

            matrix_seeds = {
                seed
                for seed in fair_seeds
                if all(
                    any(
                        int(row["seed"]) == seed and row.get("loss_matrix")
                        for row in method_rows[method]
                    )
                    for method in ("a1", "a2", "a3")
                )
            }
            if matrix_seeds:
                domains = next(
                    row.get("domains", [])
                    for row in method_rows["a1"]
                    if row.get("domains")
                )
                lines.extend(
                    [
                        "",
                        "### Mean Validation Loss Matrix",
                        "",
                        f"| Stage / checkpoint | {' | '.join(domains)} |",
                        f"|---|{'---:|' * len(domains)}",
                    ]
                )
                for method in ("a1", "a2", "a3"):
                    matrices = [
                        np.asarray(row["loss_matrix"], dtype=float)
                        for row in method_rows[method]
                        if int(row["seed"]) in matrix_seeds
                    ]
                    mean_matrix = np.mean(np.stack(matrices), axis=0)
                    for checkpoint, values in enumerate(mean_matrix):
                        cells = " | ".join(f"{value:.6g}" for value in values)
                        lines.append(
                            f"| {method.upper()} after checkpoint {checkpoint} | {cells} |"
                        )

            task_names = sorted(
                {
                    task
                    for method in ("a1", "a2", "a3")
                    for row in method_rows[method]
                    if int(row["seed"]) in fair_seeds
                    for task in row.get("deployed_evaluation", {})
                    .get("legal_tasks", {})
                    .get("final", {})
                }
            )
            if task_names:
                lines.extend(
                    [
                        "",
                        "### Final Legal Task Metrics",
                        "",
                        "| Task / metric | A1 | A2 | A3 | Matched seeds |",
                        "|---|---:|---:|---:|---:|",
                    ]
                )
                for task in task_names:
                    for key, label in (
                        ("task_primary_score", "primary score"),
                        ("task_f1", "macro-F1"),
                    ):
                        by_method = {}
                        for method in ("a1", "a2", "a3"):
                            by_method[method] = {
                                int(row["seed"]): float(value)
                                for row in method_rows[method]
                                if int(row["seed"]) in fair_seeds
                                for value in [
                                    row.get("deployed_evaluation", {})
                                    .get("legal_tasks", {})
                                    .get("final", {})
                                    .get(task, {})
                                    .get(key)
                                ]
                                if value is not None
                            }
                        task_seed_sets = [
                            set(by_method[method]) for method in ("a1", "a2", "a3")
                        ]
                        task_seeds = (
                            set.intersection(*task_seed_sets)
                            if all(task_seed_sets)
                            else set()
                        )
                        cells = [
                            _cell(
                                [by_method[method][seed] for seed in sorted(task_seeds)]
                            )
                            for method in ("a1", "a2", "a3")
                        ]
                        lines.append(
                            f"| {task} {label} | {' | '.join(cells)} | {len(task_seeds)} |"
                        )

    lines.extend(["", "## General Summary", ""])
    if not summary_candidates:
        lines.append(
            "No complete matched A1/A2/A3 cohort is available; no comparative conclusion is valid."
        )
    else:
        for title, label, direction, means in summary_candidates:
            measured = {
                method: value for method, value in means.items() if value is not None
            }
            if not measured:
                continue
            best = (
                min(measured, key=measured.get)
                if direction == "lower"
                else max(measured, key=measured.get)
            )
            lines.append(
                f"- {title}: {best.upper()} has the best measured {label.lower()} "
                f"({measured[best]:.6g}) among matched seeds."
            )
        measured_internal = sum(
            bool(_report_metric_value(row, metric, int(row.get("rank", report_rank))))
            for row in records
            for _, metric, _ in internal_specs
        )
        total_internal = len(records) * len(internal_specs)
        lines.append(
            f"- Internal metric coverage: {measured_internal}/{total_internal} run-metric cells."
        )
        lines.append(
            "- Fracture and overlap use gold legal-label semantic factors and canonical "
            "singular components, never domain identity or raw adapter coordinates."
        )
        lines.append(
            "- N/A means a prerequisite was not configured or no fully matched cohort exists; "
            "it is never interpreted as zero."
        )
        lines.append(
            "- This summary is descriptive and intentionally does not collapse heterogeneous "
            "metrics into a single composite score."
        )

    path = output_dir / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
