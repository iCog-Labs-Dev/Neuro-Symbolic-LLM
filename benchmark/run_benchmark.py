"""Run the complete A1/A2/A3 benchmark and generate internal/external reports.

    python benchmark/run_benchmark.py
    python benchmark/run_benchmark.py report --output-dir runs/benchmark

Planning, training, evaluation, artifact validation and visualization live here.
Model training and metric formulas remain in their shared project modules.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import webbrowser
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from numbers import Integral, Real
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import numpy as np
import yaml

GENERATED_REPORT_PREFIX = "report_and_visualization/"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def source_fingerprint(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    diff = git("diff", "--binary", "HEAD")
    untracked = git("ls-files", "--others", "--exclude-standard")
    untracked_paths = (
        []
        if not untracked
        else sorted(
            path
            for path in untracked.splitlines()
            if path != GENERATED_REPORT_PREFIX.rstrip("/")
            and not path.startswith(GENERATED_REPORT_PREFIX)
        )
    )
    untracked_source_hashes = {
        relative: sha256_file(root / relative)
        for relative in untracked_paths
        if Path(relative).suffix.lower() in {".py", ".yaml", ".yml", ".toml"}
        and (root / relative).is_file()
    }
    payload = {
        "git_commit": commit,
        "tracked_diff_sha256": (
            None if diff is None else sha256_bytes(diff.encode("utf-8"))
        ),
        "untracked_paths": untracked_paths,
        "untracked_source_hashes": untracked_source_hashes,
    }
    payload["source_sha256"] = fingerprint(payload)
    return payload


def effective_condition(
    *,
    method: str,
    seed: int,
    rank: int,
    k: int | None,
    protocol: str,
    task_order: list[str] | tuple[str, ...],
    heldout_tasks: list[str] | tuple[str, ...],
    a2_arm: str | None,
    stage_config_sha256: str,
    runtime_config_sha256: str,
    runtime_source: str,
    epochs: int | None,
    source_sha256: str,
) -> dict[str, Any]:
    condition = {
        "schema": "nsllm.condition/v2",
        "method": method,
        "seed": int(seed),
        "rank": int(rank),
        "training_k": k,
        "deployment_k": 0 if method == "a3" else None,
        "protocol": protocol,
        "task_order": list(task_order),
        "heldout_tasks": sorted(heldout_tasks),
        "a2_arm": a2_arm,
        "stage_config_sha256": stage_config_sha256,
        "runtime_config_sha256": runtime_config_sha256,
        "runtime_source": runtime_source,
        "epochs": epochs,
        "source_sha256": source_sha256,
    }
    condition["condition_sha256"] = fingerprint(condition)
    return condition


def next_attempt_name(condition_dir: Path) -> str:
    condition_dir.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        name = f"attempt-{attempt:04d}"
        if (condition_dir / name).exists():
            attempt += 1
            continue
        claim = condition_dir / f"{name}.claim"
        try:
            claim.open("x", encoding="utf-8").close()
        except FileExistsError:
            attempt += 1
            continue
        return name


def finalize_attempt(run_dir: Path, spec: dict[str, Any]) -> None:
    """Write immutable run identity, hash every artifact, and mark completion."""
    (run_dir / "spec.json").write_text(
        json.dumps(spec, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    ignored = {"artifacts.json", "COMPLETE", "FAILED"}
    artifacts = {}
    for path in sorted(run_dir.rglob("*")):
        relative = path.relative_to(run_dir).as_posix()
        if path.is_file() and relative not in ignored:
            artifacts[relative] = {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    (run_dir / "artifacts.json").write_text(
        json.dumps(
            {"version": 1, "artifacts": artifacts},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "COMPLETE").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "artifact_index": "artifacts.json",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def validate_complete_attempt(
    run_dir: Path,
    *,
    condition_sha256: str,
    config_sha256: str,
) -> tuple[bool, str | None]:
    required = (
        run_dir / "COMPLETE",
        run_dir / "results.json",
        run_dir / "config.yaml",
        run_dir / "spec.json",
        run_dir / "artifacts.json",
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        return False, f"missing artifacts: {missing}"
    try:
        spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
        marker = json.loads((run_dir / "COMPLETE").read_text(encoding="utf-8"))
        index = json.loads((run_dir / "artifacts.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        return False, f"invalid completion metadata: {error}"
    if spec.get("condition_sha256") != condition_sha256:
        return False, "condition fingerprint mismatch"
    if spec.get("config_sha256") != config_sha256:
        return False, "effective config fingerprint mismatch"
    materialized_config = run_dir / "provenance" / "materialized_stage.yaml"
    if (
        not materialized_config.is_file()
        or sha256_file(materialized_config) != config_sha256
    ):
        return False, "materialized stage config artifact mismatch"
    runtime_config = run_dir / "provenance" / "materialized_runtime.yaml"
    expected_runtime = spec.get("runtime_config_sha256")
    if (
        not runtime_config.is_file()
        or not isinstance(expected_runtime, str)
        or sha256_file(runtime_config) != expected_runtime
    ):
        return False, "materialized runtime config artifact mismatch"
    if marker.get("status") != "COMPLETE":
        return False, "completion marker is not successful"
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, dict):
        return False, "artifact index is invalid"
    actual_paths = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.relative_to(run_dir).as_posix()
        not in {"artifacts.json", "COMPLETE", "FAILED"}
    }
    if actual_paths != set(artifacts):
        return False, "artifact index coverage mismatch"
    for relative, metadata in artifacts.items():
        path = run_dir / relative
        if not path.is_file() or sha256_file(path) != metadata.get("sha256"):
            return False, f"artifact hash mismatch: {relative}"
    return True, None


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def aggregate_lm_rows(
    rows: Sequence[Mapping[str, float | int]],
) -> dict[str, Any]:
    """Aggregate additive language-model statistics by evaluated token count."""
    if not rows:
        return {"status": "insufficient_data", "tokens": 0}

    tokens = 0
    totals: dict[str, list[float]] = {
        name: [] for name in ("loss_sum", "base_loss_sum", "kl_sum", "correct")
    }
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"LM row {index} must be a mapping")
        missing = {"tokens", *totals} - set(row)
        if missing:
            raise ValueError(f"LM row {index} is missing {sorted(missing)}")

        row_tokens = row["tokens"]
        if isinstance(row_tokens, bool) or not isinstance(row_tokens, Integral):
            raise ValueError(f"LM row {index} tokens must be a nonnegative integer")
        row_tokens = int(row_tokens)
        if row_tokens < 0:
            raise ValueError(f"LM row {index} tokens must be a nonnegative integer")

        values = {
            name: _finite_float(row[name], field=f"LM row {index} {name}")
            for name in totals
        }
        if not 0.0 <= values["correct"] <= row_tokens:
            raise ValueError(f"LM row {index} correct must be between 0 and tokens")

        tokens += row_tokens
        for name, value in values.items():
            totals[name].append(value)

    if tokens == 0:
        return {"status": "insufficient_data", "tokens": 0}

    loss = math.fsum(totals["loss_sum"]) / tokens
    base_loss = math.fsum(totals["base_loss_sum"]) / tokens
    try:
        perplexity = math.exp(loss)
        base_perplexity = math.exp(base_loss)
    except OverflowError as exc:
        raise ValueError("aggregated LM loss is too large to exponentiate") from exc

    return {
        "status": "measured",
        "tokens": tokens,
        "loss": loss,
        "base_loss": base_loss,
        "perplexity": perplexity,
        "base_perplexity": base_perplexity,
        "kl": math.fsum(totals["kl_sum"]) / tokens,
        "token_accuracy": math.fsum(totals["correct"]) / tokens,
    }


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
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
    "task_gradient_conflict",
    "token_accuracy",
    "expected_calibration_error",
    "kl_to_base",
    "paired_seed_delta",
    "parameter_update_commutator",
    "operator_update_commutator",
    "update_commutator",
    "heldout_transfer",
    "sequential_joint_gap",
    "anchor_kl",
    "inference_latency",
    "peak_memory",
)
EXTERNAL_METRIC_SPECS = (
    ("Validation loss", "validation_loss", "lower"),
    ("Perplexity", "perplexity", "lower"),
    ("Held-out test loss", "test_loss", "lower"),
    ("Held-out test perplexity", "test_perplexity", "lower"),
    ("Deployed LM loss", "deployed_loss", "lower"),
    ("Deployed LM perplexity", "deployed_perplexity", "lower"),
    ("Token accuracy", "token_accuracy", "higher"),
    ("Token calibration ECE", "calibration", "lower"),
    ("On-domain KL drift", "domain_kl", "lower"),
    ("Validation-loss forgetting", "forgetting", "lower"),
    ("Held-out transfer", "heldout_transfer", "higher"),
    ("Sequential versus joint gap", "sequential_joint_gap", "lower"),
    ("Anchor KL drift", "anchor_kl", "lower"),
)
INTERNAL_METRIC_SPECS = (
    ("Residual compressibility", "residual_compressibility", "higher"),
    ("Effective rank", "effective_rank", "context"),
    ("Invariant redundancy", "invariant_redundancy", "lower"),
    ("Task-gradient conflict", "task_gradient_conflict", "lower"),
    ("Parameter-order commutator proxy", "update_commutator", "lower"),
)
OPERATIONAL_METRIC_SPECS = (
    ("Trainable parameters", "trainable_params", "lower"),
    ("Training tokens/second", "tokens_per_second", "higher"),
    ("Inference latency (seconds)", "inference_latency", "lower"),
    ("Peak evaluator memory (bytes)", "peak_memory", "lower"),
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


def _load_deployed_metrics(
    run: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate the final checkpoint with the common deployed function.

    Heavy JAX/model imports are intentionally local so ``plan`` remains usable
    in lightweight environments.
    """
    import jax
    import jax.numpy as jnp

    from frozenllm.substrate import FrozenSubstrate, identity_modify
    from metrics.performance import (
        calibration_stats,
        ece_from_stats,
        kl_to_base,
        token_accuracy,
    )
    from metrics.representation import (
        effective_rank,
        gradient_conflict_summary,
        invariant_redundancy,
    )
    from residual import ResidualAdapter, ResidualConfig
    from runtime.data_loader import (
        load_runtime_config,
        load_runtime_data,
        load_split_texts,
        make_eval_set,
        tokenize_texts,
    )
    from stages.results import load_params
    from stages.stage_base import TrainingConfig

    if run.get("method") not in DEFAULT_CONFIGS:
        return {"status": "unavailable", "reason": "unsupported Stage A method"}

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
    runtime_config = load_runtime_config(runtime_config_path)
    evaluation_config = runtime_config.get("evaluation", {})
    calibration_bins = int(evaluation_config.get("calibration_bins", 15))
    domains = runtime_data.tasks
    test_sets = runtime_data.get_test_sets(
        training.batch_size,
        training.eval_batches,
    )
    metrics_by_domain: dict[str, dict[str, Any]] = {}
    batches_by_domain: dict[str, tuple[jax.Array, ...]] = {}
    latency: list[float] = []

    def deployed_with_params(
        deployed_params: dict[str, dict[str, jax.Array]],
        ids: jax.Array,
    ) -> tuple[jax.Array, jax.Array, Any]:
        if run["method"] != "a3":
            base_result = substrate.run_with_interception(
                ids, modify_fn=identity_modify, intercept_layers=adapter.layers
            )
            adapted_result = substrate.run_with_interception(
                ids,
                modify_fn=adapter.modify_fn(deployed_params),
                intercept_layers=adapter.layers,
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
            ids, modify_fn=modify, intercept_layers=adapter.layers
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
        rows: list[dict[str, float | int]] = []
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
                jax.block_until_ready(base)
                start = time.perf_counter()
                adapted = substrate.run_with_interception(
                    ids,
                    modify_fn=adapter.modify_fn(params),
                    intercept_layers=adapter.layers,
                ).logits
            jax.block_until_ready(adapted)
            latency.append(time.perf_counter() - start)
            c_sum, a_sum, cnt = calibration_stats(adapted, ids, bins=calibration_bins)
            conf_sums.append(c_sum)
            acc_sums.append(a_sum)
            count_sums.append(cnt)
            token_count = int(np.asarray(ids[:, 1:] != -100).sum())
            adapted_loss = float(substrate.compute_loss(adapted, ids, losses=("CE",)))
            base_loss = float(substrate.compute_loss(base, ids, losses=("CE",)))
            rows.append(
                {
                    "tokens": token_count,
                    "loss_sum": adapted_loss * token_count,
                    "base_loss_sum": base_loss * token_count,
                    "kl_sum": float(kl_to_base(base, adapted, ids)) * token_count,
                    "correct": float(token_accuracy(adapted, ids)) * token_count,
                }
            )
        if conf_sums:
            pooled_conf = jnp.sum(jnp.stack(conf_sums), axis=0)
            pooled_acc = jnp.sum(jnp.stack(acc_sums), axis=0)
            pooled_cnt = jnp.sum(jnp.stack(count_sums), axis=0)
            pooled_ece = float(ece_from_stats(pooled_conf, pooled_acc, pooled_cnt))
        else:
            pooled_ece = None
        aggregate = aggregate_lm_rows(rows)
        metrics_by_domain[domain.name] = {
            "status": aggregate["status"],
            "loss": aggregate.get("loss"),
            "ppl": aggregate.get("perplexity"),
            "token_accuracy": aggregate.get("token_accuracy"),
            "ece": pooled_ece,
            "kl": aggregate.get("kl"),
        }

    gradients: dict[str, Any] = {}
    for name, batches in batches_by_domain.items():
        if not batches:
            continue
        gradients[name] = jax.grad(
            lambda p, task_batches=batches: jnp.mean(
                jnp.stack([loss_for(p, ids) for ids in task_batches])
            )
        )(params)
    names = list(gradients)
    conflict_pairs = {}
    if names:
        summary = gradient_conflict_summary(gradients)
        conflict_matrix = np.asarray(summary["conflict"])
        for i, left in enumerate(names):
            for j in range(i + 1, len(names)):
                if np.isfinite(conflict_matrix[i, j]):
                    conflict_pairs[f"{left}:{names[j]}"] = float(conflict_matrix[i, j])

    representation: dict[str, Any] = {
        "effective_rank": {},
        "invariant_redundancy": {},
        "task_gradient_conflict": {
            "status": "measured" if conflict_pairs else "insufficient_data",
            "estimator": "gradient_of_mean_ce_over_configured_evaluation_batches",
            "evaluation_batches": {
                name: len(batches_by_domain[name]) for name in names
            },
            "pairs": conflict_pairs,
            "mean": (
                float(np.mean(list(conflict_pairs.values())))
                if conflict_pairs
                else None
            ),
            "max": max(conflict_pairs.values()) if conflict_pairs else None,
        },
    }
    for layer_name, block in params.items():
        if not layer_name.startswith("layer_"):
            continue
        a, b = block["A"], block["B"]
        representation["effective_rank"][layer_name] = float(effective_rank(a @ b))
        representation["invariant_redundancy"][layer_name] = float(
            invariant_redundancy(a, b)
        )

    def evaluate_lm_batches(
        evaluation_params: dict[str, dict[str, jax.Array]],
        batches: tuple[jax.Array, ...],
    ) -> dict[str, Any]:
        rows = []
        for ids in batches:
            base_logits, adapted_logits, _ = deployed_with_params(
                evaluation_params, ids
            )
            token_count = int(np.asarray(ids[:, 1:] != -100).sum())
            adapted_loss = float(
                substrate.compute_loss(adapted_logits, ids, losses=("CE",))
            )
            base_loss = float(substrate.compute_loss(base_logits, ids, losses=("CE",)))
            rows.append(
                {
                    "tokens": token_count,
                    "loss_sum": adapted_loss * token_count,
                    "base_loss_sum": base_loss * token_count,
                    "kl_sum": float(kl_to_base(base_logits, adapted_logits, ids))
                    * token_count,
                    "correct": float(token_accuracy(adapted_logits, ids)) * token_count,
                }
            )
        return aggregate_lm_rows(rows)

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
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
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

    peak_bytes = None
    for device in jax.devices():
        stats = device.memory_stats()
        value = stats.get("peak_bytes_in_use") if stats else None
        if value is not None:
            peak_bytes = max(peak_bytes or 0, int(value))
    return {
        "status": "measured",
        "domains": metrics_by_domain,
        "representation": representation,
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
    profile: str = "custom",
) -> dict[str, Any]:
    selected_protocols = tuple(protocols or (protocol,))
    if len(set(selected_protocols)) != len(selected_protocols):
        raise ValueError("protocols must be unique")
    specs = []
    base_order = tuple(task_order or EXPECTED_DOMAINS)
    orders: tuple[tuple[str, ...], ...]
    heldout_variants: tuple[tuple[str, ...], ...]
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
        "profile": profile,
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
    for run_index, raw in enumerate(manifest["runs"], start=1):
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
        print(
            f"Training {run_index}/{len(manifest['runs'])}: {spec.run_id}\nLog: {log_path}",
            flush=True,
        )
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
    final_improvement = {
        str(name): float(base) - float(loss)
        for name, base, loss in zip(domains, base_losses, final_losses, strict=False)
    } or None
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
        "domain_loss_improvement": final_improvement,
        "loss_improvement": (
            base_mean_loss - final_mean_loss
            if base_mean_loss is not None
            and final_mean_loss is not None
            and len(base_losses) == len(final_losses)
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
            record.get("runtime_data_sha256"),
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
            record.get("runtime_data_sha256"),
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
            record.get("runtime_data_sha256"),
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
                record.get("runtime_data_sha256"),
            )
        )
        record["paired_delta_vs_a1"] = (
            record["final_mean_loss"] - baseline
            if baseline is not None
            and record["method"] != "a1"
            and record.get("runtime_data_sha256")
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
            and record.get("steps") is not None
            and record.get("tokens") is not None
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
        commutator_key = (
            record["method"],
            record["seed"],
            record["rank"],
            record.get("k"),
        )
        commutator_groups.setdefault(commutator_key, []).append(record)
    for group in commutator_groups.values():
        if len(group) != 2:
            continue
        first_order = tuple(group[0].get("task_order", ()))
        second_order = tuple(group[1].get("task_order", ()))
        if (
            not first_order
            or first_order != second_order[::-1]
            or any(
                group[0].get(field) is None
                or group[0].get(field) != group[1].get(field)
                for field in ("steps", "tokens")
            )
        ):
            continue
        from stages.results import load_params as load_saved_params

        parameter_vectors = []
        operator_vectors = []
        for record in group:
            run = next(
                item for item in manifest["runs"] if item["run_id"] == record["run_id"]
            )
            checkpoint_dir = Path(run["output_dir"]) / run["run_name"] / "params"
            paths = sorted(checkpoint_dir.glob("task_*.npz"))
            loaded = (
                _filter_adapter_params(load_saved_params(paths[-1])) if paths else {}
            )
            parameter_vectors.append(
                np.concatenate(
                    [
                        array.ravel()
                        for _, block in sorted(loaded.items())
                        for array in (np.asarray(block["A"]), np.asarray(block["B"]))
                    ]
                )
                if loaded
                else np.asarray([])
            )
            operator_vectors.append(
                np.concatenate(
                    [
                        (np.asarray(block["A"]) @ np.asarray(block["B"])).ravel()
                        for _, block in sorted(loaded.items())
                    ]
                )
                if loaded
                else np.asarray([])
            )
        if (
            parameter_vectors[0].shape != parameter_vectors[1].shape
            or operator_vectors[0].shape != operator_vectors[1].shape
            or not parameter_vectors[0].size
            or not operator_vectors[0].size
        ):
            continue
        parameter_scale = 0.5 * (
            np.linalg.norm(parameter_vectors[0]) + np.linalg.norm(parameter_vectors[1])
        )
        operator_scale = 0.5 * (
            np.linalg.norm(operator_vectors[0]) + np.linalg.norm(operator_vectors[1])
        )
        parameter_value = float(
            np.linalg.norm(parameter_vectors[0] - parameter_vectors[1])
            / max(parameter_scale, 1e-8)
        )
        operator_value = float(
            np.linalg.norm(operator_vectors[0] - operator_vectors[1])
            / max(operator_scale, 1e-8)
        )
        for record in group:
            record["parameter_update_commutator"] = parameter_value
            record["operator_update_commutator"] = operator_value
            record["update_commutator"] = parameter_value
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
    if isinstance(value, int | float) and math.isfinite(float(value)):
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
    value: Any
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
    elif metric in {
        "deployed_loss",
        "deployed_perplexity",
        "token_accuracy",
        "calibration",
        "domain_kl",
    }:
        field = {
            "deployed_loss": "loss",
            "deployed_perplexity": "ppl",
            "token_accuracy": "token_accuracy",
            "calibration": "ece",
            "domain_kl": "kl",
        }[metric]
        value = {
            domain: values.get(field)
            for domain, values in deployed.get("domains", {}).items()
            if isinstance(values, dict) and values.get("status") == "measured"
        }
    elif metric == "effective_rank":
        value = representation.get("effective_rank")
    elif metric == "invariant_redundancy":
        value = representation.get("invariant_redundancy")
    elif metric == "task_gradient_conflict":
        value = representation.get("task_gradient_conflict", {}).get("mean")
    elif metric == "update_commutator":
        value = record.get(
            "parameter_update_commutator", record.get("update_commutator")
        )
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
    elif metric == "trainable_params":
        value = record.get("trainable_params")
    elif metric == "mean_step_time":
        value = record.get("mean_step_time")
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
    else:
        value = None
    return _numeric_values(value)


def metric_values_for_record(
    record: dict[str, Any], metric: str, rank: int
) -> list[float]:
    """Public read-only metric extraction shared by reports and visualizations."""
    return _report_metric_value(record, metric, rank)


def select_a3_k(
    records: list[dict[str, Any]],
    *,
    protocol: str,
    rank: int,
    order: tuple[str, ...],
    heldout: tuple[str, ...],
) -> dict[str, Any]:
    """Apply the report's validation-loss A3 K-selection rule."""
    candidates: dict[int, dict[tuple[int, str], float]] = {}
    for row in records:
        if (
            row.get("method") == "a3"
            and row.get("protocol") == protocol
            and row.get("rank", 16) == rank
            and tuple(row.get("task_order", ())) == order
            and tuple(row.get("heldout_tasks", ())) == heldout
            and row.get("k") is not None
            and row.get("final_mean_loss") is not None
            and isinstance(row.get("runtime_data_sha256"), str)
            and row.get("runtime_data_sha256")
        ):
            key = (int(row["seed"]), row["runtime_data_sha256"])
            group = candidates.setdefault(int(row["k"]), {})
            if key in group:
                raise ValueError(
                    f"duplicate A3 K-selection run for K={row['k']}, seed={key[0]}"
                )
            value = float(row["final_mean_loss"])
            if math.isfinite(value):
                group[key] = value
    common = (
        set.intersection(*(set(values) for values in candidates.values()))
        if candidates
        else set()
    )
    means = (
        {
            k: float(np.mean([values[key] for key in sorted(common)]))
            for k, values in candidates.items()
        }
        if common
        else {}
    )
    selected = min(means, key=lambda value: (means[value], value)) if means else None
    return {
        "selected_k": selected,
        "mean_validation_loss_by_k": means,
        "matched_seeds": sorted(seed for seed, _ in common),
        "rule": "lowest mean validation loss on seeds with matching data across all K values; smaller K breaks ties",
    }


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
    external_specs = EXTERNAL_METRIC_SPECS
    internal_specs = INTERNAL_METRIC_SPECS
    operational_specs = OPERATIONAL_METRIC_SPECS
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
        "# A1/A2/A3 Continuous Residual Benchmark Report",
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
        selection = select_a3_k(
            records,
            protocol=protocol,
            rank=rank,
            order=order,
            heldout=heldout,
        )
        selected_k = selection["selected_k"]
        if selected_k is not None:
            selected_k_by_condition[(protocol, rank, order, heldout)] = selected_k
            lines.append(
                f"| {protocol} | {rank} | {' -> '.join(order)} | {selected_k} | "
                f"{selection['mean_validation_loss_by_k'][selected_k]:.6g} | "
                f"{selection['rule']} |"
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
                if all(
                    row.get("runtime_data_sha256")
                    for method in ("a1", "a2", "a3")
                    for row in method_rows[method]
                    if int(row["seed"]) == seed
                )
                and len(
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
                        metric in {"test_loss", "token_accuracy", "forgetting"}
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
                        matrix_cells = " | ".join(f"{value:.6g}" for value in values)
                        lines.append(
                            f"| {method.upper()} after checkpoint {checkpoint} | {matrix_cells} |"
                        )

    lines.extend(["", "## General Summary", ""])
    if not summary_candidates:
        lines.append(
            "No complete matched A1/A2/A3 cohort is available; no comparative conclusion is valid."
        )
    else:
        for summary_title, label, direction, means in summary_candidates:
            measured = {
                method: value for method, value in means.items() if value is not None
            }
            if not measured:
                continue
            best = (
                min(measured, key=lambda method: measured[method])
                if direction == "lower"
                else max(measured, key=lambda method: measured[method])
            )
            lines.append(
                f"- {summary_title}: {best.upper()} has the best measured {label.lower()} "
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
            "- N/A means a prerequisite was not configured or no fully matched cohort exists; "
            "it is never interpreted as zero."
        )
        lines.append(
            "- This summary is descriptive and intentionally does not collapse heterogeneous "
            "metrics into a single composite score."
        )

    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This report covers continuous residual methods A1, A2, and A3. "
            "Symbolic-head metrics are outside its scope.",
        ]
    )

    path = output_dir / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STAGE_FOLDER_PATTERN = "stage_{stage}"
GENERAL_REPORT_PATTERN = "stage_{stage}_general_report.md"
STAGE_REPORT_PATTERN = "{method}_result_and_visualization.md"
FIGURES_FOLDER = "figures"
METHODS = ("a1", "a2", "a3")
METHOD_LABELS = {"a1": "A1", "a2": "A2", "a3": "A3"}
METHOD_COLORS = {"a1": "#3B82F6", "a2": "#F59E0B", "a3": "#10B981"}
FIGURE_DPI = 150
VISUALIZATION_OPERATIONAL_METRIC_SPECS = OPERATIONAL_METRIC_SPECS + (
    ("Mean training step time (seconds)", "mean_step_time", "lower"),
)


@dataclass(frozen=True)
class HeadlineCohort:
    protocol: str
    rank: int
    order: tuple[str, ...]
    heldout: tuple[str, ...]
    selected_k: int | None
    selection: dict[str, Any]
    records: dict[str, tuple[dict[str, Any], ...]]
    matched_seeds: tuple[int, ...]


@dataclass(frozen=True)
class RunArtifacts:
    record: dict[str, Any]
    result: dict[str, Any]
    meta: dict[str, Any]
    events: tuple[dict[str, Any], ...]
    run_dir: Path | None


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        return ()
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object at {path}:{line_number}")
        rows.append(value)
    return tuple(rows)


def _slug(value: str) -> str:
    result = []
    for character in value.lower():
        if character.isalnum():
            result.append(character)
        elif not result or result[-1] != "_":
            result.append("_")
    return "".join(result).strip("_") or "metric"


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    flattened = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_numeric(item, name))
    else:
        number = _finite(value)
        if number is not None and prefix:
            flattened[prefix] = number
    return flattened


def _sample_cell(values: Sequence[float], *, best: bool = False) -> str:
    if not values:
        return "N/A"
    mean = fmean(values)
    deviation = stdev(values) if len(values) > 1 else None
    text = f"{mean:.6g}"
    if deviation is not None:
        text += f" +/- {deviation:.3g}"
    text += f" (n={len(values)})"
    return f"**{text}**" if best else text


def _direction_text(direction: str) -> str:
    return {"higher": "higher ^", "lower": "lower v", "context": "context"}.get(
        direction, direction
    )


def _load_plotting():
    try:
        import matplotlib
        import pandas as pd

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return plt, pd
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            "Install the visualization extra to generate report images: pip install -e '.[visualization]'"
        ) from error


def _save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", metadata={})


def _placeholder(plt: Any, path: Path, title: str, reason: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.axis("off")
    axis.text(0.5, 0.58, title, ha="center", va="center", weight="bold", fontsize=13)
    axis.text(
        0.5,
        0.42,
        f"not available: {reason}",
        ha="center",
        va="center",
        wrap=True,
        fontsize=10,
    )
    _save_figure(figure, path)
    plt.close(figure)


def _image_markdown(path: Path, report_dir: Path, alt: str, plt_available: bool) -> str:
    if not plt_available:
        return f"_Figure unavailable: plotting dependency is not installed ({alt})._"
    relative = path.relative_to(report_dir).as_posix()
    return f"![{alt}]({relative})"


def _condition_key(
    record: Mapping[str, Any],
) -> tuple[str, int, tuple[str, ...], tuple[str, ...]]:
    return (
        str(record.get("protocol", "sequential")),
        int(record.get("rank", 16)),
        tuple(record.get("task_order", EXPECTED_DOMAINS)),
        tuple(record.get("heldout_tasks", ())),
    )


def _select_headline_cohort(
    records: Sequence[dict[str, Any]], report_rank: int
) -> HeadlineCohort | None:
    valid = [record for record in records if record.get("status") != "invalid"]
    conditions = sorted({_condition_key(record) for record in valid})
    candidates = []
    for condition in conditions:
        protocol, rank, order, heldout = condition
        methods = {
            str(record.get("method"))
            for record in valid
            if _condition_key(record) == condition
        }
        if set(METHODS) <= methods:
            candidates.append(condition)
    if not candidates:
        return None

    def preference(condition):
        protocol, rank, order, heldout = condition
        return (
            protocol != "sequential",
            bool(heldout),
            order != EXPECTED_DOMAINS,
            rank != report_rank,
            -rank,
            protocol,
            order,
            heldout,
        )

    protocol, rank, order, heldout = min(candidates, key=preference)
    selection = select_a3_k(
        valid,
        protocol=protocol,
        rank=rank,
        order=order,
        heldout=heldout,
    )
    selected_k = selection["selected_k"]
    method_records = {}
    for method in METHODS:
        rows = [
            record
            for record in valid
            if record.get("method") == method
            and _condition_key(record) == (protocol, rank, order, heldout)
            and (method != "a3" or record.get("k") == selected_k)
        ]
        method_records[method] = tuple(rows)
    seed_sets = [{int(row["seed"]) for row in method_records[m]} for m in METHODS]
    common = set.intersection(*seed_sets) if all(seed_sets) else set()
    matched = []
    for seed in sorted(common):
        fingerprints = {
            row.get("runtime_data_sha256")
            for method in METHODS
            for row in method_records[method]
            if int(row["seed"]) == seed
        }
        if (
            len(fingerprints) == 1
            and None not in fingerprints
            and "" not in fingerprints
        ):
            matched.append(seed)
    return HeadlineCohort(
        protocol=protocol,
        rank=rank,
        order=order,
        heldout=heldout,
        selected_k=selected_k,
        selection=selection,
        records=method_records,
        matched_seeds=tuple(matched),
    )


def _metric_samples(
    cohort: HeadlineCohort, metric: str
) -> dict[str, list[tuple[int, float]]]:
    by_seed: dict[str, dict[int, float]] = {}
    for method in METHODS:
        samples = {}
        for record in cohort.records[method]:
            seed = int(record["seed"])
            if seed not in cohort.matched_seeds:
                continue
            values = metric_values_for_record(record, metric, cohort.rank)
            if values:
                samples[seed] = float(np.mean(values))
        by_seed[method] = samples
    seed_sets = [set(by_seed[method]) for method in METHODS]
    common = set.intersection(*seed_sets) if all(seed_sets) else set()
    return {
        method: [(seed, by_seed[method][seed]) for seed in sorted(common)]
        for method in METHODS
    }


def _na_reason(method: str, metric: str, cohort: HeadlineCohort | None) -> str:
    if cohort is None:
        return (
            "no A1/A2/A3 condition has matching protocol, rank, order, and held-out set"
        )
    rows = cohort.records.get(method, ())
    if not rows:
        return "no run exists in the selected comparison cohort"
    if not cohort.matched_seeds:
        return "no seed with matching A1/A2/A3 runtime fingerprints"
    if metric == "residual_compressibility":
        statuses = sorted(
            {str(row.get("compressibility_status", "unavailable")) for row in rows}
        )
        return "compressibility unavailable: " + ", ".join(statuses)
    if metric == "sequential_joint_gap":
        return "matched sequential and joint runs with equal token/step budgets are unavailable"
    if metric == "update_commutator":
        return "both alternate task orders are required"
    if any(metric_values_for_record(row, metric, cohort.rank) for row in rows):
        return "metric is not available for all three stages on the same matched seeds"
    deployed_reasons = sorted(
        {
            str(row.get("deployed_evaluation", {}).get("reason"))
            for row in rows
            if row.get("deployed_evaluation", {}).get("status") != "measured"
            and row.get("deployed_evaluation", {}).get("reason")
        }
    )
    if deployed_reasons:
        return "; ".join(deployed_reasons)
    return "metric was not produced by the selected run profile"


def _plot_method_metric(
    plt: Any,
    path: Path,
    label: str,
    samples: Mapping[str, Sequence[tuple[int, float]]],
    reason: str,
) -> None:
    available = {method: list(rows) for method, rows in samples.items() if rows}
    if not available:
        _placeholder(plt, path, label, reason)
        return
    figure, axis = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(METHODS))
    means = []
    errors = []
    for method in METHODS:
        values = [value for _, value in samples.get(method, ())]
        means.append(float(np.mean(values)) if values else np.nan)
        errors.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
    axis.bar(
        positions,
        means,
        yerr=errors,
        capsize=5,
        color=[METHOD_COLORS[method] for method in METHODS],
        alpha=0.82,
        width=0.62,
    )
    for index, method in enumerate(METHODS):
        rows = samples.get(method, ())
        for point, (seed, value) in enumerate(rows):
            offset = (point - (len(rows) - 1) / 2) * 0.04
            axis.scatter(
                index + offset,
                value,
                color="#111827",
                s=22,
                zorder=3,
                label=f"seed {seed}" if index == 0 else None,
            )
    axis.set_xticks(positions, [METHOD_LABELS[method] for method in METHODS])
    axis.set_title(label)
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.set_axisbelow(True)
    _save_figure(figure, path)
    plt.close(figure)


def _metric_table(
    specs: Sequence[tuple[str, str, str]],
    cohort: HeadlineCohort | None,
    plt: Any,
    pd: Any,
    figures_dir: Path,
    report_dir: Path,
    category: str,
) -> tuple[list[str], list[str]]:
    lines = [
        f"## {category.title()} Metrics",
        "",
        "| Metric | A1 | A2 | A3 | Direction |",
        "|---|---:|---:|---:|---|",
    ]
    figures = []
    for label, metric, direction in specs:
        samples = (
            _metric_samples(cohort, metric)
            if cohort is not None
            else {method: [] for method in METHODS}
        )
        values = {method: [value for _, value in samples[method]] for method in METHODS}
        means = {
            method: (
                float(pd.Series(rows, dtype="float64").mean())
                if rows and pd is not None
                else (fmean(rows) if rows else None)
            )
            for method, rows in values.items()
        }
        best_methods: set[str] = set()
        if direction in {"higher", "lower"}:
            present = {
                method: value for method, value in means.items() if value is not None
            }
            if present:
                target = (
                    max(present.values())
                    if direction == "higher"
                    else min(present.values())
                )
                best_methods = {
                    method
                    for method, value in present.items()
                    if math.isclose(value, target)
                }
        cells = []
        for method in METHODS:
            if values[method]:
                cells.append(_sample_cell(values[method], best=method in best_methods))
            else:
                cells.append(f"N/A ({_na_reason(method, metric, cohort)})")
        lines.append(
            f"| {label} | {' | '.join(cells)} | {_direction_text(direction)} |"
        )
        figure_path = figures_dir / f"{category}_{_slug(metric)}.png"
        reason = _na_reason("a1", metric, cohort)
        if plt is not None:
            _plot_method_metric(plt, figure_path, label, samples, reason)
        figures.extend(
            [
                "",
                f"### {label}",
                "",
                _image_markdown(
                    figure_path,
                    report_dir,
                    f"A1, A2, and A3 comparison for {label}",
                    plt is not None,
                ),
            ]
        )
    return lines, figures


def _resolve_path(value: Any, benchmark_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else benchmark_dir / path


def _load_run_artifacts(
    method: str,
    records: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
    benchmark_dir: Path,
) -> tuple[RunArtifacts, ...]:
    run_map = {str(run.get("run_id")): run for run in manifest.get("runs", [])}
    artifacts = []
    for record in records:
        if record.get("method") != method or record.get("status") == "invalid":
            continue
        manifest_run = run_map.get(str(record.get("run_id")), {})
        result_path = _resolve_path(
            record.get("results") or manifest_run.get("results"), benchmark_dir
        )
        result = _read_json(result_path, required=False) if result_path else {}
        run_dir = result_path.parent if result_path and result_path.is_file() else None
        meta = _read_json(run_dir / "meta.json", required=False) if run_dir else {}
        events = _read_jsonl(run_dir / "metrics.jsonl") if run_dir else ()
        artifacts.append(RunArtifacts(record, result, meta, events, run_dir))
    return tuple(artifacts)


def _event_key_inventory(artifacts: Sequence[RunArtifacts]) -> dict[str, list[str]]:
    inventory: dict[str, set[str]] = defaultdict(set)
    for artifact in artifacts:
        for event in artifact.events:
            event_name = str(event.get("event", "unknown"))
            inventory[event_name].update(str(key) for key in event)
    return {name: sorted(keys) for name, keys in sorted(inventory.items())}


def _plot_lines(
    plt: Any,
    path: Path,
    title: str,
    series: Mapping[str, Sequence[tuple[float, float]]],
    reason: str,
    *,
    x_label: str,
    y_label: str,
) -> None:
    available = {name: list(points) for name, points in series.items() if points}
    if not available:
        _placeholder(plt, path, title, reason)
        return
    figure, axis = plt.subplots(figsize=(8, 5))
    for index, (name, points) in enumerate(sorted(available.items())):
        ordered = sorted(points)
        axis.plot(
            [point[0] for point in ordered],
            [point[1] for point in ordered],
            marker="o",
            linewidth=1.8,
            label=name,
            color=list(METHOD_COLORS.values())[index % len(METHOD_COLORS)],
        )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, linestyle="--", alpha=0.3)
    if len(available) > 1:
        axis.legend(fontsize=8)
    _save_figure(figure, path)
    plt.close(figure)


def _plot_heatmap(
    plt: Any,
    path: Path,
    title: str,
    matrix: Any,
    reason: str,
    *,
    x_labels: Sequence[Any] | None = None,
    y_labels: Sequence[Any] | None = None,
) -> None:
    try:
        values = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError):
        values = np.asarray([])
    if values.ndim != 2 or not values.size or not np.isfinite(values).any():
        _placeholder(plt, path, title, reason)
        return
    figure, axis = plt.subplots(figsize=(7, 5))
    image = axis.imshow(values, aspect="auto", cmap="viridis")
    axis.set_title(title)
    if x_labels and len(x_labels) == values.shape[1]:
        axis.set_xticks(range(values.shape[1]), [str(item) for item in x_labels])
    if y_labels and len(y_labels) == values.shape[0]:
        axis.set_yticks(range(values.shape[0]), [str(item) for item in y_labels])
    figure.colorbar(image, ax=axis)
    _save_figure(figure, path)
    plt.close(figure)


def _selected_artifact(
    method: str, artifacts: Sequence[RunArtifacts], cohort: HeadlineCohort | None
) -> RunArtifacts | None:
    candidates = list(artifacts)
    if cohort is not None:
        candidates = [
            artifact
            for artifact in candidates
            if _condition_key(artifact.record)
            == (cohort.protocol, cohort.rank, cohort.order, cohort.heldout)
            and (method != "a3" or artifact.record.get("k") == cohort.selected_k)
            and int(artifact.record.get("seed", -1)) in cohort.matched_seeds
        ]
    return (
        min(candidates, key=lambda row: int(row.record.get("seed", 0)))
        if candidates
        else None
    )


def _common_stage_figures(
    method: str,
    artifacts: Sequence[RunArtifacts],
    cohort: HeadlineCohort | None,
    plt: Any,
    figures_dir: Path,
    report_dir: Path,
) -> list[str]:
    lines: list[str] = []
    selected = _selected_artifact(method, artifacts, cohort)
    plots: list[tuple[str, str, str]] = []

    compression_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for artifact in artifacts:
        curve = artifact.record.get("residual_compressibility") or {}
        for rank, value in curve.items():
            number = _finite(value)
            if number is not None:
                compression_series[
                    f"seed {artifact.record.get('seed')} K={artifact.record.get('k')}"
                ].append((float(rank), number))
    path = figures_dir / "residual_compressibility.png"
    if plt is not None:
        _plot_lines(
            plt,
            path,
            "Residual compressibility versus rank",
            compression_series,
            "requires at least two ranks with a reliable reference improvement",
            x_label="rank",
            y_label="C(r)",
        )
    plots.append(
        (
            "Residual compressibility",
            path.name,
            "Residual compressibility curve by rank",
        )
    )

    matrix = selected.record.get("loss_matrix") if selected else None
    domains = selected.record.get("domains", []) if selected else []
    path = figures_dir / "loss_matrix.png"
    if plt is not None:
        _plot_heatmap(
            plt,
            path,
            "Validation loss matrix",
            matrix,
            "loss matrix is unavailable",
            x_labels=domains,
            y_labels=range(len(matrix or [])),
        )
    plots.append(
        ("Loss matrix", path.name, "Checkpoint-by-domain validation loss heatmap")
    )

    perplexity_matrix = selected.result.get("ppl") if selected else None
    path = figures_dir / "perplexity_matrix.png"
    if plt is not None:
        _plot_heatmap(
            plt,
            path,
            "Validation perplexity matrix",
            perplexity_matrix,
            "perplexity matrix is unavailable",
            x_labels=domains,
            y_labels=range(len(perplexity_matrix or [])),
        )
    plots.append(
        (
            "Perplexity matrix",
            path.name,
            "Checkpoint-by-domain validation perplexity heatmap",
        )
    )

    forgetting_series = {}
    if selected:
        values = selected.result.get("mean_forgetting") or []
        forgetting_series["mean forgetting"] = [
            (index, float(value))
            for index, value in enumerate(values)
            if _finite(value) is not None
        ]
    path = figures_dir / "forgetting.png"
    if plt is not None:
        _plot_lines(
            plt,
            path,
            "Loss-based forgetting",
            forgetting_series,
            "forgetting trajectory is unavailable",
            x_label="training checkpoint",
            y_label="mean loss increase",
        )
    plots.append(("Forgetting", path.name, "Loss-based forgetting trajectory"))

    anchor_series: dict[str, list[tuple[float, float]]] = {}
    if selected:
        trajectory = (
            selected.record.get("deployed_evaluation", {})
            .get("anchor_kl", {})
            .get("trajectory", [])
        )
        anchor_series["anchor KL"] = [
            (float(row["checkpoint"]), float(row["kl"]))
            for row in trajectory
            if _finite(row.get("checkpoint")) is not None
            and _finite(row.get("kl")) is not None
        ]
    path = figures_dir / "anchor_kl.png"
    if plt is not None:
        _plot_lines(
            plt,
            path,
            "Anchor KL across checkpoints",
            anchor_series,
            "anchor data or checkpoint trajectory is unavailable",
            x_label="checkpoint",
            y_label="KL(base || adapted)",
        )
    plots.append(("Anchor KL", path.name, "Anchor KL drift across checkpoints"))

    on_domain_kl: dict[str, list[tuple[float, float]]] = defaultdict(list)
    if selected:
        kl_matrix = selected.result.get("kl") or []
        for checkpoint, row in enumerate(kl_matrix):
            if not isinstance(row, list):
                continue
            for index, value in enumerate(row):
                number = _finite(value)
                if number is not None:
                    domain = (
                        domains[index] if index < len(domains) else f"domain {index}"
                    )
                    on_domain_kl[domain].append((checkpoint, number))
    path = figures_dir / "on_domain_kl.png"
    if plt is not None:
        _plot_lines(
            plt,
            path,
            "On-domain KL across checkpoints",
            on_domain_kl,
            "checkpoint KL matrix is unavailable",
            x_label="training checkpoint",
            y_label="KL(base || adapted)",
        )
    plots.append(("On-domain KL", path.name, "On-domain KL across checkpoints"))

    common_scalars = (
        ("validation_loss", "Final validation loss", "validation_loss.png"),
        ("test_loss", "Final held-out test loss", "test_loss.png"),
        ("token_accuracy", "Token accuracy", "token_accuracy.png"),
        ("calibration", "Token calibration ECE", "token_ece.png"),
        ("heldout_transfer", "Held-out transfer", "heldout_transfer.png"),
        (
            "sequential_joint_gap",
            "Sequential versus joint gap",
            "sequential_joint_gap.png",
        ),
        ("mean_step_time", "Mean training step time", "mean_step_time.png"),
        ("tokens_per_second", "Training throughput", "throughput.png"),
        ("trainable_params", "Trainable parameter count", "trainable_params.png"),
        ("inference_latency", "Inference latency", "inference_latency.png"),
        ("peak_memory", "Peak evaluator memory", "evaluator_memory.png"),
    )
    for metric, title, filename in common_scalars:
        series: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for artifact in artifacts:
            values = metric_values_for_record(
                artifact.record, metric, int(artifact.record.get("rank", 16))
            )
            if not values:
                continue
            label = (
                f"seed={artifact.record.get('seed')} "
                f"{artifact.record.get('protocol', 'sequential')} "
                f"order={','.join(artifact.record.get('task_order', []))} "
                f"heldout={','.join(artifact.record.get('heldout_tasks', []))} "
                f"K={artifact.record.get('k')}"
            )
            series[label].append(
                (float(artifact.record.get("rank", 16)), float(np.mean(values)))
            )
        path = figures_dir / filename
        if plt is not None:
            _plot_lines(
                plt,
                path,
                title,
                series,
                _na_reason(method, metric, cohort),
                x_label="rank",
                y_label=title,
            )
        plots.append((title, path.name, f"{METHOD_LABELS[method]} {title}"))

    for title, filename, alt in plots:
        lines.extend(
            [
                "",
                f"### {title}",
                "",
                _image_markdown(
                    figures_dir / filename,
                    report_dir,
                    alt,
                    plt is not None,
                ),
            ]
        )
    lines.extend(
        [
            "",
            "### Task-score matrices",
            "",
            "Task-score matrices require explicit checkpoint-by-task labeled audit evidence.",
        ]
    )
    return lines


def _training_series(
    artifacts: Sequence[RunArtifacts], keys: Sequence[str]
) -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for artifact in artifacts:
        for event in artifact.events:
            if event.get("event") != "train":
                continue
            step = _finite(event.get("global_step"))
            if step is None:
                continue
            for key in keys:
                value = _finite(event.get(key))
                if value is not None:
                    label = f"{artifact.record.get('run_id')} {key}"
                    series[label].append((step, value))
    return series


def _stage_metric_table(method: str, artifacts: Sequence[RunArtifacts]) -> list[str]:
    lines = [
        "## Run Summary",
        "",
        "| Run | Seed | Rank | K | Protocol | Validation loss | Test loss | Forgetting | Step time |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for artifact in sorted(
        artifacts,
        key=lambda row: (
            int(row.record.get("seed", 0)),
            int(row.record.get("rank", 0)),
            -1 if row.record.get("k") is None else int(row.record["k"]),
        ),
    ):
        record = artifact.record
        cells = [
            str(record.get("run_id")),
            str(record.get("seed")),
            str(record.get("rank")),
            str(record.get("k", "N/A")),
            str(record.get("protocol", "N/A")),
            str(record.get("final_mean_loss", "N/A")),
            str(record.get("final_test_mean_loss", "N/A")),
            str(record.get("mean_forgetting", "N/A")),
            str(record.get("mean_step_time", "N/A")),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    if not artifacts:
        lines.append(
            f"| {METHOD_LABELS[method]} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
        )
    return lines


def _individual_metric_table(
    title: str, specs: Sequence[tuple[str, str, str]], artifacts: Sequence[RunArtifacts]
) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Metric | Run | Value | Interpretation |",
        "|---|---|---:|---|",
    ]
    for label, metric, direction in specs:
        found = False
        for artifact in artifacts:
            values = metric_values_for_record(
                artifact.record, metric, int(artifact.record.get("rank", 16))
            )
            if values:
                found = True
                lines.append(
                    f"| {label} | {artifact.record.get('run_id')} | {fmean(values):.6g} | {_direction_text(direction)} |"
                )
        if not found:
            lines.append(f"| {label} | N/A | N/A | evidence not available |")
    return lines


def _individual_metric_figure(
    plt: Any,
    path: Path,
    title: str,
    specs: Sequence[tuple[str, str, str]],
    artifacts: Sequence[RunArtifacts],
) -> None:
    available = []
    for label, metric, _ in specs:
        points = [
            (str(a.record.get("run_id")), fmean(values))
            for a in artifacts
            if (
                values := metric_values_for_record(
                    a.record, metric, int(a.record.get("rank", 16))
                )
            )
        ]
        if points:
            available.append((label, points))
    if not available:
        _placeholder(
            plt, path, title, "No measured metrics; see the table for required evidence"
        )
        return
    columns = min(3, len(available))
    rows = math.ceil(len(available) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 3.4 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, (label, points) in zip(axes.flat, available, strict=False):
        axis.scatter(
            range(len(points)), [value for _, value in points], color="#2563eb"
        )
        axis.set_xticks(
            range(len(points)),
            [name for name, _ in points],
            rotation=45,
            ha="right",
            fontsize=7,
        )
        axis.set_title(label)
        axis.grid(axis="y", linestyle="--", alpha=0.3)
    for axis in list(axes.flat)[len(available) :]:
        axis.axis("off")
    figure.suptitle(title)
    _save_figure(figure, path)
    plt.close(figure)


def _write_stage_report(
    method: str,
    records: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
    benchmark_dir: Path,
    stage_root: Path,
    cohort: HeadlineCohort | None,
    plt: Any,
) -> Path:
    method_dir = stage_root / METHOD_LABELS[method]
    figures_dir = method_dir / FIGURES_FOLDER
    figures_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _load_run_artifacts(method, records, manifest, benchmark_dir)
    inventory = _event_key_inventory(artifacts)
    report_path = method_dir / STAGE_REPORT_PATTERN.format(method=method)
    lines = [
        f"# {METHOD_LABELS[method]} Results and Visualization",
        "",
        f"- Runs available: {len(artifacts)}",
        f"- Headline rank: {cohort.rank if cohort else 'N/A'}",
        f"- Headline protocol: {cohort.protocol if cohort else 'N/A'}",
    ]
    if method == "a3":
        lines.append(f"- Selected A3 K: {cohort.selected_k if cohort else 'N/A'}")
    lines.extend(["", *_stage_metric_table(method, artifacts)])
    lines.extend(
        [
            "",
            *_individual_metric_table(
                "Internal Metrics", INTERNAL_METRIC_SPECS, artifacts
            ),
        ]
    )
    lines.extend(
        [
            "",
            *_individual_metric_table(
                "External Metrics", EXTERNAL_METRIC_SPECS, artifacts
            ),
        ]
    )
    for category, specs in (
        ("Internal", INTERNAL_METRIC_SPECS),
        ("External", EXTERNAL_METRIC_SPECS),
    ):
        path = figures_dir / f"{category.lower()}_metrics.png"
        if plt is not None:
            _individual_metric_figure(
                plt,
                path,
                f"{METHOD_LABELS[method]} {category.lower()} metrics by run",
                specs,
                artifacts,
            )
        lines.extend(
            [
                "",
                f"## {category} Metric Plots",
                "",
                _image_markdown(
                    path, method_dir, f"{method} {category} metrics", plt is not None
                ),
            ]
        )

    lines.extend(["", "## Data Availability", ""])
    if inventory:
        for event_name, event_keys in inventory.items():
            lines.append(
                f"- `{event_name}`: {', '.join(f'`{key}`' for key in event_keys)}"
            )
    else:
        lines.append("- `metrics.jsonl`: N/A (no event records found)")

    lines.extend(
        [
            "",
            "## Run Provenance",
            "",
            "| Run | Stage | Model | Python | Git commit | Dirty |",
            "|---|---|---|---|---|---|",
        ]
    )
    for artifact in artifacts:
        meta = artifact.meta
        lines.append(
            f"| {artifact.record.get('run_id')} | {meta.get('stage', 'N/A')} | "
            f"{meta.get('model', 'N/A')} | {meta.get('python', 'N/A')} | "
            f"{meta.get('git_commit', 'N/A')} | {meta.get('git_dirty', 'N/A')} |"
        )
    if not artifacts:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A |")

    if method == "a1":
        series = _training_series(artifacts, ("loss_task", "loss_total"))
        path = figures_dir / "training_task_loss.png"
        if plt is not None:
            _plot_lines(
                plt,
                path,
                "A1 training task loss",
                series,
                "loss_task/loss_total train events were not recorded",
                x_label="global step",
                y_label="loss",
            )
        lines.extend(
            [
                "",
                "## A1 Training Loss",
                "",
                _image_markdown(
                    path, method_dir, "A1 task training-loss curves", plt is not None
                ),
            ]
        )
    elif method == "a2":
        loss_keys = ("loss_task", "loss_total", "loss_cpc", "loss_hid", "loss_stab")
        series = _training_series(artifacts, loss_keys)
        path = figures_dir / "auxiliary_losses.png"
        if plt is not None:
            _plot_lines(
                plt,
                path,
                "A2 task and auxiliary losses",
                series,
                "A2 loss_cpc/loss_hid/loss_stab events were not recorded",
                x_label="global step",
                y_label="loss",
            )
        lines.extend(
            [
                "",
                "## A2 Objective Terms",
                "",
                _image_markdown(
                    path,
                    method_dir,
                    "A2 task, CPC, hidden, and stability losses",
                    plt is not None,
                ),
            ]
        )
        diagnostics: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for artifact in artifacts:
            for index, event in enumerate(artifact.events):
                if event.get("event") != "a2_diagnostics":
                    continue
                for key, value in event.items():
                    number = _finite(value)
                    if key not in {"event", "when", "domain"} and number is not None:
                        diagnostics[key].append((float(index), number))
            if not any(diagnostics.values()):
                for key, value in _flatten_numeric(
                    artifact.result.get("a2_diagnostics", {})
                ).items():
                    diagnostics[key].append((float(len(diagnostics[key])), value))
        path = figures_dir / "a2_diagnostics.png"
        if plt is not None:
            _plot_lines(
                plt,
                path,
                "A2 predictor and monitor diagnostics",
                dict(sorted(diagnostics.items())[:12]),
                "a2_diagnostics events were not recorded",
                x_label="diagnostic event",
                y_label="value",
            )
        lines.extend(
            [
                "",
                "## A2 Diagnostics",
                "",
                _image_markdown(
                    path,
                    method_dir,
                    "A2 predictor and monitor diagnostics",
                    plt is not None,
                ),
                "",
                "## A2 vs A1 Paired Deltas",
                "",
                "| Run | Seed | Rank | Delta vs A1 |",
                "|---|---:|---:|---:|",
            ]
        )
        for artifact in artifacts:
            record = artifact.record
            lines.append(
                f"| {record.get('run_id')} | {record.get('seed')} | {record.get('rank')} | "
                f"{record.get('paired_delta_vs_a1', 'N/A')} |"
            )
    else:
        # K/rank heatmaps must not average different training protocols or task orders.
        sweep_artifacts = [
            a
            for a in artifacts
            if cohort is not None
            and _condition_key(a.record)[0] == cohort.protocol
            and _condition_key(a.record)[2:] == (cohort.order, cohort.heldout)
            and int(a.record.get("seed", -1)) in cohort.matched_seeds
            and any(
                other.get("seed") == a.record.get("seed")
                and other.get("runtime_data_sha256")
                == a.record.get("runtime_data_sha256")
                for other in cohort.records["a1"]
            )
        ]
        for metric, title in (
            (
                "final_mean_loss",
                "A3 final validation loss over K and rank (available matched seeds)",
            ),
            (
                "mean_forgetting",
                "A3 forgetting over K and rank (available matched seeds)",
            ),
            ("compressibility_c4", "A3 residual compressibility C(4) over K and rank"),
        ):
            ranks = sorted({int(a.record["rank"]) for a in sweep_artifacts})
            ks = sorted(
                {
                    int(a.record["k"])
                    for a in sweep_artifacts
                    if a.record.get("k") is not None
                }
            )
            grid = np.full((len(ks), len(ranks)), np.nan)
            for row_index, k in enumerate(ks):
                for column_index, rank in enumerate(ranks):
                    metric_values: list[float] = []
                    for artifact in sweep_artifacts:
                        if (
                            artifact.record.get("k") != k
                            or artifact.record.get("rank") != rank
                        ):
                            continue
                        raw_value = (
                            (artifact.record.get("residual_compressibility") or {}).get(
                                "4"
                            )
                            if metric == "compressibility_c4"
                            else artifact.record.get(metric)
                        )
                        number = _finite(raw_value)
                        if number is not None:
                            metric_values.append(number)
                    if metric_values:
                        grid[row_index, column_index] = fmean(metric_values)
            path = figures_dir / f"k_rank_{_slug(metric)}.png"
            if plt is not None:
                _plot_heatmap(
                    plt,
                    path,
                    title,
                    grid,
                    "multiple A3 K/rank conditions are unavailable",
                    x_labels=ranks,
                    y_labels=ks,
                )
            lines.extend(
                [
                    "",
                    f"## {title}",
                    "",
                    _image_markdown(path, method_dir, title, plt is not None),
                ]
            )
        step_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for artifact in sweep_artifacts:
            value = _finite(artifact.record.get("mean_step_time"))
            k_value = _finite(artifact.record.get("k"))
            if value is not None and k_value is not None:
                step_series[f"rank {artifact.record.get('rank')}"].append(
                    (k_value, value)
                )
        path = figures_dir / "step_time_vs_k.png"
        if plt is not None:
            _plot_lines(
                plt,
                path,
                "A3 mean training step time versus K",
                step_series,
                "A3 K/step-time sweep is unavailable",
                x_label="K",
                y_label="seconds",
            )
        lines.extend(
            [
                "",
                "## A3 Step-Time Overhead",
                "",
                _image_markdown(
                    path, method_dir, "A3 training step time versus K", plt is not None
                ),
            ]
        )
        energy_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for artifact in artifacts:
            for event in artifact.events:
                if event.get("event") != "energy_curve" or not isinstance(
                    event.get("curve"), list
                ):
                    continue
                for index, value in enumerate(event["curve"]):
                    number = _finite(value)
                    if number is not None:
                        energy_series[
                            f"seed {artifact.record.get('seed')} K={artifact.record.get('k')}"
                        ].append((index, number))
        path = figures_dir / "energy_curve.png"
        if plt is not None:
            _plot_lines(
                plt,
                path,
                "A3 predictive-coding energy curve",
                energy_series,
                "run_stage_a3.py was not run with --energy-curve",
                x_label="inference step",
                y_label="energy",
            )
        lines.extend(
            [
                "",
                "## A3 Energy Curve",
                "",
                _image_markdown(
                    path,
                    method_dir,
                    "A3 predictive-coding energy curve",
                    plt is not None,
                ),
                "",
                "## A3 K Selection",
                "",
                f"- Selected K: {cohort.selected_k if cohort else 'N/A'}",
                f"- Rule: {cohort.selection.get('rule') if cohort else 'N/A'}",
                f"- Mean validation loss by K: {cohort.selection.get('mean_validation_loss_by_k') if cohort else 'N/A'}",
            ]
        )

    lines.extend(
        [
            "",
            "## Metrics Common To All Stages",
            *_common_stage_figures(
                method,
                artifacts,
                cohort,
                plt,
                figures_dir,
                method_dir,
            ),
            "",
            "Scope: A1/A2/A3 language-model and continuous-residual measurements; symbolic and category/probe audits are excluded.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _write_general_report(
    manifest: dict[str, Any],
    evaluation: dict[str, Any],
    output_root: Path,
    stage_root: Path,
    report_rank: int,
    plt: Any,
    pd: Any,
    manifest_sha256: str,
) -> tuple[Path, HeadlineCohort | None]:
    records = [
        record
        for record in evaluation.get("records", [])
        if isinstance(record, dict) and record.get("status") != "invalid"
    ]
    cohort = _select_headline_cohort(records, report_rank)
    report_path = stage_root / GENERAL_REPORT_PATTERN.format(stage="a")
    figures_dir = stage_root / FIGURES_FOLDER
    figures_dir.mkdir(parents=True, exist_ok=True)
    actual_seeds = sorted(
        {int(record["seed"]) for record in records if "seed" in record}
    )
    protocols = sorted(
        {str(record.get("protocol")) for record in records if record.get("protocol")}
    )
    source_sha = manifest.get("source", {}).get("source_sha256", "N/A")
    lines = [
        "# Stage A General Report",
        "",
        f"- Profile: {manifest.get('profile', 'custom')}",
        f"- Seeds actually evaluated: {actual_seeds or 'N/A'}",
        f"- Protocols present: {protocols or 'N/A'}",
        f"- Source fingerprint: `{source_sha}`",
        f"- Manifest SHA-256: `{manifest_sha256}`",
        f"- Run completeness: {evaluation.get('completed_runs', 0)}/{evaluation.get('planned_runs', len(manifest.get('runs', [])))} valid; {evaluation.get('invalid_runs', 0)} invalid",
    ]
    if cohort is None:
        lines.extend(
            [
                "- Headline comparison: N/A (no complete A1/A2/A3 cohort)",
                "- A3 K selection: N/A",
            ]
        )
    else:
        lines.extend(
            [
                f"- Headline condition: protocol={cohort.protocol}, rank={cohort.rank}, order={' -> '.join(cohort.order)}, heldout={list(cohort.heldout)}",
                f"- Matched seeds: {list(cohort.matched_seeds) or 'N/A'}",
                f"- A3 selected K: {cohort.selected_k if cohort.selected_k is not None else 'N/A'}",
                f"- A3 K-selection rule: {cohort.selection['rule']}",
                f"- A3 mean validation loss by K: {cohort.selection['mean_validation_loss_by_k']}",
            ]
        )

    for category, specs in (
        ("internal", INTERNAL_METRIC_SPECS),
        ("external", EXTERNAL_METRIC_SPECS),
        ("operational", VISUALIZATION_OPERATIONAL_METRIC_SPECS),
    ):
        table, figures = _metric_table(
            specs,
            cohort,
            plt,
            pd,
            figures_dir,
            stage_root,
            category,
        )
        lines.extend(["", *table, *figures])

    lines.extend(
        [
            "",
            "## General Summary",
            "",
        ]
    )
    if cohort is None or not cohort.matched_seeds:
        lines.append(
            "No complete fingerprint-matched A1/A2/A3 seed cohort is available, so no cross-stage best-method conclusion is reported."
        )
    else:
        for label, metric, direction in EXTERNAL_METRIC_SPECS + INTERNAL_METRIC_SPECS:
            if direction not in {"higher", "lower"}:
                continue
            samples = _metric_samples(cohort, metric)
            means = {
                method: fmean(value for _, value in rows)
                for method, rows in samples.items()
                if rows
            }
            if not means:
                continue
            best_value = (
                max(means.values()) if direction == "higher" else min(means.values())
            )
            best = ", ".join(
                METHOD_LABELS[method]
                for method, value in means.items()
                if math.isclose(value, best_value)
            )
            lines.append(
                f"- {label}: {best} has the best observed mean on the matched cohort ({best_value:.6g}; {_direction_text(direction)})."
            )
        lines.append(
            "- Operational metrics are reported separately and are not folded into a composite score."
        )
        lines.append(
            "- Observed rankings are descriptive and do not establish statistical significance or the full research hypothesis."
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path, cohort


def generate_reports_and_visualizations(
    benchmark_dir: Path | str,
    *,
    output_dir: Path | str | None = None,
    report_rank: int = 4,
    repository_root: Path | str | None = None,
) -> dict[str, Any]:
    """Generate deterministic read-only benchmark reports outside run artifacts."""
    benchmark_root = Path(benchmark_dir).resolve()
    repository = Path(repository_root or REPOSITORY_ROOT).resolve()
    destination = (
        Path(output_dir).resolve()
        if output_dir
        else repository / "report_and_visualization"
    )
    if destination == repository or destination in repository.parents:
        raise ValueError(
            "visualization output must not be the repository root or its ancestor"
        )
    if destination == benchmark_root or benchmark_root in destination.parents:
        raise ValueError(
            "visualization output must not be inside the benchmark artifact root"
        )
    manifest_path = benchmark_root / "manifest.json"
    evaluation_path = benchmark_root / "evaluation.json"
    manifest = _read_json(manifest_path)
    evaluation = _read_json(evaluation_path)
    manifest_sha = sha256(manifest_path.read_bytes()).hexdigest()

    stage_root = destination / STAGE_FOLDER_PATTERN.format(stage="A")
    stage_root.mkdir(parents=True, exist_ok=True)
    plotting = _load_plotting()
    plt, pd = plotting if plotting is not None else (None, None)
    general_report, cohort = _write_general_report(
        manifest,
        evaluation,
        destination,
        stage_root / "comparison",
        report_rank,
        plt,
        pd,
        manifest_sha,
    )
    records = [
        record for record in evaluation.get("records", []) if isinstance(record, dict)
    ]
    stage_reports = {}
    for method in METHODS:
        stage_reports[method] = _write_stage_report(
            method,
            records,
            manifest,
            benchmark_root,
            stage_root,
            cohort,
            plt,
        )

    readme = destination / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Benchmark Reports and Visualizations",
                "",
                "Generated from benchmark artifacts without modifying immutable run directories.",
                "",
                f"- [Stage A general report]({general_report.relative_to(destination).as_posix()})",
                f"- [A1 report]({stage_reports['a1'].relative_to(destination).as_posix()})",
                f"- [A2 report]({stage_reports['a2'].relative_to(destination).as_posix()})",
                f"- [A3 report]({stage_reports['a3'].relative_to(destination).as_posix()})",
                "",
            ]
        ),
        encoding="utf-8",
    )
    dashboard = _write_dashboard(destination, general_report, stage_reports)
    return {
        "output_dir": str(destination),
        "dashboard": str(dashboard),
        "readme": str(readme),
        "general_report": str(general_report),
        "stage_reports": {method: str(path) for method, path in stage_reports.items()},
        "figures_enabled": plt is not None,
    }


def _write_dashboard(
    destination: Path,
    general_report: Path,
    stage_reports: Mapping[str, Path],
) -> Path:
    """Put both metric groups and every generated figure in one browser view."""

    def inline(value: str) -> str:
        value = html.escape(value)
        value = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", value)
        return re.sub(r"`([^`]+)`", r"<code>\1</code>", value)

    sections = []
    for label, report in [
        ("Comparison", general_report),
        *[(key.upper(), value) for key, value in stage_reports.items()],
    ]:
        blocks = []
        table_open = False
        for line in report.read_text(encoding="utf-8").splitlines():
            if line.startswith("|"):
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    continue
                if not table_open:
                    blocks.append('<div class="table-wrap"><table>')
                tag = "td" if table_open else "th"
                blocks.append(
                    "<tr>"
                    + "".join(f"<{tag}>{inline(cell)}</{tag}>" for cell in cells)
                    + "</tr>"
                )
                table_open = True
                continue
            if table_open:
                blocks.append("</table></div>")
                table_open = False
            image_match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line.strip())
            if image_match:
                alt, relative = image_match.groups()
                source = (report.parent / relative).resolve()
                url = source.relative_to(destination).as_posix()
                blocks.append(
                    f'<img loading="lazy" src="{html.escape(url, quote=True)}" alt="{html.escape(alt, quote=True)}">'
                )
            elif line.startswith("#"):
                level = min(len(line) - len(line.lstrip("#")) + 1, 6)
                blocks.append(
                    f"<h{level}>{inline(line.lstrip('# ').strip())}</h{level}>"
                )
            elif line.strip():
                blocks.append(f"<p>{inline(line)}</p>")
        if table_open:
            blocks.append("</table></div>")
        sections.append(f'<section id="{label.lower()}">{"".join(blocks)}</section>')
    banner = "Measured benchmark outputs. Missing measurements are shown as N/A."
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    page += "<title>A1 / A2 / A3 benchmark</title><style>body{margin:0;background:#f3f6fb;color:#142338;font:16px system-ui}header,main{max-width:1280px;margin:auto;padding:28px}header{background:#142338;color:white}nav{display:flex;gap:24px;flex-wrap:wrap}nav a{color:#b9d9ff}section{background:white;margin:24px 0;padding:24px;border-radius:12px}img{width:100%;height:auto;margin:16px 0}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #d9e2ee}th{background:#eaf1fb}code{overflow-wrap:anywhere}p{line-height:1.6;overflow-wrap:anywhere}</style>"
    page += f'<header><h1>A1 / A2 / A3 benchmark</h1><p>{html.escape(banner)}</p><nav><a href="#comparison">Comparison</a><a href="#a1">A1</a><a href="#a2">A2</a><a href="#a3">A3</a></nav></header><main>{"".join(sections)}</main></html>'
    path = destination / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=("run", "plan", "train", "eval", "report"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/benchmark"))
    parser.add_argument(
        "--profile", choices=("smoke", "headline", "full"), default="headline"
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--k", nargs="+", type=int)
    parser.add_argument("--ranks", nargs="+", type=int)
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=("sequential", "joint", "heldout", "commutator"),
    )
    parser.add_argument("--task-order", nargs="+")
    parser.add_argument("--heldout-task", action="append")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--report-rank", type=int, default=16)
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--visualization-dir", type=Path)
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated HTML report in a browser.",
    )
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    visualization_dir = (
        args.visualization_dir or ROOT / "report_and_visualization"
    ).resolve()
    # Fail before expensive training when the requested plotting stack is missing.
    if not args.no_visualization and args.command in {"run", "eval", "report"}:
        _load_plotting()
    profiles = {
        "smoke": ([0], [16], [4], ["sequential"]),
        "headline": ([0, 1, 2], [16], [1, 2, 4, 8], ["sequential", "joint"]),
        "full": (
            [0, 1, 2, 3, 4],
            [1, 2, 4, 8, 16, 32, 64],
            [1, 2, 4, 8],
            ["sequential", "joint", "heldout", "commutator"],
        ),
    }
    seeds, ranks, ks, protocols = profiles[args.profile]
    manifest_path = output_dir / "manifest.json"
    if args.command == "plan" or args.command == "run" and not manifest_path.is_file():
        manifest = make_manifest(
            output_dir,
            seeds=args.seeds or seeds,
            methods=args.methods,
            k_values=args.k or ks,
            rank_values=args.ranks or ranks,
            epochs=args.epochs,
            protocols=args.protocols or protocols,
            task_order=args.task_order,
            heldout_tasks=args.heldout_task,
            profile=args.profile,
        )
        print(f"Planned {len(manifest['runs'])} runs: {manifest_path}", flush=True)
    elif args.command == "run":
        print(
            f"Resuming existing manifest: {manifest_path}. Use a new output directory for a different experiment.",
            flush=True,
        )
    if args.command in {"run", "train"}:
        train_manifest(output_dir, resume=not args.no_resume, epochs=args.epochs)
        print(f"Training complete: {output_dir}", flush=True)
    if args.command in {"run", "eval"}:
        evaluation = evaluate_manifest(output_dir)
        print(
            f"Evaluated {evaluation['completed_runs']}/{evaluation['planned_runs']} runs",
            flush=True,
        )
    if args.command in {"run", "eval", "report"}:
        print(report_manifest(output_dir, report_rank=args.report_rank), flush=True)
        if not args.no_visualization:
            generated = generate_reports_and_visualizations(
                output_dir, output_dir=visualization_dir, report_rank=args.report_rank
            )
            print(f"Visual report: {generated['dashboard']}", flush=True)
            if args.open:
                webbrowser.open(Path(generated["dashboard"]).as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
