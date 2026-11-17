"""Plan, train and evaluate Stage A benchmarks. Run with python -m benchmark.benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import yaml

GENERATED_REPORT_PREFIX = "docs/stage_A_results/"


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
    diff = git(
        "diff", "--binary", "HEAD", "--", ".", ":(exclude)docs/stage_A_results/**"
    )
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


def evaluate_long_range(
    config: Mapping[str, Any],
    score_choice: Callable[[str, str], float],
) -> dict[str, Any]:
    """Evaluate inline multiple-choice prompts with an external scoring callback."""
    if not isinstance(config, Mapping):
        raise ValueError("long-range configuration must be a mapping")
    if config.get("schema") != "long-range/v1":
        raise ValueError("long-range probes must use schema 'long-range/v1'")
    if not callable(score_choice):
        raise ValueError("score_choice must be callable")

    items = config.get("items", [])
    if not isinstance(items, list):
        raise ValueError("long-range items must be a list")
    if not items:
        return {"status": "not_configured", "reason": "no long-range probes"}

    required = {
        "id",
        "family",
        "distance",
        "context",
        "question",
        "choices",
        "answer",
    }
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"long-range item {index} must be a mapping")
        missing = required - set(item)
        if missing:
            raise ValueError(f"long-range item is missing {sorted(missing)}")

        text_fields = ("id", "family", "distance", "context", "question")
        for field in text_fields:
            if not isinstance(item[field], str) or not item[field]:
                raise ValueError(
                    f"long-range item {index} {field} must be nonempty text"
                )
        item_id = item["id"]
        if item_id in seen_ids:
            raise ValueError(f"long-range item id {item_id!r} is duplicated")
        seen_ids.add(item_id)

        choices = item["choices"]
        if (
            not isinstance(choices, list)
            or len(choices) < 2
            or any(not isinstance(choice, str) or not choice for choice in choices)
        ):
            raise ValueError(
                f"long-range item {item_id!r} requires at least two text choices"
            )

        answer = item["answer"]
        if isinstance(answer, bool) or not isinstance(answer, Integral):
            raise ValueError(f"long-range item {item_id!r} answer must be an index")
        answer = int(answer)
        if not 0 <= answer < len(choices):
            raise ValueError(f"long-range item {item_id!r} answer is out of range")

        filler = item.get("filler", "")
        if not isinstance(filler, str):
            raise ValueError(f"long-range item {item_id!r} filler must be text")
        repeat = item.get("filler_repeat", 0)
        if isinstance(repeat, bool) or not isinstance(repeat, Integral) or repeat < 0:
            raise ValueError(
                f"long-range item {item_id!r} filler_repeat must be a nonnegative integer"
            )

        prompt = item["context"] + (filler * int(repeat)) + item["question"]
        scores = [
            _finite_float(
                score_choice(prompt, choice),
                field=f"long-range item {item_id!r} score {choice!r}",
            )
            for choice in choices
        ]
        predicted = max(range(len(scores)), key=scores.__getitem__)
        distractors = [score for choice, score in enumerate(scores) if choice != answer]
        rows.append(
            {
                "id": item_id,
                "family": item["family"],
                "distance": item["distance"],
                "answer": answer,
                "predicted": predicted,
                "correct": predicted == answer,
                "margin": scores[answer] - max(distractors),
                "scores": scores,
            }
        )

    by_family: dict[str, dict[str, float]] = {}
    for family in sorted({row["family"] for row in rows}):
        selected = [row for row in rows if row["family"] == family]
        by_family[family] = {
            "accuracy": math.fsum(row["correct"] for row in selected) / len(selected),
            "mean_margin": math.fsum(row["margin"] for row in selected) / len(selected),
        }

    by_distance: dict[str, float] = {}
    for distance in sorted({row["distance"] for row in rows}):
        selected = [row for row in rows if row["distance"] == distance]
        by_distance[distance] = math.fsum(row["correct"] for row in selected) / len(
            selected
        )

    near_far_gap = None
    if "near" in by_distance and "far" in by_distance:
        near_far_gap = by_distance["near"] - by_distance["far"]

    return {
        "status": "measured",
        "accuracy": math.fsum(row["correct"] for row in rows) / len(rows),
        "mean_margin": math.fsum(row["margin"] for row in rows) / len(rows),
        "by_family": by_family,
        "by_distance": by_distance,
        "near_far_gap": near_far_gap,
        "items": rows,
    }


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_CONFIG = ROOT / "configs" / "runtime_datasets.yaml"
BENCHMARK_RUNTIME_SOURCE = None
METHODS = ("a1", "a2", "a3")
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
EXTERNAL_METRIC_SPECS = (
    ("Validation loss", "validation_loss", "lower"),
    ("Perplexity", "perplexity", "lower"),
    ("Held-out test loss", "test_loss", "lower"),
    ("Held-out test perplexity", "test_perplexity", "lower"),
    ("Validation-loss forgetting", "forgetting", "lower"),
    ("On-domain KL drift", "domain_kl", "lower"),
    ("Token accuracy", "token_accuracy", "higher"),
    ("Calibration ECE", "calibration", "lower"),
)
INTERNAL_METRIC_SPECS = (
    ("Residual compressibility", "residual_compressibility", "higher"),
    ("Task-gradient conflict", "task_gradient_conflict", "lower"),
    ("Invariant redundancy", "invariant_redundancy", "lower"),
    ("Ablation prediction MAE", "ablation_mae", "lower"),
    ("Ablation rank correlation (Spearman)", "ablation_spearman", "higher"),
)
MEASURED_METRICS = tuple(
    key for _, key, _ in INTERNAL_METRIC_SPECS + EXTERNAL_METRIC_SPECS
)
AUDIT_METRICS = ("ablation_mae", "ablation_spearman")


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
    runtime_source: str | None = BENCHMARK_RUNTIME_SOURCE
    runtime_config: str = str(DEFAULT_RUNTIME_CONFIG)
    task_order: tuple[str, ...] = ()
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
        gradient_conflict_summary,
        invariant_redundancy,
    )
    from residual import ResidualAdapter, ResidualConfig
    from runtime.data_loader import (
        load_runtime_config,
        load_runtime_data,
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
    runtime_config_path = Path(run["materialized_runtime_config"])
    runtime_data = load_runtime_data(
        config_path=runtime_config_path,
        tokenizer=substrate.tokenizer,
        source=run.get("runtime_source"),
        task_order=run.get("task_order") or None,
        heldout_tasks=run.get("heldout_tasks") or None,
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
        for ids in batches:
            if run["method"] == "a3":
                base, adapted, _ = deployed(ids)
            else:
                base = substrate.run_with_interception(
                    ids,
                    modify_fn=identity_modify,
                    intercept_layers=adapter.layers,
                ).logits
                jax.block_until_ready(base)
                adapted = substrate.run_with_interception(
                    ids,
                    modify_fn=adapter.modify_fn(params),
                    intercept_layers=adapter.layers,
                ).logits
            jax.block_until_ready(adapted)
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
        representation["invariant_redundancy"][layer_name] = float(
            invariant_redundancy(a, b)
        )

    return {
        "status": "measured",
        "domains": metrics_by_domain,
        "representation": representation,
    }


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _config_fingerprint(config: dict[str, Any]) -> str:
    return _sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def measure_mixed_curvature(
    loss_fn: Callable[[Any], Any],
    params: dict[str, Any],
    *,
    samples: int = 2,
    seed: int = 0,
) -> dict[str, Any]:
    """Estimate ||d²L/dA dB||_F with Rademacher HVPs, without dense Hessians.

    Blocks are the A/B factors within each adapter layer. The reported value is
    the RMS of their Frobenius estimates; it is parameterization dependent.
    """
    import jax
    import jax.numpy as jnp

    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("curvature samples must be a positive integer")
    blocks = {}
    key = jax.random.PRNGKey(seed)
    for name, block in sorted(params.items()):
        if not isinstance(block, dict) or not {"A", "B"} <= block.keys():
            continue

        def block_loss(a: Any, b: Any, name: str = name, block: Any = block) -> Any:
            return loss_fn({**params, name: {**block, "A": a, "B": b}})

        squared = []
        for _ in range(samples):
            key, draw = jax.random.split(key)
            direction = jax.random.rademacher(draw, block["B"].shape).astype(
                block["B"].dtype
            )
            _, product = jax.jvp(
                lambda b, a=block["A"], objective=block_loss: jax.grad(
                    objective, argnums=0
                )(a, b),
                (block["B"],),
                (direction,),
            )
            squared.append(float(jnp.sum(jnp.square(product.astype(jnp.float32)))))
        blocks[name] = float(np.sqrt(np.mean(squared)))
    if not blocks:
        return {"status": "unavailable", "reason": "no adapter A/B blocks found"}
    if not all(math.isfinite(value) for value in blocks.values()):
        return {"status": "undefined", "reason": "non-finite mixed Hessian product"}
    return {
        "status": "measured",
        "estimator": "Rademacher HVP Frobenius norm; RMS across within-layer A/B blocks",
        "samples_per_block": samples,
        "seed": seed,
        "blocks": blocks,
        "value": float(np.sqrt(np.mean(np.square(list(blocks.values()))))),
    }


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
    runtime_config: Path = DEFAULT_RUNTIME_CONFIG,
    runtime_source: str | None = None,
) -> list[RunSpec]:
    runtime_config = Path(runtime_config).resolve()
    runtime_settings = yaml.safe_load(runtime_config.read_text(encoding="utf-8")) or {}
    runtime_source = runtime_source or runtime_settings.get("runtime", {}).get("source")
    if not runtime_source:
        raise ValueError("Specify runtime.source in YAML or --runtime-source")
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
    order = tuple(task_order or ())
    heldout = tuple(heldout_tasks or ())
    for method in methods:
        for seed in seeds:
            ks: list[int | None] = (
                [int(k) for k in k_values] if method == "a3" else [None]
            )
            for rank in ranks:
                for k in ks:
                    suffix = f"-r{rank}" + (f"-k{k}" if k is not None else "")
                    order_tag = "-".join(order) or re.sub(
                        r"[^a-zA-Z0-9_-]", "_", runtime_source
                    )
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
                            runtime_config=str(runtime_config),
                            runtime_source=runtime_source,
                            task_order=order,
                            heldout_tasks=heldout,
                            a2_arm="full" if method == "a2" else None,
                        )
                    )
    return specs


def _materialize(spec: RunSpec, root: Path, epochs: int | None) -> Path:
    source = Path(spec.config)
    config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
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
    if spec.task_order:
        config["runtime"]["task_order"] = list(spec.task_order)
    if spec.heldout_tasks:
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
    runtime_config: Path = DEFAULT_RUNTIME_CONFIG,
    runtime_source: str | None = None,
) -> dict[str, Any]:
    runtime_config = Path(runtime_config).resolve()
    runtime_settings = yaml.safe_load(runtime_config.read_text(encoding="utf-8")) or {}
    runtime_source = runtime_source or runtime_settings.get("runtime", {}).get("source")
    if not runtime_source:
        raise ValueError("Specify runtime.source in YAML or --runtime-source")
    selected_protocols = tuple(protocols or (protocol,))
    if len(set(selected_protocols)) != len(selected_protocols):
        raise ValueError("protocols must be unique")
    specs = []
    base_order = tuple(task_order or ())
    orders: tuple[tuple[str, ...], ...]
    heldout_variants: tuple[tuple[str, ...], ...]
    for selected in selected_protocols:
        if selected == "commutator" and not task_order:
            raise ValueError(
                "commutator requires explicit --task-order names; the loader owns task discovery"
            )
        if selected == "heldout" and not heldout_tasks:
            raise ValueError(
                "heldout requires explicit --heldout-task names; the loader owns task discovery"
            )
        if selected == "commutator":
            orders = (base_order, tuple(reversed(base_order)))
            heldout_variants = ((),)
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
                        runtime_config=runtime_config,
                        runtime_source=runtime_source,
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
                "requested_task_order": list(spec.task_order),
                "requested_heldout_tasks": list(spec.heldout_tasks),
                "planned_run_id": spec.run_id,
                "planned_condition_sha256": condition["condition_sha256"],
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
        "domains": list(base_order),
        "runtime_config": str(runtime_config),
        "runtime_source": runtime_source,
        "source": source,
        "measured_metrics": list(MEASURED_METRICS),
        "runs": run_rows,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _update_manifest(path: Path, run_id: str, **fields: Any) -> None:
    manifest = _read_json(path)
    for run in manifest["runs"]:
        if run.get("planned_run_id", run["run_id"]) == run_id:
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
        "--runtime-config",
        str(runtime_config),
        "--training-mode",
        "joint" if spec.protocol == "joint" else "sequential",
    ]
    if spec.runtime_source:
        command.extend(["--runtime-source", spec.runtime_source])
    if spec.task_order:
        command.extend(["--task-order", *spec.task_order])
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
            run_id=raw.get("planned_run_id", raw["run_id"]),
            config=raw["config"],
            output_dir=raw["output_dir"],
            run_name=raw["run_name"],
            k=raw.get("k"),
            rank=raw.get("rank", 16),
            runtime_source=raw.get("runtime_source", BENCHMARK_RUNTIME_SOURCE),
            runtime_config=raw.get("runtime_config", str(DEFAULT_RUNTIME_CONFIG)),
            task_order=tuple(
                raw.get("requested_task_order", raw.get("task_order", ()))
            ),
            heldout_tasks=tuple(
                raw.get("requested_heldout_tasks", raw.get("heldout_tasks", ()))
            ),
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
                condition_sha256=raw.get(
                    "condition_sha256", condition["condition_sha256"]
                ),
                config_sha256=config_sha,
            )
            if condition["condition_sha256"] != raw.get(
                "planned_condition_sha256", condition["condition_sha256"]
            ):
                reusable, reason = (
                    False,
                    "requested runtime or training condition changed",
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
            planned_condition_sha256=condition["condition_sha256"],
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
        meta = _read_json(run_dir / "meta.json")
        realized_order = meta.get("runtime_data", {}).get("task_order")
        if not realized_order:
            raise RuntimeError(
                f"{spec.run_id}: meta.json has no runtime_data.task_order"
            )
        realized_id = spec.run_id + "-tasks-" + fingerprint(realized_order)[:12]
        condition["task_order"] = list(realized_order)
        condition["heldout_tasks"] = sorted(
            meta["runtime_data"].get("heldout_tasks", spec.heldout_tasks)
        )
        condition.pop("condition_sha256", None)
        condition["condition_sha256"] = fingerprint(condition)
        _update_manifest(
            manifest_path, spec.run_id, realized_run_id=realized_id, **condition
        )
        completed_manifest = _read_json(manifest_path)
        completed_manifest["domains"] = list(
            dict.fromkeys(
                name
                for row in completed_manifest["runs"]
                for name in row.get("task_order", [])
            )
        )
        _write_json(manifest_path, completed_manifest)
        completion_spec = {
            **condition,
            "run_id": realized_id,
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
            materialized_config=str(provenance_dir / "materialized_stage.yaml"),
            materialized_runtime_config=str(
                provenance_dir / "materialized_runtime.yaml"
            ),
            log=str(run_dir / "logs" / "stdout.log"),
            run_dir=str(run_dir),
            finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        completed_manifest = _read_json(manifest_path)
        for row in completed_manifest["runs"]:
            if row.get("planned_run_id") == spec.run_id:
                row["run_id"] = realized_id
        _write_json(manifest_path, completed_manifest)


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
        "task_order": run.get("task_order", []),
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
                record["task_order"] = meta["runtime_data"].get(
                    "task_order", record["task_order"]
                )
                record["runtime_data_sha256"] = _config_fingerprint(
                    meta["runtime_data"]
                )
        try:
            evaluation_run = {
                **run,
                "task_order": record["task_order"],
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
        if not record.get("runtime_data_sha256"):
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
    evaluation = {
        "schema": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "domains": manifest.get("domains", []),
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
    if record.get("status") == "invalid":
        return []
    fields = {
        "validation_loss": "final_mean_loss",
        "perplexity": "final_mean_ppl",
        "test_loss": "final_test_mean_loss",
        "test_perplexity": "final_test_mean_ppl",
        "forgetting": "mean_forgetting",
    }
    deployed = record.get("deployed_evaluation", {})
    representation = deployed.get("representation", {})
    if metric in fields:
        value = record.get(fields[metric])
    elif metric == "residual_compressibility":
        value = (record.get(metric) or {}).get(str(rank))
    elif metric == "invariant_redundancy":
        value = representation.get(metric)
    elif metric == "task_gradient_conflict":
        value = representation.get(metric, {}).get("mean")
    elif metric in {"domain_kl", "token_accuracy", "calibration"}:
        field = {
            "domain_kl": "kl",
            "token_accuracy": "token_accuracy",
            "calibration": "ece",
        }[metric]
        value = [
            row.get(field)
            for row in deployed.get("domains", {}).values()
            if row.get("status") == "measured"
        ]
    else:
        value = None
    return _numeric_values(value)


def measurement_reason(record: dict[str, Any], metric: str) -> str:
    if record.get("status") == "invalid":
        return str(record.get("reason", "invalid run artifacts"))
    if metric in AUDIT_METRICS:
        return "No validated held-out component-effect predictor and paired ablation measurements; prediction fidelity is not measured."
    if metric == "residual_compressibility":
        return (
            "Requires a matched rank sweep with positive reference improvement: "
            + str(record.get("compressibility_status", "unavailable"))
        )
    return str(
        record.get("deployed_evaluation", {}).get("reason")
        or "Required measurement absent or insufficient held-out data."
    )


def resolve_report_dir(value: Path | str | None = None) -> Path:
    path = Path(value or GENERATED_REPORT_PREFIX)
    destination = (path if path.is_absolute() else ROOT / path).resolve()
    if destination == ROOT or destination in ROOT.parents:
        raise ValueError(
            "Report directory must not be the repository root or its ancestor"
        )
    return destination


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


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _condition_key(
    record: Mapping[str, Any],
) -> tuple[str, int, tuple[str, ...], tuple[str, ...]]:
    return (
        str(record.get("protocol", "sequential")),
        int(record.get("rank", 16)),
        tuple(record.get("task_order", ())),
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=("run", "plan", "train", "eval"),
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
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--runtime-source")
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
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
            runtime_config=args.runtime_config,
            runtime_source=args.runtime_source,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
