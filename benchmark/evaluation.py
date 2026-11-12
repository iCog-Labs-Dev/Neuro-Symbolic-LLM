"""Post-hoc legal, semantic, and external benchmark evaluation."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from metrics.performance import (
    MultilabelPlattCalibrator,
    RidgeProbe,
    apply_multilabel_platt,
    fit_multilabel_platt,
    fit_ridge_probe,
    fit_temperature,
    multilabel_label_losses,
    multilabel_metrics,
    per_example_task_losses,
    ridge_probe_scores,
    select_multilabel_threshold,
    single_label_metrics,
    temperature_scale,
)
from metrics.representation import (
    ablation_fidelity,
    causal_fingerprint_separability,
    gradient_conflict_summary,
    mixed_curvature_coupling,
    off_support_leakage,
    semantic_factor_metrics,
)
from metrics.symbolic import retrieval_report
from runtime.data_loader import LabeledSplit


@dataclass(frozen=True)
class TaskProbe:
    task_name: str
    multilabel: bool
    label_names: tuple[str, ...]
    probe: RidgeProbe
    temperature: float | None
    calibrator: MultilabelPlattCalibrator | None
    threshold: float | None
    bins: int
    acquisition_checkpoint: int
    train_fingerprint: str
    validation_fingerprint: str

    def probabilities(self, features: Any) -> np.ndarray:
        scores = ridge_probe_scores(self.probe, features)
        if self.multilabel:
            if self.calibrator is None:
                raise ValueError("multilabel probe has no calibrator")
            return apply_multilabel_platt(self.calibrator, scores)
        if self.temperature is None:
            raise ValueError("single-label probe has no temperature")
        return temperature_scale(scores, self.temperature)


def fit_task_probe(
    train: LabeledSplit,
    validation: LabeledSplit,
    train_features: Any,
    validation_features: Any,
    *,
    alpha: float,
    bins: int,
    acquisition_checkpoint: int,
) -> tuple[TaskProbe, dict[str, float]]:
    if train.task_name != validation.task_name:
        raise ValueError("train and validation splits must belong to one task")
    if (
        train.label_names != validation.label_names
        or train.multilabel != validation.multilabel
    ):
        raise ValueError("train and validation label schemas differ")
    labels = None if train.multilabel else np.arange(len(train.label_names))
    probe = fit_ridge_probe(
        train_features,
        train.probe_targets(),
        alpha=alpha,
        labels=labels,
    )
    validation_scores = ridge_probe_scores(probe, validation_features)
    if train.multilabel:
        calibrator = fit_multilabel_platt(validation_scores, validation.targets)
        probabilities = apply_multilabel_platt(calibrator, validation_scores)
        threshold = select_multilabel_threshold(validation.targets, probabilities)
        temperature = None
        metrics = multilabel_metrics(
            validation.targets, probabilities, threshold=threshold
        )
    else:
        validation_targets = validation.probe_targets()
        temperature = fit_temperature(
            validation_scores,
            validation_targets,
            labels=labels,
        )
        probabilities = temperature_scale(validation_scores, temperature)
        threshold = None
        calibrator = None
        metrics = single_label_metrics(
            validation_targets,
            probabilities,
            labels=labels,
            bins=bins,
        )
    artifact = TaskProbe(
        task_name=train.task_name,
        multilabel=train.multilabel,
        label_names=train.label_names,
        probe=probe,
        temperature=temperature,
        calibrator=calibrator,
        threshold=threshold,
        bins=bins,
        acquisition_checkpoint=acquisition_checkpoint,
        train_fingerprint=train.fingerprint,
        validation_fingerprint=validation.fingerprint,
    )
    return artifact, {str(key): float(value) for key, value in metrics.items()}


def evaluate_task_probe(
    probe: TaskProbe,
    split: LabeledSplit,
    features: Any,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    if (
        probe.task_name != split.task_name
        or probe.multilabel != split.multilabel
        or probe.label_names != split.label_names
    ):
        raise ValueError("probe and labeled split schemas differ")
    probabilities = probe.probabilities(features)
    labels = None if split.multilabel else np.arange(len(split.label_names))
    if split.multilabel:
        if probe.threshold is None:
            raise ValueError("multilabel probe has no frozen threshold")
        metrics = multilabel_metrics(
            split.targets,
            probabilities,
            threshold=probe.threshold,
        )
        primary = metrics["micro_f1"]
        task_accuracy = metrics["subset_accuracy"]
    else:
        metrics = single_label_metrics(
            split.probe_targets(),
            probabilities,
            labels=labels,
            bins=probe.bins,
        )
        primary = metrics["accuracy"]
        task_accuracy = metrics["accuracy"]
    losses = (
        multilabel_label_losses(split.targets, probabilities)
        if split.multilabel
        else per_example_task_losses(
            split.probe_targets(),
            probabilities,
            multilabel=False,
            labels=labels,
        )
    )
    normalized = {str(key): float(value) for key, value in metrics.items()}
    normalized.update(
        task_accuracy=float(task_accuracy),
        task_f1=float(metrics["macro_f1"]),
        task_primary_score=float(primary),
        task_ece=float(metrics.get("ece", np.nan)),
    )
    return normalized, probabilities, losses


def extract_split_features(
    split: LabeledSplit,
    feature_fn: Callable[[np.ndarray, np.ndarray], Any],
    *,
    batch_size: int,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    rows = []
    positions = np.maximum(np.sum(split.attention_mask, axis=1) - 1, 0).astype(np.int32)
    for start in range(0, len(split.input_ids), batch_size):
        end = start + batch_size
        values = np.asarray(
            feature_fn(split.input_ids[start:end], positions[start:end]),
            dtype=np.float64,
        )
        if values.ndim != 2 or values.shape[0] != len(split.input_ids[start:end]):
            raise ValueError("feature_fn must return [batch, hidden] features")
        rows.append(values)
    return np.concatenate(rows, axis=0)


def semantic_factor_effects(
    full_losses: Any,
    ablated_losses: Any,
    factor_mask: Any,
    *,
    min_support: int,
) -> dict[str, Any]:
    """Aggregate signed component effects by gold semantic factor."""
    full = np.asarray(full_losses, dtype=np.float64)
    ablated = np.asarray(ablated_losses, dtype=np.float64)
    factors = np.asarray(factor_mask, dtype=np.bool_)
    if full.ndim not in {1, 2}:
        raise ValueError("full losses must be [examples] or [examples, factors]")
    if factors.ndim != 2 or factors.shape[0] != full.shape[0]:
        raise ValueError("factor_mask must be [examples, factors]")
    if full.ndim == 1 and (ablated.ndim != 2 or ablated.shape[1] != full.shape[0]):
        raise ValueError("expected ablated losses [components, examples]")
    if full.ndim == 2 and (
        full.shape != factors.shape
        or ablated.ndim != 3
        or ablated.shape[1:] != full.shape
    ):
        raise ValueError(
            "multilabel losses require full [examples, factors] and "
            "ablated [components, examples, factors]"
        )
    if min_support < 1:
        raise ValueError("min_support must be >= 1")
    support = np.sum(factors, axis=0)
    valid = support >= min_support
    effects = np.full((ablated.shape[0], factors.shape[1]), np.nan, dtype=np.float64)
    deltas = ablated - full[None, :]
    for factor in np.flatnonzero(valid):
        selected = factors[:, factor]
        values = deltas[:, selected] if full.ndim == 1 else deltas[:, selected, factor]
        effects[:, factor] = np.mean(values, axis=1)
    return {
        "signed_effects": effects,
        "support": support,
        "valid": valid,
        "status": "measured" if np.any(valid) else "insufficient_data",
    }


def save_task_probe(path: Path, probe: TaskProbe) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "weights": probe.probe.weights,
        "intercept": probe.probe.intercept,
        "scaler_mean": probe.probe.scaler.mean,
        "scaler_scale": probe.probe.scaler.scale,
        "classes": (
            np.asarray([]) if probe.probe.classes is None else probe.probe.classes
        ),
        "label_names": np.asarray(probe.label_names),
        "multilabel": np.asarray(probe.multilabel),
        "temperature": np.asarray(
            np.nan if probe.temperature is None else probe.temperature
        ),
        "threshold": np.asarray(np.nan if probe.threshold is None else probe.threshold),
        "acquisition_checkpoint": np.asarray(probe.acquisition_checkpoint),
        "train_fingerprint": np.asarray(probe.train_fingerprint),
        "validation_fingerprint": np.asarray(probe.validation_fingerprint),
    }
    if probe.calibrator is not None:
        payload.update(
            platt_slopes=probe.calibrator.slopes,
            platt_intercepts=probe.calibrator.intercepts,
            platt_constant=probe.calibrator.constant_labels,
        )
    np.savez(path, **payload)
    return path


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray) or hasattr(value, "shape"):
        array = np.asarray(value)
        return array.item() if array.ndim == 0 else array.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _coverage_support(effects: np.ndarray, coverage: float = 0.9) -> np.ndarray:
    importance = np.maximum(effects, 0.0)
    support = np.zeros_like(importance, dtype=np.bool_)
    for factor in range(importance.shape[1]):
        values = importance[:, factor]
        total = float(np.sum(values))
        if total <= 0.0:
            continue
        order = np.argsort(-values, kind="stable")
        cumulative = np.cumsum(values[order]) / total
        count = int(np.searchsorted(cumulative, coverage, side="left")) + 1
        support[order[:count], factor] = True
    return support


def evaluate_legal_tasks(
    *,
    labeled_splits: dict[str, dict[str, LabeledSplit]],
    checkpoints: list[dict[str, Any]],
    feature_fn: Callable[[Any, LabeledSplit], np.ndarray],
    components: list[dict[str, Any]],
    ablate_fn: Callable[[Any, dict[str, Any], float], Any],
    output_dir: Path,
    alpha: float,
    bins: int,
    min_factor_support: int,
) -> dict[str, Any]:
    """Fit acquisition probes, measure forgetting, and audit semantic components."""
    if not checkpoints:
        return {"status": "unavailable", "reason": "no task checkpoints"}
    by_task = {str(row["task"]): row for row in checkpoints}
    joint_checkpoint = by_task.get("joint")
    ordered_tasks = [
        name
        for name in labeled_splits
        if name in by_task or joint_checkpoint is not None
    ]
    if not ordered_tasks:
        return {"status": "unavailable", "reason": "no labeled task checkpoints"}
    probes: dict[str, TaskProbe] = {}
    validation_metrics = {}
    cache: dict[tuple[int, str, str], np.ndarray] = {}

    def features(checkpoint: dict[str, Any], name: str, split: str) -> np.ndarray:
        key = (int(checkpoint["index"]), name, split)
        if key not in cache:
            cache[key] = feature_fn(checkpoint["params"], labeled_splits[name][split])
        return cache[key]

    for name in ordered_tasks:
        checkpoint = by_task.get(name, joint_checkpoint)
        assert checkpoint is not None
        train = labeled_splits[name]["train"]
        validation = labeled_splits[name]["validation"]
        probe, metrics = fit_task_probe(
            train,
            validation,
            features(checkpoint, name, "train"),
            features(checkpoint, name, "validation"),
            alpha=alpha,
            bins=bins,
            acquisition_checkpoint=int(checkpoint["index"]),
        )
        probes[name] = probe
        validation_metrics[name] = metrics
        save_task_probe(output_dir / "probes" / f"{name}.npz", probe)

    primary_matrix: list[list[float | None]] = []
    checkpoint_metrics = []
    final_by_task: dict[str, dict[str, float]] = {}
    final_checkpoint = checkpoints[-1]
    for checkpoint in checkpoints:
        row: list[float | None] = []
        task_rows = {}
        for name in ordered_tasks:
            probe = probes[name]
            if int(checkpoint["index"]) < probe.acquisition_checkpoint:
                row.append(None)
                continue
            metrics, _, _ = evaluate_task_probe(
                probe,
                labeled_splits[name]["test"],
                features(checkpoint, name, "test"),
            )
            row.append(metrics["task_primary_score"])
            task_rows[name] = metrics
            if checkpoint is final_checkpoint:
                final_by_task[name] = metrics
        primary_matrix.append(row)
        checkpoint_metrics.append(
            {"checkpoint": int(checkpoint["index"]), "tasks": task_rows}
        )

    forgetting: list[float | None] = []
    matrix = np.asarray(
        [
            [np.nan if value is None else value for value in row]
            for row in primary_matrix
        ],
        dtype=np.float64,
    )
    for current in range(len(checkpoints)):
        drops = []
        for task_index, name in enumerate(ordered_tasks):
            acquired = probes[name].acquisition_checkpoint
            if acquired >= current or task_index >= matrix.shape[1]:
                continue
            history = matrix[acquired:current, task_index]
            current_value = matrix[current, task_index]
            if np.isfinite(history).any() and np.isfinite(current_value):
                drops.append(float(np.nanmax(history) - current_value))
        forgetting.append(float(np.mean(drops)) if drops else None)

    representation: dict[str, Any] = {
        "status": "unavailable",
        "reason": "no canonical components",
    }
    if components:
        factor_names = []
        factor_support = []
        full_effect_columns = []
        quarter_effect_columns = []
        half_effect_columns = []
        support_effect_columns = []
        sample_gradient_rows = []
        naturality_errors = {}
        final_params = final_checkpoint["params"]
        for name in ordered_tasks:
            split = labeled_splits[name]["test"]
            probe = probes[name]
            full_features = feature_fn(final_params, split)
            _, full_probabilities, full_losses = evaluate_task_probe(
                probe, split, full_features
            )
            transformed_split = replace(
                split,
                input_ids=split.transformed_input_ids,
                attention_mask=split.transformed_attention_mask,
            )
            transformed_features = feature_fn(final_params, transformed_split)
            _, transformed_probabilities, _ = evaluate_task_probe(
                probe, transformed_split, transformed_features
            )
            naturality_errors[name] = float(
                np.mean(
                    np.sum(
                        (transformed_probabilities - full_probabilities) ** 2, axis=1
                    )
                )
            )
            support_split = labeled_splits[name]["validation"]
            support_features = feature_fn(final_params, support_split)
            _, _, support_full_losses = evaluate_task_probe(
                probe, support_split, support_features
            )
            ablated_losses = []
            quarter_losses = []
            half_losses = []
            support_ablated_losses = []
            for component in components:
                rows = []
                for retained in (0.0, 0.75, 0.5):
                    changed = ablate_fn(final_params, component, retained)
                    changed_features = feature_fn(changed, split)
                    _, _, losses = evaluate_task_probe(probe, split, changed_features)
                    rows.append(losses)
                ablated_losses.append(rows[0])
                quarter_losses.append(rows[1])
                half_losses.append(rows[2])
                support_changed = ablate_fn(final_params, component, 0.0)
                support_changed_features = feature_fn(support_changed, support_split)
                _, _, support_losses = evaluate_task_probe(
                    probe, support_split, support_changed_features
                )
                support_ablated_losses.append(support_losses)
            full_result = semantic_factor_effects(
                full_losses,
                np.asarray(ablated_losses),
                split.factor_mask,
                min_support=min_factor_support,
            )
            quarter_result = semantic_factor_effects(
                full_losses,
                np.asarray(quarter_losses),
                split.factor_mask,
                min_support=min_factor_support,
            )
            half_result = semantic_factor_effects(
                full_losses,
                np.asarray(half_losses),
                split.factor_mask,
                min_support=min_factor_support,
            )
            quarter_array = np.asarray(quarter_losses, dtype=np.float64)
            full_array = np.asarray(full_losses, dtype=np.float64)
            sample_gradient_rows.append(
                (4.0 * (quarter_array - full_array[None, ...]))
                .reshape((len(components), -1))
                .T
            )
            support_result = semantic_factor_effects(
                support_full_losses,
                np.asarray(support_ablated_losses),
                support_split.factor_mask,
                min_support=min_factor_support,
            )
            full_effects = np.asarray(full_result["signed_effects"])
            support_effects = np.asarray(support_result["signed_effects"])
            valid = (
                np.asarray(full_result["valid"], dtype=bool)
                & np.asarray(support_result["valid"], dtype=bool)
                & (np.nansum(np.maximum(full_effects, 0.0), axis=0) > 0.0)
                & (np.nansum(np.maximum(support_effects, 0.0), axis=0) > 0.0)
            )
            factor_names.extend(
                factor
                for factor, keep in zip(split.factor_names, valid, strict=True)
                if keep
            )
            factor_support.extend(np.asarray(full_result["support"])[valid].tolist())
            full_effect_columns.append(full_effects[:, valid])
            quarter_effect_columns.append(
                np.asarray(quarter_result["signed_effects"])[:, valid]
            )
            half_effect_columns.append(
                np.asarray(half_result["signed_effects"])[:, valid]
            )
            support_effect_columns.append(support_effects[:, valid])

        if factor_names:
            actual = np.concatenate(full_effect_columns, axis=1)
            quarter = np.concatenate(quarter_effect_columns, axis=1)
            half = np.concatenate(half_effect_columns, axis=1)
            support_fit = np.concatenate(support_effect_columns, axis=1)
            predicted = 6.0 * half - 8.0 * quarter
            structure = semantic_factor_metrics(actual)
            gradients = 4.0 * quarter
            sample_gradients = np.concatenate(sample_gradient_rows, axis=0)
            conflict = gradient_conflict_summary(gradients.T)
            support = _coverage_support(support_fit)
            leakage = off_support_leakage(gradients, support=support)
            component_layers = np.asarray([int(item["layer"]) for item in components])
            role_labels = np.argmax(np.maximum(support_fit, 0.0), axis=1)
            role_mass = np.sum(np.maximum(support_fit, 0.0), axis=1)
            role_counts = np.bincount(role_labels, minlength=support_fit.shape[1])
            fingerprint_mask = (role_mass > 0.0) & (role_counts[role_labels] >= 2)
            if len(np.unique(role_labels[fingerprint_mask])) >= 2:
                fingerprints = causal_fingerprint_separability(
                    actual[fingerprint_mask],
                    labels=role_labels[fingerprint_mask],
                )
                fingerprints["label_source"] = (
                    "validation_semantic_factor_argmax_test_fingerprint"
                )
            else:
                fingerprints = {
                    "status": "unavailable",
                    "reason": "fewer than two validation-defined roles with two components",
                }
            fidelity = ablation_fidelity(
                predicted.ravel(),
                actual.ravel(),
                top_k=min(20, actual.size),
            )
            curvature = {}
            unique_layers = sorted(set(component_layers.tolist()))
            for left_index, left in enumerate(unique_layers):
                for right in unique_layers[left_index + 1 :]:
                    curvature[f"layer_{left}:layer_{right}"] = float(
                        mixed_curvature_coupling(
                            sample_gradients[:, component_layers == left],
                            sample_gradients[:, component_layers == right],
                        )
                    )
            overlap = np.asarray(structure["overlap"], dtype=np.float64)
            fracture = np.asarray(structure["fracture"], dtype=np.float64)
            representation = {
                "status": str(structure["status"]),
                "estimator": "canonical_svd_semantic_label_ablation",
                "component_names": [str(item["name"]) for item in components],
                "factor_names": factor_names,
                "factor_support": factor_support,
                "signed_effects": actual.tolist(),
                "fracture": {
                    name: float(value)
                    for name, value in zip(factor_names, fracture, strict=True)
                },
                "fracture_mean": float(np.nanmean(fracture)),
                "overlap_matrix": overlap.tolist(),
                "overlap_mean_off_diagonal": (
                    float(np.nanmean(overlap[~np.eye(len(factor_names), dtype=bool)]))
                    if len(factor_names) > 1
                    else None
                ),
                "gradient_conflict": _json_value(conflict),
                "mixed_curvature_coupling": curvature,
                "mixed_curvature_estimator": (
                    "per_observation_quarter_ablation_empirical_fisher_proxy"
                ),
                "off_support_leakage": {
                    name: float(value)
                    for name, value in zip(
                        factor_names, np.asarray(leakage), strict=True
                    )
                },
                "support_protocol": "validation_fit_test_evaluation_90pct_positive_mass",
                "causal_fingerprint_separability": _json_value(fingerprints),
                "ablation_fidelity": {
                    **_json_value(fidelity),
                    "estimator": "quarter_half_intervention_quadratic_extrapolation",
                },
                "category_naturality": {
                    "status": "measured_proxy",
                    "transformation": "lowercase_input_identity_label_action",
                    "by_task": naturality_errors,
                    "value": float(np.mean(list(naturality_errors.values()))),
                },
            }

    single_label_accuracies = [
        final_by_task[name]["task_accuracy"]
        for name in final_by_task
        if not probes[name].multilabel
    ]
    aggregate = {
        "task_accuracy": (
            float(np.mean(single_label_accuracies)) if single_label_accuracies else None
        ),
        "task_macro_f1": float(
            np.mean([row["task_f1"] for row in final_by_task.values()])
        ),
        "task_primary_score": float(
            np.mean([row["task_primary_score"] for row in final_by_task.values()])
        ),
        "task_score_forgetting": forgetting[-1] if forgetting else None,
    }
    return {
        "status": "measured",
        "protocol": "frozen_acquisition_ridge_probe",
        "primary_metric": {
            name: "micro_f1" if probes[name].multilabel else "accuracy"
            for name in ordered_tasks
        },
        "validation": validation_metrics,
        "checkpoints": checkpoint_metrics,
        "primary_score_matrix": primary_matrix,
        "task_score_forgetting": forgetting,
        "final": final_by_task,
        "aggregate": aggregate,
        "representation": representation,
    }


def aggregate_lm_rows(
    rows: list[dict[str, float | int]],
) -> dict[str, float | int]:
    if not rows:
        return {"status": "insufficient_data", "tokens": 0}
    tokens = sum(int(row["tokens"]) for row in rows)
    if tokens <= 0:
        return {"status": "insufficient_data", "tokens": 0}
    loss = sum(float(row["loss_sum"]) for row in rows) / tokens
    base_loss = sum(float(row["base_loss_sum"]) for row in rows) / tokens
    return {
        "status": "measured",
        "tokens": tokens,
        "loss": loss,
        "base_loss": base_loss,
        "perplexity": float(np.exp(loss)),
        "base_perplexity": float(np.exp(base_loss)),
        "kl": sum(float(row["kl_sum"]) for row in rows) / tokens,
        "token_accuracy": sum(float(row["correct"]) for row in rows) / tokens,
    }


def evaluate_long_range(
    config: dict[str, Any],
    score_choice: Callable[[str, str], float],
) -> dict[str, Any]:
    if config.get("schema") != "long-range/v1":
        raise ValueError("long-range probes must use schema 'long-range/v1'")
    items = config.get("items", [])
    if not items:
        return {"status": "not_configured", "reason": "no long-range probes"}
    rows = []
    for item in items:
        required = {
            "id",
            "family",
            "distance",
            "context",
            "question",
            "choices",
            "answer",
        }
        missing = required - set(item)
        if missing:
            raise ValueError(f"long-range item is missing {sorted(missing)}")
        if not isinstance(item["choices"], list) or len(item["choices"]) < 2:
            raise ValueError("long-range item requires at least two choices")
        filler = str(item.get("filler", ""))
        repeat = int(item.get("filler_repeat", 0))
        if repeat < 0:
            raise ValueError("filler_repeat must be >= 0")
        prompt = str(item["context"]) + (filler * repeat) + str(item["question"])
        scores = [float(score_choice(prompt, choice)) for choice in item["choices"]]
        answer = int(item["answer"])
        predicted = int(np.argmax(scores))
        distractors = [score for index, score in enumerate(scores) if index != answer]
        rows.append(
            {
                "id": str(item["id"]),
                "family": str(item["family"]),
                "distance": str(item["distance"]),
                "correct": predicted == answer,
                "margin": scores[answer] - max(distractors),
                "scores": scores,
            }
        )
    by_family = {}
    for family in sorted({row["family"] for row in rows}):
        selected = [row for row in rows if row["family"] == family]
        by_family[family] = {
            "accuracy": float(np.mean([row["correct"] for row in selected])),
            "mean_margin": float(np.mean([row["margin"] for row in selected])),
        }
    by_distance = {}
    for distance in sorted({row["distance"] for row in rows}):
        selected = [row for row in rows if row["distance"] == distance]
        by_distance[distance] = float(np.mean([row["correct"] for row in selected]))
    return {
        "status": "measured",
        "accuracy": float(np.mean([row["correct"] for row in rows])),
        "mean_margin": float(np.mean([row["margin"] for row in rows])),
        "by_family": by_family,
        "by_distance": by_distance,
        "near_far_gap": (
            by_distance.get("near", np.nan) - by_distance.get("far", np.nan)
        ),
        "items": rows,
    }


def evaluate_symbolic_rankings(
    path: Path | None, cutoffs=(1, 4, 8, 16)
) -> dict[str, Any]:
    if path is None:
        return {
            "status": "not_applicable",
            "reason": "A1/A2/A3 have no symbolic retrieval head",
        }
    retrieved = []
    relevant = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            values = list(row["retrieved"])
            if len(values) != len(set(values)):
                raise ValueError("symbolic rankings must contain unique IDs")
            retrieved.append(values)
            relevant.append(set(row["relevant"]))
    return {"status": "measured", **retrieval_report(retrieved, relevant, cutoffs)}


__all__ = [
    "TaskProbe",
    "aggregate_lm_rows",
    "evaluate_task_probe",
    "evaluate_legal_tasks",
    "evaluate_long_range",
    "evaluate_symbolic_rankings",
    "extract_split_features",
    "fit_task_probe",
    "save_task_probe",
    "semantic_factor_effects",
]
