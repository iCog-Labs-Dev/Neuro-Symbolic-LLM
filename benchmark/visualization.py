"""Deterministic Markdown reports and figures for completed benchmark outputs."""

from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import numpy as np

from benchmark.core import (
    EXPECTED_DOMAINS,
    EXTERNAL_METRIC_SPECS,
    INTERNAL_METRIC_SPECS,
    OPERATIONAL_METRIC_SPECS,
    metric_values_for_record,
    select_a3_k,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VISUALIZATION_DIR = REPOSITORY_ROOT / "report_and_visualization"
STAGE_FOLDER_PATTERN = "stage_{stage}"
GENERAL_REPORT_PATTERN = "stage_{stage}_general_report.md"
STAGE_REPORT_PATTERN = "{method}_result_and_visualization.md"
FIGURES_FOLDER = "figures"
METHODS = ("a1", "a2", "a3")
METHOD_LABELS = {"a1": "A1", "a2": "A2", "a3": "A3"}
METHOD_COLORS = {"a1": "#3B82F6", "a2": "#F59E0B", "a3": "#10B981"}
FIGURE_DPI = 150
SYMBOLIC_SCOPE_REASON = "requires a trained A4 symbolic head and measurement artifacts"
UNAVAILABLE_INTERNAL_METRIC_SPECS = (
    ("Representation fracture", "representation_fracture_unavailable", "context"),
    ("Factor overlap", "factor_overlap_unavailable", "lower"),
    ("Mixed curvature coupling proxy", "mixed_curvature_unavailable", "lower"),
    ("Off-support leakage", "off_support_leakage_unavailable", "lower"),
    (
        "Causal fingerprint separability",
        "causal_fingerprint_unavailable",
        "higher",
    ),
    ("Naturality proxy", "naturality_unavailable", "context"),
    ("Ablation fidelity proxy", "ablation_fidelity_unavailable", "higher"),
)
UNAVAILABLE_EXTERNAL_METRIC_SPECS = (
    ("Task primary score", "task_primary_unavailable", "higher"),
    ("Task macro-F1", "task_macro_f1_unavailable", "higher"),
    ("Accuracy forgetting", "accuracy_forgetting_unavailable", "lower"),
)
VISUALIZATION_OPERATIONAL_METRIC_SPECS = OPERATIONAL_METRIC_SPECS + (
    ("Mean training step time (seconds)", "mean_step_time", "lower"),
)
UNAVAILABLE_REASONS = {
    "representation_fracture_unavailable": "semantic factors are not configured for A1-A3",
    "factor_overlap_unavailable": "semantic factors are not configured for A1-A3",
    "mixed_curvature_unavailable": "no validated per-example curvature protocol is enabled",
    "off_support_leakage_unavailable": "A1-A3 do not define intended component supports",
    "causal_fingerprint_unavailable": "independent component-role labels are unavailable",
    "naturality_unavailable": "category labels and declared transformations are unavailable",
    "ablation_fidelity_unavailable": "the sampled intervention audit protocol is not enabled",
    "task_primary_unavailable": "the current benchmark does not use labeled task targets",
    "task_macro_f1_unavailable": "the current benchmark does not use labeled task targets",
    "accuracy_forgetting_unavailable": "task-score matrices are unavailable without labeled targets",
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
        print(f"warning: visualization figures disabled: {error}")
        return None


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
    record: Mapping[str, Any]
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
        if len(fingerprints) == 1:
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
    if metric in UNAVAILABLE_REASONS:
        return UNAVAILABLE_REASONS[metric]
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
        ("long_range", "Long-range accuracy", "long_range.png"),
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
                f"{artifact.record.get('protocol', 'sequential')} "
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
            "_N/A: the current benchmark does not use labeled task targets._",
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
                    label = f"seed {artifact.record.get('seed')} {key}"
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

    lines.extend(["", "## Data Availability", ""])
    if inventory:
        for event, keys in inventory.items():
            lines.append(f"- `{event}`: {', '.join(f'`{key}`' for key in keys)}")
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
        keys = ("loss_task", "loss_total", "loss_cpc", "loss_hid", "loss_stab")
        series = _training_series(artifacts, keys)
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
        for metric, title in (
            ("final_mean_loss", "A3 final validation loss over K and rank"),
            ("mean_forgetting", "A3 forgetting over K and rank"),
            ("compressibility_c4", "A3 residual compressibility C(4) over K and rank"),
        ):
            ranks = sorted({int(a.record["rank"]) for a in artifacts})
            ks = sorted(
                {int(a.record["k"]) for a in artifacts if a.record.get("k") is not None}
            )
            grid = np.full((len(ks), len(ranks)), np.nan)
            for row_index, k in enumerate(ks):
                for column_index, rank in enumerate(ranks):
                    values = []
                    for artifact in artifacts:
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
                        values.append(_finite(raw_value))
                    values = [value for value in values if value is not None]
                    if values:
                        grid[row_index, column_index] = fmean(values)
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
        for artifact in artifacts:
            value = _finite(artifact.record.get("mean_step_time"))
            k = _finite(artifact.record.get("k"))
            if value is not None and k is not None:
                step_series[f"rank {artifact.record.get('rank')}"].append((k, value))
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
            "Task-score, macro-F1, and accuracy-forgetting metrics are N/A because the current benchmark does not produce label-based task metrics.",
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
        ("internal", INTERNAL_METRIC_SPECS + UNAVAILABLE_INTERNAL_METRIC_SPECS),
        ("external", EXTERNAL_METRIC_SPECS + UNAVAILABLE_EXTERNAL_METRIC_SPECS),
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
            "## A4 Symbolic Scope",
            "",
            f"- Symbolic metrics: N/A ({SYMBOLIC_SCOPE_REASON}).",
            f"- Blocked A4 child plans in manifest: {len(manifest.get('a4_runs', []))}.",
            "- Pure metric contracts under `metrics/a4` are not measured results.",
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
            best = (
                max(means, key=means.get)
                if direction == "higher"
                else min(means, key=means.get)
            )
            lines.append(
                f"- {label}: {METHOD_LABELS[best]} is best on the matched cohort ({means[best]:.6g}; {_direction_text(direction)})."
            )
        lines.append(
            "- Operational metrics are reported separately and are not folded into a composite score."
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
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)
    plotting = _load_plotting()
    plt, pd = plotting if plotting is not None else (None, None)
    general_report, cohort = _write_general_report(
        manifest,
        evaluation,
        destination,
        stage_root,
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
    return {
        "output_dir": str(destination),
        "readme": str(readme),
        "general_report": str(general_report),
        "stage_reports": {method: str(path) for method, path in stage_reports.items()},
        "figures_enabled": plt is not None,
    }


__all__ = [
    "DEFAULT_VISUALIZATION_DIR",
    "GENERAL_REPORT_PATTERN",
    "METHOD_COLORS",
    "STAGE_FOLDER_PATTERN",
    "STAGE_REPORT_PATTERN",
    "generate_reports_and_visualizations",
]
