"""Generate figures from saved evaluation artifacts, without loading models."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import fmean

import numpy as np

from benchmark.benchmark import (
    EXTERNAL_METRIC_SPECS,
    INTERNAL_METRIC_SPECS,
    METHODS,
    _metric_samples,
    _read_json,
    _select_headline_cohort,
    metric_values_for_record,
    resolve_report_dir,
)
from benchmark.summary import report_manifest


def _load_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Figure generation requires matplotlib; install the visualization extra."
        ) from exc
    return plt


def generate_visualizations(
    output_dir: Path, report_dir: Path | str | None = None, report_rank: int = 16
):
    destination = resolve_report_dir(report_dir)
    if (
        destination == output_dir.resolve()
        or output_dir.resolve() in destination.parents
    ):
        raise ValueError("Reports must be outside the raw runs directory")
    evaluation = _read_json(output_dir / "evaluation.json")
    records = evaluation.get("records", [])
    cohort = _select_headline_cohort(records, report_rank)
    plt = _load_plotting()
    for section in (*METHODS, "comparison"):
        folder = (
            destination
            / "stage_A"
            / (section.upper() if section in METHODS else section)
            / "figures"
        )
        folder.mkdir(parents=True, exist_ok=True)
        for label, key, _ in INTERNAL_METRIC_SPECS + EXTERNAL_METRIC_SPECS:
            points = []
            if section == "comparison":
                samples = _metric_samples(cohort, key) if cohort else {}
                points = [
                    (method.upper(), fmean(value for _, value in values))
                    for method, values in samples.items()
                    if values
                ]
            else:
                for row in records:
                    if row.get("method") != section:
                        continue
                    values = metric_values_for_record(
                        row, key, int(row.get("rank", 16))
                    )
                    if values:
                        points.append((row["run_id"], fmean(values)))
            figure, axis = plt.subplots(figsize=(9, 5))
            axis.set_title(label)
            if points:
                axis.bar([name for name, _ in points], [value for _, value in points])
                axis.tick_params(axis="x", labelrotation=60)
            else:
                axis.text(
                    0.5,
                    0.5,
                    "N/A — see report for missing evidence",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                axis.set_axis_off()
            figure.tight_layout()
            figure.savefig(folder / f"{key}.png", dpi=150)
            plt.close(figure)
        if section in METHODS:
            candidates = [
                row
                for row in records
                if row.get("method") == section
                and row.get("status") != "invalid"
                and row.get("loss_matrix")
            ]
            for row in candidates:
                matrix = np.asarray(row["loss_matrix"], dtype=float)
                if matrix.ndim != 2 or not matrix.size:
                    continue
                figure, axis = plt.subplots(figsize=(8, 5))
                heatmap = axis.imshow(matrix, aspect="auto")
                axis.set_title(f"Validation loss: {row['run_id']}")
                axis.set_xlabel("Evaluation task")
                axis.set_ylabel("Training checkpoint")
                names = row.get("domains") or row.get("task_order", [])
                if len(names) == matrix.shape[1]:
                    axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
                figure.colorbar(heatmap, ax=axis, label="Validation CE")
                figure.tight_layout()
                # Hash identity to avoid interpreting task names as path segments.
                from benchmark.benchmark import fingerprint

                figure.savefig(
                    folder
                    / f"validation_loss_heatmap_{fingerprint(row['run_id'])[:16]}.png",
                    dpi=150,
                )
                plt.close(figure)
    # Refresh links and dashboard after figures exist; summary never imports plotting.
    report_manifest(output_dir, report_rank, destination)
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/benchmark"))
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--report-rank", type=int, default=16)
    args = parser.parse_args(argv)
    print(
        generate_visualizations(
            args.output_dir.resolve(), args.report_dir, args.report_rank
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
