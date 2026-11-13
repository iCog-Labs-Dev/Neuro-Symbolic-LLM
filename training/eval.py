"""Global post-hoc evaluation for matched A1, A2, and A3 benchmark runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

STAGE_METHODS = ("a1", "a2", "a3")


def _read_manifest(benchmark_dir: Path) -> dict[str, Any]:
    path = benchmark_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark manifest not found: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Benchmark manifest is not valid JSON: {path}") from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("runs"), list):
        raise ValueError("Benchmark manifest must contain a runs list")
    return manifest


def _validate_methods(
    methods: Sequence[str], manifest: dict[str, Any]
) -> tuple[str, ...]:
    selected = tuple(str(method).lower() for method in methods)
    if len(set(selected)) != len(selected):
        raise ValueError("methods must be unique")
    if set(selected) != set(STAGE_METHODS):
        raise ValueError(
            "Global stage comparison requires exactly A1, A2, and A3; "
            f"got {list(selected)}"
        )
    planned = {str(run.get("method", "")).lower() for run in manifest["runs"]}
    missing = sorted(set(selected) - planned)
    if missing:
        raise ValueError(f"Manifest has no planned runs for stages: {missing}")
    return selected


def _incomplete_runs(
    benchmark_dir: Path,
    manifest: dict[str, Any],
    methods: tuple[str, ...],
) -> list[str]:
    incomplete = []
    for run in manifest["runs"]:
        if str(run.get("method", "")).lower() not in methods:
            continue
        result_path = Path(run.get("results", ""))
        if result_path and not result_path.is_absolute():
            result_path = benchmark_dir / result_path
        if run.get("status") != "complete" or not result_path.is_file():
            incomplete.append(str(run.get("run_id", "unknown")))
    return incomplete


def evaluate_all_stages(
    benchmark_dir: Path | str,
    *,
    methods: Sequence[str] = STAGE_METHODS,
    generate_report: bool = True,
    report_rank: int = 4,
    require_complete: bool = True,
    generate_visualizations: bool = True,
    visualization_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Evaluate and compare all matched A1/A2/A3 runs in one benchmark root.

    Training remains stage-local. This function validates the shared manifest,
    invokes the common deployed evaluator for every completed run, aggregates
    matched conditions, and optionally writes ``REPORT.md``.
    """
    root = Path(benchmark_dir).resolve()
    manifest = _read_manifest(root)
    selected = _validate_methods(methods, manifest)
    incomplete = _incomplete_runs(root, manifest, selected)
    if incomplete and require_complete:
        preview = ", ".join(incomplete[:10])
        suffix = "" if len(incomplete) <= 10 else f" (+{len(incomplete) - 10} more)"
        raise RuntimeError(
            "Global evaluation requires complete A1/A2/A3 runs; incomplete: "
            f"{preview}{suffix}"
        )

    # Keep heavy model/JAX imports out of module import and CLI help paths.
    from benchmark.core import evaluate_manifest, report_manifest

    evaluation = evaluate_manifest(root)
    report_path = (
        report_manifest(root, report_rank=report_rank) if generate_report else None
    )
    visualization = None
    if generate_report and generate_visualizations:
        from benchmark.visualization import generate_reports_and_visualizations

        visualization = generate_reports_and_visualizations(
            root,
            output_dir=visualization_dir,
            report_rank=report_rank,
        )
    if require_complete and evaluation["completed_runs"] != evaluation["planned_runs"]:
        raise RuntimeError(
            "Evaluation excluded invalid or missing runs: "
            f"{evaluation['completed_runs']}/{evaluation['planned_runs']} valid"
        )
    evaluation["global_evaluation"] = {
        "methods": list(selected),
        "incomplete_runs": incomplete,
        "evaluation_path": str(root / "evaluation.json"),
        "report_path": None if report_path is None else str(report_path),
        "visualization": visualization,
    }
    return evaluation


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("runs") / "benchmark",
        help="Benchmark root containing manifest.json and stage run artifacts.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=STAGE_METHODS,
        default=list(STAGE_METHODS),
    )
    parser.add_argument("--report-rank", type=int, default=4)
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Write evaluation.json without regenerating REPORT.md.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Evaluate available runs while retaining explicit coverage gaps.",
    )
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--visualization-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    evaluation = evaluate_all_stages(
        args.benchmark_dir,
        methods=args.methods,
        generate_report=not args.no_report,
        report_rank=args.report_rank,
        require_complete=not args.allow_incomplete,
        generate_visualizations=not args.no_visualization,
        visualization_dir=args.visualization_dir,
    )
    global_result = evaluation["global_evaluation"]
    print(
        f"evaluated {evaluation['completed_runs']}/{evaluation['planned_runs']} "
        f"A1/A2/A3 runs: {global_result['evaluation_path']}"
    )
    if global_result["report_path"] is not None:
        print(f"report: {global_result['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
