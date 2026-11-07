"""Inspect A4 metrics or measure a completed A4 checkpoint's latency."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from metrics.a4.latency import summarize_latency, summarize_transfers
from metrics.a4.retrieval import retrieval_metrics


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON number: {value}")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return result


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8-sig"),
        parse_constant=_invalid_constant,
        parse_float=_finite_float,
    )
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _optional(path: Path) -> dict[str, Any]:
    return _read(path) if path.is_file() else {}


def _source(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def inspect_run(path: Path) -> dict[str, Any]:
    """Read recorded metrics; absent measurements remain absent."""
    status = _read(path / "status.json")
    result = _optional(path / "results.json")
    if status.get("stage") != "A4" or (result and result.get("stage") != "A4"):
        raise ValueError("Expected a Stage A4 run")
    latest_train: dict[str, Any] = {}
    evaluations = []
    metrics_path = path / "metrics.jsonl"
    if metrics_path.is_file():
        with metrics_path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(
                    line, parse_constant=_invalid_constant, parse_float=_finite_float
                )
                if not isinstance(row, dict):
                    raise ValueError("Training metrics must be JSON objects")
                if row.get("event") == "train":
                    latest_train = row
                elif row.get("event") == "eval":
                    evaluations.append(row)
    return {
        "stage": "A4",
        "status": status,
        "results_available": bool(result),
        "results": result,
        "latest_training": latest_train,
        "evaluations": evaluations,
        "retrieval_quality": "Not established by training losses; score labelled probes separately",
        "scope": "Recorded observations only; no new training, inference or qualification",
    }


def list_runs(root: Path) -> dict[str, Any]:
    runs = []
    if root.exists():
        for path in sorted(root.iterdir()):
            status_path = path / "status.json"
            if not path.is_dir() or not status_path.is_file():
                continue
            status = _read(status_path)
            if status.get("stage") == "A4":
                runs.append(
                    {
                        "path": str(path),
                        "status": status.get("status"),
                        "checkpoint_available": (
                            path / "checkpoints/a4-final.npz"
                        ).is_file(),
                    }
                )
    return {"stage": "A4", "runs": runs, "scope": "A4 runs directly under " + str(root)}


def latest_completed_run(root: Path) -> Path:
    candidates = [
        Path(row["path"])
        for row in list_runs(root)["runs"]
        if row["status"] == "completed" and row["checkpoint_available"]
    ]
    if not candidates:
        raise ValueError(f"No completed A4 checkpoint found under {root}")
    return max(
        candidates,
        key=lambda path: ((path / "status.json").stat().st_mtime_ns, str(path)),
    )


def score_probes(path: Path) -> dict[str, Any]:
    payload = _read(path)
    kind, split = payload.get("label_kind"), payload.get("split")
    if kind not in {"semantic_firing", "exact_neighbor"}:
        raise ValueError("label_kind must be semantic_firing or exact_neighbor")
    if split not in {"train", "validation", "test"}:
        raise ValueError("Probe split must be train, validation or test")
    probes = payload["probes"]
    if not isinstance(probes, list) or not all(isinstance(row, dict) for row in probes):
        raise ValueError("probes must be a list of labelled retrieval records")
    result = retrieval_metrics(
        [row["retrieved_ids"] for row in probes],
        [row["relevant_ids"] for row in probes],
    )
    return {
        "stage": "A4",
        "measurement": "semantic_retrieval"
        if kind == "semantic_firing"
        else "ann_agreement",
        "split": split,
        "metrics": asdict(result),
        "source": _source(path),
        "scope": "Scores supplied ranked IDs, not new model inference; label completeness and held-out independence require separate audit",
    }


def inspect_latency(path: Path) -> dict[str, Any]:
    path = path / "result.json" if path.is_dir() else path
    payload = _read(path)
    if payload.get("status") != "completed" or not payload.get("checkpoint_sha256"):
        raise ValueError("Expected a completed A4 checkpoint latency measurement")
    samples = payload["samples"]
    if not isinstance(samples, list) or not samples:
        raise ValueError("Raw paired latency samples are required")
    modes = {row["mode"] for row in samples}
    if modes - {"training", "deployment"}:
        raise ValueError("Unsupported latency measurement mode")
    reports = []
    for mode in sorted(modes):
        rows = [row for row in samples if row["mode"] == mode]
        report = {"mode": mode, **summarize_latency(rows)}
        transfers_path = path.parent / f"cuda-transfers-{mode}.json"
        if transfers_path.is_file():
            transfers = _read(transfers_path)["samples"]
            if len(transfers) != len(rows):
                raise ValueError("CUDA transfer and primary samples must align")
            for original, transfer in zip(rows, transfers, strict=True):
                if any(
                    original[key] != transfer[key] for key in ("sample", "batch")
                ) or not math.isclose(
                    original["step_seconds"],
                    transfer["unprofiled_step_seconds"],
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ):
                    raise ValueError(
                        "CUDA transfer denominator does not match primary sample"
                    )
            report["cuda_transfers"] = {
                **summarize_transfers(transfers),
                "source": _source(transfers_path),
            }
        reports.append(report)
    return {
        "stage": "A4",
        "reports": reports,
        "source": _source(path),
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "bank_snapshot": payload.get("bank_snapshot"),
        "bank_records": payload.get("bank_records"),
        "scope": "Recomputed from recorded samples; not a new benchmark or an overall <5% qualification",
    }


def _print_result(command: str, result: dict[str, Any]) -> None:
    if command == "list":
        if not result["runs"]:
            print("No A4 training runs found")
        for row in result["runs"]:
            print(f"{row['status']:12} {row['path']}")
    elif command == "show":
        print(f"A4 status: {result['status']['status']}")
        if not result["results_available"]:
            print("Final results: not recorded")
        for row in result["evaluations"]:
            print(
                f"{row['split']} epoch {row['epoch']}: "
                f"CE={row['loss']:.6f} PPL={row['ppl']:.4f}"
            )
        row = result["latest_training"]
        if row:
            print(
                f"Last update {row['step']}: CE={row['task']:.6f} "
                f"key={row['key']:.6f} value={row['value']:.6f}"
            )
        print(result["retrieval_quality"])
    elif command == "retrieval":
        scores = result["metrics"]
        print(
            f"A4 {result['measurement']} | split={result['split']} | probes={scores['query_count']}"
        )
        print(
            f"Macro precision={scores['macro_precision']:.4f} recall={scores['macro_recall']:.4f}"
        )
        print(
            f"Micro precision={scores['micro_precision']:.4f} recall={scores['micro_recall']:.4f}"
        )
        print(
            f"Hit rate={scores['hit_rate']:.4f} MRR={scores['mean_reciprocal_rank']:.4f}"
        )
    else:
        for report in result["reports"]:
            print(
                f"A4 {report['mode']} | samples={report['samples']} | mean step={report['step_mean_ms']:.3f} ms"
            )
            for name, prefix in (
                ("Full bridge", "full_round_trip"),
                ("Bridge minus CPU search", "communication"),
            ):
                print(
                    f"{name}: {report[f'{prefix}_mean_ms']:.3f} ms | "
                    f"mean share={report[f'{prefix}_ratio_of_means_percent']:.2f}% | "
                    f"paired p95 share={report[f'{prefix}_paired_ratio_p95_percent']:.2f}%"
                )
            if "cuda_transfers" in report:
                copies = report["cuda_transfers"]
                print(
                    f"CUDA copy intervals only: {copies['transfer_only_mean_ms']:.3f} ms | "
                    f"mean share={copies['transfer_only_ratio_of_means_percent']:.2f}% | "
                    f"paired p95 share={copies['transfer_only_paired_ratio_p95_percent']:.2f}%"
                )
            else:
                print("CUDA copy intervals: not recorded")
    print(result["scope"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List A4 training runs")
    listing.add_argument("path", type=Path, nargs="?", default=Path("runs"))
    listing.add_argument("--json", action="store_true")
    for name, help_text in (
        ("show", "Show a saved A4 run's losses, evaluation and status"),
        ("retrieval", "Score explicitly labelled, already retrieved template IDs"),
        ("latency", "Recompute timing statistics from saved raw A4 samples"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("path", type=Path)
        command.add_argument(
            "--json", action="store_true", help="Print structured metrics"
        )
    measure = commands.add_parser(
        "measure-latency",
        help="Run a fresh GPU/CPU/GPU benchmark on a completed A4 run",
    )
    measure.add_argument(
        "path", type=Path, nargs="?", help="Completed A4 training run directory"
    )
    measure.add_argument(
        "--latest",
        action="store_true",
        help="Select the completed A4 checkpoint with the newest status-file modification time",
    )
    measure.add_argument("--runs-root", type=Path, default=Path("runs"))
    measure.add_argument(
        "--output",
        type=Path,
        help="Fresh output directory; defaults inside the run's evaluation directory",
    )
    measure.add_argument("--samples", type=int, default=100)
    measure.add_argument("--batches", type=int, default=10)
    measure.add_argument("--warmups", type=int, default=10)
    measure.add_argument(
        "--mode", choices=("both", "training", "deployment"), default="both"
    )
    measure.add_argument(
        "--native-library", type=Path, help="Explicit replacement native library"
    )
    measure.add_argument(
        "--skip-transfers",
        action="store_true",
        help="Skip the separate CUDA-event copy measurement",
    )
    measure.add_argument(
        "--allow-source-change",
        action="store_true",
        help="Record changed training sources for a runtime comparison, not an exact historical replay",
    )
    retrieval = commands.add_parser(
        "measure-retrieval",
        help="Evaluate a completed head on source-disjoint reference probes",
    )
    retrieval.add_argument("path", type=Path, nargs="?")
    retrieval.add_argument("--latest", action="store_true")
    retrieval.add_argument("--runs-root", type=Path, default=Path("runs"))
    retrieval.add_argument("--probes", type=Path, required=True)
    retrieval.add_argument("--output", type=Path)
    retrieval.add_argument("--native-library", type=Path)
    retrieval.add_argument("--allow-source-change", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command in {"measure-latency", "measure-retrieval"}:
            if (args.path is None) == (not args.latest):
                raise ValueError("Supply a run directory or --latest, not both")
            selected = (
                latest_completed_run(args.runs_root) if args.latest else args.path
            )
            print(f"Selected A4 run: {selected.resolve()}", flush=True)
            if args.command == "measure-retrieval":
                from stages.stage_A.a4.evaluation import measure_retrieval

                output = measure_retrieval(
                    selected,
                    args.probes,
                    output=args.output,
                    native_library=args.native_library,
                    allow_source_change=args.allow_source_change,
                )
                result = _read(output / "result.json")
                for report in result["reports"]:
                    print(
                        f"{report['condition']}: causal-window CE={report['causal_window_ce']:.6f}"
                    )
                    for layer in report["layers"]:
                        for row in layer["scores"]:
                            print(
                                f"  layer {layer['layer']} top-{row['top_k']}: "
                                f"all-probe precision={row['all_probe_micro_precision']:.4f} "
                                f"positive-probe recall={row['positive_only']['macro_recall']:.4f}"
                            )
                if "support_prior" in result:
                    print("Bank-source support prior (no held-out fitting):")
                    for row in result["support_prior"]["scores"]:
                        print(
                            f"  top-{row['top_k']}: precision={row['all_probe_micro_precision']:.4f} "
                            f"positive-probe recall={row['positive_only']['macro_recall']:.4f}"
                        )
                print(result["scope"])
                return 0
            from stages.stage_A.a4.benchmark import measure_latency

            output = measure_latency(
                selected,
                output=args.output,
                samples=args.samples,
                batches=args.batches,
                warmups=args.warmups,
                mode=args.mode,
                native_library=args.native_library,
                profile_transfers=not args.skip_transfers,
                allow_source_change=args.allow_source_change,
            )
            _print_result("latency", inspect_latency(output))
            return 0
        result = {
            "list": list_runs,
            "show": inspect_run,
            "retrieval": score_probes,
            "latency": inspect_latency,
        }[args.command](args.path)
        if args.json:
            print(json.dumps(result, indent=2, allow_nan=False))
        else:
            _print_result(args.command, result)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        ImportError,
        subprocess.SubprocessError,
    ) as error:
        parser.error(str(error))
    return 0
