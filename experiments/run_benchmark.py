"""Thin CLI for the compact Stage A A1/A2/A3 benchmark harness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.core import (  # noqa: E402
    evaluate_manifest,
    make_manifest,
    report_manifest,
    train_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "train", "eval", "report"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "benchmark")
    parser.add_argument(
        "--profile", choices=("smoke", "headline", "full"), default="smoke"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("a1", "a2", "a3"),
        default=["a1", "a2", "a3"],
    )
    parser.add_argument("--k", nargs="+", type=int, default=None)
    parser.add_argument("--ranks", nargs="+", type=int, default=None)
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=("sequential", "joint", "heldout", "commutator"),
        default=None,
    )
    parser.add_argument("--task-order", nargs="+", default=None)
    parser.add_argument("--heldout-task", action="append", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--include-a4-plan", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--report-rank", type=int, default=4)
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=None,
        help="Generated report root; defaults to repository/report_and_visualization.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    defaults = {
        "smoke": {"seeds": [0], "ranks": [16], "k": [4], "protocols": ["sequential"]},
        "headline": {
            "seeds": [0, 1, 2],
            "ranks": [16],
            "k": [1, 2, 4, 8],
            "protocols": ["sequential", "joint"],
        },
        "full": {
            "seeds": [0, 1, 2, 3, 4],
            "ranks": [1, 2, 4, 8, 16, 32, 64],
            "k": [1, 2, 4, 8],
            "protocols": ["sequential", "joint", "heldout", "commutator"],
        },
    }[args.profile]
    seeds = args.seeds or defaults["seeds"]
    ranks = args.ranks or defaults["ranks"]
    k_values = args.k or defaults["k"]
    protocols = args.protocols or defaults["protocols"]
    if args.command == "plan":
        manifest = make_manifest(
            output_dir,
            seeds=seeds,
            methods=args.methods,
            k_values=k_values,
            rank_values=ranks,
            epochs=args.epochs,
            protocols=protocols,
            task_order=args.task_order,
            heldout_tasks=args.heldout_task,
            include_a4_plan=args.include_a4_plan,
            profile=args.profile,
        )
        print(f"planned {len(manifest['runs'])} runs: {output_dir / 'manifest.json'}")
    elif args.command == "train":
        train_manifest(output_dir, resume=not args.no_resume, epochs=args.epochs)
        print(f"training complete: {output_dir}")
    elif args.command == "eval":
        evaluation = evaluate_manifest(output_dir)
        print(
            f"evaluated {evaluation['completed_runs']}/{evaluation['planned_runs']} runs"
        )
    else:
        report_path = report_manifest(output_dir, report_rank=args.report_rank)
        print(report_path)
        if not args.no_visualization:
            from benchmark.visualization import generate_reports_and_visualizations

            generated = generate_reports_and_visualizations(
                output_dir,
                output_dir=args.visualization_dir,
                report_rank=args.report_rank,
                repository_root=ROOT,
            )
            print(f"visualizations: {generated['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
