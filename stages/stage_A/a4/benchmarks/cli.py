"""Command-line entry point for opt-in A4 infrastructure comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmark", choices=("search", "tcp-local", "native-prototype")
    )
    parser.add_argument("--patterns", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fresh directory, preferably under runs/",
    )
    parser.add_argument("--mork-url", default="http://127.0.0.1:18000")
    parser.add_argument(
        "--queries",
        type=Path,
        help="Optional float32 .npy [query-count, bank dimension]",
    )
    parser.add_argument("--query-count", type=positive, default=256)
    parser.add_argument("--top-m", type=positive, default=8)
    parser.add_argument("--threads", type=positive, default=2)
    parser.add_argument("--samples", type=positive, default=100)
    parser.add_argument("--warmups", type=positive, default=10)
    parser.add_argument("--ef-search", type=positive, nargs="+", default=[16, 50, 100])
    parser.add_argument(
        "--cuda-root",
        type=Path,
        help="Optional CUDA toolkit root for prototype compilation",
    )
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output must not already exist")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Lazy imports keep --help usable without JAX, FAISS, CUDA or model weights.
    from symbolic_memory.bank import load_bank
    from symbolic_memory.storage import MorkTemplateStore

    from .common import metadata, query_batch, write_artifacts

    bank = load_bank(args.patterns, args.embeddings)
    if args.top_m > min(64, len(bank.records)):
        raise ValueError("top-m must not exceed 64 or the bank size")
    queries = query_batch(bank.records, args.query_count, args.queries)
    args.output.mkdir(parents=True, exist_ok=False)
    result = metadata(args, bank, queries)
    samples: list[dict[str, Any]] = []
    write_artifacts(args.output, result, samples)
    try:
        # Never install or mutate the user's MORK bank as a benchmarking side effect.
        with MorkTemplateStore(args.mork_url, timeout_sec=300) as store:
            print("Reading and verifying the existing MORK bank...", flush=True)
            if args.benchmark == "search":
                from .search import measure as measure_search

                index = store.load_index(bank)
                measure_search(args, index, queries, result, samples)
            elif args.benchmark == "tcp-local":
                from .transport import measure as measure_transport

                measure_transport(args, bank, store, queries, result, samples)
            else:
                from .prototype import measure as measure_prototype

                index = store.load_index(bank)
                measure_prototype(args, index, queries, result, samples)
            result["mork_readback_verified"] = True
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_artifacts(args.output, result, samples)
    for row in result["rows"]:
        print(
            f"{row['variant']:20s} {row['metric']:26s} mean={row['mean_ms']:.3f} ms  p50={row['p50_ms']:.3f}  p95={row['p95_ms']:.3f}",
            flush=True,
        )
    print(f"Saved: {args.output}", flush=True)
    return 0
