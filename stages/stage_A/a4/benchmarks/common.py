"""Shared inputs, paired sampling and artifacts for infrastructure benchmarks."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def query_batch(records: Sequence[Any], count: int, path: Path | None) -> np.ndarray:
    if path is None:
        positions = np.linspace(0, len(records) - 1, count, dtype=int)
        queries = np.stack([records[i].key_vector for i in positions])
    else:
        queries = np.load(path, allow_pickle=False)
    dimension = len(records[0].key_vector)
    if (
        not isinstance(queries, np.ndarray)
        or queries.dtype != np.float32
        or queries.shape != (count, dimension)
        or not np.isfinite(queries).all()
        or np.any(np.linalg.norm(queries, axis=1) == 0)
    ):
        raise ValueError(
            f"Queries must be finite nonzero float32 [{count}, {dimension}]"
        )
    return cast(np.ndarray, np.ascontiguousarray(queries))


def paired_samples(
    calls: Mapping[str, Callable[[], Mapping[str, float]]],
    samples: int,
    warmups: int,
    rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Calls must synchronize device completion before returning CPU timings."""
    if not calls or samples < 1 or warmups < 1:
        raise ValueError("Provide variants and positive sample/warmup counts")
    for call in calls.values():
        for _ in range(warmups):
            call()
    names = list(calls)
    if rows is None:
        rows = []
    for trial in range(samples):
        order = names[trial % len(names) :] + names[: trial % len(names)]
        if trial % 2:
            order.reverse()
        for name in order:
            started = time.perf_counter_ns()
            details = dict(calls[name]())
            total = (time.perf_counter_ns() - started) / 1e9
            if any(not np.isfinite(v) or v < 0 for v in details.values()):
                raise ValueError("Invalid component timing")
            if "total_seconds" in details:
                raise ValueError("Variants cannot replace the synchronized wall timer")
            if "cpu_index_seconds" in details:
                search = details["cpu_index_seconds"]
                if search > total:
                    raise ValueError("CPU interval exceeds synchronized boundary")
                details["search_excluded_seconds"] = total - search
            rows.append(dict(variant=name, trial=trial, total_seconds=total, **details))
    return rows


def summarize(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for name in dict.fromkeys(row["variant"] for row in samples):
        selected = [row for row in samples if row["variant"] == name]
        metrics = sorted(
            {key for row in selected for key in row if key.endswith("_seconds")}
        )
        for metric in metrics:
            values = np.asarray([row[metric] for row in selected], dtype=float) * 1000
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Timing samples must be finite and nonnegative")
            rows.append(
                {
                    "variant": name,
                    "metric": metric,
                    "samples": len(values),
                    "mean_ms": float(values.mean()),
                    "p50_ms": float(np.percentile(values, 50)),
                    "p95_ms": float(np.percentile(values, 95)),
                }
            )
    return rows


def verify_outputs(expected: Sequence[Any], actual: Sequence[Any]) -> None:
    """Compare keys, values, scores and registry indices, including dtypes."""
    if len(expected) != 4 or len(actual) != 4:
        raise ValueError("Expected keys, values, scores and indices")
    for left, right in zip(expected, actual, strict=True):
        if np.asarray(left).dtype != np.asarray(right).dtype:
            raise ValueError("Retrieval output dtypes differ")
        np.testing.assert_array_equal(left, right)


def expected_outputs(index: Any, queries: np.ndarray, top_m: int) -> tuple[Any, ...]:
    matches = index.query_top_k(queries, top_m)
    lookup = {name: i for i, name in enumerate(index.id_store)}
    indices = np.asarray(
        [[lookup[name] for name in row] for row in matches.template_ids], dtype=np.int32
    )
    return matches.keys, matches.values, matches.scores, indices


class CpuBackend:
    """Set FAISS threads in the actual callback/server thread; time index work."""

    def __init__(self, index: Any, info: dict[str, Any], threads: int):
        self.index, self.info, self.threads = index, info, threads
        self.last_seconds = 0.0
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.index, name)

    def query_top_k(self, queries: np.ndarray, top_m: int) -> Any:
        import faiss

        previous = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(self.threads)
        try:
            started = time.perf_counter_ns()
            result = self.index.query_top_k(queries, top_m)
            self.last_seconds = (time.perf_counter_ns() - started) / 1e9
            self.calls += 1
            return result
        finally:
            faiss.omp_set_num_threads(previous)


def gpu_queries(queries: np.ndarray) -> Any:
    import jax

    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not devices:
        raise RuntimeError("Transport benchmarks require a CUDA-enabled JAX GPU")
    return jax.block_until_ready(jax.device_put(queries, devices[0]))


def write_artifacts(output: Path, result: dict[str, Any], samples: list) -> None:
    result["rows"] = summarize(samples)
    with (output / "result.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for name, rows in (("samples.csv", samples), ("summary.csv", result["rows"])):
        if rows:
            with (output / name).open("w", encoding="utf-8", newline="") as stream:
                fields = list(dict.fromkeys(key for row in rows for key in row))
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)


def metadata(args: Any, bank: Any, queries: np.ndarray) -> dict[str, Any]:
    import faiss

    return {
        "status": "running",
        "benchmark": args.benchmark,
        "protocol": "a4-infrastructure-v1",
        "python": platform.python_version(),
        "host": platform.platform(),
        "faiss": faiss.__version__,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "bank_snapshot": bank.input_snapshot_sha256,
        "bank_records": len(bank.records),
        "query_sha256": hashlib.sha256(queries.tobytes()).hexdigest(),
        "query_source": "supplied .npy; provenance supplied separately"
        if args.queries
        else "bank-key control; not held-out semantic relevance",
        "source_sha256": {
            path.name: digest(path)
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "ordering": "rotated and reversed variants; compilation/setup/warmups excluded",
        "training_performed": False,
        "model_time_denominator": None,
        "under_five_percent_verified": False,
    }
