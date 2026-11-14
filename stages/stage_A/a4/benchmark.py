"""Synchronized GPU/CPU/GPU latency measurement for a fixed A4 checkpoint."""

from __future__ import annotations

import os
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path
from statistics import fmean
from typing import Any

from metrics.a4.latency import (
    interval_breakdown,
    summarize_latency,
    summarize_transfers,
)
from stages.stage_A.a4.config import TRAINING_STATE_SOURCE
from stages.stage_A.a4.replay import (
    checkpoint_replay,
    digest,
    recorded_training_state_source,
)


def training_state_comparison(meta, result, *, mode, allow_source_change):
    """Keep historical training protocols distinct from the current kernel."""
    recorded = recorded_training_state_source(meta, result)
    matches = recorded == TRAINING_STATE_SOURCE
    if mode in {"training", "both"} and not matches and not allow_source_change:
        raise ValueError(
            "Recorded training state source differs from frozen A3 prediction; "
            "use --allow-source-change for an explicit current-protocol comparison"
        )
    return {
        "recorded": recorded,
        "runtime": TRAINING_STATE_SOURCE,
        "matches": matches,
    }


def require_idle_gpu() -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    pids = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if any(pid != os.getpid() for pid in pids):
        raise RuntimeError("Another GPU process is active; measure after it exits")


def _paired_samples(
    mode,
    invoke,
    bridge,
    retriever,
    batches,
    samples,
    warmups,
    ready,
    to_device,
    *,
    expected_calls,
    clock=time.perf_counter,
):
    print(f"Compiling and warming {mode}; these calls are not timed.", flush=True)
    for index in range(max(warmups, len(batches))):
        invoke(mode, batches[index % len(batches)])
    schedules = []
    for batch in batches:
        with retriever.capture_queries() as captured:
            invoke(mode, batch)
        if len(captured) != expected_calls:
            raise RuntimeError(
                "Captured retrieval calls do not match the attached layers"
            )
        schedules.append([to_device(query) for query in captured])
    if len({len(schedule) for schedule in schedules}) != 1:
        raise RuntimeError("Attached-layer retrieval schedule changed between batches")
    ready(schedules)
    for index in range(max(warmups, len(batches))):
        for query in schedules[index % len(batches)]:
            ready(bridge(query))
    print(f"Measuring {mode}: {samples} paired samples.", flush=True)
    rows = []
    for index in range(samples):
        batch_index = index % len(batches)
        queries = schedules[batch_index]
        measurements = {}
        for kind in ("step", "bridge") if index % 2 == 0 else ("bridge", "step"):
            before = retriever.stats()
            started = clock()
            if kind == "step":
                invoke(mode, batches[batch_index])
            else:
                for query in queries:
                    ready(bridge(query))
            seconds = clock() - started
            after = retriever.stats()
            delta = {key: after[key] - before[key] for key in before}
            if delta["calls"] != len(queries):
                raise RuntimeError(
                    "Measured retrieval schedule differs from the captured layers"
                )
            measurements[kind] = seconds, delta
        step_seconds, integrated = measurements["step"]
        full_seconds, replay = measurements["bridge"]
        rows.append(
            {
                "mode": mode,
                "sample": index,
                "batch": batch_index,
                "step_seconds": step_seconds,
                "full_round_trip_seconds": full_seconds,
                "cpu_index_seconds": replay["cpu_index_seconds"],
                "communication_seconds": full_seconds - replay["cpu_index_seconds"],
                "retrieval_calls_per_step": len(queries),
                "bridge_breakdown_seconds": interval_breakdown(
                    full_seconds, replay, retriever.info["return_mode"]
                ),
                "model_breakdown_seconds": interval_breakdown(
                    step_seconds, integrated, retriever.info["return_mode"]
                ),
                "query_bytes": sum(query.size * 4 for query in queries),
                "result_bytes": sum(
                    query.shape[0]
                    * min(retriever.info["max_top_m"], retriever.info["record_count"])
                    * (2 * query.shape[1] + 2)
                    * 4
                    for query in queries
                ),
            }
        )
    summarize_latency(rows)
    return rows


def _transfer_samples(mode, invoke, retriever, batches, primary, warmups, layer_count):
    print(f"Measuring {mode} CUDA copy intervals in a separate model pass.", flush=True)
    rows = []
    with retriever.transfer_timing():
        for index in range(max(warmups, len(batches))):
            invoke(mode, batches[index % len(batches)])
        retriever.transfer_stats(reset=True)
        for original in primary:
            before = retriever.transfer_stats()
            started = time.perf_counter()
            invoke(mode, batches[original["batch"]])
            seconds = time.perf_counter() - started
            after = retriever.transfer_stats()
            if after["calls"] - before["calls"] != layer_count:
                raise RuntimeError(
                    "CUDA transfer timing did not cover every attached layer"
                )
            rows.append(
                {
                    "sample": original["sample"],
                    "batch": original["batch"],
                    "d2h_gpu_seconds": after["d2h_gpu_seconds"]
                    - before["d2h_gpu_seconds"],
                    "h2d_gpu_seconds": after["h2d_gpu_seconds"]
                    - before["h2d_gpu_seconds"],
                    "profiled_model_seconds": seconds,
                    "unprofiled_step_seconds": original["step_seconds"],
                }
            )
    return rows


def measure_latency(
    run: Path,
    *,
    output: Path | None = None,
    samples: int = 100,
    batches: int = 10,
    warmups: int = 10,
    mode: str = "both",
    native_library: Path | None = None,
    profile_transfers: bool = True,
    allow_source_change: bool = False,
    substrate: Any = None,
) -> Path:
    """Replay fixed parameters and optimizer state without persisting updates."""
    if min(samples, batches, warmups) < 1 or samples < batches:
        raise ValueError(
            "Positive sample/batch/warmup counts are required; samples must cover all batches"
        )
    if mode not in {"training", "deployment", "both"}:
        raise ValueError("Unsupported measurement mode")
    run = run.resolve(strict=True)
    output = output or run / "evaluation" / time.strftime(
        "latency-%Y%m%d-%H%M%S", time.gmtime()
    )
    output = output.resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh measurement output directory")
    require_idle_gpu()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import jax
    import jax.numpy as jnp
    import numpy as np

    from stages.results import RunWriter
    from stages.stage_A.a4.training import make_train_step

    with ExitStack() as runtime:
        replay = runtime.enter_context(
            checkpoint_replay(
                run,
                native_library=native_library,
                allow_source_change=allow_source_change,
                substrate=substrate,
            )
        )
        config = replay.config
        state_comparison = training_state_comparison(
            replay.metadata,
            replay.result,
            mode=mode,
            allow_source_change=allow_source_change,
        )
        batch_size = config.training.batch_size
        if len(replay.data.train) < batches * batch_size:
            raise ValueError("Not enough examples for the requested distinct batches")
        step = make_train_step(
            replay.model,
            replay.frozen,
            replay.optimizer,
            jnp.asarray(np.stack([row.key_vector for row in replay.bank.records])),
            jnp.asarray(np.stack([row.val_vector for row in replay.bank.records])),
            config,
        )

        @jax.jit
        def deploy(params, ids, mask):
            clean, _ = replay.model.continuous.clean_pass(ids)
            latents = replay.model.continuous.predict_latents(replay.frozen, clean)
            forward = replay.model.forward_at_latents(
                replay.frozen, params["symbolic"], ids, latents
            )
            return replay.model.continuous.sub.compute_loss(
                forward.logits, jnp.where(mask, ids, -100), losses=("CE",)
            )

        order = np.random.default_rng(config.training.seed).permutation(
            len(replay.data.train)
        )
        selected = []
        for index in range(batches):
            indices = order[index * batch_size : (index + 1) * batch_size]
            positions, targets = replay.data.batch_labels(indices)
            mask = (
                np.ones_like(replay.data.train[indices], dtype=bool)
                if replay.data.train_mask is None
                else replay.data.train_mask[indices]
            )
            selected.append(
                tuple(
                    jnp.asarray(value)
                    for value in (replay.data.train[indices], positions, targets, mask)
                )
            )
        jax.block_until_ready((replay.params, replay.state, selected))

        def invoke(phase, batch):
            ids, positions, targets, mask = batch
            value = (
                step(replay.params, replay.state, ids, positions, targets, mask)
                if phase == "training"
                else deploy(replay.params, ids, mask)
            )
            jax.block_until_ready(value)
            if (phase == "training" and not bool(value[-1])) or (
                phase == "deployment" and not np.isfinite(float(value))
            ):
                raise FloatingPointError("Nonfinite checkpoint replay")

        bridge = jax.jit(replay.retriever)
        with RunWriter(output.parent, output.name) as writer:
            writer.write_config(
                {
                    "source_run": str(run),
                    "training_state_source": state_comparison,
                    "samples": samples,
                    "batches": batches,
                    "warmups": warmups,
                    "mode": mode,
                    "profile_transfers": profile_transfers,
                    "native_library": str(native_library) if native_library else None,
                    "allow_source_change": allow_source_change,
                }
            )
            writer.write_json("status.json", {"stage": "A4", "status": "running"})
            try:
                reports, observations = [], []
                for phase in ("training", "deployment") if mode == "both" else (mode,):
                    rows = _paired_samples(
                        phase,
                        invoke,
                        bridge,
                        replay.retriever,
                        selected,
                        samples,
                        warmups,
                        jax.block_until_ready,
                        jax.device_put,
                        expected_calls=len(replay.model.continuous.layers),
                    )
                    report = {"mode": phase, **summarize_latency(rows)}
                    for prefix in ("bridge", "model"):
                        parts = rows[0][f"{prefix}_breakdown_seconds"]
                        report[f"{prefix}_breakdown_mean_ms"] = {
                            name: 1000
                            * fmean(
                                row[f"{prefix}_breakdown_seconds"][name] for row in rows
                            )
                            for name in parts
                        }
                    report["mean_query_bytes_per_sample"] = fmean(
                        row["query_bytes"] for row in rows
                    )
                    report["mean_result_bytes_per_sample"] = fmean(
                        row["result_bytes"] for row in rows
                    )
                    if profile_transfers:
                        transfers = _transfer_samples(
                            phase,
                            invoke,
                            replay.retriever,
                            selected,
                            rows,
                            warmups,
                            len(replay.model.continuous.layers),
                        )
                        report["cuda_transfers"] = summarize_transfers(transfers)
                        writer.write_json(
                            f"cuda-transfers-{phase}.json",
                            {"samples": transfers, "report": report["cuda_transfers"]},
                        )
                    reports.append(report)
                    observations.extend(rows)
                    writer.log("latency", **report)
                summary = {
                    "contract_version": "a4-latency-v1",
                    "stage": "A4",
                    "status": "completed",
                    "run": str(run),
                    "training_state_source": state_comparison,
                    "checkpoint_sha256": replay.checkpoint_sha256,
                    "bank_snapshot": replay.bank.input_snapshot_sha256,
                    "bank_records": len(replay.bank.records),
                    "ten_thousand_templates_verified": len(replay.bank.records)
                    >= 10000,
                    "model": config.training.model,
                    "precision": replay.arguments.precision,
                    "top_m": config.top_m,
                    "batch_size": batch_size,
                    "seq_len": config.training.seq_len,
                    "layers": list(replay.model.continuous.layers),
                    "retrieval_backend": replay.retriever.info,
                    "measurement_source_sha256": digest(Path(__file__).resolve()),
                    "training_native_library_sha256": replay.metadata[
                        "native_library_sha256"
                    ],
                    "measurement_native_library_sha256": digest(
                        native_library.resolve()
                        if native_library
                        else replay.arguments.native_library
                    ),
                    "explicit_runtime_replacement": native_library is not None,
                    "changed_training_sources": replay.changed_sources,
                    "compilation_and_warmups_excluded": True,
                    "parameters_and_optimizer_state_advanced": False,
                    "checkpoints_written": False,
                    "frozen_weights_verified_unchanged": True,
                    "reports": reports,
                    "samples": observations,
                    "scope": "Fixed-checkpoint current-runtime measurement; training uses frozen A3 predictions. Shares of complete A4 training or deployment CE steps, not incremental head overhead or autoregressive generation latency. A different recorded training protocol is an explicit comparison, not historical-step replay.",
                }
                runtime.close()
                writer.write_json("result.json", summary)
                writer.write_json("status.json", {"stage": "A4", "status": "completed"})
            except BaseException as error:
                writer.write_json(
                    "status.json",
                    {"stage": "A4", "status": "failed", "error": str(error)},
                )
                raise
    print(f"Saved: {output / 'result.json'}", flush=True)
    return output
