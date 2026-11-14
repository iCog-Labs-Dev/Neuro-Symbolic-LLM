"""Restore a completed A4 run for read-only checkpoint evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def recorded_training_state_source(meta: dict[str, Any], result: dict[str, Any]) -> str:
    """Validate an explicit protocol identity without guessing for old runs."""
    source = meta.get("training_state_source")
    if source != result.get("training_state_source"):
        raise ValueError("Training state source differs between metadata and results")
    if source is None:
        return "unspecified-legacy"
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Training state source must be a nonempty string")
    return source


def training_arguments(
    meta: dict[str, Any], *, native_library: Path | None = None
) -> argparse.Namespace:
    """Resolve recorded inputs independently of the measurement's working directory."""
    values = meta.get("training_arguments")
    if values is None:
        from experiments.run_stage_a4 import parse_args

        argv = list(meta["argv"][1:])
        if native_library is not None:
            if "--native-library" in argv:
                index = argv.index("--native-library")
                argv[index + 1] = str(native_library.resolve(strict=True))
            else:
                argv.extend(
                    ("--native-library", str(native_library.resolve(strict=True)))
                )
        values = vars(parse_args(argv))
    else:
        values = dict(values)
    base = Path(meta.get("working_directory", REPO_ROOT))
    for name in (
        "config",
        "warm_start",
        "train_sources",
        "validation_sources",
        "test_sources",
        "alignment_labels",
        "patterns",
        "embeddings",
        "native_library",
    ):
        if values.get(name) is not None:
            values[name] = (base / values[name]).resolve()
    return argparse.Namespace(**values)


def source_changes(meta: dict[str, Any], *, root: Path = REPO_ROOT) -> list[str]:
    changed = []
    for name, expected in meta["source_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Recorded source path leaves the checkout")
        if not path.is_file() or digest(path) != expected:
            changed.append(name)
    return changed


def _data_identity(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _data_identity(item) for key, item in value.items() if key != "path"
        }
    if isinstance(value, list):
        return [_data_identity(item) for item in value]
    return value


@dataclass
class A4Replay:
    model: Any
    params: Any
    state: Any
    optimizer: Any
    frozen: Any
    data: Any
    bank: Any
    config: Any
    retriever: Any
    arguments: argparse.Namespace
    metadata: dict[str, Any]
    result: dict[str, Any]
    changed_sources: list[str]
    checkpoint_sha256: str


@contextmanager
def checkpoint_replay(
    run: Path,
    *,
    native_library: Path | None = None,
    allow_source_change: bool = False,
    substrate: Any = None,
) -> Iterator[A4Replay]:
    status, result = read_json(run / "status.json"), read_json(run / "results.json")
    if (
        any(row.get("stage") != "A4" for row in (status, result))
        or status.get("status") != "completed"
    ):
        raise ValueError("A completed Stage A4 run is required")
    meta = read_json(run / "meta.json")
    training_state_source = recorded_training_state_source(meta, result)
    changed = source_changes(meta)
    if changed and not allow_source_change:
        raise ValueError(
            "Training source changed: "
            + ", ".join(changed)
            + "; use --allow-source-change only for an explicitly recorded runtime comparison"
        )
    if meta["retrieval_backend"] != "native":
        raise ValueError("Latency measurement requires a native-backend A4 run")
    original = training_arguments(meta, native_library=native_library)
    library = (
        native_library.resolve(strict=True)
        if native_library
        else original.native_library
    )
    if native_library is None and digest(library) != meta["native_library_sha256"]:
        raise ValueError("Native library changed; supply an explicit replacement")

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import torch

    from frozenllm.substrate import FrozenSubstrate
    from residual import ResidualAdapter
    from residual.symbolic_head.jax_head import SymbolicHead
    from stages.results import load_params
    from stages.stage_A.A3_model import A3Config, A3Model
    from stages.stage_A.a4.checkpoint import restore_a3_weights
    from stages.stage_A.a4.config import load_config
    from stages.stage_A.a4.data import load_data
    from stages.stage_A.a4.model import A4Model
    from stages.stage_A.a4.training import init_params, residual_fingerprint
    from stages.stage_base import load_configs
    from symbolic_memory.bank import load_bank
    from symbolic_memory.native_bridge import NativeJaxRetriever

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("A real JAX GPU is required")
    config = load_config(run / "config.yaml")
    if not config.training.jit:
        raise ValueError("Latency measurement requires the compiled A4 path")
    jax.config.update("jax_default_matmul_precision", original.precision)
    torch.set_num_threads(2)
    torch.manual_seed(config.training.seed)
    bank = load_bank(original.patterns, original.embeddings)
    if (
        bank.input_snapshot_sha256 != meta["bank_snapshot"]
        or len(bank.records) < config.minimum_templates
    ):
        raise ValueError(
            "Bank differs from the run or does not meet its configured size requirement"
        )
    if digest(config.residual_config) != meta["residual_config_sha256"]:
        raise ValueError("Inherited residual configuration changed")
    inherited = load_configs(config.residual_config, A3Config)
    if substrate is None:
        substrate = FrozenSubstrate(config.training.model, dtype=torch.float32)
    adapter = ResidualAdapter(inherited.adapter, substrate.architecture)
    continuous = A3Model(
        substrate,
        layers=meta["layers"],
        rank=inherited.adapter.rank,
        seq_len=config.training.seq_len,
        scale=adapter.scale,
        activation=inherited.adapter.activation,
        config=inherited.objective,
    )
    frozen = restore_a3_weights(continuous, load_params(run / "params/a3-frozen.npz"))
    data = load_data(original, substrate.tokenizer, bank, config.training)
    if _data_identity(data.provenance) != _data_identity(read_json(run / "data.json")):
        raise ValueError("Replay data differs from the completed training run")
    checkpoint = run / "checkpoints/a4-final.npz"
    checkpoint_hash = digest(checkpoint)
    backend = meta.get("retrieval", {})
    with NativeJaxRetriever.from_mork(
        bank,
        server_url=original.mork_url,
        library_path=library,
        top_m=config.top_m,
        max_queries=max(1024, config.training.batch_size * config.training.seq_len),
        output_layout=backend.get("output_layout", "packed"),
        return_mode=backend.get("return_mode", "sync"),
    ) as retriever:
        try:
            model = A4Model(
                continuous,
                SymbolicHead(
                    continuous.hidden_size,
                    bank.vector_dim,
                    continuous.layers,
                    retriever,
                ),
            )
            params: dict[str, Any] = {"symbolic": {}, "value": {}}
            for name, weights in load_params(checkpoint).items():
                group, layer = name.split(".", 1)
                params[group][layer] = {
                    key: jnp.asarray(value) for key, value in weights.items()
                }
            expected = init_params(model, config.training.seed)
            if jax.tree.structure(params) != jax.tree.structure(expected):
                raise ValueError("Checkpoint parameter structure differs from A4")
            for actual, template in zip(
                jax.tree.leaves(params), jax.tree.leaves(expected), strict=True
            ):
                if (
                    actual.shape != template.shape
                    or actual.dtype != template.dtype
                    or not np.isfinite(np.asarray(actual)).all()
                ):
                    raise ValueError(
                        "Invalid checkpoint parameter shape, dtype or value"
                    )
            optimizer = optax.chain(
                optax.clip_by_global_norm(original.max_grad_norm),
                optax.adam(config.training.learning_rate),
            )
            leaves, tree = jax.tree.flatten(optimizer.init(params))
            with np.load(checkpoint, allow_pickle=False) as archive:
                if {name for name in archive.files if name.startswith("opt/")} != {
                    f"opt/leaf_{i}" for i in range(len(leaves))
                }:
                    raise ValueError(
                        "Checkpoint optimizer does not match the A4 optimizer"
                    )
                saved = []
                for index, template in enumerate(leaves):
                    value = archive[f"opt/leaf_{index}"]
                    if (
                        value.shape != template.shape
                        or value.dtype != template.dtype
                        or not np.isfinite(value).all()
                    ):
                        raise ValueError("Invalid optimizer state")
                    saved.append(jnp.asarray(value))
                checkpoint_meta = json.loads(str(archive["__metadata__"]))
            fingerprint = residual_fingerprint(model, frozen)
            if (
                checkpoint_meta["bank_snapshot_sha256"] != bank.input_snapshot_sha256
                or any(
                    fingerprint != record["frozen_residual_sha256"]
                    for record in (checkpoint_meta, result)
                )
                or checkpoint_meta["updates"] != result["updates"]
                or checkpoint_meta.get("training_state_source", "unspecified-legacy")
                != training_state_source
            ):
                raise ValueError(
                    "Checkpoint bank, frozen residual, update or training-state identity mismatch"
                )
            state = jax.tree.unflatten(tree, saved)
            before = [
                np.asarray(value).copy() for value in jax.tree.leaves((params, state))
            ]
            yield A4Replay(
                model,
                params,
                state,
                optimizer,
                frozen,
                data,
                bank,
                config,
                retriever,
                original,
                meta,
                result,
                changed,
                checkpoint_hash,
            )
            jax.effects_barrier()
            for left, right in zip(
                before, jax.tree.leaves((params, state)), strict=True
            ):
                np.testing.assert_array_equal(left, np.asarray(right))
            if (
                residual_fingerprint(model, frozen) != fingerprint
                or not substrate.verify_frozen()["params_unchanged"]
            ):
                raise RuntimeError(
                    "Frozen model or residual changed during measurement"
                )
            if digest(checkpoint) != checkpoint_hash:
                raise RuntimeError("Checkpoint file changed during measurement")
        finally:
            jax.effects_barrier()
