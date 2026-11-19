"""Run Stage A4 symbolic-head training."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from frozenllm.substrate import FrozenSubstrate
    from stages.stage_A.a4.config import A4Config

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = REPO_ROOT / "configs/stage_A/a4_symbolic_head.yaml"
DEFAULT_RUNTIME_CONFIG = REPO_ROOT / "configs/runtime_datasets.yaml"

# Temporary synthetic bank for A4 integration testing.
# Replace with the real miner-generated bank for production.
DEFAULT_BANK_DIR = REPO_ROOT / "data/a4_test_bank"
DEFAULT_PATTERNS = DEFAULT_BANK_DIR / "patterns.jsonl"
DEFAULT_EMBEDDINGS = DEFAULT_BANK_DIR / "embeddings.jsonl"

DEFAULT_RUNS_DIR = REPO_ROOT / "runs"


def find_latest_a3_checkpoint(runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    """Find the latest completed A3 checkpoint."""
    valid = []
    for checkpoint in runs_dir.glob("a3hybrid-*/checkpoints/a3hybrid_final.npz"):
        if not checkpoint.is_file():
            continue
        meta_path = checkpoint.parents[1] / "meta.json"
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(metadata, dict) and metadata.get("stage") == "A3":
            valid.append(checkpoint)

    if not valid:
        raise FileNotFoundError(
            f"No valid A3 checkpoint found under {runs_dir}. "
            "Run A3 training first or provide --warm-start."
        )

    return max(valid, key=lambda path: path.stat().st_mtime_ns)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--benchmark-pair", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--data-mode",
        choices=("runtime", "aligned"),
        default="runtime",
        help="Runtime sequential datasets or legacy aligned source data.",
    )

    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=DEFAULT_RUNTIME_CONFIG,
        help="Shared runtime dataset configuration.",
    )

    parser.add_argument(
        "--warm-start",
        type=Path,
        default=None,
        help="A3 checkpoint. Defaults to the latest completed A3 checkpoint.",
    )
    parser.add_argument(
        "--patterns",
        type=Path,
        default=DEFAULT_PATTERNS,
        help="Symbolic pattern bank JSONL.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=DEFAULT_EMBEDDINGS,
        help="Template key/value embedding JSONL.",
    )

    for name in (
        "train-sources",
        "validation-sources",
        "alignment-labels",
    ):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--test-sources", type=Path)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="Disable CUDA copy timing during actual training updates",
    )
    parser.add_argument(
        "--latency-every",
        type=int,
        default=100,
        help="Measure CUDA copies on every Nth real update after warmup",
    )
    parser.add_argument("--model")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rank", type=int, help="Expected rank of the A3 warm start.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--eval-batches", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--run-name")
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--no-jit", action="store_true")
    parser.add_argument("--mork-url", default="http://127.0.0.1:18000")
    parser.add_argument(
        "--retrieval-backend", choices=("native", "tcp"), default="native"
    )
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--retrieval-host", default="127.0.0.1")
    parser.add_argument("--retrieval-port", type=int, default=19091)
    parser.add_argument("--max-train-rows", type=int, default=20000)
    parser.add_argument("--max-eval-rows", type=int, default=2000)
    parser.add_argument("--max-train-blocks", type=int, default=50000)
    parser.add_argument("--minimum-examples", type=int, default=5000)
    parser.add_argument("--max-eval-blocks", type=int, default=5000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--precision", choices=("default", "highest"), default="highest"
    )
    args = parser.parse_args(argv)
    if args.benchmark_pair is not None and (
        args.warm_start is None
        or not args.benchmark_pair.is_file()
        or not args.runtime_config.is_file()
    ):
        parser.error("A benchmark pair requires --warm-start, pairing metadata and runtime YAML")
    if args.data_mode == "aligned":
        required_sources = (
            "train_sources",
            "validation_sources",
            "alignment_labels",
        )

        for name in required_sources:
            if getattr(args, name) is None:
                parser.error(
                    f"--{name.replace('_', '-')} is required "
                    "in aligned mode"
                )

        if not args.skip_test and args.test_sources is None:
            parser.error(
                "--test-sources is required unless --skip-test "
                "is selected"
            )
    else:
        if not args.runtime_config.is_file():
            parser.error(
                f"Runtime dataset config not found: {args.runtime_config}"
            )
    for name in (
        "batch_size",
        "seq_len",
        "eval_batches",
        "log_every",
        "latency_every",
        "max_train_rows",
        "max_eval_rows",
        "max_train_blocks",
        "minimum_examples",
        "max_eval_blocks",
    ):
        if getattr(args, name) is not None and getattr(args, name) < (
            2 if name == "seq_len" else 1
        ):
            parser.error(f"Invalid --{name.replace('_', '-')}")
    if args.epochs is not None and args.epochs < 0:
        parser.error("--epochs must be nonnegative")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error("--seed must be an unsigned 32-bit integer")
    if args.rank is not None and args.rank < 1:
        parser.error("--rank must be positive")
    for name in ("learning_rate", "max_grad_norm"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.retrieval_backend == "native" and (
        args.native_library is None
        or not args.native_library.is_file()
        or not args.native_library.with_suffix(
            args.native_library.suffix + ".json"
        ).is_file()
    ):
        parser.error(
            "Native retrieval requires --native-library and its build manifest"
        )
    return args


def _apply_training_overrides(config: A4Config, args: argparse.Namespace) -> A4Config:
    overrides = {
        name: getattr(args, name)
        for name in (
            "model",
            "seed",
            "epochs",
            "batch_size",
            "seq_len",
            "learning_rate",
            "eval_batches",
            "log_every",
            "output_dir",
        )
        if getattr(args, name) is not None
    }
    return replace(
        config,
        training=replace(
            config.training, **overrides, jit=config.training.jit and not args.no_jit
        ),
    )


def _resolve_substrate(
    model_id: str, substrate: FrozenSubstrate | None, *, revision: str | None = None
) -> FrozenSubstrate:
    if substrate is not None:
        return substrate
    if revision is not None:
        from frozenllm.substrate import FrozenSubstrate

        return FrozenSubstrate(model_id, revision=revision)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load native weights before the substrate enables TorchAX dispatch.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    from frozenllm.substrate import FrozenSubstrate

    return FrozenSubstrate(model, tokenizer=tokenizer)


def _aligned_source_texts(args: argparse.Namespace) -> dict[str, set[str]]:
    sources: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    for split, texts in sources.items():
        path = getattr(args, f"{split}_sources")
        if path is not None and (split != "test" or not args.skip_test):
            with path.open(encoding="utf-8-sig") as stream:
                texts.update(
                    json.loads(line)["source_text"].strip()
                    for line in stream
                    if line.strip()
                )
    return sources


def main(
    argv: Sequence[str] | None = None, *, substrate: FrozenSubstrate | None = None
) -> int:
    args = parse_args(argv)
    if args.warm_start is None:
        args.warm_start = find_latest_a3_checkpoint()

    if not args.warm_start.is_file():
        raise FileNotFoundError(f"A3 checkpoint not found: {args.warm_start}")
    if not args.patterns.is_file():
        raise FileNotFoundError(f"Pattern bank not found: {args.patterns}")
    if not args.embeddings.is_file():
        raise FileNotFoundError(f"Embedding bank not found: {args.embeddings}")

    import jax
    import torch

    from residual import ResidualAdapter
    from residual.symbolic_head.jax_head import Retriever, SymbolicHead
    from runtime.data_loader import load_runtime_data, runtime_pairing_fingerprint
    from stages.results import RunWriter, default_run_name, load_params, run_metadata
    from stages.stage_A.a3 import A3Model
    from stages.stage_A.a4.checkpoint import restore_a3_weights
    from stages.stage_A.a4.config import (
        TRAINING_STATE_SOURCE,
        load_config,
        load_residual_config,
    )
    from stages.stage_A.a4.data import load_data
    from stages.stage_A.a4.model import A4Model
    from stages.stage_A.a4.training import train, train_sequential
    from stages.stage_base import configure_logging
    from symbolic_memory.bank import load_bank
    from symbolic_memory.jax_bridge import JaxRetriever
    from symbolic_memory.service import RetrievalClient
    from symbolic_memory.storage import MorkTemplateStore

    configure_logging(level=logging.INFO)
    log = logging.getLogger("stage_a4")
    log.info("A3 checkpoint: %s", args.warm_start)
    log.info("Pattern bank: %s", args.patterns)
    log.info("Embedding bank: %s", args.embeddings)

    config = load_config(args.config)
    config = _apply_training_overrides(config, args)
    jax.config.update("jax_default_matmul_precision", args.precision)
    torch.set_num_threads(2)
    torch.manual_seed(config.training.seed)
    bank = load_bank(args.patterns, args.embeddings)
    if len(bank.records) < config.minimum_templates:
        raise ValueError(f"Bank needs at least {config.minimum_templates} templates")
    inherited = load_residual_config(config.residual_config)
    if args.rank is not None and inherited.adapter.rank != args.rank:
        raise ValueError("Requested rank differs from the inherited A3 rank")
    warm_start = args.warm_start.resolve()
    recorded_path = warm_start.parent.parent / "config.yaml"
    if warm_start.parent.name in {"checkpoints", "params"} and recorded_path.is_file():
        recorded = load_residual_config(recorded_path)
        inherited_adapter = replace(inherited.adapter, seed=recorded.adapter.seed)
        if (
            recorded.adapter != inherited_adapter
            or recorded.objective != inherited.objective
            or recorded.training.model != config.training.model
        ):
            raise ValueError(
                "Inherited residual configuration differs from the A3 warm start"
            )
    pair = (
        json.loads(args.benchmark_pair.read_text(encoding="utf-8"))
        if args.benchmark_pair
        else None
    )
    checkpoint_sha256 = hashlib.sha256(warm_start.read_bytes()).hexdigest()
    residual_config_sha256 = hashlib.sha256(
        config.residual_config.read_bytes()
    ).hexdigest()
    if pair is not None:
        common = pair["common"]
        runtime_config_sha256 = hashlib.sha256(
            args.runtime_config.read_bytes()
        ).hexdigest()
        if (
            pair["checkpoint"] != str(warm_start)
            or pair["checkpoint_sha256"] != checkpoint_sha256
            or pair["residual_config_sha256"] != residual_config_sha256
            or common["runtime_config_sha256"] != runtime_config_sha256
            or common["training_mode"] != "sequential"
            or inherited.adapter.rank != common["rank"]
            or (
                args.data_mode == "runtime"
                and config.training.test_batches != common["eval_batches"]
            )
            or any(
                getattr(config.training, key) != common[key]
                for key in ("model", "seed", "seq_len", "batch_size", "eval_batches")
            )
        ):
            raise ValueError(
                "A4 shared configuration or checkpoint differs from its A3 pair"
            )
    substrate_reused = substrate is not None
    substrate = _resolve_substrate(
        config.training.model,
        substrate,
        revision=pair["frozen"]["revision"] if pair is not None else None,
    )
    adapter = ResidualAdapter(inherited.adapter, substrate.architecture)
    continuous = A3Model(
        substrate,
        layers=args.layers or adapter.layers,
        rank=inherited.adapter.rank,
        seq_len=config.training.seq_len,
        scale=adapter.scale,
        activation=inherited.adapter.activation,
        config=inherited.objective,
    )
    if pair is not None and (
        list(continuous.layers) != pair["layers"]
        or substrate.pairing_identity() != pair["frozen"]
    ):
        raise ValueError(
            "A4 residual layers, frozen model revision or tokenizer differ from its A3 pair"
        )
    frozen_residual = restore_a3_weights(continuous, load_params(warm_start))
    runtime_data = None
    if args.data_mode == "runtime" or pair is not None:
        runtime_data = load_runtime_data(
            args.runtime_config,
            substrate.tokenizer,
            sequence_length=config.training.seq_len,
            batch_size=config.training.batch_size,
            eval_batches=config.training.eval_batches,
            default_epochs=config.training.epochs,
            epochs_override=args.epochs,
            training_mode="sequential",
            source_texts=_aligned_source_texts(args)
            if args.data_mode == "aligned"
            else None,
        )
        if pair is not None and (
            list(runtime_data.task_order) != pair["task_order"]
            or runtime_pairing_fingerprint(
                runtime_data, runtime_config_sha256=runtime_config_sha256
            )
            != pair["runtime_data_sha256"]
        ):
            raise ValueError(
                "A4 runtime dataset, splits or task order differ from its A3 pair"
            )

    if args.data_mode == "runtime":
        assert runtime_data is not None
        data = runtime_data

        # Runtime-only training uses next-token CE.
        # The template bank remains active for retrieval.
        config = replace(
            config,
            key_weight=0.0,
            value_weight=0.0,
        )

        logging.getLogger("stage_a4").info(
            "Loaded %d runtime tasks: %s",
            len(data.tasks),
            ", ".join(data.task_order),
        )
    else:
        data = load_data(
            args,
            substrate.tokenizer,
            bank,
            config.training,
        )
    retriever: Retriever
    with ExitStack() as stack:
        if args.retrieval_backend == "native":
            from symbolic_memory.native_bridge import NativeJaxRetriever

            retriever = stack.enter_context(
                NativeJaxRetriever.from_mork(
                    bank,
                    server_url=args.mork_url,
                    library_path=args.native_library,
                    top_m=config.top_m,
                    max_queries=max(
                        1024, config.training.batch_size * config.training.seq_len
                    ),
                )
            )
            backend = retriever.info
        else:
            with MorkTemplateStore(args.mork_url) as store:
                if not store.check_bank(bank)["matches"]:
                    raise ValueError("MORK readback does not match the bank")
            client = stack.enter_context(
                RetrievalClient(
                    args.retrieval_host, args.retrieval_port, bank.input_snapshot_sha256
                )
            )
            retriever = JaxRetriever(
                client, [row.template_id for row in bank.records], top_m=config.top_m
            )
            backend = client.info
        model = A4Model(
            continuous,
            SymbolicHead(
                continuous.hidden_size, bank.vector_dim, continuous.layers, retriever
            ),
        )
        with RunWriter(
            config.training.output_dir, args.run_name or default_run_name("a4")
        ) as writer:
            writer.write_config(config.to_dict())
            writer.write_json(
                "data.json",
                data.metadata
                if args.data_mode == "runtime"
                else data.provenance,
            )
            writer.write_json(
                "meta.json",
                {
                    **run_metadata(),
                    "stage": "A4",
                    "training_state_source": TRAINING_STATE_SOURCE,
                    "working_directory": str(Path.cwd().resolve()),
                    "training_arguments": {
                        key: str(value.resolve()) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                    "source_sha256": {
                        name: hashlib.sha256(
                            (REPO_ROOT / name).read_bytes()
                        ).hexdigest()
                        for name in (
                            "experiments/run_stage_a4.py",
                            "stages/stage_A/a3/a3_model.py",
                            "stages/stage_A/a3/a3_graph.py",
                            "stages/stage_A/a3/a3_tail.py",
                            "stages/stage_A/a4/data.py",
                            "stages/stage_A/a4/model.py",
                            "stages/stage_A/a4/objective.py",
                            "stages/stage_A/a4/training.py",
                            "stages/stage_A/a4/config.py",
                            "stages/stage_A/a4/checkpoint.py",
                            "runtime/data_loader.py",
                            "residual/symbolic_head/alignment.py",
                            "frozenllm/substrate/loader.py",
                            "frozenllm/substrate/substrate.py",
                            "symbolic_memory/native_bridge.py",
                            "symbolic_memory/native/retrieval.cc",
                            "symbolic_memory/storage.py",
                            "residual/symbolic_head/retrieval/faiss_index.py",
                            "configs/tiers.yaml",
                        )
                    },
                    "model": config.training.model,
                    "seed": config.training.seed,
                    "rank": inherited.adapter.rank,
                    "data_mode": args.data_mode,
                    "runtime_data": data.metadata if args.data_mode == "runtime" else None,
                    "substrate_reused": substrate_reused,
                    "layers": list(continuous.layers),
                    "bank_snapshot": bank.input_snapshot_sha256,
                    "bank_count": len(bank.records),
                    "retrieval_backend": args.retrieval_backend,
                    "retrieval": backend,
                    "warm_start": str(args.warm_start),
                    "warm_start_sha256": checkpoint_sha256,
                    "residual_config_sha256": residual_config_sha256,
                    **({"a3_pairing": pair} if pair is not None else {}),
                    "native_library_sha256": (
                        hashlib.sha256(args.native_library.read_bytes()).hexdigest()
                        if args.retrieval_backend == "native"
                        else None
                    ),
                },
            )
            writer.write_json("status.json", {"stage": "A4", "status": "running"})
            logger = logging.getLogger("stage_a4")
            training_log = logging.FileHandler(
                writer.dir / "training.log", encoding="utf-8"
            )
            training_log.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(training_log)
            try:
                if args.data_mode == "runtime":
                    results = train_sequential(
                        model,
                        frozen_residual,
                        data,
                        bank,
                        config,
                        writer,
                        max_grad_norm=args.max_grad_norm,
                    )
                else:
                    results = train(
                        model,
                        frozen_residual,
                        data,
                        bank,
                        config,
                        writer,
                        max_grad_norm=args.max_grad_norm,
                        latency_every=(
                            0 if args.skip_latency else args.latency_every
                        ),
                    )
                if pair is not None:
                    writer.write_json("results.json", {**results, "a3_pairing": pair})
                writer.write_json("status.json", {"stage": "A4", "status": "completed"})
            except BaseException as error:
                writer.write_json(
                    "status.json",
                    {"stage": "A4", "status": "failed", "error": str(error)},
                )
                raise
            finally:
                logger.removeHandler(training_log)
                training_log.close()
                jax.effects_barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
