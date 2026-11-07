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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = REPO_ROOT / "configs/stage_A/a4_symbolic_head.yaml"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    for name in (
        "warm-start",
        "train-sources",
        "validation-sources",
        "alignment-labels",
        "patterns",
        "embeddings",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--test-sources", type=Path)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--eval-batches", type=int)
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
    if not args.skip_test and args.test_sources is None:
        parser.error("--test-sources is required unless --skip-test is selected")
    for name in (
        "batch_size",
        "seq_len",
        "eval_batches",
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


def _resolve_substrate(
    model_id: str, substrate: FrozenSubstrate | None
) -> FrozenSubstrate:
    if substrate is not None:
        return substrate
    import torch

    from frozenllm.substrate import FrozenSubstrate

    return FrozenSubstrate(model_id, dtype=torch.float32)


def main(
    argv: Sequence[str] | None = None, *, substrate: FrozenSubstrate | None = None
) -> int:
    args = parse_args(argv)
    import jax
    import torch

    from residual import ResidualAdapter
    from residual.symbolic_head.jax_head import SymbolicHead
    from stages.results import RunWriter, default_run_name, load_params, run_metadata
    from stages.stage_A.A3_model import A3Config, A3Model
    from stages.stage_A.a4.checkpoint import restore_a3_weights
    from stages.stage_A.a4.config import TRAINING_STATE_SOURCE, load_config
    from stages.stage_A.a4.data import load_data
    from stages.stage_A.a4.model import A4Model
    from stages.stage_A.a4.training import train
    from stages.stage_base import configure_logging, load_configs
    from symbolic_memory.bank import load_bank
    from symbolic_memory.jax_bridge import JaxRetriever
    from symbolic_memory.service import RetrievalClient
    from symbolic_memory.storage import MorkTemplateStore

    configure_logging(level=logging.INFO)
    config = load_config(args.config)
    overrides = {
        name: getattr(args, name)
        for name in (
            "model",
            "epochs",
            "batch_size",
            "seq_len",
            "learning_rate",
            "eval_batches",
            "output_dir",
        )
        if getattr(args, name) is not None
    }
    config = replace(
        config,
        training=replace(
            config.training, **overrides, jit=config.training.jit and not args.no_jit
        ),
    )
    jax.config.update("jax_default_matmul_precision", args.precision)
    torch.set_num_threads(2)
    torch.manual_seed(config.training.seed)
    bank = load_bank(args.patterns, args.embeddings)
    if len(bank.records) < config.minimum_templates:
        raise ValueError(f"Bank needs at least {config.minimum_templates} templates")
    inherited = load_configs(config.residual_config, A3Config)
    substrate_reused = substrate is not None
    substrate = _resolve_substrate(config.training.model, substrate)
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
    frozen_residual = restore_a3_weights(continuous, load_params(args.warm_start))
    data = load_data(args, substrate.tokenizer, bank, config.training)
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
            writer.write_json("data.json", data.provenance)
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
                            "stages/stage_A/A3_model.py",
                            "stages/stage_A/a4/data.py",
                            "stages/stage_A/a4/model.py",
                            "stages/stage_A/a4/objective.py",
                            "stages/stage_A/a4/training.py",
                            "stages/stage_A/a4/config.py",
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
                    "substrate_reused": substrate_reused,
                    "layers": list(continuous.layers),
                    "bank_snapshot": bank.input_snapshot_sha256,
                    "bank_count": len(bank.records),
                    "retrieval_backend": args.retrieval_backend,
                    "retrieval": backend,
                    "warm_start": str(args.warm_start),
                    "warm_start_sha256": hashlib.sha256(
                        args.warm_start.read_bytes()
                    ).hexdigest(),
                    "residual_config_sha256": hashlib.sha256(
                        config.residual_config.read_bytes()
                    ).hexdigest(),
                    "native_library_sha256": hashlib.sha256(
                        args.native_library.read_bytes()
                    ).hexdigest()
                    if args.retrieval_backend == "native"
                    else None,
                },
            )
            writer.write_json("status.json", {"stage": "A4", "status": "running"})
            try:
                results = train(
                    model,
                    frozen_residual,
                    data,
                    bank,
                    config,
                    writer,
                    max_grad_norm=args.max_grad_norm,
                )
                writer.write_json("status.json", {"stage": "A4", "status": "completed"})
            except BaseException as error:
                writer.write_json(
                    "status.json",
                    {"stage": "A4", "status": "failed", "error": str(error)},
                )
                raise
            finally:
                jax.effects_barrier()
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
