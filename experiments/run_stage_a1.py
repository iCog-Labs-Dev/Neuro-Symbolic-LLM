"""Run Stage A1: standard residual adapter training."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import jax

os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import optax  # noqa: E402

from frozenllm.substrate import FrozenSubstrate  # noqa: E402
from residual import ResidualAdapter  # noqa: E402
from runtime.data_loader import load_runtime_data  # noqa: E402
from stages.results import (  # noqa: E402
    RunWriter,
    default_run_name,
    run_metadata,
)
from stages.stage_A.a1 import (  # noqa: E402
    A1Config,
    a1_forward,
    a1_loss,
)
from stages.stage_base import (  # noqa: E402
    RunConfig,
    apply_run_overrides,
    configure_logging,
    evaluate,
    initialize_stage,
    load_configs,
    log_experiment_header,
    log_final_summary,
)
from training.train import train  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "stage_A" / "a1_lora_baseline.yaml"
DEFAULT_RUNTIME_CONFIG = REPO_ROOT / "configs" / "runtime_datasets.yaml"

log = logging.getLogger("stage_a1")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--epochs", "--epoch", dest="epochs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--eval-checkpoint",
        type=Path,
        default=None,
        help="Evaluate a saved checkpoint.",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--no-jit", action="store_true", default=False)

    return parser.parse_args(argv)


def apply_overrides(cfg: RunConfig, args: argparse.Namespace) -> RunConfig:
    return apply_run_overrides(cfg, args)


def main(argv: Sequence[str] | None = None) -> int:
    start_time = time.time()

    configure_logging(level=logging.INFO)

    args = parse_args(argv)

    cfg = apply_overrides(
        load_configs(args.config, A1Config),
        args,
    )

    train_cfg = cfg.training

    substrate = FrozenSubstrate(train_cfg.model)
    arch = substrate.architecture

    adapter = initialize_stage(substrate, cfg)
    params = adapter.init_params()

    n_trainable = ResidualAdapter.num_params(params)
    n_frozen = sum(p.numel() for p in substrate.params.values())

    eval_batches = (
        args.eval_batches if args.eval_batches is not None else train_cfg.eval_batches
    )

    # Prepare shared runtime datasets.
    runtime_data = load_runtime_data(
        config_path=DEFAULT_RUNTIME_CONFIG,
        tokenizer=substrate.tokenizer,
        sequence_length=train_cfg.seq_len,
        batch_size=train_cfg.batch_size,
        eval_batches=eval_batches,
        default_epochs=train_cfg.epochs,
    )

    # Checkpoint evaluation.
    if args.eval_checkpoint is not None:
        results = {}

        test_sets = runtime_data.get_test_sets(
            train_cfg.batch_size,
            eval_batches,
        )

        if args.split == "test":
            selected_sets = test_sets
        elif args.split in ("eval", "validation"):
            selected_sets = {task.name: task.eval_set for task in runtime_data.tasks}
        else:
            raise ValueError(
                "Runtime checkpoint evaluation supports 'test' and 'validation' splits."
            )

        for name, batches in selected_sets.items():
            results[name] = evaluate(
                substrate,
                adapter,
                params,
                batches,
                jit=train_cfg.jit,
            )

        print(json.dumps(results, indent=2))
        return 0

    domains = runtime_data.tasks

    run_name = args.run_name or default_run_name("a1")

    optimizer = optax.adam(train_cfg.learning_rate)

    with RunWriter(train_cfg.output_dir, run_name) as writer:
        log.info("run_dir=%s", writer.dir)

        writer.write_config(cfg.to_dict())

        writer.write_json(
            "meta.json",
            {
                **run_metadata(),
                "stage": "A1",
                "model": train_cfg.model,
                "num_layers": arch.num_layers,
                "hidden_size": arch.hidden_size,
                "adapted_layers": list(adapter.layers),
                "trainable_params": n_trainable,
                "frozen_params": n_frozen,
                "epochs": {domain.name: domain.epochs for domain in domains},
                "jit": train_cfg.jit,
                "runtime_data": runtime_data.metadata,
            },
        )

        log_experiment_header(
            log,
            cfg=cfg,
            arch=substrate.architecture,
            adapter=adapter,
            n_trainable=sum(p.size for p in jax.tree_util.tree_leaves(params)),
            n_frozen=n_frozen,
            run_dir=writer.dir,
            domains=domains,
            stage="A1",
        )

        # Shared training engine with A1-specific forward and loss functions.
        params, summary = train(
            params=params,
            tasks=domains,
            forward_fn=partial(
                a1_forward,
                substrate=substrate,
                adapter=adapter,
                config=cfg.objective,
            ),
            loss_fn=partial(
                a1_loss,
                substrate=substrate,
            ),
            loss_config=cfg.objective,
            optimizer=optimizer,
            config=train_cfg,
            writer=writer,
            eval_fn=partial(
                evaluate,
                substrate,
                adapter,
                jit=train_cfg.jit,
            ),
        )

        frozen = substrate.verify_frozen()

        summary["params_unchanged"] = frozen["params_unchanged"]
        summary["runtime_data"] = runtime_data.metadata

        # Final test evaluation.
        test_results = {}

        for name, test_set in runtime_data.get_test_sets(
            train_cfg.batch_size,
            eval_batches,
        ).items():
            test_results[name] = evaluate(
                substrate,
                adapter,
                params,
                test_set,
                jit=train_cfg.jit,
            )

            writer.log(
                "test",
                domain=name,
                **test_results[name],
            )

        summary["test"] = test_results

        writer.write_json("results.json", summary)

    log_final_summary(
        log,
        summary,
        frozen,
        start_time=start_time,
    )

    if not frozen["params_unchanged"]:
        log.error("Frozen substrate parameters changed during training.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
