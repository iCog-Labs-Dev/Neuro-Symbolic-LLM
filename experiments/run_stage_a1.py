"""Run Stage A1: train a hidden-state residual adapter on a frozen LM.

Wiring only: builds the configs, substrate and adapter, then runs the shared
loop in ``stages/stage_base.py`` and checks the base is unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

# Silence verbose C++ runtime warnings (e.g. XLA slow_operation_alarm) before JAX initializes
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
# Run as a script: put the repo root on sys.path.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frozenllm.substrate import FrozenSubstrate  # noqa: E402
from residual import ResidualAdapter  # noqa: E402
from stages.results import RunWriter, default_run_name, run_metadata  # noqa: E402
from stages.stage_A.a1 import A1Config, a1_objective  # noqa: E402
from stages.stage_base import (  # noqa: E402
    Domain,
    RunConfig,
    configure_logging,
    evaluate,
    evaluate_checkpoint,
    load_configs,
    load_token_blocks,
    log_experiment_header,
    log_final_summary,
    make_eval_set,
    run_sequential,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "stage_A" / "a1_lora_baseline.yaml"

log = logging.getLogger("stage_a1")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--rank", type=int, default=None, help="Residual adapter rank override."
    )
    parser.add_argument(
        "--epochs",
        "--epoch",
        dest="epochs",
        type=int,
        default=None,
        help="Epochs for every domain (smoke runs).",
    )
    parser.add_argument("--steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--eval-checkpoint",
        type=Path,
        default=None,
        help="Evaluate a saved checkpoint instead of training.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split to evaluate when using --eval-checkpoint (default: test).",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=None,
        help="Override eval_batches setting.",
    )
    parser.add_argument(
        "--no-jit",
        action="store_true",
        default=False,
        help="Disable JAX JIT compilation for training and evaluation.",
    )
    return parser.parse_args(argv)


def apply_overrides(cfg: RunConfig, args: argparse.Namespace) -> RunConfig:
    adapter, training, domains = cfg.adapter, cfg.training, cfg.domains
    if args.rank is not None:
        adapter = replace(adapter, rank=args.rank)
    if args.model is not None:
        training = replace(training, model=args.model)
    if args.output_dir is not None:
        training = replace(training, output_dir=args.output_dir)
    if getattr(args, "no_jit", False):
        training = replace(training, jit=False)
    if args.epochs is not None and args.steps is not None:
        raise ValueError("Cannot specify both --epochs and --steps. Use --epochs.")
    if args.steps is not None:
        raise ValueError("--steps has been replaced by --epochs. Use --epochs.")
    if args.epochs is not None:
        if args.epochs < 0:
            raise ValueError(f"epochs must be >= 0, got {args.epochs}")
        training = replace(training, epochs=args.epochs)
        domains = tuple(replace(d, epochs=args.epochs) for d in domains)
    if getattr(args, "eval_batches", None) is not None:
        if args.eval_batches < 1:
            raise ValueError(f"eval_batches must be >= 1, got {args.eval_batches}")
        training = replace(training, eval_batches=args.eval_batches)
    return replace(cfg, adapter=adapter, training=training, domains=domains)


def main(argv: Sequence[str] | None = None) -> int:
    start_time = time.time()
    configure_logging(level=logging.INFO)
    args = parse_args(argv)
    cfg = apply_overrides(load_configs(args.config, A1Config), args)
    train_cfg = cfg.training

    substrate = FrozenSubstrate(train_cfg.model)
    arch = substrate.architecture
    max_pos = arch.max_position_embeddings
    if max_pos is not None and train_cfg.seq_len > max_pos:
        raise ValueError(f"seq_len={train_cfg.seq_len} exceeds {max_pos} positions.")

    adapter = ResidualAdapter(cfg.adapter, arch)
    params = adapter.init_params()
    n_trainable = ResidualAdapter.num_params(params)
    n_frozen = sum(p.numel() for p in substrate.params.values())

    frozen = substrate.verify_frozen()
    if not frozen["params_unchanged"]:
        log.error("Substrate not frozen before training: %s", frozen)
        return 1

    if args.eval_checkpoint is not None:
        log.info(
            "Evaluating checkpoint %s on split=%s",
            args.eval_checkpoint,
            args.split,
        )
        eval_batches = (
            args.eval_batches
            if args.eval_batches is not None
            else train_cfg.eval_batches
        )
        results = evaluate_checkpoint(
            substrate,
            adapter,
            args.eval_checkpoint,
            cfg.domains,
            split=args.split,
            seq_len=train_cfg.seq_len,
            batch_size=train_cfg.batch_size,
            eval_batches=eval_batches,
            jit=train_cfg.jit,
        )
        print(json.dumps(results, indent=2))
        return 0

    # Load data before creating the run directory, so bad datasets fail fast.
    log.info("Preparing datasets across %d domain(s)...", len(cfg.domains))
    domains = []
    for d in cfg.domains:
        train_blocks = load_token_blocks(substrate, d, d.train_split, train_cfg.seq_len)
        eval_blocks = load_token_blocks(substrate, d, d.eval_split, train_cfg.seq_len)
        domains.append(
            Domain(
                name=d.name,
                train_blocks=train_blocks,
                eval_set=make_eval_set(
                    eval_blocks, train_cfg.batch_size, train_cfg.eval_batches
                ),
                epochs=train_cfg.epochs if d.epochs is None else d.epochs,
            )
        )

    run_name = args.run_name or default_run_name("a1")
    with RunWriter(train_cfg.output_dir, run_name) as writer:
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
                "epochs": {d.name: d.epochs for d in domains},
                "jit": train_cfg.jit,
            },
        )
        log_experiment_header(
            log,
            cfg=cfg,
            arch=arch,
            adapter=adapter,
            n_trainable=n_trainable,
            n_frozen=n_frozen,
            run_dir=writer.dir,
            domains=domains,
        )
        params, summary = run_sequential(
            substrate,
            adapter,
            params,
            domains,
            a1_objective,
            cfg.objective,
            train_cfg,
            writer,
        )
        frozen = substrate.verify_frozen()
        summary["params_unchanged"] = frozen["params_unchanged"]

        # Final test-set evaluation across domains for paper-table results
        test_eval_batches = (
            args.eval_batches
            if args.eval_batches is not None
            else train_cfg.eval_batches
        )
        log.info("Running final test evaluation across %d domains...", len(cfg.domains))
        test_results = {}
        for d in cfg.domains:
            test_blocks = load_token_blocks(
                substrate, d, d.test_split, train_cfg.seq_len
            )
            test_set = make_eval_set(
                test_blocks, train_cfg.batch_size, test_eval_batches
            )
            test_results[d.name] = evaluate(
                substrate, adapter, params, test_set, jit=train_cfg.jit
            )
            writer.log("test", domain=d.name, **test_results[d.name])
        summary["test"] = test_results
        writer.write_json("results.json", summary)

    log_final_summary(log, summary, frozen, start_time=start_time)
    if not frozen["params_unchanged"]:
        log.error("Base parameters changed during training.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
