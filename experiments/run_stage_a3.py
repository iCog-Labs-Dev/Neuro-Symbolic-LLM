"""Run Stage A3: hybrid predictive-coding residual training."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from frozenllm.substrate import FrozenSubstrate  # noqa: E402
from residual import ResidualAdapter  # noqa: E402
from runtime.data_loader import load_runtime_data  # noqa: E402
from stages.results import (  # noqa: E402
    RunWriter,
    default_run_name,
    run_metadata,
)
from stages.stage_A.a3 import A3HybridConfig, A3Model  # noqa: E402
from stages.stage_base import (  # noqa: E402
    RunConfig,
    apply_run_overrides,
    configure_logging,
    format_params,
    load_configs,
    log_experiment_header,
    log_final_summary,
    make_eval_set,
)
from training.train import train  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "stage_A" / "a3_hybrid_pc.yaml"
DEFAULT_RUNTIME_CONFIG = REPO_ROOT / "configs" / "runtime_datasets.yaml"

log = logging.getLogger("stage_a3")


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--rank", type=int, default=None)

    parser.add_argument(
        "--K",
        type=int,
        nargs="+",
        default=None,
        help="Run separate experiments for each PC inference-step count.",
    )

    parser.add_argument("--eta-z", type=float, default=None)
    parser.add_argument("--lambda-kl", type=float, default=None)

    parser.add_argument(
        "--epochs",
        "--epoch",
        dest="epochs",
        type=int,
        default=None,
    )

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument(
        "--energy-curve",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--no-jit",
        action="store_true",
        default=False,
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------
# Configuration overrides
# ---------------------------------------------------------------------


def apply_overrides(
    cfg: RunConfig,
    args: argparse.Namespace,
) -> RunConfig:
    """Apply shared training and A3-specific CLI overrides."""

    cfg = apply_run_overrides(cfg, args)

    objective: A3HybridConfig = cfg.objective

    if args.eta_z is not None:
        objective = replace(
            objective,
            eta_z=args.eta_z,
        )

    if args.lambda_kl is not None:
        objective = replace(
            objective,
            lambda_kl=args.lambda_kl,
        )

    return replace(
        cfg,
        objective=objective,
    )


# ---------------------------------------------------------------------
# A3 evaluation
# ---------------------------------------------------------------------


def evaluate_deployment(
    substrate: FrozenSubstrate,
    model: A3Model,
    params,
    eval_batches,
    *,
    jit: bool = True,
) -> dict[str, float]:
    """Evaluate the deployed A3 model using its adapter."""

    if not eval_batches:
        return {
            "base_loss": 0.0,
            "base_ppl": 1.0,
            "loss": 0.0,
            "ppl": 1.0,
            "kl": 0.0,
        }

    def evaluate_step(p, ids):
        adapted, base = model.deploy_logits(p, ids)

        base_loss = substrate.compute_loss(
            base,
            ids,
            losses=("CE",),
        )

        loss = substrate.compute_loss(
            adapted,
            ids,
            losses=("CE",),
        )

        kl = substrate.compute_loss(
            adapted,
            ids,
            losses=("KL",),
            base_logits=base,
        )

        return base_loss, loss, kl

    step = jax.jit(evaluate_step) if jit else evaluate_step

    rows = [tuple(float(v) for v in step(params, ids)) for ids in eval_batches]

    base_loss, loss, kl = (
        float(np.mean(column)) for column in zip(*rows, strict=False)
    )

    return {
        "base_loss": base_loss,
        "base_ppl": float(np.exp(base_loss)),
        "loss": loss,
        "ppl": float(np.exp(loss)),
        "kl": kl,
    }


# ---------------------------------------------------------------------
# A3 PC gradient adapter
# ---------------------------------------------------------------------


def make_a3_gradient_fn(model: A3Model):
    """Adapt A3's custom PC gradients to the shared training interface."""

    def a3_gradient_fn(params, batch, key):
        grads, aux = model.train_grads(
            params,
            batch,
            key,
        )

        metrics = {
            "total": aux["energy_after_exact"],
            "task": aux["task"],
            "pc_energy": aux["spring"],
        }

        if "kl" in aux:
            metrics["kl"] = aux["kl"]

        return grads, metrics

    return a3_gradient_fn


# ---------------------------------------------------------------------
# Run one A3 experiment
# ---------------------------------------------------------------------


def run_one(
    cfg: RunConfig,
    args: argparse.Namespace,
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    runtime_data,
    test_sets,
    n_frozen: int,
    run_name: str,
    start_time: float,
) -> int:
    train_cfg = cfg.training
    objective: A3HybridConfig = cfg.objective

    # -------------------------------------------------------------
    # Build A3 model
    # -------------------------------------------------------------

    model = A3Model(
        substrate,
        layers=adapter.layers,
        rank=cfg.adapter.rank,
        seq_len=train_cfg.seq_len,
        scale=adapter.scale,
        activation=cfg.adapter.activation,
        config=objective,
    )

    params = model.init_params(jax.random.PRNGKey(train_cfg.seed))

    n_trainable = sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(params))

    domains = runtime_data.tasks

    # -------------------------------------------------------------
    # Evaluation callback
    # -------------------------------------------------------------

    def eval_fn(p, eval_set):
        return evaluate_deployment(
            substrate,
            model,
            p,
            eval_set,
            jit=train_cfg.jit,
        )

    # -------------------------------------------------------------
    # A3-specific PC energy curve callback
    # -------------------------------------------------------------

    base_key = jax.random.PRNGKey(train_cfg.seed + 100)

    def after_task_fn(p, task_idx, task):
        if not args.energy_curve or not task.eval_set:
            return

        curve = model.energy_curve(
            p,
            task.eval_set[0],
            jax.random.fold_in(base_key, 0),
        )

        writer.log(
            "energy_curve",
            after_task=task_idx,
            domain=task.name,
            curve=curve,
        )

    # -------------------------------------------------------------
    # Run artifacts
    # -------------------------------------------------------------

    with RunWriter(
        train_cfg.output_dir,
        run_name,
    ) as writer:
        writer.write_config(cfg.to_dict())

        writer.write_json(
            "meta.json",
            {
                **run_metadata(),
                "stage": "A3",
                "model": train_cfg.model,
                "num_layers": substrate.architecture.num_layers,
                "hidden_size": substrate.architecture.hidden_size,
                "adapted_layers": list(adapter.layers),
                "trainable_params": n_trainable,
                "frozen_params": n_frozen,
                "infer_steps": objective.infer_steps,
                "eta_z": objective.eta_z,
                "lambda_kl": objective.lambda_kl,
                "epochs": {domain.name: domain.epochs for domain in domains},
                "jit": train_cfg.jit,
                "runtime_data": runtime_data.metadata,
            },
        )

        # ---------------------------------------------------------
        # Experiment header
        # ---------------------------------------------------------

        log_experiment_header(
            log,
            cfg=cfg,
            arch=substrate.architecture,
            adapter=adapter,
            n_trainable=n_trainable,
            n_frozen=n_frozen,
            run_dir=writer.dir,
            domains=domains,
            stage="A3",
        )

        log.info(
            "A3 parameters      : Total=%s",
            format_params(n_trainable),
        )

        # ---------------------------------------------------------
        # Shared training engine with A3-specific PC gradients
        # ---------------------------------------------------------

        optimizer = optax.adam(train_cfg.learning_rate)

        params, summary = train(
            params=params,
            tasks=domains,
            forward_fn=None,
            loss_fn=None,
            loss_config=None,
            optimizer=optimizer,
            config=train_cfg,
            writer=writer,
            eval_fn=eval_fn,
            pc_gradient_fn=make_a3_gradient_fn(model),
            checkpoint_fn=model.adapter_params,
            after_task_fn=after_task_fn,
        )

        # ---------------------------------------------------------
        # Final A3 adapter checkpoint
        # ---------------------------------------------------------

        writer.save_checkpoint(
            "a3hybrid_final",
            model.adapter_params(params),
            global_step=summary["steps"],
            stage="A3",
            model=train_cfg.model,
            adapted_layers=list(adapter.layers),
            rank=cfg.adapter.rank,
            scale=adapter.scale,
            infer_steps=objective.infer_steps,
            final=True,
        )

        # ---------------------------------------------------------
        # Substrate integrity
        # ---------------------------------------------------------

        frozen = substrate.verify_frozen()

        summary["params_unchanged"] = frozen["params_unchanged"]
        summary["infer_steps"] = objective.infer_steps
        summary["runtime_data"] = runtime_data.metadata

        # ---------------------------------------------------------
        # Final test evaluation
        # ---------------------------------------------------------

        test_results = {}

        for name, test_set in test_sets.items():
            result = eval_fn(
                params,
                test_set,
            )

            test_results[name] = result

            writer.log(
                "test",
                domain=name,
                **result,
            )

        summary["test"] = test_results

        writer.write_json(
            "results.json",
            summary,
        )

    # -------------------------------------------------------------
    # Final experiment summary
    # -------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging(level=logging.INFO)

    args = parse_args(argv)
    start_time = time.time()

    cfg = apply_overrides(
        load_configs(
            args.config,
            A3HybridConfig,
            objective_section="a3_hybrid",
        ),
        args,
    )

    train_cfg = cfg.training

    # -------------------------------------------------------------
    # Frozen substrate and adapter
    # -------------------------------------------------------------

    substrate = FrozenSubstrate(train_cfg.model)

    arch = substrate.architecture

    adapter = ResidualAdapter(
        cfg.adapter,
        arch,
    )

    n_frozen = sum(parameter.numel() for parameter in substrate.params.values())

    frozen = substrate.verify_frozen()

    if not frozen["params_unchanged"]:
        raise RuntimeError(f"Frozen substrate verification failed: {frozen}")

    # -------------------------------------------------------------
    # Shared runtime datasets
    # -------------------------------------------------------------

    eval_batches = train_cfg.eval_batches

    runtime_data = load_runtime_data(
        config_path=DEFAULT_RUNTIME_CONFIG,
        tokenizer=substrate.tokenizer,
        sequence_length=train_cfg.seq_len,
        batch_size=train_cfg.batch_size,
        eval_batches=eval_batches,
        default_epochs=train_cfg.epochs,
    )

    domains = runtime_data.tasks

    # Select the final evaluation split.
    if args.split == "test":
        test_sets = runtime_data.get_test_sets(
            train_cfg.batch_size,
            eval_batches,
        )

    elif args.split in ("eval", "validation"):
        test_sets = {task.name: task.eval_set for task in domains}

    elif args.split == "train":
        test_sets = {
            task.name: make_eval_set(
                task.train_blocks,
                train_cfg.batch_size,
                eval_batches,
            )
            for task in domains
        }

    else:
        raise ValueError("split must be 'test', 'validation', 'eval', or 'train'.")

    # -------------------------------------------------------------
    # Multiple inference-step experiments
    # -------------------------------------------------------------

    base_name = args.run_name or default_run_name("a3hybrid")

    inference_steps = args.K if args.K is not None else [cfg.objective.infer_steps]

    if any(k < 0 for k in inference_steps):
        raise ValueError("Every K value must be >= 0.")

    if len(set(inference_steps)) != len(inference_steps):
        raise ValueError("K values must be unique.")

    status = 0

    for k in inference_steps:
        run_cfg = replace(
            cfg,
            objective=replace(
                cfg.objective,
                infer_steps=k,
            ),
        )

        run_name = base_name if len(inference_steps) == 1 else f"{base_name}-K{k}"

        status |= run_one(
            run_cfg,
            args,
            substrate,
            adapter,
            runtime_data,
            test_sets,
            n_frozen,
            run_name,
            start_time,
        )

    return status


if __name__ == "__main__":
    sys.exit(main())
