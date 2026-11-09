"""Run Stage A3: Hybrid predictive-coding residual training."""

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
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from frozenllm.substrate import FrozenSubstrate  # noqa: E402
from metrics.performance import training_step_time  # noqa: E402
from residual import ResidualAdapter  # noqa: E402
from stages.results import RunWriter, default_run_name, run_metadata  # noqa: E402
from stages.stage_A.a3 import A3HybridConfig, A3Model  # noqa: E402
from stages.stage_base import (  # noqa: E402
    Domain,
    RunConfig,
    TrainingConfig,
    batches,
    configure_logging,
    load_configs,
    load_token_blocks,
    make_eval_set,
    summarize,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "stage_A" / "a3_hybrid_pc.yaml"

log = logging.getLogger("stage_a3")


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
        help="Run separate experiments for each inference-step count.",
    )
    parser.add_argument("--eta-z", type=float, default=None)
    parser.add_argument("--lambda-kl", type=float, default=None)
    parser.add_argument("--epochs", "--epoch", dest="epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--energy-curve", action="store_true", default=False)
    parser.add_argument("--no-jit", action="store_true", default=False)

    return parser.parse_args(argv)


def apply_overrides(cfg: RunConfig, args: argparse.Namespace) -> RunConfig:
    adapter = cfg.adapter
    objective = cfg.objective
    training = cfg.training
    domains = cfg.domains

    if args.rank is not None:
        adapter = replace(adapter, rank=args.rank)

    if args.eta_z is not None:
        objective = replace(objective, eta_z=args.eta_z)

    if args.lambda_kl is not None:
        objective = replace(objective, lambda_kl=args.lambda_kl)

    if args.model is not None:
        training = replace(training, model=args.model)

    if args.output_dir is not None:
        training = replace(training, output_dir=args.output_dir)

    if args.no_jit:
        training = replace(training, jit=False)

    if args.epochs is not None:
        if args.epochs < 0:
            raise ValueError(f"epochs must be >= 0, got {args.epochs}")

        training = replace(training, epochs=args.epochs)
        domains = tuple(replace(domain, epochs=args.epochs) for domain in domains)

    if args.eval_batches is not None:
        if args.eval_batches < 1:
            raise ValueError(f"eval_batches must be >= 1, got {args.eval_batches}")

        training = replace(
            training,
            eval_batches=args.eval_batches,
        )

    return replace(
        cfg,
        adapter=adapter,
        objective=objective,
        training=training,
        domains=domains,
    )


def evaluate_deployment(
    substrate,
    model: A3Model,
    params,
    eval_batches,
    *,
    jit: bool = True,
):
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


def train_sequential(
    substrate,
    model: A3Model,
    params,
    domains,
    training: TrainingConfig,
    writer: RunWriter,
    *,
    energy_curve: bool = False,
):
    names = [domain.name for domain in domains]

    def evaluate_all(after_task: int):
        results = {}

        for domain in domains:
            results[domain.name] = evaluate_deployment(
                substrate,
                model,
                params,
                domain.eval_set,
                jit=training.jit,
            )

            writer.log(
                "eval",
                after_task=after_task,
                domain=domain.name,
                **results[domain.name],
            )

        return results

    initial_results = evaluate_all(-1)

    optimizer = optax.adam(training.learning_rate)
    base_key = jax.random.PRNGKey(training.seed + 100)

    def train_step(params, opt_state, ids, key):
        grads, aux = model.train_grads(params, ids, key)

        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            params,
        )

        metrics = {
            "total_loss": aux["loss"] + aux["spring"],
            "pc_energy": aux["spring"],
            "ppl": jnp.exp(aux["task"]),
        }

        params = optax.apply_updates(params, updates)

        return params, opt_state, metrics

    if training.jit:
        train_step = jax.jit(train_step)

    opt_state = optimizer.init(params)

    rows = []
    step_times = []

    global_step = 0
    tokens = 0

    for task_idx, domain in enumerate(domains):
        if task_idx > 0 and training.reset_optimizer:
            opt_state = optimizer.init(params)

        num_epochs = domain.epochs

        if num_epochs > 0 and len(domain.train_blocks) < training.batch_size:
            raise ValueError(
                f"Domain {domain.name!r} has insufficient training blocks "
                f"for batch_size={training.batch_size}."
            )

        batches_per_epoch = (
            len(domain.train_blocks) // training.batch_size if num_epochs > 0 else 0
        )

        log.info("\nTask %d: %s", task_idx + 1, domain.name)

        domain_step = 0

        for epoch in range(1, num_epochs + 1):
            log.info("  Epoch %d/%d", epoch, num_epochs)

            epoch_start = time.time()

            epoch_step_times = []
            epoch_metrics = []
            epoch_tokens = 0

            stream = batches(
                domain.train_blocks,
                training.batch_size,
                training.seed + task_idx * 10000 + (epoch - 1),
            )

            for batch_idx, ids in enumerate(stream, start=1):
                domain_step += 1
                global_step += 1

                batch_tokens = int(ids.size)
                tokens += batch_tokens
                epoch_tokens += batch_tokens

                key = jax.random.fold_in(
                    base_key,
                    global_step,
                )

                (params, opt_state, metrics), step_time = training_step_time(
                    train_step,
                    params,
                    opt_state,
                    ids,
                    key,
                )

                step_times.append(step_time)
                epoch_step_times.append(step_time)

                scalars = {name: float(value) for name, value in metrics.items()}

                epoch_metrics.append(scalars)

                writer.log(
                    "train",
                    task=task_idx,
                    domain=domain.name,
                    epoch=epoch,
                    batch=batch_idx,
                    step=domain_step,
                    global_step=global_step,
                    tokens=tokens,
                    step_time=step_time,
                    **scalars,
                )

                if (
                    batch_idx == 1
                    or batch_idx == batches_per_epoch
                    or domain_step % training.log_every == 0
                ):
                    log.info(
                        "    Batch %d | Total loss: %.4f | "
                        "PC energy: %.6f | PPL: %.2f",
                        batch_idx,
                        scalars["total_loss"],
                        scalars["pc_energy"],
                        scalars["ppl"],
                    )

            if epoch_metrics:
                duration = time.time() - epoch_start

                mean_metrics = {
                    name: float(np.mean([metrics[name] for metrics in epoch_metrics]))
                    for name in epoch_metrics[0]
                }

                mean_step_time = float(np.mean(epoch_step_times))

                writer.log(
                    "epoch",
                    task=task_idx,
                    domain=domain.name,
                    epoch=epoch,
                    total_epochs=num_epochs,
                    global_step=global_step,
                    batches=len(epoch_metrics),
                    blocks=len(epoch_metrics) * training.batch_size,
                    epoch_tokens=epoch_tokens,
                    tokens=tokens,
                    epoch_time=duration,
                    mean_step_time=mean_step_time,
                    **mean_metrics,
                )

                log.info(
                    "    Epoch average | Total loss: %.4f | "
                    "PC energy: %.4f | PPL: %.2f",
                    mean_metrics["total_loss"],
                    mean_metrics["pc_energy"],
                    mean_metrics["ppl"],
                )

        writer.save_checkpoint(
            f"task_{task_idx}_{domain.name}",
            model.adapter_params(params),
            opt_state=opt_state,
            global_step=global_step,
            task=task_idx,
            domain=domain.name,
            epoch=num_epochs,
            step=domain_step,
        )

        rows.append(evaluate_all(task_idx))

        if energy_curve and domain.eval_set:
            curve = model.energy_curve(
                params,
                domain.eval_set[0],
                jax.random.fold_in(base_key, 0),
            )

            writer.log(
                "energy_curve",
                after_task=task_idx,
                domain=domain.name,
                curve=curve,
            )

    summary = summarize(names, initial_results, rows)

    summary.update(
        epochs={domain.name: domain.epochs for domain in domains},
        steps=global_step,
        tokens=tokens,
        mean_step_time=(
            float(np.mean(step_times[1:])) if len(step_times) > 1 else None
        ),
    )

    return params, summary


def run_one(
    cfg: RunConfig,
    args,
    substrate,
    adapter,
    domains,
    n_frozen: int,
    run_name: str,
    start_time: float,
) -> int:
    train_cfg = cfg.training
    objective = cfg.objective

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

    with RunWriter(train_cfg.output_dir, run_name) as writer:
        writer.write_config(cfg.to_dict())

        writer.write_json(
            "meta.json",
            {
                **run_metadata(),
                "stage": "A3",
                "model": train_cfg.model,
                "adapted_layers": list(adapter.layers),
                "trainable_params": n_trainable,
                "frozen_params": n_frozen,
                "infer_steps": objective.infer_steps,
                "epochs": {domain.name: domain.epochs for domain in domains},
                "jit": train_cfg.jit,
            },
        )

        params, summary = train_sequential(
            substrate,
            model,
            params,
            domains,
            train_cfg,
            writer,
            energy_curve=args.energy_curve,
        )
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

        log.info("Final A3 Hybrid adapter saved: a3hybrid_final")

        frozen = substrate.verify_frozen()

        summary["params_unchanged"] = frozen["params_unchanged"]
        summary["infer_steps"] = objective.infer_steps

        test_results = {}

        for domain in cfg.domains:
            split_name = {
                "test": domain.test_split,
                "train": domain.train_split,
            }.get(args.split, domain.eval_split)

            blocks = load_token_blocks(
                substrate,
                domain,
                split_name,
                train_cfg.seq_len,
            )

            test_set = make_eval_set(
                blocks,
                train_cfg.batch_size,
                train_cfg.eval_batches,
            )

            test_results[domain.name] = evaluate_deployment(
                substrate,
                model,
                params,
                test_set,
                jit=train_cfg.jit,
            )

            writer.log(
                "test",
                domain=domain.name,
                **test_results[domain.name],
            )

        summary["test"] = test_results

        writer.write_json("results.json", summary)

    if not frozen["params_unchanged"]:
        log.error("Frozen substrate parameters changed during training.")
        return 1

    return 0


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

    substrate = FrozenSubstrate(train_cfg.model)
    arch = substrate.architecture

    max_pos = arch.max_position_embeddings

    if max_pos is not None and train_cfg.seq_len > max_pos:
        raise ValueError(f"seq_len={train_cfg.seq_len} exceeds {max_pos} positions.")

    adapter = ResidualAdapter(cfg.adapter, arch)

    n_frozen = sum(parameter.numel() for parameter in substrate.params.values())

    frozen = substrate.verify_frozen()

    if not frozen["params_unchanged"]:
        raise RuntimeError(f"Frozen substrate verification failed: {frozen}")

    domains = []

    for domain in cfg.domains:
        train_blocks = load_token_blocks(
            substrate,
            domain,
            domain.train_split,
            train_cfg.seq_len,
        )

        eval_blocks = load_token_blocks(
            substrate,
            domain,
            domain.eval_split,
            train_cfg.seq_len,
        )

        domains.append(
            Domain(
                name=domain.name,
                train_blocks=train_blocks,
                eval_set=make_eval_set(
                    eval_blocks,
                    train_cfg.batch_size,
                    train_cfg.eval_batches,
                ),
                epochs=(train_cfg.epochs if domain.epochs is None else domain.epochs),
            )
        )

    base_name = args.run_name or default_run_name("a3hybrid")
    ks = args.K or [cfg.objective.infer_steps]

    status = 0

    for k in ks:
        run_cfg = replace(
            cfg,
            objective=replace(
                cfg.objective,
                infer_steps=k,
            ),
        )

        run_name = base_name if len(ks) == 1 else f"{base_name}-K{k}"

        status |= run_one(
            run_cfg,
            args,
            substrate,
            adapter,
            domains,
            n_frozen,
            run_name,
            start_time,
        )

    return status


if __name__ == "__main__":
    sys.exit(main())
