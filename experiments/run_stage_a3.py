"""Run Stage A3: train a predictive-coding latent residual on a frozen LM (FabricPC 0.6.0).

Orchestration only: builds the configs, substrate and A3 graph, runs the optimiser loop, evaluates
and logs in the same format as stage A1. The architecture lives in stages/stage_A/A3_model.py and
a3_tail.py; losses are the substrate's; data, configs, summaries and logging helpers are the shared
ones in stages/stage_base.py.

    python experiments/run_stage_a3.py --config configs/stage_A/a3_true_pc.yaml --K 1 2 4
"""

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
from metrics.performance import perplexity, training_step_time  # noqa: E402
from residual import ResidualAdapter  # noqa: E402
from stages.results import RunWriter, default_run_name, run_metadata  # noqa: E402
from stages.stage_A.A3_model import A3Config, A3Model  # noqa: E402
from stages.stage_base import (  # noqa: E402
    Domain,
    RunConfig,
    TrainingConfig,
    _fmt_losses,
    _fmt_terms,
    batches,
    configure_logging,
    load_configs,
    load_token_blocks,
    log_experiment_header,
    log_final_summary,
    make_eval_set,
    summarize,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "stage_A" / "a3_true_pc.yaml"

log = logging.getLogger("stage_a3")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--rank", type=int, default=None, help="Latent size r override."
    )
    parser.add_argument(
        "--K",
        type=int,
        nargs="+",
        default=None,
        help="Inference steps; several values run one training per K.",
    )
    parser.add_argument(
        "--eta-z", type=float, default=None, help="Per-token inference step."
    )
    parser.add_argument(
        "--lambda-kl", type=float, default=None, help="KL trust-region weight."
    )
    parser.add_argument(
        "--epochs",
        "--epoch",
        dest="epochs",
        type=int,
        default=None,
        help="Epochs for every domain (smoke runs).",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--eval-batches", type=int, default=None, help="Override eval_batches."
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Split for the final evaluation."
    )
    parser.add_argument(
        "--energy-curve",
        action="store_true",
        default=False,
        help="After each domain, log the exact energy at z^(0)..z^(K) (eager, slower).",
    )
    parser.add_argument(
        "--no-jit",
        action="store_true",
        default=False,
        help="Disable JAX JIT compilation for training and evaluation.",
    )
    return parser.parse_args(argv)


def apply_overrides(cfg: RunConfig, args: argparse.Namespace) -> RunConfig:
    adapter, objective, training, domains = (
        cfg.adapter,
        cfg.objective,
        cfg.training,
        cfg.domains,
    )
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
        domains = tuple(replace(d, epochs=args.epochs) for d in domains)
    if args.eval_batches is not None:
        if args.eval_batches < 1:
            raise ValueError(f"eval_batches must be >= 1, got {args.eval_batches}")
        training = replace(training, eval_batches=args.eval_batches)
    return replace(
        cfg, adapter=adapter, objective=objective, training=training, domains=domains
    )


def evaluate_deployment(
    substrate, model: A3Model, params, eval_batches, *, jit: bool = True
):
    """Label-free deployment (z = f_phi(h0), no inference): same metrics as A1's ``evaluate``."""
    if not eval_batches:
        return {"base_loss": 0.0, "base_ppl": 1.0, "loss": 0.0, "ppl": 1.0, "kl": 0.0}

    def _step(p, ids):
        adapted, base = model.deploy_logits(p, ids)
        base_loss = substrate.compute_loss(base, ids, losses=("CE",))
        loss = substrate.compute_loss(adapted, ids, losses=("CE",))
        kl = substrate.compute_loss(adapted, ids, losses=("KL",), base_logits=base)
        return base_loss, loss, kl

    step = jax.jit(_step) if jit else _step
    rows = [tuple(float(v) for v in step(params, ids)) for ids in eval_batches]
    base_loss, loss, kl = (float(np.mean(col)) for col in zip(*rows, strict=False))
    return {
        "base_loss": base_loss,
        "base_ppl": float(perplexity(base_loss)),
        "loss": loss,
        "ppl": float(perplexity(loss)),
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
    """Sequential-domain loop with A1's schedule, logging and artifacts. The gradient is the
    local predictive-coding gradient from ``model.train_grads``, not ``jax.grad``."""
    names = [d.name for d in domains]

    def eval_all(after_task: int):
        row = {}
        for d in domains:
            row[d.name] = evaluate_deployment(
                substrate, model, params, d.eval_set, jit=training.jit
            )
            writer.log("eval", after_task=after_task, domain=d.name, **row[d.name])
        if after_task < 0:
            log.info(
                "[Validation] Baseline zero-shot eval after_task=%d: %s",
                after_task,
                _fmt_losses(row),
            )
        else:
            log.info(
                "[Validation] eval after_task=%d (%s): %s",
                after_task,
                domains[after_task].name,
                _fmt_losses(row),
            )
        return row

    init = eval_all(-1)

    optimizer = optax.adam(training.learning_rate)
    base_key = jax.random.PRNGKey(training.seed + 100)

    def train_step(params, opt_state, ids, key):
        grads, aux = model.train_grads(params, ids, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        terms = {
            "total": aux["energy_after"],
            "task": aux["task"],
            "spring": aux["spring"],
            "energy_before": aux["energy_before"],
        }
        if "kl" in aux:
            terms["kl"] = aux["kl"]
        return optax.apply_updates(params, updates), opt_state, terms

    if training.jit:
        train_step = jax.jit(train_step)

    opt_state = optimizer.init(params)
    rows, step_times = [], []
    global_step = tokens = 0

    for t, domain in enumerate(domains):
        if t > 0 and training.reset_optimizer:
            opt_state = optimizer.init(params)
        num_epochs = domain.epochs
        if num_epochs > 0 and len(domain.train_blocks) < training.batch_size:
            raise ValueError(
                f"Domain {domain.name!r} has {len(domain.train_blocks)} blocks, "
                f"which cannot fill batch_size={training.batch_size}."
            )
        batches_per_epoch = (
            len(domain.train_blocks) // training.batch_size if num_epochs > 0 else 0
        )
        log.info("=" * 78)
        log.info(
            ">>> DOMAIN [%d/%d]: '%s' | Epochs: %d | Batches/Epoch: %d (Total: %d)",
            t + 1,
            len(domains),
            domain.name,
            num_epochs,
            batches_per_epoch,
            batches_per_epoch * num_epochs,
        )
        log.info("=" * 78)

        domain_step = 0
        for epoch in range(1, num_epochs + 1):
            log.info("  EPOCH %d/%d ['%s']", epoch, num_epochs, domain.name)
            epoch_start = time.time()
            epoch_step_times: list[float] = []
            epoch_scalars: list[dict[str, float]] = []
            epoch_tokens = 0
            stream = batches(
                domain.train_blocks,
                training.batch_size,
                training.seed + t * 10000 + (epoch - 1),
            )

            for batch_idx, ids in enumerate(stream, start=1):
                domain_step += 1
                global_step += 1
                tokens += int(ids.size)
                epoch_tokens += int(ids.size)
                key = jax.random.fold_in(base_key, global_step)

                (params, opt_state, terms), step_time = training_step_time(
                    train_step, params, opt_state, ids, key
                )
                step_times.append(step_time)
                epoch_step_times.append(step_time)
                scalars = {k: float(v) for k, v in terms.items()}
                epoch_scalars.append(scalars)

                metrics = {f"loss_{k}": v for k, v in scalars.items()}
                metrics["ppl"] = float(perplexity(scalars["task"]))
                writer.log(
                    "train",
                    task=t,
                    domain=domain.name,
                    epoch=epoch,
                    batch=batch_idx,
                    step=domain_step,
                    global_step=global_step,
                    tokens=tokens,
                    step_time=step_time,
                    **metrics,
                )
                if (
                    batch_idx == 1
                    or batch_idx == batches_per_epoch
                    or domain_step % training.log_every == 0
                ):
                    log.info(
                        "  Batch %d/%d %s  ppl=%.2f",
                        batch_idx,
                        batches_per_epoch,
                        _fmt_terms(scalars),
                        metrics["ppl"],
                    )

            if epoch_scalars:
                duration = float(time.time() - epoch_start)
                mean_terms = {
                    k: float(np.mean([s[k] for s in epoch_scalars]))
                    for k in epoch_scalars[0]
                }
                mean_time = float(np.mean(epoch_step_times))
                epoch_ppl = float(perplexity(mean_terms["task"]))
                writer.log(
                    "epoch",
                    task=t,
                    domain=domain.name,
                    epoch=epoch,
                    total_epochs=num_epochs,
                    global_step=global_step,
                    batches=len(epoch_scalars),
                    blocks=len(epoch_scalars) * training.batch_size,
                    epoch_tokens=epoch_tokens,
                    tokens=tokens,
                    epoch_time=duration,
                    mean_step_time=mean_time,
                    ppl=epoch_ppl,
                    **{f"loss_{k}": v for k, v in mean_terms.items()},
                )
                log.info(
                    "  Epoch %d/%d Finished | Duration: %.2fs (avg %.3fs/batch) | %s | ppl=%.2f",
                    epoch,
                    num_epochs,
                    duration,
                    mean_time,
                    _fmt_terms(mean_terms),
                    epoch_ppl,
                )

        writer.save_checkpoint(
            f"task_{t}_{domain.name}",
            model.adapter_params(params),
            opt_state=opt_state,
            global_step=global_step,
            task=t,
            domain=domain.name,
            epoch=num_epochs,
            step=domain_step,
        )
        log.info(
            "Checkpoint saved: task_%d_%s (global_step=%d)", t, domain.name, global_step
        )
        rows.append(eval_all(t))

        if energy_curve and domain.eval_set:
            curve = model.energy_curve(
                params, domain.eval_set[0], jax.random.fold_in(base_key, 0)
            )
            writer.log("energy_curve", after_task=t, domain=domain.name, curve=curve)
            log.info(
                "[Energy] after_task=%d (%s): %s",
                t,
                domain.name,
                " -> ".join(f"{e:.4f}" for e in curve),
            )

    summary = summarize(names, init, rows)
    summary.update(
        epochs={d.name: d.epochs for d in domains},
        steps=global_step,
        tokens=tokens,
        mean_step_time=float(np.mean(step_times[1:])) if len(step_times) > 1 else None,
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
    train_cfg, obj = cfg.training, cfg.objective
    arch = substrate.architecture
    model = A3Model(
        substrate,
        layers=adapter.layers,
        rank=cfg.adapter.rank,
        seq_len=train_cfg.seq_len,
        scale=adapter.scale,
        activation=cfg.adapter.activation,
        config=obj,
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
                "num_layers": arch.num_layers,
                "hidden_size": arch.hidden_size,
                "adapted_layers": list(adapter.layers),
                "trainable_params": n_trainable,
                "frozen_params": n_frozen,
                "infer_steps": obj.infer_steps,
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
        log.info(
            "Predictive coding  : K=%d | eta_z=%s | sigma2=%s | lambda_kl=%s",
            obj.infer_steps,
            obj.eta_z,
            obj.sigma2,
            obj.lambda_kl,
        )
        log.info("-" * 78)

        params, summary = train_sequential(
            substrate,
            model,
            params,
            domains,
            train_cfg,
            writer,
            energy_curve=args.energy_curve,
        )
        frozen = substrate.verify_frozen()
        summary["params_unchanged"] = frozen["params_unchanged"]
        summary["infer_steps"] = obj.infer_steps

        log.info(
            "Running final %s evaluation across %d domains...",
            args.split,
            len(cfg.domains),
        )
        test_results = {}
        for d in cfg.domains:
            split_name = {"test": d.test_split, "train": d.train_split}.get(
                args.split, d.eval_split
            )
            blocks = load_token_blocks(substrate, d, split_name, train_cfg.seq_len)
            test_set = make_eval_set(
                blocks, train_cfg.batch_size, train_cfg.eval_batches
            )
            test_results[d.name] = evaluate_deployment(
                substrate, model, params, test_set, jit=train_cfg.jit
            )
            writer.log("test", domain=d.name, **test_results[d.name])
        summary["test"] = test_results
        writer.write_json("results.json", summary)

    log_final_summary(log, summary, frozen, start_time=start_time)
    if not frozen["params_unchanged"]:
        log.error("Base parameters changed during training.")
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    start_time = time.time()
    configure_logging(level=logging.INFO)
    args = parse_args(argv)
    cfg = apply_overrides(load_configs(args.config, A3Config), args)
    train_cfg = cfg.training

    substrate = FrozenSubstrate(train_cfg.model)
    arch = substrate.architecture
    max_pos = arch.max_position_embeddings
    if max_pos is not None and train_cfg.seq_len > max_pos:
        raise ValueError(f"seq_len={train_cfg.seq_len} exceeds {max_pos} positions.")
    adapter = ResidualAdapter(
        cfg.adapter, arch
    )  # late-block selection and scale, as in A1
    n_frozen = sum(p.numel() for p in substrate.params.values())

    frozen = substrate.verify_frozen()
    if not frozen["params_unchanged"]:
        log.error("Substrate not frozen before training: %s", frozen)
        return 1

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

    base_name = args.run_name or default_run_name("a3")
    ks = args.K or [cfg.objective.infer_steps]
    status = 0
    for k in ks:
        run_cfg = replace(cfg, objective=replace(cfg.objective, infer_steps=k))
        run_name = base_name if len(ks) == 1 else f"{base_name}-K{k}"
        status |= run_one(
            run_cfg, args, substrate, adapter, domains, n_frozen, run_name, start_time
        )
    return status


if __name__ == "__main__":
    sys.exit(main())
