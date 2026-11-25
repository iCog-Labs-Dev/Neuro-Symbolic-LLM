"""Run Stage A2: residual adapter with local predictors and auxiliary losses."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frozenllm.substrate import FrozenSubstrate  # noqa: E402
from frozenllm.substrate.memory import get_memory_status  # noqa: E402
from residual.stability import StabilityMonitor  # noqa: E402
from runtime.data_loader import load_runtime_data  # noqa: E402
from stages.results import (  # noqa: E402
    RunWriter,
    default_run_name,
    load_params,
    run_metadata,
)
from stages.stage_A.a2 import (  # noqa: E402
    ARMS,
    A2Config,
    a2_audit,
    a2_diagnostics,
    adapter_params,
    apply_arm,
    audit_to_dict,
    init_a2_params,
    make_a2_objective,
    make_a2_optimizer,
    param_counts,
    predictor_bank,
)
from stages.stage_A.a2.a2_reporting import internal_metrics  # noqa: E402
from stages.stage_base import (  # noqa: E402
    RunConfig,
    apply_run_overrides,
    configure_logging,
    evaluate,
    format_params,
    initialize_stage,
    load_configs,
    log_experiment_header,
    log_final_summary,
)
from training.train import train  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "stage_A" / "a2_pc_inspired.yaml"
DEFAULT_RUNTIME_CONFIG = REPO_ROOT / "configs" / "runtime_datasets.yaml"

log = logging.getLogger("stage_a2")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--epochs", "--epoch", dest="epochs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--arm", choices=ARMS, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--lambda-kl", type=float, default=None)
    parser.add_argument("--lambda-wd", type=float, default=None)
    parser.add_argument("--lambda-cp", type=float, default=None)
    parser.add_argument("--lambda-stab", type=float, default=None)

    parser.add_argument("--lambda-hid", type=float, default=None)
    parser.add_argument(
        "--hidden-target",
        choices=("corrected_teacher", "base_anchor"),
        default=None,
    )
    parser.add_argument("--ff-dim", type=int, default=None)
    parser.add_argument(
        "--hid-detach-input",
        "--detach-input",
        dest="detach_input",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--predict-delta",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--cpc-detach-query",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--predictor-lr-mult", type=float, default=None)

    parser.add_argument(
        "--stability-mode",
        choices=("off", "diagnostic"),
        default=None,
    )
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--audit-only", action="store_true")

    parser.add_argument("--eval-checkpoint", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--no-jit", action="store_true")
    parser.add_argument("--report-internals", action="store_true")

    return parser.parse_args(argv)


def apply_overrides(cfg: RunConfig, args: argparse.Namespace) -> RunConfig:
    cfg = apply_run_overrides(cfg, args)

    adapter = cfg.adapter
    training = cfg.training
    obj: A2Config = cfg.objective

    if args.arm is not None:
        obj = apply_arm(obj, args.arm)

    if args.seed is not None:
        adapter = replace(adapter, seed=args.seed)
        training = replace(training, seed=args.seed)
        obj = replace(
            obj,
            seed=args.seed,
            hid=replace(
                obj.hid,
                predictor=replace(
                    obj.hid.predictor,
                    seed=args.seed,
                ),
            ),
        )

    obj = replace(
        obj,
        lambda_kl=(args.lambda_kl if args.lambda_kl is not None else obj.lambda_kl),
        lambda_wd=(args.lambda_wd if args.lambda_wd is not None else obj.lambda_wd),
        cpc=replace(
            obj.cpc,
            weight=(args.lambda_cp if args.lambda_cp is not None else obj.cpc.weight),
        ),
        stab=replace(
            obj.stab,
            weight=(
                args.lambda_stab if args.lambda_stab is not None else obj.stab.weight
            ),
        ),
    )

    hid_overrides = {
        key: value
        for key, value in {
            "lambda_hid": args.lambda_hid,
            "hidden_target": args.hidden_target,
            "predictor_lr_mult": args.predictor_lr_mult,
        }.items()
        if value is not None
    }

    predictor_overrides = {
        key: value
        for key, value in {
            "ff_dim": args.ff_dim,
            "detach_input": args.detach_input,
            "predict_delta": args.predict_delta,
        }.items()
        if value is not None
    }

    monitor_overrides = {
        key: value
        for key, value in {
            "mode": args.stability_mode,
        }.items()
        if value is not None
    }

    obj = replace(
        obj,
        hid_detach_input=(
            obj.hid_detach_input if args.detach_input is None else args.detach_input
        ),
        cpc_detach_query=(
            obj.cpc_detach_query
            if args.cpc_detach_query is None
            else args.cpc_detach_query
        ),
        hid=replace(
            obj.hid,
            **hid_overrides,
            predictor=replace(
                obj.hid.predictor,
                **predictor_overrides,
            ),
        ),
        monitor=replace(
            obj.monitor,
            **monitor_overrides,
        ),
    )

    return replace(
        cfg,
        adapter=adapter,
        objective=obj,
        training=training,
    )


def main(argv: Sequence[str] | None = None) -> int:
    start_time = time.time()
    configure_logging(level=logging.INFO)

    args = parse_args(argv)

    cfg = apply_overrides(
        load_configs(args.config, A2Config),
        args,
    )

    train_cfg = cfg.training
    obj: A2Config = cfg.objective

    if args.report_internals and args.eval_checkpoint is None:
        raise ValueError("--report-internals requires --eval-checkpoint.")

    if obj.cpc.enabled and args.eval_checkpoint is None:
        if train_cfg.batch_size < 2:
            raise ValueError("Enabled CPC requires batch_size >= 2.")

        if max(obj.cpc.offsets) >= train_cfg.seq_len:
            raise ValueError("Every CPC offset must be smaller than seq_len.")

    # ---------------------------------------------------------
    # Frozen substrate and A2 components
    # ---------------------------------------------------------

    substrate = FrozenSubstrate(train_cfg.model)
    arch = substrate.architecture
    adapter = initialize_stage(substrate, cfg)

    bank = (
        predictor_bank(obj, adapter, train_cfg.seq_len)
        if obj.uses_hidden_pred
        else None
    )

    monitor = (
        StabilityMonitor(obj.stability, adapter.layers)
        if obj.stability.enabled
        else None
    )

    params = init_a2_params(
        obj,
        adapter,
        train_cfg.seq_len,
    )

    counts = param_counts(params)

    n_adapter = counts["adapter"]
    n_predictor = counts["pred"]
    n_frozen = sum(p.numel() for p in substrate.params.values())

    # ---------------------------------------------------------
    # Shared runtime datasets
    # ---------------------------------------------------------

    eval_batches = (
        args.eval_batches if args.eval_batches is not None else train_cfg.eval_batches
    )

    runtime_data = load_runtime_data(
        config_path=DEFAULT_RUNTIME_CONFIG,
        tokenizer=substrate.tokenizer,
        sequence_length=train_cfg.seq_len,
        batch_size=train_cfg.batch_size,
        eval_batches=eval_batches,
        default_epochs=train_cfg.epochs,
    )

    domains = runtime_data.tasks

    test_sets = runtime_data.get_test_sets(
        train_cfg.batch_size,
        eval_batches,
    )

    # ---------------------------------------------------------
    # A2 evaluation uses adapter parameters only
    # ---------------------------------------------------------

    def evaluate_a2(p, eval_set):
        return evaluate(
            substrate,
            adapter,
            adapter_params(p),
            eval_set,
            jit=train_cfg.jit,
        )

    # ---------------------------------------------------------
    # Checkpoint evaluation
    # ---------------------------------------------------------

    if args.eval_checkpoint is not None:
        saved = load_params(args.eval_checkpoint)

        if args.split == "test":
            selected_sets = test_sets
        elif args.split in ("eval", "validation"):
            selected_sets = {task.name: task.eval_set for task in domains}
        else:
            raise ValueError(
                "Runtime checkpoint evaluation supports "
                "'test' and 'validation' splits."
            )

        results = {
            name: evaluate_a2(saved, eval_set)
            for name, eval_set in selected_sets.items()
        }

        if args.report_internals:
            reports = {}

            for name, eval_set in selected_sets.items():
                row = internal_metrics(
                    substrate,
                    adapter,
                    saved,
                    obj,
                    eval_set,
                )

                if bank is not None:
                    row.update(
                        a2_diagnostics(
                            substrate,
                            adapter,
                            bank,
                            obj,
                            saved,
                            eval_set,
                            jit=train_cfg.jit,
                        )
                    )

                reports[name] = row

            print(
                json.dumps(
                    {
                        "evaluation": results,
                        "internal_metrics": reports,
                        "params_unchanged": substrate.params_unchanged(),
                    },
                    indent=2,
                )
            )
        else:
            print(json.dumps(results, indent=2))

        return 0

    # ---------------------------------------------------------
    # A2 audit
    # ---------------------------------------------------------

    audit_data = None

    if args.audit or args.audit_only:
        audit_data = audit_to_dict(
            a2_audit(
                substrate,
                adapter,
                obj,
                domains[0].eval_set[0],
            )
        )

        if args.audit_only:
            print(json.dumps(audit_data, indent=2))
            return 0 if audit_data["ok"] else 1

        if not audit_data["ok"]:
            print(json.dumps(audit_data, indent=2))
            return 1

    # ---------------------------------------------------------
    # A2 diagnostics
    # ---------------------------------------------------------

    def diagnostics(p, eval_set):
        if bank is None:
            return {}

        return a2_diagnostics(
            substrate,
            adapter,
            bank,
            obj,
            p,
            eval_set,
            jit=train_cfg.jit,
        )

    # ---------------------------------------------------------
    # A2 objective and optimizer
    # ---------------------------------------------------------

    objective_fn = make_a2_objective(
        bank,
        monitor,
    )

    optimizer = make_a2_optimizer(
        obj,
        train_cfg.learning_rate,
    )

    # The shared trainer expects:
    #   forward_fn(params, batch, key)
    #   loss_fn(predictions, batch, params, loss_config)
    #
    # A2's existing objective already computes the forward pass,
    # all auxiliary losses, and the total loss.
    # The small wrappers below preserve that behavior.

    def a2_forward(p, batch, key):
        return batch

    def a2_loss(predictions, batch, p, loss_config):
        return objective_fn(
            p,
            substrate=substrate,
            adapter=adapter,
            config=loss_config,
            input_ids=batch,
        )

    # ---------------------------------------------------------
    # Run artifacts
    # ---------------------------------------------------------

    run_name = args.run_name or default_run_name("a2")

    with RunWriter(train_cfg.output_dir, run_name) as writer:
        log.info("run_dir=%s", writer.dir)

        writer.write_config(cfg.to_dict())

        if audit_data is not None:
            writer.write_json(
                "audit.json",
                audit_data,
            )

        writer.write_json(
            "meta.json",
            {
                **run_metadata(),
                "stage": "A2",
                "model": train_cfg.model,
                "num_layers": arch.num_layers,
                "hidden_size": arch.hidden_size,
                "adapted_layers": list(adapter.layers),
                "predictor_layers": (list(bank.layers) if bank else []),
                "adapter_params": n_adapter,
                "predictor_params": n_predictor,
                "trainable_params": counts["total"],
                "trainable_params_by_group": counts,
                "frozen_params": n_frozen,
                "arm": args.arm,
                "hidden_target": obj.hidden_target,
                "hidden_updates_adapter": (obj.uses_hidden_pred and obj.hid_coupled),
                "hid_detach_input": obj.hid_detach_input,
                "cpc_detach_query": obj.cpc_detach_query,
                "seed": train_cfg.seed,
                "epochs": {domain.name: domain.epochs for domain in domains},
                "jit": train_cfg.jit,
                "runtime_data": runtime_data.metadata,
            },
        )

        log_experiment_header(
            log,
            cfg=cfg,
            arch=arch,
            adapter=adapter,
            n_trainable=counts["total"],
            n_frozen=n_frozen,
            run_dir=writer.dir,
            domains=domains,
            stage="A2",
        )

        log.info(
            "A2 parameters      : Adapter=%s | Predictors=%s | CPC=%s",
            format_params(n_adapter),
            format_params(n_predictor),
            format_params(counts["cpc"]),
        )

        # Initial predictor diagnostics.
        init_diag = diagnostics(
            params,
            domains[0].eval_set,
        )

        writer.log(
            "a2_diagnostics",
            when="init",
            domain=domains[0].name,
            **init_diag,
        )

        # -----------------------------------------------------
        # Shared A1/A2/A3 training engine
        # -----------------------------------------------------

        params, summary = train(
            params=params,
            tasks=domains,
            forward_fn=a2_forward,
            loss_fn=a2_loss,
            loss_config=obj,
            optimizer=optimizer,
            config=train_cfg,
            writer=writer,
            eval_fn=evaluate_a2,
        )

        # -----------------------------------------------------
        # Substrate integrity
        # -----------------------------------------------------

        frozen = substrate.verify_frozen()

        summary["params_unchanged"] = frozen["params_unchanged"]
        summary["runtime_data"] = runtime_data.metadata
        summary["device_memory"] = asdict(get_memory_status())

        # -----------------------------------------------------
        # Final A2 diagnostics
        # -----------------------------------------------------

        final_diag = {}

        for domain in domains:
            row = diagnostics(
                params,
                domain.eval_set,
            )

            final_diag[domain.name] = row

            writer.log(
                "a2_diagnostics",
                when="final",
                domain=domain.name,
                **row,
            )

        summary["a2_diagnostics"] = {
            "init": {
                domains[0].name: init_diag,
            },
            "final": final_diag,
        }

        # -----------------------------------------------------
        # Final test evaluation
        # -----------------------------------------------------

        test_results = {}

        for name, test_set in test_sets.items():
            result = evaluate_a2(
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

    # ---------------------------------------------------------
    # Final experiment summary
    # ---------------------------------------------------------

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
