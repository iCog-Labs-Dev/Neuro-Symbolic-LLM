"""Run Stage A2 (local-predictor part): residual adapter + FabricPC predictor + stability diagnostics.

Wiring only, mirroring ``run_stage_a1.py``: builds the configs, substrate, adapter, local
predictor bank and stability monitor, then runs the shared loop in ``stages/stage_base.py`` with the
A2 objective. Ordinary backprop; there is no settling anywhere in this stage.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Run as a script: put the repo root on sys.path.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frozenllm.substrate import FrozenSubstrate  # noqa: E402
from frozenllm.substrate.memory import get_memory_status  # noqa: E402
from residual.stability import (  # noqa: E402
    StabilityMonitor,
)
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
from stages.stage_base import (  # noqa: E402
    Domain,
    RunConfig,
    apply_run_overrides,
    configure_logging,
    evaluate_checkpoint,
    initialize_stage,
    load_configs,
    load_token_blocks,
    log_final_summary,
    make_eval_set,
    prepare_domain,
    resolve_split,
    run_sequential,
    test_domain,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "stage_A" / "a2_pc_inspired.yaml"

log = logging.getLogger("stage_a2")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--rank", type=int, default=None, help="Adapter rank override.")
    parser.add_argument(
        "--epochs", type=int, default=None, help="Epochs for every domain (smoke runs)."
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--arm", choices=ARMS, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lambda-kl", type=float, default=None)
    parser.add_argument("--lambda-wd", type=float, default=None)
    parser.add_argument("--lambda-cp", type=float, default=None)
    parser.add_argument("--lambda-stab", type=float, default=None)
    parser.add_argument(
        "--eval-checkpoint",
        type=Path,
        default=None,
        help="Evaluate a saved checkpoint instead of training.",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--no-jit", action="store_true", default=False)
    # Local predictor
    parser.add_argument("--lambda-hid", type=float, default=None)
    parser.add_argument(
        "--hidden-target", choices=("corrected_teacher", "base_anchor"), default=None
    )
    parser.add_argument("--ff-dim", type=int, default=None)
    parser.add_argument(
        "--hid-detach-input",
        "--detach-input",
        dest="detach_input",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stop L_hidden-pred gradients into the adapter.",
    )
    parser.add_argument(
        "--predict-delta", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--cpc-detach-query",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Detach the causal CPC input from column parameters (default: true).",
    )
    parser.add_argument("--predictor-lr-mult", type=float, default=None)
    # Stability
    parser.add_argument("--stability-mode", choices=("off", "diagnostic"), default=None)
    parser.add_argument("--audit", action="store_true", default=False)
    parser.add_argument("--audit-only", action="store_true", default=False)
    parser.add_argument(
        "--report-internals",
        action="store_true",
        help="Include checkpoint internal metrics; requires --eval-checkpoint and the saved config.",
    )
    return parser.parse_args(argv)


def apply_overrides(cfg: RunConfig, args: argparse.Namespace) -> RunConfig:
    cfg = apply_run_overrides(cfg, args)
    adapter, training, domains = cfg.adapter, cfg.training, cfg.domains
    obj: A2Config = cfg.objective
    if args.arm is not None:
        obj = apply_arm(obj, args.arm)
    if args.seed is not None:
        adapter = replace(adapter, seed=args.seed)
        training = replace(training, seed=args.seed)
        obj = replace(
            obj,
            seed=args.seed,
            hid=replace(obj.hid, predictor=replace(obj.predictor, seed=args.seed)),
        )
    obj = replace(
        obj,
        lambda_kl=args.lambda_kl if args.lambda_kl is not None else obj.lambda_kl,
        lambda_wd=args.lambda_wd if args.lambda_wd is not None else obj.lambda_wd,
        cpc=replace(
            obj.cpc,
            weight=args.lambda_cp if args.lambda_cp is not None else obj.cpc.weight,
        ),
        stab=replace(
            obj.stab,
            weight=(
                args.lambda_stab if args.lambda_stab is not None else obj.stab.weight
            ),
        ),
    )
    hid = {
        k: v
        for k, v in {
            "lambda_hid": args.lambda_hid,
            "hidden_target": args.hidden_target,
            "predictor_lr_mult": args.predictor_lr_mult,
        }.items()
        if v is not None
    }
    pred = {
        k: v
        for k, v in {
            "ff_dim": args.ff_dim,
            "detach_input": args.detach_input,
            "predict_delta": args.predict_delta,
        }.items()
        if v is not None
    }
    stab = {
        k: v
        for k, v in {
            "mode": args.stability_mode,
        }.items()
        if v is not None
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
            **hid,
            predictor=replace(obj.hid.predictor, **pred),
        ),
        monitor=replace(obj.monitor, **stab),
    )
    return replace(
        cfg, adapter=adapter, objective=obj, training=training, domains=domains
    )


def main(argv: Sequence[str] | None = None) -> int:
    start_time = time.time()
    configure_logging(level=logging.INFO)
    args = parse_args(argv)
    cfg = apply_overrides(load_configs(args.config, A2Config), args)
    train_cfg = cfg.training
    obj: A2Config = cfg.objective
    if args.report_internals and args.eval_checkpoint is None:
        raise ValueError("--report-internals requires --eval-checkpoint.")
    if obj.cpc.enabled and args.eval_checkpoint is None:
        if train_cfg.batch_size < 2:
            raise ValueError(
                "Enabled CPC requires batch_size >= 2 for cross-sequence negatives."
            )
        if max(obj.cpc.offsets) >= train_cfg.seq_len:
            raise ValueError("Every CPC offset must be smaller than training.seq_len.")
    log.info("adapter=%s", cfg.adapter)
    log.info("objective=%s", obj)
    log.info("training=%s", train_cfg)
    log.info("domains=%s", [d.name for d in cfg.domains])

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
    params = init_a2_params(obj, adapter, train_cfg.seq_len)
    counts = param_counts(params)
    n_adapter = counts["adapter"]
    n_pred = counts["pred"]
    n_frozen = sum(p.numel() for p in substrate.params.values())
    log.info(
        "model=%s L=%d d=%d adapter_layers=%s predictor_layers=%s rank=%d "
        "adapter_params=%d predictor_params=%d frozen=%d",
        train_cfg.model,
        arch.num_layers,
        arch.hidden_size,
        list(adapter.layers),
        list(bank.layers) if bank else [],
        cfg.adapter.rank,
        n_adapter,
        n_pred,
        n_frozen,
    )

    if args.eval_checkpoint is not None:
        log.info(
            "Evaluating checkpoint %s on split=%s", args.eval_checkpoint, args.split
        )
        saved = load_params(args.eval_checkpoint)
        results = evaluate_checkpoint(
            substrate,
            adapter,
            adapter_params(saved),
            cfg.domains,
            split=args.split,
            seq_len=train_cfg.seq_len,
            batch_size=train_cfg.batch_size,
            eval_batches=train_cfg.eval_batches,
            jit=train_cfg.jit,
        )
        if args.report_internals:
            from stages.stage_A.a2_reporting import internal_metrics

            reports = {}
            for d_cfg in cfg.domains:
                blocks = load_token_blocks(
                    substrate,
                    d_cfg,
                    resolve_split(d_cfg, args.split),
                    train_cfg.seq_len,
                )
                batches = make_eval_set(
                    blocks, train_cfg.batch_size, train_cfg.eval_batches
                )
                row = internal_metrics(substrate, adapter, saved, obj, batches)
                if bank is not None:
                    row.update(
                        a2_diagnostics(
                            substrate,
                            adapter,
                            bank,
                            obj,
                            saved,
                            batches,
                            jit=train_cfg.jit,
                        )
                    )
                reports[d_cfg.name] = row
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

    # Shared data preparation; fail before creating a run directory.
    domains = [prepare_domain(substrate, d, train_cfg) for d in cfg.domains]

    audit_data = None
    if args.audit or args.audit_only:
        audit_data = audit_to_dict(
            a2_audit(substrate, adapter, obj, domains[0].eval_set[0])
        )
        if args.audit_only:
            print(json.dumps(audit_data, indent=2))
            return 0 if audit_data["ok"] else 1
        if not audit_data["ok"]:
            print(json.dumps(audit_data, indent=2))
            return 1

    def diagnostics(p: dict, domain: Domain) -> dict[str, float]:
        if bank is None:
            return {}
        return a2_diagnostics(
            substrate,
            adapter,
            bank,
            obj,
            p,
            domain.eval_set,
            jit=train_cfg.jit,
        )

    run_name = args.run_name or default_run_name("a2")
    with RunWriter(train_cfg.output_dir, run_name) as writer:
        log.info("run_dir=%s", writer.dir)
        writer.write_config(cfg.to_dict())
        if audit_data is not None:
            writer.write_json("audit.json", audit_data)
        writer.write_json(
            "meta.json",
            {
                **run_metadata(),
                "stage": "A2",
                "model": train_cfg.model,
                "num_layers": arch.num_layers,
                "hidden_size": arch.hidden_size,
                "adapted_layers": list(adapter.layers),
                "predictor_layers": list(bank.layers) if bank else [],
                "adapter_params": n_adapter,
                "predictor_params": n_pred,
                "trainable_params": counts["total"],
                "trainable_params_by_group": counts,
                "frozen_params": n_frozen,
                "arm": args.arm,
                "hidden_target": obj.hidden_target,
                "hidden_updates_adapter": obj.uses_hidden_pred and obj.hid_coupled,
                "hid_detach_input": obj.hid_detach_input,
                "cpc_detach_query": obj.cpc_detach_query,
                "seed": train_cfg.seed,
                "epochs": {d.name: d.epochs for d in domains},
                "jit": train_cfg.jit,
            },
        )
        init_diag = diagnostics(params, domains[0])
        writer.log("a2_diagnostics", when="init", domain=domains[0].name, **init_diag)

        params, summary = run_sequential(
            substrate,
            adapter,
            params,
            domains,
            make_a2_objective(bank, monitor),
            obj,
            train_cfg,
            writer,
            optimizer=make_a2_optimizer(obj, train_cfg.learning_rate),
        )
        frozen = substrate.verify_frozen()
        summary["params_unchanged"] = frozen["params_unchanged"]
        summary["device_memory"] = asdict(get_memory_status())

        final_diag = {}
        for domain in domains:
            final_diag[domain.name] = diagnostics(params, domain)
            writer.log(
                "a2_diagnostics",
                when="final",
                domain=domain.name,
                **final_diag[domain.name],
            )
        summary["a2_diagnostics"] = {
            "init": {domains[0].name: init_diag},
            "final": final_diag,
        }

        test_results = {
            d.name: test_domain(substrate, adapter, params, d, train_cfg, writer=writer)
            for d in cfg.domains
        }
        summary["test"] = test_results
        writer.write_json("results.json", summary)

    log_final_summary(log, summary, frozen, start_time=start_time)
    if not frozen["params_unchanged"]:
        log.error("Base parameters changed during training.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
