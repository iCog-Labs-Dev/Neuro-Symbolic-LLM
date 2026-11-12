"""Console A4 evaluation summaries without a numerical runtime."""

from __future__ import annotations

from typing import Any


def training_row(batch: int, batches: int, metrics: dict[str, Any]) -> str:
    """Format one training update; PPL is computed from task CE, not total loss."""
    ppl = "overflow" if metrics["ppl"] is None else f"{metrics['ppl']:.4f}"
    return (
        f"{f'{batch}/{batches}':>12} {metrics['total']:>14.6f} "
        f"{metrics['task']:>14.6f} {metrics['symbolic']:>14.6f} {ppl:>12}"
    )


def latency_lines(reports: list[dict[str, Any]]):
    """Shared readable timings for standalone and post-training measurements."""
    for report in reports:
        yield (
            f"A4 {report['mode']} | samples={report['samples']} | "
            f"mean step={report['step_mean_ms']:.3f} ms"
        )
        for name, prefix in (
            ("Full bridge", "full_round_trip"),
            ("Bridge minus CPU search", "communication"),
        ):
            yield (
                f"{name}: {report[f'{prefix}_mean_ms']:.3f} ms | "
                f"mean share={report[f'{prefix}_ratio_of_means_percent']:.2f}% | "
                f"paired p95 share={report[f'{prefix}_paired_ratio_p95_percent']:.2f}%"
            )
        if "cuda_transfers" in report:
            copies = report["cuda_transfers"]
            yield (
                f"CUDA copy intervals only: {copies['transfer_only_mean_ms']:.3f} ms | "
                f"mean share={copies['transfer_only_ratio_of_means_percent']:.2f}% | "
                f"paired p95 share={copies['transfer_only_paired_ratio_p95_percent']:.2f}%"
            )
        else:
            yield "CUDA copy intervals: not recorded"


def print_result(command: str, result: dict[str, Any]) -> None:
    if command == "list":
        if not result["runs"]:
            print("No A4 training runs found")
        for row in result["runs"]:
            print(f"{row['status']:12} {row['path']}")
    elif command == "show":
        print(f"A4 status: {result['status']['status']}")
        if not result["results_available"]:
            print("Final results: not recorded")
        for row in result["evaluations"]:
            print(
                f"{row['split']} epoch {row['epoch']}: "
                f"CE={row['loss']:.6f} PPL={row['ppl']:.4f}"
            )
        row = result["latest_training"]
        if row:
            print(
                f"Last update {row['step']}: CE={row['task']:.6f} "
                f"key={row['key']:.6f} value={row['value']:.6f}"
            )
            if "total" in row:
                print(f"  Total loss={row['total']:.6f}")
            if "gradient_norm" in row:
                print(f"  Gradient norm (before clipping)={row['gradient_norm']:.6f}")
            if "step_time" in row:
                print(f"  Last update time={1000 * row['step_time']:.3f} ms")
        print(result["retrieval_quality"])
    elif command == "retrieval":
        scores = result["metrics"]
        print(
            f"A4 {result['measurement']} | split={result['split']} | probes={scores['query_count']}"
        )
        print(
            f"Macro precision={scores['macro_precision']:.4f} recall={scores['macro_recall']:.4f}"
        )
        print(
            f"Micro precision={scores['micro_precision']:.4f} recall={scores['micro_recall']:.4f}"
        )
        print(
            f"Hit rate={scores['hit_rate']:.4f} MRR={scores['mean_reciprocal_rank']:.4f}"
        )
    else:
        for line in latency_lines(result["reports"]):
            print(line)
    print(result["scope"])


def number(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "unavailable"
    return f"{100 * value:.2f}%" if percent else f"{value:.6f}"


def print_rankings(rows: list[dict[str, Any]], *, indent: str = "  ") -> None:
    for row in rows:
        scores = row["positive_only"]
        print(
            f"{indent}top-{row['top_k']}: "
            f"precision(all)={number(row['all_probe_micro_precision'], percent=True)} "
            f"recall(positive)={number(scores['macro_recall'], percent=True)} "
            f"hit(positive)={number(scores['hit_rate'], percent=True)} "
            f"MRR(positive)={number(scores['mean_reciprocal_rank'])}"
        )
    if rows:
        row = rows[0]
        print(
            f"{indent}Cases: {row['positive_probes']} positive, "
            f"{row['known_negative_probes']} known negative"
        )


def print_retrieval(result: dict[str, Any]) -> None:
    print("A4 retrieval evaluation")
    if "bank_records" in result:
        print(f"Bank: {result['bank_records']} templates")
    for report in result["reports"]:
        print(
            f"{report['condition']}: CE={number(report['causal_window_ce'])} "
            f"targets={report['next_token_targets']}"
        )
        for layer in report["layers"]:
            print(f"  Layer {layer['layer']}")
            print_rankings(layer["scores"], indent="    ")
    if "support_prior" in result:
        print("Bank-support baseline (same ranked IDs for every case)")
        print_rankings(result["support_prior"]["scores"])
    print(result.get("scope", "Fixed-checkpoint retrieval evaluation"))


def print_representation(
    result: dict[str, Any], *, capacity: dict[str, Any] | None = None
) -> None:
    print("A4 prediction, alignment and representation evaluation")
    if "bank_records" in result:
        print(
            f"Bank: {result['bank_records']} templates | "
            f"Evaluation cases: {result['probes']}"
        )
    for report in result["reports"]:
        print(f"\n{report['condition']}")
        print(
            f"  Prediction: CE={number(report['ce'])} "
            f"PPL={number(report['ppl'])} "
            f"accuracy={number(report['next_token_accuracy'], percent=True)} "
            f"targets={report['next_token_targets']}"
        )
        print(
            f"  Prediction drift: KL(base||model)={number(report['kl_base_to_model'])}"
        )
        for layer in report["layers"]:
            print(f"  Layer {layer['layer']}")
            print(
                f"    Alignment: key NLL={number(layer['reference_pair_key_nll'])} "
                f"value squared error={number(layer['reference_pair_value_squared_error'])} "
                f"reference pairs={layer['alignment_reference_pairs']}"
            )
            print_rankings(layer["retrieval"], indent="    ")
            clusters = layer["query_clusters"]
            print(f"    Query silhouette={number(clusters.get('silhouette'))}")
            if clusters.get("reason"):
                print(f"      {clusters['reason']}")
            print(
                "    Writeback: mean adjustment/hidden norm="
                f"{number(layer['writeback_to_hidden_norm_mean'])} "
                "nonzero fraction="
                f"{number(layer['writeback_nonzero_fraction'], percent=True)}"
            )
            organization = layer.get("factor_organization")
            if organization is None:
                print(
                    "    Channel organization: "
                    + report.get("factor_organization_status", "not measured")
                )
                continue
            fracture = [v for v in organization["fracture"] if v is not None]
            mean_fracture = sum(fracture) / len(fracture) if fracture else None
            print(
                f"    Channel organization: mean fracture={number(mean_fracture)} "
                "mean pair overlap="
                f"{number(organization['mean_off_diagonal_overlap'])}"
            )
            print(
                f"      Factors={organization['factors']} "
                f"components={organization['components']} "
                f"zero-effect factors={len(organization['zero_effect_factor_indices'])}"
            )
    geometry = result.get("fixed_bank_geometry")
    if geometry:
        print("\nFixed bank geometry (not a learned-head improvement)")
        print(
            f"  Vectors={geometry['vectors']} dimensions={geometry['dimensions']} "
            f"unordered pairs={geometry['pairs']}"
        )
        print(
            f"  Key norms: min={number(geometry['norm_min'])} "
            f"mean={number(geometry['norm_mean'])} max={number(geometry['norm_max'])}"
        )
        print(
            f"  Cosine: mean={number(geometry['mean_cosine'])} "
            f"mean absolute={number(geometry['mean_absolute_cosine'])} "
            f"mean squared={number(geometry['mean_squared_cosine'])} "
            f"max={number(geometry['max_cosine'])}"
        )
    controls = capacity if capacity is not None else result.get("capacity", {})
    if controls:
        print("\nCapacity sensitivity (fixed weights, exact cosine, no retraining)")
        for name, measured in controls.items():
            print(f"  Layer {name}")
            for control in measured["controls"]:
                print(
                    f"    dimensions={control['dimensions']} bank={control['bank_size']} "
                    "reference coverage="
                    f"{number(control['available_reference_pair_fraction'], percent=True)}"
                )
                for score in control["scores"]:
                    print(
                        f"      top-{score['top_k']}: "
                        f"precision(all)={number(score['all_probe_precision'], percent=True)} "
                        "recall(positive)="
                        f"{number(score['positive_macro_recall'], percent=True)}"
                    )
            print(
                "    HNSW/exact top-8 ID agreement="
                f"{number(measured.get('ann_exact_id_agreement_at_8'), percent=True)} "
                "(tie-sensitive; not semantic precision)"
            )
    print(result.get("scope", "Fixed-checkpoint representation evaluation"))
