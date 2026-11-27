"""Render saved benchmark measurements as Markdown, JSON and HTML."""

from __future__ import annotations

import argparse
import html
import re
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean
from typing import Any

from benchmark.benchmark import (
    EXTERNAL_METRIC_SPECS,
    INTERNAL_METRIC_SPECS,
    METHODS,
    _metric_samples,
    _read_json,
    _select_headline_cohort,
    _write_json,
    a4_comparison_groups,
    measurement_reason,
    metric_values_for_record,
    resolve_report_dir,
)


def _cell(values: list[float]) -> str:
    return f"{fmean(values):.6g} (n={len(values)})" if values else "N/A"


def _safe_text(value: Any) -> str:
    return str(value).replace("|", "/").replace("\n", " ")


def _format_task_order(order: list[str]) -> str:
    matches = [re.fullmatch(r"(.+)_task_(\d+)", name) for name in order]
    if matches and all(matches):
        prefixes = {match.group(1) for match in matches if match is not None}
        if len(prefixes) == 1:
            prefix = next(iter(prefixes)).replace("_", " ")
            indices = ", ".join(
                match.group(2) for match in matches if match is not None
            )
            return f"{prefix} task {indices}"
    return ", ".join(name.replace("_", " ") for name in order) or "N/A"


def _realized_task_orders(records: list[dict[str, Any]]) -> str:
    orders: list[tuple[str, ...]] = []
    for record in records:
        order = tuple(str(name) for name in record.get("task_order", []))
        if order and order not in orders:
            orders.append(order)
    return "; ".join(_format_task_order(list(order)) for order in orders) or "N/A"


def _comparison_table(
    cohort, specs, records: list[dict[str, Any]] | None = None
) -> list[str]:
    labels = [label for label, _, _ in specs]
    lines = [
        f"| Method | {' | '.join(labels)} |",
        f"|---|{'---:|' * len(labels)}",
    ]
    metric_samples = {
        key: (
            _metric_samples(cohort, key)
            if cohort
            else {method: [] for method in METHODS}
        )
        for _, key, _ in specs
    }
    for method in METHODS:
        if method == "a4":
            continue
        cells = [
            _cell([value for _, value in metric_samples[key][method]])
            for _, key, _ in specs
        ]
        lines.append(f"| {method.upper()} | {' | '.join(cells)} |")
    groups = a4_comparison_groups(records or []) or {"A4": []}
    for label, rows in groups.items():
        cells = []
        for _, key, _ in specs:
            values = [
                value
                for row in rows
                for value in metric_values_for_record(
                    row, key, int(row.get("rank", 16))
                )
            ]
            cell = _cell(values)
            if key == "residual_direction_redundancy" and values:
                cell += " — inherited from frozen A3 residual"
            cells.append(cell)
        lines.append(f"| {label} | {' | '.join(cells)} |")
    lines.append("")
    return lines


def _combined_evidence_table(records: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Run evidence, dataset token accuracy, and N/A reasons",
        "",
        "Next-token accuracy is measured on loader-provided test splits; it is not labeled task accuracy.",
        "",
        "| Run | Method | Seed | Rank | K | Dataset/test split | Token accuracy | N/A metrics and reasons |",
        "|---|---|---:|---:|---:|---|---:|---|",
    ]
    specs = INTERNAL_METRIC_SPECS + EXTERNAL_METRIC_SPECS
    for record in records:
        missing = []
        for label, key, _ in specs:
            values = metric_values_for_record(record, key, int(record.get("rank", 16)))
            if not values:
                reason = measurement_reason(record, key)
                missing.append(f"{label}: {reason}")
        reason_cell = _safe_text("; ".join(missing) if missing else "None")
        domains = record.get("deployed_evaluation", {}).get("domains", {})
        rows = list(domains.items()) if domains else [("N/A", {})]
        for dataset, measurement in rows:
            value = measurement.get("token_accuracy")
            accuracy = (
                _cell([float(value)])
                if value is not None and measurement.get("status") == "measured"
                else "N/A"
            )
            lines.append(
                f"| {_safe_text(record.get('run_id', 'N/A'))} | "
                f"{_safe_text(str(record.get('method', 'N/A')).upper())} | "
                f"{record.get('seed', 'N/A')} | {record.get('rank', 'N/A')} | "
                f"{record.get('k', 'N/A')} | {_safe_text(dataset)} | {accuracy} | "
                f"{reason_cell} |"
            )
    if not records:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A | No evaluated runs |")
    lines.append("")
    return lines


def _figure_notice() -> list[str]:
    return [
        "Figures are stored separately in the [`figures/`](figures/) folder.",
        "",
    ]


def _a4_detail_table(records: list[dict[str, Any]]) -> list[str]:
    lines = ["## A4 Run Evidence", ""]
    for record in records:
        if record.get("method") != "a4" or record.get("status") == "invalid":
            continue
        lines.extend(
            [
                f"### {_safe_text(record['run_id'])}",
                "",
                f"Mode {_safe_text(record.get('a4_data_mode', 'aligned'))}, "
                f"arm {_safe_text(record.get('a4_arm', 'full'))}, seed {record['seed']}.",
                "",
            ]
        )
        history = record.get("after_task", [])
        if history:
            lines.extend(
                [
                    f"Validation scope is {_safe_text(record['validation_scope'])}.",
                    "",
                    "| After task | Validation set | CE | Perplexity |",
                    "|---|---|---:|---:|",
                ]
            )
            for name, evaluations in zip(
                ["Initial", *record["task_order"]],
                [record["initial"], *history],
                strict=True,
            ):
                for dataset, metrics in evaluations.items():
                    lines.append(
                        f"| {_safe_text(name)} | {_safe_text(dataset)} | "
                        f"{metrics['loss']:.4g} | {metrics['ppl']:.4g} |"
                    )
            if record.get("backward_transfer_ce") is not None:
                lines.extend(
                    [
                        "",
                        f"Backward transfer in CE is {record['backward_transfer_ce']:.4g}.",
                    ]
                )
            lines.append("")
        baseline = record.get("frozen_a3_baseline", {})
        if baseline:
            lines.extend(
                [
                    "| Frozen A3 split | CE | Perplexity |",
                    "|---|---:|---:|",
                    *(
                        f"| {_safe_text(split)} | {metrics['loss']:.4g} | {metrics['ppl']:.4g} |"
                        for split, metrics in baseline.items()
                    ),
                    "",
                ]
            )
        detail = (
            record.get("deployed_evaluation", {})
            .get("a4", {})
            .get("symbolic_detail", {})
        )
        if detail:
            lines.extend(
                [
                    "| Symbolic metric | Value |",
                    "|---|---:|",
                    *(
                        f"| {_safe_text(label)} | {_cell([float(value)])} |"
                        for label, value in detail.items()
                    ),
                    "",
                ]
            )
    return lines if len(lines) > 2 else []


def report_manifest(
    output_dir: Path, report_rank: int = 16, report_dir: Path | str | None = None
) -> Path:
    destination = resolve_report_dir(report_dir)
    if (
        destination == output_dir.resolve()
        or output_dir.resolve() in destination.parents
    ):
        raise ValueError("Reports must be outside the raw runs directory")
    manifest = _read_json(output_dir / "manifest.json")
    evaluation = _read_json(output_dir / "evaluation.json")
    records = evaluation.get("records", [])
    destination.mkdir(parents=True, exist_ok=True)
    stage_root = destination / "stage_A"
    comparison = stage_root / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    cohort = _select_headline_cohort(records, report_rank)
    lines = [
        "# Stage A General Report",
        "",
        f"Runtime source: {manifest.get('runtime_source', 'unknown')}",
        f"Realized task order: {_realized_task_orders(records)}",
        "",
        "Only measured values are reported; missing evidence is N/A.",
        "",
    ]
    if cohort is None or not cohort.matched_seeds:
        lines += [
            "No complete matched A1/A2/A3 cohort is available; A4 is reported independently.",
            "",
        ]
    else:
        lines += [
            f"Selected A3 K: {cohort.selected_k}. Selection rule: {cohort.selection['rule']}",
            f"Matched seeds: {list(cohort.matched_seeds)}",
            "",
        ]
    for category, specs in (
        ("Internal", INTERNAL_METRIC_SPECS),
        ("External", EXTERNAL_METRIC_SPECS),
    ):
        lines += [
            f"## {category} Metrics",
            "",
            *_comparison_table(cohort, specs, records),
        ]
    lines += [
        "Measurement notes: task-gradient conflict uses fixed task-specific training "
        "batches at the final checkpoint. Residual direction redundancy averages "
        "read/write overlap across layers. Ablation fidelity compares predicted and "
        "actual held-out loss changes for sampled rank channels.",
        "",
        *_combined_evidence_table(records),
        *_figure_notice(),
    ]
    general = comparison / "stage_a_general_report.md"
    general.write_text("\n".join(lines) + "\n", encoding="utf-8")
    reports = {}
    for method in METHODS:
        folder = stage_root / method.upper()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{method}_result_and_visualization.md"
        content = [
            f"# {method.upper()} Results",
            "",
            "Numeric comparisons and run evidence are consolidated in the "
            "[Stage A general report](../comparison/stage_a_general_report.md).",
            "",
            *_figure_notice(),
        ]
        if method == "a4":
            content[4:4] = _a4_detail_table(records)
        path.write_text("\n".join(content) + "\n", encoding="utf-8")
        reports[method] = path
    report = destination / "REPORT.md"
    report.write_text(
        general.read_text(encoding="utf-8").replace(
            "](figures/", "](stage_A/comparison/figures/"
        ),
        encoding="utf-8",
    )
    _write_json(
        destination / "metrics.json",
        {
            "metrics": [
                key for _, key, _ in (INTERNAL_METRIC_SPECS + EXTERNAL_METRIC_SPECS)
            ],
            "runs": [
                {
                    "run_id": row["run_id"],
                    "task_order": row.get("task_order", []),
                    **(
                        {
                            key: row.get(key)
                            for key in (
                                "a4_data_mode",
                                "a4_arm",
                                "seed",
                                "rank",
                                "protocol",
                                "validation_scope",
                                "after_task",
                                "backward_transfer_ce",
                                "task_epochs",
                                "checkpoints",
                                "frozen_a3_baseline",
                            )
                        }
                        if row.get("method") == "a4"
                        else {}
                    ),
                    "params_unchanged": row.get("params_unchanged"),
                    "metrics": {
                        key: {
                            "values": metric_values_for_record(
                                row, key, int(row.get("rank", 16))
                            ),
                            "reason": (
                                None
                                if metric_values_for_record(
                                    row, key, int(row.get("rank", 16))
                                )
                                else measurement_reason(row, key)
                            ),
                        }
                        for _, key, _ in (INTERNAL_METRIC_SPECS + EXTERNAL_METRIC_SPECS)
                    },
                }
                for row in records
            ],
        },
    )
    (destination / "README.md").write_text(
        "# Stage A results\n\n[Dashboard](index.html) · [Comparison](stage_A/comparison/stage_a_general_report.md)\n",
        encoding="utf-8",
    )
    _write_dashboard(destination, general, reports)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/benchmark"))
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--report-rank", type=int, default=16)
    args = parser.parse_args(argv)
    print(report_manifest(args.output_dir.resolve(), args.report_rank, args.report_dir))
    return 0


def _write_dashboard(
    destination: Path,
    general_report: Path,
    stage_reports: Mapping[str, Path],
) -> Path:
    """Put both metric groups and every generated figure in one browser view."""

    def inline(value: str) -> str:
        value = html.escape(value)
        value = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", value)
        return re.sub(r"`([^`]+)`", r"<code>\1</code>", value)

    sections = []
    for label, report in [
        ("Comparison", general_report),
        *[(key.upper(), value) for key, value in stage_reports.items()],
    ]:
        blocks = []
        table_open = False
        for line in report.read_text(encoding="utf-8").splitlines():
            if line.startswith("|"):
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    continue
                if not table_open:
                    blocks.append('<div class="table-wrap"><table>')
                tag = "td" if table_open else "th"
                blocks.append(
                    "<tr>"
                    + "".join(f"<{tag}>{inline(cell)}</{tag}>" for cell in cells)
                    + "</tr>"
                )
                table_open = True
                continue
            if table_open:
                blocks.append("</table></div>")
                table_open = False
            image_match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line.strip())
            if image_match:
                continue
            elif line.startswith("#"):
                level = min(len(line) - len(line.lstrip("#")) + 1, 6)
                blocks.append(
                    f"<h{level}>{inline(line.lstrip('# ').strip())}</h{level}>"
                )
            elif line.strip():
                blocks.append(f"<p>{inline(line)}</p>")
        if table_open:
            blocks.append("</table></div>")
        sections.append(f'<section id="{label.lower()}">{"".join(blocks)}</section>')
    banner = "Measured benchmark outputs. Missing measurements are shown as N/A."
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    page += "<title>A1 / A2 / A3 / A4 benchmark</title><style>body{margin:0;background:#f3f6fb;color:#142338;font:16px system-ui}header,main{max-width:1280px;margin:auto;padding:28px}header{background:#142338;color:white}nav{display:flex;gap:24px;flex-wrap:wrap}nav a{color:#b9d9ff}section{background:white;margin:24px 0;padding:24px;border-radius:12px}img{width:100%;height:auto;margin:16px 0}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #d9e2ee}th{background:#eaf1fb}code{overflow-wrap:anywhere}p{line-height:1.6;overflow-wrap:anywhere}</style>"
    page += f'<header><h1>A1 / A2 / A3 / A4 benchmark</h1><p>{html.escape(banner)}</p><nav><a href="#comparison">Comparison</a><a href="#a1">A1</a><a href="#a2">A2</a><a href="#a3">A3</a><a href="#a4">A4</a></nav></header><main>{"".join(sections)}</main></html>'
    path = destination / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
