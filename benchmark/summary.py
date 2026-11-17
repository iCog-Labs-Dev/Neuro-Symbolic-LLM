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
    fingerprint,
    measurement_reason,
    metric_values_for_record,
    resolve_report_dir,
)


def _cell(values: list[float]) -> str:
    return f"{fmean(values):.6g} (n={len(values)})" if values else "N/A"


def _tables(records: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Run conditions",
        "",
        "| Run | Protocol | Seed | Rank | K | Realized task order | Held-out tasks |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['run_id']} | {record.get('protocol', 'N/A')} | {record.get('seed', 'N/A')} | {record.get('rank', 'N/A')} | {record.get('k', 'N/A')} | {' → '.join(record.get('task_order', []))} | {', '.join(record.get('heldout_tasks', []))} |"
        )
    lines.append("")
    for category, specs in (
        ("Internal", INTERNAL_METRIC_SPECS),
        ("External", EXTERNAL_METRIC_SPECS),
    ):
        lines += [
            f"## {category} Metrics",
            "",
            "| Run | Metric | Value | Missing measurement reason |",
            "|---|---|---|---|",
        ]
        for record in records:
            for label, key, _ in specs:
                values = metric_values_for_record(
                    record, key, int(record.get("rank", 16))
                )
                reason = "" if values else measurement_reason(record, key)
                lines.append(
                    f"| {record['run_id']} | {label} | {_cell(values)} | {reason.replace('|', '/')} |"
                )
        if not records:
            for label, _, _ in specs:
                lines.append(f"| N/A | {label} | N/A | No evaluated runs |")
        lines.append("")
    lines += [
        "## Per-task token accuracy",
        "",
        "Next-token prediction accuracy on each task's test split; this is not labeled task accuracy.",
        "",
        "| Run | Task | Token accuracy |",
        "|---|---|---|",
    ]
    for record in records:
        for task, measurement in (
            record.get("deployed_evaluation", {}).get("domains", {}).items()
        ):
            value = measurement.get("token_accuracy")
            cell = (
                _cell([value])
                if value is not None and measurement.get("status") == "measured"
                else "N/A"
            )
            lines.append(f"| {record['run_id']} | {task} | {cell} |")
    lines.append("")
    return lines


def _figure_links(folder: Path) -> list[str]:
    # Link only the figures this version owns, not unrelated/stale PNGs.
    return [
        f"![{label}](figures/{key}.png)"
        for label, key, _ in INTERNAL_METRIC_SPECS + EXTERNAL_METRIC_SPECS
        if (folder / "figures" / f"{key}.png").is_file()
    ]


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
        f"Realized task orders: {[r.get('task_order', []) for r in records]}",
        "",
        "Only measured values are reported; missing evidence is N/A.",
        "",
    ]
    if cohort is None or not cohort.matched_seeds:
        lines += ["No complete matched A1/A2/A3 cohort is available.", ""]
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
            "| Metric | A1 | A2 | A3 | A2-A1 paired delta | A3-A1 paired delta |",
            "|---|---|---|---|---|---|",
        ]
        for label, key, _ in specs:
            samples = (
                _metric_samples(cohort, key)
                if cohort
                else {method: [] for method in METHODS}
            )
            columns = [
                _cell([value for _, value in samples[method]]) for method in METHODS
            ]
            base = dict(samples["a1"])
            columns += [
                _cell(
                    [
                        value - base[seed]
                        for seed, value in samples[method]
                        if seed in base
                    ]
                )
                for method in ("a2", "a3")
            ]
            lines.append(f"| {label} | " + " | ".join(columns) + " |")
        lines.append("")
    lines += [
        "## Per-run evidence and N/A reasons",
        "",
        *_tables(records),
        *_figure_links(comparison),
    ]
    general = comparison / "stage_a_general_report.md"
    general.write_text("\n".join(lines) + "\n", encoding="utf-8")
    reports = {}
    for method in METHODS:
        folder = stage_root / method.upper()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{method}_result_and_visualization.md"
        selected = [row for row in records if row.get("method") == method]
        content = [
            f"# {method.upper()} Results",
            "",
            *_tables(selected),
            *_figure_links(folder),
        ]
        for row in selected:
            relative = (
                f"figures/validation_loss_heatmap_{fingerprint(row['run_id'])[:16]}.png"
            )
            if (folder / relative).is_file():
                content.append(f"![Validation loss heatmap]({relative})")
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
                key for _, key, _ in INTERNAL_METRIC_SPECS + EXTERNAL_METRIC_SPECS
            ],
            "runs": [
                {
                    "run_id": row["run_id"],
                    "task_order": row.get("task_order", []),
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
                        for _, key, _ in INTERNAL_METRIC_SPECS + EXTERNAL_METRIC_SPECS
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
                alt, relative = image_match.groups()
                source = (report.parent / relative).resolve()
                url = source.relative_to(destination).as_posix()
                blocks.append(
                    f'<img loading="lazy" src="{html.escape(url, quote=True)}" alt="{html.escape(alt, quote=True)}">'
                )
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
    page += "<title>A1 / A2 / A3 benchmark</title><style>body{margin:0;background:#f3f6fb;color:#142338;font:16px system-ui}header,main{max-width:1280px;margin:auto;padding:28px}header{background:#142338;color:white}nav{display:flex;gap:24px;flex-wrap:wrap}nav a{color:#b9d9ff}section{background:white;margin:24px 0;padding:24px;border-radius:12px}img{width:100%;height:auto;margin:16px 0}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #d9e2ee}th{background:#eaf1fb}code{overflow-wrap:anywhere}p{line-height:1.6;overflow-wrap:anywhere}</style>"
    page += f'<header><h1>A1 / A2 / A3 benchmark</h1><p>{html.escape(banner)}</p><nav><a href="#comparison">Comparison</a><a href="#a1">A1</a><a href="#a2">A2</a><a href="#a3">A3</a></nav></header><main>{"".join(sections)}</main></html>'
    path = destination / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
