"""Reports expose exactly the agreed metrics, without importing plotting."""

from benchmark.benchmark import (
    EXTERNAL_METRIC_SPECS,
    INTERNAL_METRIC_SPECS,
    MEASURED_METRICS,
    ROOT,
    _write_json,
    resolve_report_dir,
)
from benchmark.summary import report_manifest


def test_report_exact_metrics_and_missing_evidence(tmp_path):
    runs = tmp_path / "runs"
    reports = tmp_path / "reports"
    _write_json(runs / "manifest.json", {"runtime_source": "stories"})
    _write_json(runs / "evaluation.json", {"records": []})
    path = report_manifest(runs, report_dir=reports)
    content = path.read_text()
    assert len(MEASURED_METRICS) == 13
    assert len(INTERNAL_METRIC_SPECS) == 5
    assert len(EXTERNAL_METRIC_SPECS) == 8
    for label, _, _ in INTERNAL_METRIC_SPECS + EXTERNAL_METRIC_SPECS:
        assert label in content
    assert "No evaluated runs" in content
    assert "![" not in content
    assert not (runs / "REPORT.md").exists()
    assert (reports / "index.html").is_file()
    assert not any(
        word in content
        for word in (
            "Effective rank",
            "Peak memory",
            "Trainable parameters",
            "Curvature",
        )
    )


def test_report_dir_resolves_from_repo(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert resolve_report_dir() == ROOT / "docs/stage_A_results"
    assert resolve_report_dir("custom") == ROOT / "custom"


def test_effect_magnitudes_are_not_prediction_fidelity():
    from benchmark.benchmark import measurement_reason, metric_values_for_record

    record = {
        "deployed_evaluation": {
            "representation": {"adapter_ablations": {"mean_loss_increase": 1.5}}
        }
    }
    for key in ("ablation_mae", "ablation_spearman"):
        assert metric_values_for_record(record, key, 16) == []
        assert "predictor" in measurement_reason(record, key)
    for key in ("adapter_ablation", "causal_fingerprint", "cross_adapter_curvature"):
        assert metric_values_for_record(record, key, 16) == []


def test_summary_has_no_plotting_dependency():
    import ast
    from pathlib import Path

    import benchmark.summary as summary

    tree = ast.parse(Path(summary.__file__).read_text())
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all(
        "matplotlib" not in ast.unparse(node)
        and "visualization" not in ast.unparse(node)
        for node in imports
    )
