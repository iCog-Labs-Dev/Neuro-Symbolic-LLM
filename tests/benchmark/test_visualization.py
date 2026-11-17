"""Figures use saved artifacts and preserve unrelated report files."""

import pytest

from benchmark.benchmark import _write_json
from benchmark.visualization import generate_visualizations


def test_empty_artifacts_generate_explicit_na_figures(tmp_path):
    pytest.importorskip("matplotlib")
    runs = tmp_path / "runs"
    reports = tmp_path / "reports"
    reports.mkdir()
    sentinel = reports / "keep.txt"
    sentinel.write_text("keep")
    _write_json(runs / "manifest.json", {})
    _write_json(runs / "evaluation.json", {"records": []})
    generate_visualizations(runs, reports)
    assert sentinel.read_text() == "keep"
    for section in ("A1", "A2", "A3", "comparison"):
        folder = reports / "stage_A" / section
        assert len(list((folder / "figures").glob("*.png"))) == 13
        assert "![" in next(folder.glob("*.md")).read_text()
    assert "<img" in (reports / "index.html").read_text()
