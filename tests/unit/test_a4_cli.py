"""Stage A4 command-line validation."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from experiments.run_stage_a4 import DEFAULT_CONFIG, _resolve_substrate, parse_args


def test_reuses_substrate_without_loading(monkeypatch):
    constructor = Mock(side_effect=AssertionError("Unexpected model loading"))
    monkeypatch.setitem(
        sys.modules, "frozenllm.substrate", SimpleNamespace(FrozenSubstrate=constructor)
    )
    existing = Mock()
    assert _resolve_substrate("configured-model", existing) is existing
    constructor.assert_not_called()


def test_standalone_run_loads_one_substrate(monkeypatch):
    existing = Mock()
    constructor = Mock(return_value=existing)
    dtype = object()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32=dtype))
    monkeypatch.setitem(
        sys.modules, "frozenllm.substrate", SimpleNamespace(FrozenSubstrate=constructor)
    )
    assert _resolve_substrate("configured-model", None) is existing
    constructor.assert_called_once_with("configured-model", dtype=dtype)


def arguments(*extra):
    return [
        "--warm-start",
        "a3.npz",
        "--train-sources",
        "train.jsonl",
        "--validation-sources",
        "validation.jsonl",
        "--alignment-labels",
        "labels.json",
        "--patterns",
        "patterns.jsonl",
        "--embeddings",
        "embeddings.jsonl",
        "--retrieval-backend",
        "tcp",
        "--skip-test",
        *extra,
    ]


def test_defaults_use_a4_configuration():
    args = parse_args(arguments())
    assert args.config == DEFAULT_CONFIG
    assert args.epochs is None
    assert not hasattr(args, "compare")
    assert not hasattr(args, "comparison_steps")


@pytest.mark.parametrize("required", ["--warm-start", "--alignment-labels"])
def test_requires_checkpoint_and_firing_labels(required):
    argv = arguments()
    index = argv.index(required)
    del argv[index : index + 2]
    with pytest.raises(SystemExit):
        parse_args(argv)


@pytest.mark.parametrize(
    "extra",
    [
        ("--epochs", "-1"),
        ("--batch-size", "0"),
        ("--seq-len", "1"),
        ("--learning-rate", "nan"),
        ("--max-grad-norm", "0"),
        ("--max-train-blocks", "0"),
        ("--minimum-examples", "0"),
        ("--eval-batches", "0"),
        ("--task-only",),
        ("--comparison-steps", "10"),
        ("--steps", "10"),
    ],
)
def test_rejects_invalid_or_nontraining_options(extra):
    with pytest.raises(SystemExit):
        parse_args(arguments(*extra))


def test_test_split_is_explicit():
    argv = arguments()
    argv.remove("--skip-test")
    with pytest.raises(SystemExit):
        parse_args(argv)
    assert parse_args([*argv, "--test-sources", "test.jsonl"]).test_sources == Path(
        "test.jsonl"
    )


def test_native_requires_library_and_manifest():
    with pytest.raises(SystemExit):
        parse_args(arguments("--retrieval-backend", "native"))


def test_help_does_not_import_numeric_runtime():
    path = Path(__file__).resolve().parents[2] / "experiments/run_stage_a4.py"
    code = (
        "import runpy, sys\n"
        "class Guard:\n"
        " def find_spec(self, fullname, path=None, target=None):\n"
        "  if fullname.split('.')[0] in {'jax','torch','numpy','optax'}:\n"
        "   raise AssertionError(fullname)\n"
        "sys.meta_path.insert(0, Guard())\n"
        f"sys.argv = [{str(path)!r}, '--help']\n"
        f"runpy.run_path({str(path)!r}, run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "--alignment-labels" in result.stdout


def test_inspection_tools_are_outside_experiments():
    root = Path(__file__).resolve().parents[2]
    for name in (
        "audit_mining_depth",
        "benchmark_mining_depth",
        "compare_a3_a4",
        "mine_reference_shards",
        "analyze_a4_run",
    ):
        assert not (root / "experiments" / f"{name}.py").exists()
