"""Stage A4 command-line validation."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from experiments.run_stage_a4 import (
    DEFAULT_CONFIG,
    DEFAULT_EMBEDDINGS,
    DEFAULT_PATTERNS,
    DEFAULT_RUNTIME_CONFIG,
    _apply_training_overrides,
    _resolve_substrate,
    find_latest_a3_checkpoint,
    parse_args,
)


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
    native, tokenizer = Mock(), Mock()
    model_loader = Mock(return_value=native)
    tokenizer_loader = Mock(return_value=tokenizer)
    constructor = Mock(return_value=existing)
    dtype = object()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32=dtype))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_loader),
            AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "frozenllm.substrate", SimpleNamespace(FrozenSubstrate=constructor)
    )
    assert _resolve_substrate("configured-model", None) is existing
    model_loader.assert_called_once_with(
        "configured-model", dtype=dtype, attn_implementation="eager"
    )
    tokenizer_loader.assert_called_once_with("configured-model")
    constructor.assert_called_once_with(native, tokenizer=tokenizer)


def arguments(*extra):
    return [
        "--train-sources",
        "train.jsonl",
        "--validation-sources",
        "validation.jsonl",
        "--alignment-labels",
        "labels.json",
        "--retrieval-backend",
        "tcp",
        "--skip-test",
        *extra,
    ]


def test_defaults_use_a4_configuration():
    args = parse_args(arguments())
    assert args.config == DEFAULT_CONFIG
    assert args.runtime_config == DEFAULT_RUNTIME_CONFIG
    assert args.data_mode == "runtime"
    assert args.warm_start is None
    assert args.patterns == DEFAULT_PATTERNS
    assert args.embeddings == DEFAULT_EMBEDDINGS
    assert args.epochs is None
    assert not args.skip_latency
    assert parse_args(arguments("--skip-latency")).skip_latency
    assert args.latency_every == 100
    assert parse_args(arguments("--latency-every", "20")).latency_every == 20
    assert parse_args(arguments("--log-every", "1")).log_every == 1
    assert not hasattr(args, "compare")
    assert not hasattr(args, "comparison_steps")


def test_effective_seed_and_epochs_round_trip(tmp_path):
    import yaml

    from stages.stage_A.a4.config import load_config

    args = parse_args(arguments("--seed", "879", "--epochs", "2", "--rank", "16"))
    config = _apply_training_overrides(load_config(args.config), args)
    assert config.training.seed == 879
    assert config.training.epochs == 2
    recorded = tmp_path / "recorded.yaml"
    recorded.write_text(yaml.safe_dump(config.to_dict()))
    restored = load_config(recorded, recorded=True)
    assert restored == config


def test_accepts_checkpoint_and_bank_overrides():
    args = parse_args(
        arguments(
            "--warm-start",
            "a3.npz",
            "--patterns",
            "patterns.jsonl",
            "--embeddings",
            "embeddings.jsonl",
        )
    )
    assert args.warm_start == Path("a3.npz")
    assert args.patterns == Path("patterns.jsonl")
    assert args.embeddings == Path("embeddings.jsonl")


def test_requires_firing_labels_in_aligned_mode():
    argv = arguments("--data-mode", "aligned")
    index = argv.index("--alignment-labels")
    del argv[index : index + 2]
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_finds_newest_completed_a3_checkpoint(tmp_path):
    def checkpoint(run, name, metadata, timestamp):
        run_dir = tmp_path / run
        path = run_dir / "checkpoints" / name
        path.parent.mkdir(parents=True)
        path.touch()
        (run_dir / "meta.json").write_text(metadata, encoding="utf-8")
        os.utime(path, (timestamp, timestamp))
        return path

    final = "a3hybrid_final.npz"
    a3 = '{"stage": "A3"}'
    checkpoint("a3hybrid-old", final, a3, 1)
    latest = checkpoint("a3hybrid-new", final, a3, 2)
    checkpoint("a3hybrid-incomplete", "task_0_stories.npz", a3, 3)
    checkpoint("a3hybrid-wrong-stage", final, '{"stage": "A2"}', 4)
    checkpoint("a3hybrid-invalid", final, "[]", 5)
    assert find_latest_a3_checkpoint(tmp_path) == latest


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
        ("--log-every", "0"),
        ("--latency-every", "0"),
        ("--task-only",),
        ("--comparison-steps", "10"),
        ("--steps", "10"),
    ],
)
def test_rejects_invalid_or_nontraining_options(extra):
    with pytest.raises(SystemExit):
        parse_args(arguments(*extra))


def test_test_split_is_explicit():
    argv = arguments("--data-mode", "aligned")
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
    path = Path(__file__).resolve().parents[3] / "experiments/run_stage_a4.py"
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
    root = Path(__file__).resolve().parents[3]
    for name in (
        "audit_mining_depth",
        "benchmark_mining_depth",
        "compare_a3_a4",
        "mine_reference_shards",
        "analyze_a4_run",
    ):
        assert not (root / "experiments" / f"{name}.py").exists()
