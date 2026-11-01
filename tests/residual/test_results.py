"""Unit tests for the run-artifact writer. No model needed."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
import optax
import pytest
import yaml

if "torchax" not in sys.modules:
    sys.modules["torchax"] = MagicMock()
if "torchax.interop" not in sys.modules:
    sys.modules["torchax.interop"] = MagicMock()

from frozenllm.substrate.architecture import Architecture
from residual import ResidualAdapter, ResidualConfig
from stages.results import (
    RunWriter,
    default_run_name,
    load_checkpoint,
    load_params,
    run_metadata,
)
from stages.stage_base import evaluate_checkpoint, validate_checkpoint_params


def test_writes_all_artifacts(tmp_path: Path) -> None:
    params = {"layer_7": {"A": jnp.ones((4, 2)), "B": jnp.zeros((2, 4))}}
    with RunWriter(tmp_path, "run") as writer:
        writer.write_config({"training": {"steps": 3}, "domains": [{"name": "d"}]})
        writer.write_json("results.json", {"loss": jnp.asarray([[1.5]]), "x": None})
        writer.log("train", step=1, loss=jnp.float32(2.0), ids=np.arange(3))
        writer.log("eval", after_task=0)
        path = writer.save_params("task_0_d", params)

    run = tmp_path / "run"
    assert yaml.safe_load((run / "config.yaml").read_text()) == {
        "training": {"steps": 3},
        "domains": [{"name": "d"}],
    }
    assert json.loads((run / "results.json").read_text()) == {
        "loss": [[1.5]],
        "x": None,
    }
    lines = [
        json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()
    ]
    assert lines == [
        {"event": "train", "step": 1, "loss": 2.0, "ids": [0, 1, 2]},
        {"event": "eval", "after_task": 0},
    ]
    loaded = load_params(path)
    assert set(loaded) == {"layer_7"}
    np.testing.assert_array_equal(loaded["layer_7"]["A"], np.ones((4, 2)))
    np.testing.assert_array_equal(loaded["layer_7"]["B"], np.zeros((2, 4)))
    # Atomic writes leave no temp files behind.
    assert not [p for p in run.iterdir() if p.name.startswith(".")]


def test_refuses_existing_run_dir(tmp_path: Path) -> None:
    RunWriter(tmp_path, "run").close()
    with pytest.raises(FileExistsError):
        RunWriter(tmp_path, "run")


@pytest.mark.parametrize("name", ["", "a/b", "../x"])
def test_rejects_non_plain_run_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="plain directory name"):
        RunWriter(tmp_path, name)


def test_log_after_close_raises(tmp_path: Path) -> None:
    writer = RunWriter(tmp_path, "run")
    writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        writer.log("train")


def test_metadata_and_run_name() -> None:
    meta = run_metadata()
    assert {"started_at", "git_commit", "git_dirty", "python", "versions"} <= set(meta)
    assert default_run_name("a1").startswith("a1-")


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    params = {"layer_7": {"A": jnp.ones((4, 2)), "B": jnp.zeros((2, 4))}}
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    with RunWriter(tmp_path, "run") as writer:
        ckpt_path = writer.save_checkpoint(
            "task_0_eurlex",
            params,
            opt_state=opt_state,
            global_step=42,
            domain="eurlex",
        )

    # 1. Untrusted loading refuses to unpickle __opt_treedef__ and raises ValueError
    with pytest.raises(
        ValueError, match="Refusing to deserialize untrusted '__opt_treedef__'"
    ):
        load_checkpoint(ckpt_path)
    with pytest.raises(
        ValueError, match="Refusing to deserialize untrusted '__opt_treedef__'"
    ):
        load_checkpoint(ckpt_path, trusted=False)

    # 2. Trusted loading restores params, opt_state, and metadata
    loaded_params, loaded_opt, meta = load_checkpoint(ckpt_path, trusted=True)
    assert meta["global_step"] == 42
    assert meta["domain"] == "eurlex"
    np.testing.assert_array_equal(loaded_params["layer_7"]["A"], np.ones((4, 2)))
    np.testing.assert_array_equal(loaded_params["layer_7"]["B"], np.zeros((2, 4)))
    assert loaded_opt is not None

    # Verify optimizer step works with restored opt_state
    grads = {"layer_7": {"A": jnp.zeros((4, 2)), "B": jnp.zeros((2, 4))}}
    updates, next_opt = optimizer.update(grads, loaded_opt, loaded_params)
    assert next_opt is not None

    # 3. Backwards-compatible load_params works on the checkpoint path too
    params_only = load_params(ckpt_path)
    np.testing.assert_array_equal(params_only["layer_7"]["A"], np.ones((4, 2)))


def test_checkpoint_without_optimizer_state_loads_untrusted(tmp_path: Path) -> None:
    params = {"layer_7": {"A": jnp.ones((4, 2)), "B": jnp.zeros((2, 4))}}

    with RunWriter(tmp_path, "run") as writer:
        ckpt_path = writer.save_checkpoint(
            "no_opt",
            params,
            opt_state=None,
            global_step=10,
        )

    # Checkpoint without optimizer state can be loaded safely with trusted=False
    loaded_params, loaded_opt, meta = load_checkpoint(ckpt_path, trusted=False)
    assert loaded_opt is None
    assert meta["global_step"] == 10
    np.testing.assert_array_equal(loaded_params["layer_7"]["A"], np.ones((4, 2)))


def test_checkpoint_malicious_treedef_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Construct a crafted checkpoint with a malicious payload in __opt_treedef__
    malicious_bytes = b"cos\nsystem\n(S'echo exploit_ran'\ntR."
    crafted_path = tmp_path / "malicious.npz"
    np.savez(
        crafted_path,
        **{
            "params/layer_0/w": np.ones((2, 2)),
            "__opt_treedef__": np.frombuffer(malicious_bytes, dtype=np.uint8),
        },
    )

    unpickle_called = []

    def fail_on_unpickle(*args, **kwargs):
        unpickle_called.append(True)
        raise AssertionError("pickle.loads must NOT be called on untrusted checkpoint!")

    monkeypatch.setattr("stages.results.pickle.loads", fail_on_unpickle)

    # Loading without trusted=True must raise ValueError and NOT execute pickle.loads
    with pytest.raises(
        ValueError, match="Refusing to deserialize untrusted '__opt_treedef__'"
    ):
        load_checkpoint(crafted_path)

    assert not unpickle_called


def test_atomic_savez_leaves_no_truncated_file_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stages.results import _atomic_savez

    target = tmp_path / "target.npz"

    def crashing_savez(*args, **kwargs):
        raise OSError("Disk full or simulated crash during write")

    monkeypatch.setattr("stages.results.np.savez", crashing_savez)

    with pytest.raises(IOError, match="simulated crash"):
        _atomic_savez(target, a=np.array([1, 2, 3]))

    # Target path must not exist or be left truncated
    assert not target.exists()
    # Temporary files should be cleaned up
    assert list(tmp_path.glob(".*")) == []


def test_interrupted_pair_params_fallback_to_checkpoint(tmp_path: Path) -> None:
    params = {"layer_7": {"A": jnp.ones((4, 2)), "B": jnp.zeros((2, 4))}}
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    with RunWriter(tmp_path, "run") as writer:
        ckpt_path = writer.save_checkpoint(
            "task_0_dom",
            params,
            opt_state=opt_state,
            global_step=10,
        )

    assert ckpt_path.exists()

    # Simulate an interruption where params/<tag>.npz was deleted/never written
    params_path = tmp_path / "run" / "params" / "task_0_dom.npz"
    assert params_path.exists()
    params_path.unlink()

    # load_params on the missing params path should transparently fall back to checkpoint
    recovered_params = load_params(params_path)
    np.testing.assert_array_equal(recovered_params["layer_7"]["A"], np.ones((4, 2)))
    np.testing.assert_array_equal(recovered_params["layer_7"]["B"], np.zeros((2, 4)))


def test_interrupted_pair_checkpoint_fallback_to_params(tmp_path: Path) -> None:
    params = {"layer_7": {"A": jnp.ones((4, 2)), "B": jnp.zeros((2, 4))}}

    with RunWriter(tmp_path, "run") as writer:
        params_path = writer.save_params("task_0_dom", params)

    # Simulate an interruption where checkpoints/<tag>.npz was never written
    ckpt_path = tmp_path / "run" / "checkpoints" / "task_0_dom.npz"
    assert not ckpt_path.exists()
    assert params_path.exists()

    # load_checkpoint on missing checkpoint path should recover params from companion params archive
    recovered_params, opt_state, meta = load_checkpoint(ckpt_path)
    assert opt_state is None
    assert meta == {}
    np.testing.assert_array_equal(recovered_params["layer_7"]["A"], np.ones((4, 2)))
    np.testing.assert_array_equal(recovered_params["layer_7"]["B"], np.zeros((2, 4)))


def test_validate_checkpoint_params_exact_layers_and_shapes() -> None:
    arch = Architecture(
        model_family="gpt2",
        num_layers=4,
        hidden_size=16,
        num_heads=2,
        head_dim=8,
        vocab_size=32,
    )
    adapter = ResidualAdapter(ResidualConfig(rank=2, layers=[1, 2]), arch)
    params = adapter.init_params()

    # 1. Valid params pass
    validate_checkpoint_params(adapter, params)

    # 2. Missing layer is rejected
    missing_layer_params = {k: v for k, v in params.items() if k != "layer_2"}
    with pytest.raises(ValueError, match="missing layers: \\['layer_2'\\]"):
        validate_checkpoint_params(adapter, missing_layer_params)

    # 3. Extra layer is rejected
    extra_layer_params = {
        **params,
        "layer_3": {
            "A": jnp.zeros((16, 2), dtype=jnp.float32),
            "B": jnp.zeros((2, 16), dtype=jnp.float32),
        },
    }
    with pytest.raises(ValueError, match="extra layers: \\['layer_3'\\]"):
        validate_checkpoint_params(adapter, extra_layer_params)

    # 4. Missing A or B key is rejected
    missing_key_params = {
        "layer_1": {"A": params["layer_1"]["A"]},
        "layer_2": params["layer_2"],
    }
    with pytest.raises(ValueError, match="missing keys: \\['B'\\]"):
        validate_checkpoint_params(adapter, missing_key_params)

    # 5. Extra key in layer is rejected
    extra_key_params = {
        "layer_1": {**params["layer_1"], "C": jnp.zeros((2, 2))},
        "layer_2": params["layer_2"],
    }
    with pytest.raises(ValueError, match="extra keys: \\['C'\\]"):
        validate_checkpoint_params(adapter, extra_key_params)

    # 6. Mismatched A shape is rejected
    wrong_a_params = {
        "layer_1": {"A": jnp.zeros((16, 4)), "B": params["layer_1"]["B"]},
        "layer_2": params["layer_2"],
    }
    with pytest.raises(
        ValueError, match="parameter 'A' has shape \\(16, 4\\), expected \\(16, 2\\)"
    ):
        validate_checkpoint_params(adapter, wrong_a_params)

    # 7. Mismatched B shape is rejected
    wrong_b_params = {
        "layer_1": {"A": params["layer_1"]["A"], "B": jnp.zeros((4, 16))},
        "layer_2": params["layer_2"],
    }
    with pytest.raises(
        ValueError, match="parameter 'B' has shape \\(4, 16\\), expected \\(2, 16\\)"
    ):
        validate_checkpoint_params(adapter, wrong_b_params)


def test_evaluate_checkpoint_fails_fast_on_mismatched_checkpoint() -> None:
    arch = Architecture(
        model_family="gpt2",
        num_layers=4,
        hidden_size=16,
        num_heads=2,
        head_dim=8,
        vocab_size=32,
    )
    adapter = ResidualAdapter(ResidualConfig(rank=2, layers=[1, 2]), arch)
    # Checkpoint omits layer_2
    invalid_checkpoint = {
        "layer_1": {
            "A": np.zeros((16, 2), dtype=np.float32),
            "B": np.zeros((2, 16), dtype=np.float32),
        }
    }
    mock_substrate = MagicMock()
    with pytest.raises(ValueError, match="missing layers: \\['layer_2'\\]"):
        evaluate_checkpoint(mock_substrate, adapter, invalid_checkpoint, domains=[])
