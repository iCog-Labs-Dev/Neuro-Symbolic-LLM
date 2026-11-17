"""Checkpoint evaluation contracts on tiny analytic models, not benchmark scores."""

import sys
from types import ModuleType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmark.benchmark import (
    _load_deployed_metrics,
    measure_ablation_prediction,
    measurement_reason,
    metric_values_for_record,
)


def test_quadratic_gate_predictor_and_true_ablations_are_independent():
    params = {"layer_0": {"A": jnp.eye(2), "B": jnp.array([[1.0], [2.0]])}}
    original = np.asarray(params["layer_0"]["B"]).copy()
    ids = jnp.array([[0, 1, 1], [1, 0, 0]])

    def loss(p, batch):
        # Sequence gradients differ, exposing an incorrect square-of-mean Fisher.
        return 10 + (1 + batch[0, 0]) * p["layer_0"]["B"].sum()

    audit = measure_ablation_prediction(loss, params, {"shared_test": (ids,)})
    np.testing.assert_allclose([p["gradient"] for p in audit["pairs"]], [1.5, 3])
    np.testing.assert_allclose(
        [p["fisher_diagonal"] for p in audit["pairs"]], [2.5, 10]
    )
    np.testing.assert_allclose([p["predicted"] for p in audit["pairs"]], [-0.25, 2])
    np.testing.assert_allclose([p["actual"] for p in audit["pairs"]], [-1.5, -3])
    assert audit["mae"] == pytest.approx(3.125)
    assert audit["spearman"] == pytest.approx(-1)
    split = measure_ablation_prediction(
        loss, params, {"shared_test": (ids[:1], ids[1:])}
    )
    assert split["pairs"] == audit["pairs"]
    np.testing.assert_array_equal(params["layer_0"]["B"], original)


def test_constant_ablation_effects_have_mae_but_no_spearman():
    params = {"layer_0": {"A": jnp.eye(2), "B": jnp.zeros((2, 2))}}
    audit = measure_ablation_prediction(
        lambda p, ids: jnp.sum(p["layer_0"]["B"]),
        params,
        {"test": (jnp.ones((1, 3), dtype=jnp.int32),)},
    )
    assert audit["mae"] == 0
    assert audit["spearman"] is None
    record = {"deployed_evaluation": {"ablation_fidelity": audit}}
    assert metric_values_for_record(record, "ablation_mae", 2) == [0]
    assert metric_values_for_record(record, "ablation_spearman", 2) == []
    assert "nonconstant" in measurement_reason(record, "ablation_spearman")


def test_ablation_sampling_and_missing_test_data():
    params = {"layer_0": {"A": jnp.ones((1, 4)), "B": jnp.arange(4.0)[:, None]}}

    def loss(p, ids):
        return jnp.sum(p["layer_0"]["B"])

    with pytest.raises(ValueError, match="held-out"):
        measure_ablation_prediction(loss, params, {})
    with pytest.raises(ValueError, match="finite"):
        measure_ablation_prediction(
            lambda p, ids: loss(p, ids) * jnp.nan,
            params,
            {"test": (jnp.ones((1, 3)),)},
        )
    audit = measure_ablation_prediction(
        loss, params, {"test": (jnp.ones((1, 3)),)}, max_components=2
    )
    assert [row["channel"] for row in audit["pairs"]] == [0, 3]


@pytest.mark.parametrize("method", ["a1", "a2", "a3"])
@pytest.mark.parametrize("source", ["stories", "separate"])
def test_checkpoint_evaluation_respects_loader_test_keys(
    monkeypatch, tmp_path, method, source
):
    """Exercise the actual evaluator without downloading a model or datasets."""

    def module(name, **members):
        value = ModuleType(name)
        value.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, value)

    class Substrate:
        def __init__(self, model):
            self.tokenizer = object()
            self.architecture = object()

        def run_with_interception(self, ids, modify_fn, intercept_layers):
            hidden = jax.nn.one_hot(ids, 2)
            logits = modify_fn(hidden, 0)
            return SimpleNamespace(logits=logits, hidden_state=lambda layer: hidden)

        def compute_loss(self, logits, ids, losses):
            probs = jax.nn.log_softmax(logits[:, :-1], axis=-1)
            return -jnp.take_along_axis(probs, ids[:, 1:, None], axis=-1).mean()

    class Adapter:
        layers = (0,)
        scale = 1.0

        def __init__(self, config, architecture):
            pass

        def validate_params(self, params):
            assert params["layer_0"]["B"].shape == (2, 2)

        def encode(self, block, hidden):
            return hidden @ block["A"]

        def modify_fn(self, p):
            return (
                lambda hidden, layer: hidden
                + self.encode(p["layer_0"], hidden) @ p["layer_0"]["B"]
            )

    ids = jnp.array([[0, 1, 0], [1, 0, 1]])
    names = (
        ["rocstories_task_1", "rocstories_task_2"]
        if source == "stories"
        else ["left", "right"]
    )
    test_sets = (
        {"rocstories": (ids,)} if source == "stories" else dict.fromkeys(names, (ids,))
    )
    tasks = [
        SimpleNamespace(name=name, train_blocks=np.asarray([[i, i, 1 - i]]))
        for i, name in enumerate(names)
    ]
    runtime = SimpleNamespace(tasks=tasks, get_test_sets=lambda *args: test_sets)
    calls = []

    def load_runtime_data(**kwargs):
        calls.append(kwargs)
        return runtime

    module(
        "frozenllm.substrate",
        FrozenSubstrate=Substrate,
        identity_modify=lambda h, layer: h,
    )
    module(
        "residual",
        ResidualAdapter=Adapter,
        ResidualConfig=SimpleNamespace(from_dict=lambda c: c),
    )
    module(
        "runtime.data_loader",
        load_runtime_data=load_runtime_data,
        load_runtime_config=lambda path: {},
    )
    module(
        "stages.stage_base",
        TrainingConfig=SimpleNamespace(
            from_dict=lambda c: SimpleNamespace(
                model="tiny", seq_len=3, batch_size=2, eval_batches=1, epochs=1
            )
        ),
        make_eval_set=lambda blocks, *args: (jnp.asarray(blocks),),
    )
    module(
        "stages.results",
        load_params=lambda path: {
            "layer_0": {"A": jnp.eye(2), "B": jnp.array([[0.2, -0.1], [0.4, 0.3]])}
        },
    )
    config = tmp_path / "config.yaml"
    config.write_text("adapter: {}\n")
    checkpoint = tmp_path / "attempt" / "params"
    checkpoint.mkdir(parents=True)
    (checkpoint / "task_1_finished.npz").touch()
    run = {
        "method": method,
        "config": str(config),
        "materialized_runtime_config": str(config),
        "runtime_source": source,
        "task_order": names,
        "output_dir": str(tmp_path),
        "run_name": "attempt",
    }
    measured = _load_deployed_metrics(run, {})
    assert measured["status"] == "measured"
    assert set(measured["domains"]) == set(test_sets)
    assert calls[0]["task_order"] == names
    conflict = measured["representation"]["task_gradient_conflict"]
    assert conflict["status"] == "measured"
    assert list(conflict["evaluation_batches"]) == names
    assert conflict["split"].startswith("train")
    redundancy = measured["representation"]["residual_direction_redundancy"]["layer_0"]
    assert redundancy["read"] == pytest.approx(0)
    assert redundancy["write"] == pytest.approx(1 / np.sqrt(5))
    assert measured["ablation_fidelity"]["status"] == "measured"
    assert measured["ablation_fidelity"]["pairs"]
    record = {"deployed_evaluation": measured}
    for key in (
        "token_accuracy",
        "calibration",
        "domain_kl",
        "residual_direction_redundancy",
        "task_gradient_conflict",
        "ablation_mae",
        "ablation_spearman",
    ):
        assert metric_values_for_record(record, key, 2), key
    assert metric_values_for_record(record, "invariant_redundancy", 2) == []
