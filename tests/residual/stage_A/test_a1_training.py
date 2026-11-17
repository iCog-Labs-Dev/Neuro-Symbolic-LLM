"""Stage A1 integration tests on the GPT-2 and Pythia fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch

from frozenllm.substrate import FrozenSubstrate
from metrics.performance import cross_entropy_loss, kl_to_base
from residual import AdapterParams, ResidualAdapter, ResidualConfig, layer_key
from runtime.data_loader import load_runtime_config
from stages.stage_A.a1 import A1Config, a1_objective
from stages.stage_base import base_logits, load_configs
from tests.frozenllm.conftest import BATCH, NUM_TOKENS, SEQ, make_substrate

FAMILIES = ("gpt2", "neox")
REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module", params=FAMILIES)
def substrate(request: pytest.FixtureRequest) -> FrozenSubstrate:
    sub: FrozenSubstrate = make_substrate(request.param)[1]
    return sub


@pytest.fixture(scope="module")
def adapter(substrate: FrozenSubstrate) -> ResidualAdapter:
    return ResidualAdapter(ResidualConfig(rank=4), substrate.architecture)


@pytest.fixture(scope="module")
def input_ids() -> jax.Array:
    rng = np.random.default_rng(0)
    return jnp.asarray(
        rng.integers(0, NUM_TOKENS, (BATCH, SEQ)),
        dtype=jnp.int32,
    )


def _with_random_b(
    params: AdapterParams,
    scale: float = 0.05,
) -> AdapterParams:
    """Initialize nonzero B matrices so A receives gradients."""
    rng = np.random.default_rng(1)
    return {
        key: {
            "A": value["A"],
            "B": jnp.asarray(
                scale * rng.normal(size=value["B"].shape),
                jnp.float32,
            ),
        }
        for key, value in params.items()
    }


def _task_loss(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    ids: jax.Array,
) -> Any:
    def loss(params: AdapterParams) -> jax.Array:
        logits = substrate.run_with_interception(
            ids,
            modify_fn=adapter.modify_fn(params),
            intercept_layers=adapter.layers,
        ).logits
        return cross_entropy_loss(logits, ids)

    return loss


# ---------------------------------------------------------------------
# Identity at initialization and pristine intermediates
# ---------------------------------------------------------------------


def test_fixture_resolves_late_layers(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
) -> None:
    num_layers = substrate.architecture.num_layers
    assert adapter.layers == tuple(range(num_layers // 2 + 1, num_layers - 1))
    assert adapter.hidden_size == substrate.architecture.hidden_size


def test_initial_logits_equal_base(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    input_ids: jax.Array,
) -> None:
    base = substrate.run_with_interception(input_ids).logits

    adapted = substrate.run_with_interception(
        input_ids,
        modify_fn=adapter.modify_fn(adapter.init_params()),
        intercept_layers=adapter.layers,
    ).logits

    np.testing.assert_allclose(
        np.asarray(adapted),
        np.asarray(base),
        atol=1e-5,
    )
    assert float(kl_to_base(base, adapted)) == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_intermediates_are_pristine(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    input_ids: jax.Array,
) -> None:
    params = _with_random_b(adapter.init_params(), scale=0.5)
    inner = adapter.modify_fn(params)
    seen: dict[int, tuple[jax.Array, jax.Array]] = {}

    def recording(
        hidden: jax.Array,
        layer_idx: int,
    ) -> jax.Array:
        out = inner(hidden, layer_idx)
        seen[layer_idx] = (hidden, out)
        return out

    result = substrate.run_with_interception(
        input_ids,
        modify_fn=recording,
        intercept_layers=adapter.layers,
    )

    base = substrate.run_with_interception(
        input_ids,
        intercept_layers=adapter.layers,
    )

    assert result.layer_indices() == adapter.layers

    for idx in adapter.layers:
        h_in, h_out = seen[idx]
        np.testing.assert_array_equal(
            np.asarray(result.hidden_state(idx)),
            np.asarray(h_in),
        )
        assert not np.allclose(
            np.asarray(result.hidden_state(idx)),
            np.asarray(h_out),
        )

    first = adapter.layers[0]

    np.testing.assert_allclose(
        np.asarray(result.hidden_state(first)),
        np.asarray(base.hidden_state(first)),
        atol=1e-6,
    )


# ---------------------------------------------------------------------
# Gradient path
# ---------------------------------------------------------------------


def test_grad_b_nonzero_and_grad_a_zero_at_init(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    input_ids: jax.Array,
) -> None:
    grads = jax.grad(_task_loss(substrate, adapter, input_ids))(adapter.init_params())

    for idx in (adapter.layers[0], adapter.layers[-1]):
        grad = grads[layer_key(idx)]

        assert np.all(np.isfinite(np.asarray(grad["B"])))
        assert float(jnp.abs(grad["B"]).max()) > 0.0
        assert float(jnp.abs(grad["A"]).max()) == 0.0

    assert substrate.params_unchanged()


def test_grad_a_nonzero_after_b_set(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    input_ids: jax.Array,
) -> None:
    params = _with_random_b(adapter.init_params())
    grads = jax.grad(_task_loss(substrate, adapter, input_ids))(params)

    for idx in (adapter.layers[0], adapter.layers[-1]):
        grad_a = np.asarray(grads[layer_key(idx)]["A"])
        assert np.all(np.isfinite(grad_a))
        assert np.abs(grad_a).max() > 0.0

    assert substrate.params_unchanged()


def test_grad_a_matches_finite_difference(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    input_ids: jax.Array,
) -> None:
    loss = _task_loss(substrate, adapter, input_ids)
    params = _with_random_b(adapter.init_params())

    key = layer_key(adapter.layers[0])
    grad_a = jax.grad(loss)(params)[key]["A"]

    i, j = np.unravel_index(
        int(jnp.argmax(jnp.abs(grad_a))),
        grad_a.shape,
    )

    eps = 0.1

    def shifted(delta: float) -> float:
        a = params[key]["A"].at[i, j].add(delta)
        return float(
            loss(
                {
                    **params,
                    key: {
                        **params[key],
                        "A": a,
                    },
                }
            )
        )

    finite_difference = (shifted(eps) - shifted(-eps)) / (2 * eps)

    np.testing.assert_allclose(
        finite_difference,
        float(grad_a[i, j]),
        rtol=2e-2,
        atol=1e-4,
    )


def test_substrate_params_receive_no_gradient(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    input_ids: jax.Array,
) -> None:
    params = _with_random_b(adapter.init_params())
    jax.grad(_task_loss(substrate, adapter, input_ids))(params)

    for param in substrate.params.values():
        assert not param.requires_grad
        assert param.grad is None


# ---------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------


def test_objective_terms_at_init(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    input_ids: jax.Array,
) -> None:
    params = adapter.init_params()
    cfg = A1Config(lambda_kl=0.5, lambda_wd=0.1)

    total, terms = a1_objective(
        params,
        substrate=substrate,
        adapter=adapter,
        config=cfg,
        input_ids=input_ids,
    )

    base = base_logits(substrate, input_ids)

    assert set(terms) == {"total", "task", "kl", "wd"}

    np.testing.assert_allclose(
        float(terms["task"]),
        float(cross_entropy_loss(base, input_ids)),
        rtol=1e-6,
    )

    assert float(terms["kl"]) == pytest.approx(0.0, abs=1e-6)

    np.testing.assert_allclose(
        float(terms["wd"]),
        float(ResidualAdapter.l2_norm_sq(params)),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        float(total),
        float(terms["task"] + 0.5 * terms["kl"] + 0.1 * terms["wd"]),
        rtol=1e-6,
    )


def test_objective_kl_and_wd_contribute_gradients(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    input_ids: jax.Array,
) -> None:
    params = _with_random_b(adapter.init_params())

    def grad_b(cfg: A1Config) -> np.ndarray:
        (_, _), grads = jax.value_and_grad(
            a1_objective,
            has_aux=True,
        )(
            params,
            substrate=substrate,
            adapter=adapter,
            config=cfg,
            input_ids=input_ids,
        )

        return np.asarray(grads[layer_key(adapter.layers[0])]["B"])

    plain = grad_b(A1Config())

    assert not np.allclose(
        grad_b(A1Config(lambda_kl=1.0)),
        plain,
    )

    assert not np.allclose(
        grad_b(A1Config(lambda_wd=1.0)),
        plain,
    )


def test_objective_skips_base_forward_when_kl_unused(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    input_ids: jax.Array,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = _with_random_b(adapter.init_params())
    cfg = A1Config(lambda_wd=0.1)

    with_kl, with_kl_terms = a1_objective(
        params,
        substrate=substrate,
        adapter=adapter,
        config=cfg,
        input_ids=input_ids,
        base=base_logits(substrate, input_ids),
    )

    def no_base(*args: Any, **kwargs: Any) -> jax.Array:
        raise AssertionError("base forward must be skipped when lambda_kl == 0")

    monkeypatch.setattr(
        "stages.stage_A.a1.a1.base_logits",
        no_base,
    )

    total, terms = a1_objective(
        params,
        substrate=substrate,
        adapter=adapter,
        config=cfg,
        input_ids=input_ids,
    )

    assert set(terms) == {"total", "task", "wd"}
    assert set(with_kl_terms) == {"total", "task", "kl", "wd"}

    np.testing.assert_allclose(
        float(total),
        float(with_kl),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        float(terms["task"]),
        float(with_kl_terms["task"]),
    )


# ---------------------------------------------------------------------
# Optimizer isolation and training
# ---------------------------------------------------------------------


def test_optimizer_state_holds_only_adapter_leaves(
    adapter: ResidualAdapter,
) -> None:
    params = adapter.init_params()
    opt_state = optax.adam(1e-3).init(params)

    adam_state = opt_state[0]
    params_def = jax.tree_util.tree_structure(params)

    assert isinstance(adam_state, optax.ScaleByAdamState)
    assert jax.tree_util.tree_structure(adam_state.mu) == params_def
    assert jax.tree_util.tree_structure(adam_state.nu) == params_def

    leaves = jax.tree_util.tree_leaves(opt_state)

    assert not any(isinstance(leaf, torch.Tensor) for leaf in leaves)

    assert sum(int(np.size(leaf)) for leaf in leaves) == (
        2 * ResidualAdapter.num_params(params) + 1
    )


def test_training_moves_only_the_adapter(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    input_ids: jax.Array,
) -> None:
    assert substrate.verify_frozen()["params_unchanged"]

    params = adapter.init_params()
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)
    grad_fn = jax.value_and_grad(a1_objective, has_aux=True)
    cfg = A1Config()

    task_losses = []

    for _ in range(5):
        (_, terms), grads = grad_fn(
            params,
            substrate=substrate,
            adapter=adapter,
            config=cfg,
            input_ids=input_ids,
        )

        assert jax.tree_util.tree_structure(grads) == jax.tree_util.tree_structure(
            params
        )

        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            params,
        )

        params = optax.apply_updates(params, updates)
        task_losses.append(float(terms["task"]))

    (_, final), _ = grad_fn(
        params,
        substrate=substrate,
        adapter=adapter,
        config=cfg,
        input_ids=input_ids,
        base=base_logits(substrate, input_ids),
    )

    assert float(final["task"]) < task_losses[0]
    assert float(final["kl"]) > 0.0

    for idx in adapter.layers:
        assert float(jnp.abs(params[layer_key(idx)]["B"]).max()) > 0.0

    assert substrate.params_unchanged()
    assert substrate.verify_frozen()["params_unchanged"]


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lambda_kl": -0.1},
        {"lambda_wd": -1.0},
        {"lambda_kl": "big"},
    ],
)
def test_a1_config_rejects_invalid(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        A1Config(**kwargs)  # type: ignore[arg-type]


def test_a1_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="Unknown A1Config keys"):
        A1Config.from_dict(
            {
                "lambda_kl": 0.1,
                "lambda_pc": 1.0,
            }
        )


def test_shipped_yaml_loads_and_rejects_unknown_keys(
    tmp_path: Path,
) -> None:
    cfg = load_configs(
        REPO_ROOT / "configs" / "stage_A" / "a1_lora_baseline.yaml",
        A1Config,
    )

    assert cfg.adapter == ResidualConfig(alpha=16)
    assert cfg.objective == A1Config()
    assert cfg.training.model == "gpt2"

    # Dataset definitions now live in the shared runtime configuration.
    assert cfg.domains == ()

    runtime_cfg = load_runtime_config(REPO_ROOT / "configs" / "runtime_datasets.yaml")

    separate_datasets = [
        item["name"]
        for item in runtime_cfg["datasets"]
        if item.get("type") == "separate"
    ]

    assert separate_datasets == ["eurlex", "ledgar", "scotus"]

    domains = "domains:\n  - {name: d, dataset: x}\n"

    bad_key = tmp_path / "bad_key.yaml"
    bad_key.write_text("adapter:\n  rank: 4\n  dropout: 0.1\n" + domains)

    with pytest.raises(ValueError, match="Unknown ResidualConfig keys"):
        load_configs(bad_key, A1Config)

    bad_section = tmp_path / "bad_section.yaml"
    bad_section.write_text("optimizer:\n  name: adamw\n")

    with pytest.raises(ValueError, match="Unknown config sections"):
        load_configs(bad_section, A1Config)
