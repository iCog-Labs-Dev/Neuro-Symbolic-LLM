"""Essential A2 contracts: detach routing, live task terms, and runner wiring."""

import importlib.util
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from losses.a2_loss import drift_metrics
from residual import ResidualAdapter, ResidualConfig, layer_key
from residual.cpc_heads import q_key
from residual.local_predictor import PredictorConfig
from runtime.data_loader import RuntimeData
from stages.results import load_params
from stages.stage_A.a2 import (
    A2Config,
    CPCConfig,
    HiddenConfig,
    StabLossConfig,
    a2_objective,
    adapter_params,
    init_a2_params,
)
from stages.stage_base import Domain, evaluate, load_configs, make_eval_set
from tests.frozenllm.conftest import BATCH, make_substrate

REPO_ROOT = Path(__file__).resolve().parents[3]
SEQ = 12


def _domain(name, token, num_blocks=8):
    blocks = np.random.default_rng(token).integers(
        0,
        64,
        (num_blocks, SEQ),
        dtype=np.int32,
    )

    return Domain(
        name,
        blocks,
        make_eval_set(blocks, BATCH, 1),
        1,
    )


def tree_norm(tree):
    return sum(float(jnp.sum(x**2)) for x in jax.tree_util.tree_leaves(tree))


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def setup():
    substrate = make_substrate("gpt2")[1]

    adapter = ResidualAdapter(
        ResidualConfig(rank=2, layers=[5]),
        substrate.architecture,
    )

    config = A2Config(
        lambda_kl=0.1,
        cpc=CPCConfig(
            weight=0.1,
            dim=4,
            max_anchors_per_seq=3,
        ),
        hid=HiddenConfig(
            weight=0.1,
            predictor=PredictorConfig(ff_dim=8),
        ),
        stab=StabLossConfig(
            weight=0.1,
            w_anch=1.0,
            w_smooth=0.1,
        ),
    )

    ids = jnp.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ],
        dtype=jnp.int32,
    )

    params = init_a2_params(
        config,
        adapter,
        seq_len=4,
    )

    params["layer_5"]["B"] = 0.05 * jax.random.normal(
        jax.random.PRNGKey(7),
        params["layer_5"]["B"].shape,
    )

    for index, block in enumerate(key for key in params if key.startswith("pred_")):
        for offset, name in enumerate(params[block]):
            params[block][name] = 0.1 * jax.random.normal(
                jax.random.PRNGKey(100 + 20 * index + offset),
                params[block][name].shape,
            )

    return substrate, adapter, config, ids, params


# ---------------------------------------------------------------------
# Gradient routing
# ---------------------------------------------------------------------


def test_detach_flags_block_column_gradients_but_heads_and_task_terms_stay_live(
    setup,
):
    """Each auxiliary loss is checked independently."""

    substrate, adapter, config, ids, params = setup

    detached = replace(
        config,
        hid_detach_input=True,
        cpc_detach_query=True,
    )

    assert not config.hid_detach_input
    assert not config.cpc_detach_query

    def terms(p, cfg=config):
        return a2_objective(
            p,
            substrate=substrate,
            adapter=adapter,
            config=cfg,
            input_ids=ids,
        )[1]

    for term, prefix in (
        ("hid", "pred_"),
        ("cpc", "cpc"),
    ):
        grad = jax.grad(lambda p, term=term: terms(p, detached)[term])(params)

        assert tree_norm(adapter_params(grad)) == 0.0

        assert (
            tree_norm(
                {key: value for key, value in grad.items() if key.startswith(prefix)}
            )
            > 0
        )

        # Without detach flags, the auxiliary loss must also
        # contribute gradients to the adapter.
        grad = jax.grad(lambda p, term=term: terms(p)[term])(params)

        assert tree_norm(adapter_params(grad)) > 0

    for term in (
        "task",
        "kl",
        "stab_anch",
        "stab_smooth",
    ):
        grad = jax.grad(lambda p, term=term: terms(p)[term])(params)

        assert tree_norm(adapter_params(grad)) > 0

    assert substrate.params_unchanged()


def test_targets_are_detached_and_drift_is_measurement_only(setup):
    from losses.a2_loss import (
        anchor_loss,
        contrastive_future_loss,
        hidden_prediction_loss,
    )
    from residual.cpc_heads import (
        a_key,
        init_cpc_heads,
        q_key,
    )

    substrate, _, config, _, _ = setup

    source = jax.random.normal(
        jax.random.PRNGKey(0),
        (2, 4, 8),
    )

    target = source + 0.5

    heads = init_cpc_heads(
        0,
        [5],
        8,
        4,
    )

    losses = (
        lambda y: hidden_prediction_loss(
            source,
            y,
            source,
            config,
        )["loss"],
        lambda y: contrastive_future_loss(
            heads[q_key(5)],
            heads[a_key(5, 1)],
            source,
            y,
        )[0],
        lambda y: anchor_loss(source, y)[0],
        lambda y: drift_metrics(
            {5: y},
            {5: source},
        )["drift_mean"],
    )

    for loss in losses:
        np.testing.assert_array_equal(
            jax.grad(loss)(target),
            jnp.zeros_like(target),
        )

    from stages.stage_base import model_loss_terms

    labels = jnp.zeros(
        (2, 4),
        dtype=jnp.int32,
    )

    base_grad = jax.grad(
        lambda base: model_loss_terms(
            substrate,
            source,
            labels,
            0.1,
            base,
        )[0]
    )(target)

    np.testing.assert_array_equal(
        base_grad,
        jnp.zeros_like(target),
    )


# ---------------------------------------------------------------------
# A2 configuration schema
# ---------------------------------------------------------------------


def test_a2_schema_defaults_and_settling_separation():
    import ast

    root = Path(__file__).resolve().parents[3]

    cfg = load_configs(
        root / "configs" / "stage_A" / "a2_pc_inspired.yaml",
        A2Config,
    ).objective

    # Verify the actual shipped defaults without modifying them.
    for candidate in (
        cfg,
        A2Config(),
        A2Config.from_dict({}),
    ):
        assert candidate.hid_detach_input is False
        assert candidate.cpc_detach_query is False
        assert candidate.predictor.detach_input is False

    assert PredictorConfig().detach_input is False

    assert A2Config.from_dict(asdict(cfg)) == cfg

    legacy = A2Config.from_dict(
        {
            "hid": {
                "predictor": {
                    "detach_input": True,
                }
            }
        }
    )

    assert legacy.hid_detach_input is True
    assert legacy.predictor.detach_input is True

    for name in ("w_step", "w_hep", "hep"):
        with pytest.raises(ValueError, match="Unknown"):
            A2Config.from_dict({"stab": {name: 1}})

    for name in (
        "hid_detach_input",
        "cpc_detach_query",
    ):
        with pytest.raises(ValueError, match="boolean"):
            A2Config.from_dict({name: 1})

    settling_helpers = {
        "settling_loss",
        "hep_nudge",
        "hep_nudge_loss",
        "output_error",
        "v_hidden_size",
    }

    paths = [
        *(root / "stages" / "stage_A").glob("a2*.py"),
        root / "experiments" / "run_stage_a2.py",
    ]

    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "metrics.representation"
            ):
                assert settling_helpers.isdisjoint(alias.name for alias in node.names)
                assert all(alias.name != "*" for alias in node.names)


# ---------------------------------------------------------------------
# A2 runner overrides
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "flags, detached",
    [
        ([], False),
        (
            ["--hid-detach-input", "--cpc-detach-query"],
            True,
        ),
        (
            ["--no-hid-detach-input", "--no-cpc-detach-query"],
            False,
        ),
    ],
)
def test_runner_detach_overrides_and_predictor_multiplier(
    runner,
    flags,
    detached,
):
    cfg = load_configs(
        REPO_ROOT / "configs" / "stage_A" / "a2_pc_inspired.yaml",
        A2Config,
    )

    # Enable the hidden-prediction objective so hid_coupled
    # is meaningful in this test.
    cfg = replace(
        cfg,
        objective=replace(
            cfg.objective,
            hid=replace(
                cfg.objective.hid,
                weight=0.1,
            ),
        ),
    )

    if "--no-hid-detach-input" in flags:
        cfg = replace(
            cfg,
            objective=replace(
                cfg.objective,
                hid_detach_input=True,
                cpc_detach_query=True,
            ),
        )

    args = runner.parse_args([*flags, "--predictor-lr-mult", "2"])

    out = runner.apply_overrides(
        cfg,
        args,
    ).objective

    assert out.hid_detach_input is detached
    assert out.cpc_detach_query is detached
    assert out.hid_coupled is (not detached)
    assert out.predictor.detach_input is detached
    assert out.predictor_lr_mult == 2.0


# ---------------------------------------------------------------------
# Runner fixture
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def runner() -> Any:
    path = REPO_ROOT / "experiments" / "run_stage_a2.py"

    spec = importlib.util.spec_from_file_location(
        "run_stage_a2_test",
        path,
    )

    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "main"):
        sys.modules.pop(spec.name, None)
        pytest.skip("experiments/run_stage_a2.py is not implemented yet (runner PR)")

    yield module

    sys.modules.pop(spec.name, None)


# ---------------------------------------------------------------------
# End-to-end A2 runner integration
# ---------------------------------------------------------------------


def test_runner_main_end_to_end(
    runner: Any,
    setup,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    gpt2 = setup[0]

    import stages.stage_base as stage_base

    cfg_path = tmp_path / "cfg.yaml"

    cfg_path.write_text(
        "adapter: {rank: 4, layers: [5, 6]}\n"
        "objective:\n"
        "  lambda_kl: 0.1\n"
        "  cpc: {weight: 0.2, dim: 8}\n"
        "  stab: {weight: 0.2}\n"
        f"training: {{model: gpt2, seq_len: {SEQ}, "
        f"batch_size: {BATCH}, epochs: 1, "
        "learning_rate: 1.0e-2, eval_batches: 1, log_every: 1}\n"
        "domains:\n"
        "  - {name: a, dataset: x}\n"
    )

    # Build a synthetic runtime dataset. No Hugging Face download.
    domain = _domain(
        "a",
        5,
        num_blocks=4,
    )

    blocks = domain.train_blocks

    runtime_data = RuntimeData(
        tasks=(domain,),
        test_blocks={"a": blocks},
        task_order=("a",),
        metadata={},
    )

    monkeypatch.setattr(
        runner,
        "FrozenSubstrate",
        lambda _name: gpt2,
    )

    # Mock the new shared runtime loader, not the legacy
    # stages.stage_base.load_token_blocks function.
    monkeypatch.setattr(
        runner,
        "load_runtime_data",
        lambda **kwargs: runtime_data,
    )

    code = runner.main(
        [
            "--arm",
            "full",
            "--seed",
            "3",
            "--config",
            str(cfg_path),
            "--output-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "r",
        ]
    )

    assert code == 0

    run = tmp_path / "runs" / "r"

    results = json.loads((run / "results.json").read_text())

    assert results["params_unchanged"] is True
    assert np.isfinite(results["test"]["a"]["loss"])

    meta = json.loads((run / "meta.json").read_text())

    assert meta["stage"] == "A2"
    assert meta["arm"] == "full"
    assert meta["seed"] == 3

    groups = meta["trainable_params_by_group"]

    assert groups["cpc"] > 0
    assert "hep" not in groups

    assert groups["pred"] > 0
    assert meta["predictor_params"] == groups["pred"]
    assert meta["predictor_layers"] == [5, 6]
    assert groups["total"] == meta["trainable_params"]

    saved = load_params(run / "params" / "task_0_a.npz")

    assert q_key(5) in saved
    assert layer_key(5) in saved

    config = load_configs(
        run / "config.yaml",
        A2Config,
    )

    assert config.objective.cpc.weight == 0.2

    assert config.objective.seed == config.training.seed == config.adapter.seed == 3

    # Exercise checkpoint evaluation through the new runtime loader.
    checkpoint = run / "params" / "task_0_a.npz"

    capsys.readouterr()

    assert (
        runner.main(
            [
                "--config",
                str(run / "config.yaml"),
                "--eval-checkpoint",
                str(checkpoint),
            ]
        )
        == 0
    )

    reloaded = json.loads(capsys.readouterr().out)

    expected = evaluate(
        gpt2,
        ResidualAdapter(
            config.adapter,
            gpt2.architecture,
        ),
        saved,
        make_eval_set(
            blocks,
            BATCH,
            1,
        ),
    )

    assert reloaded["a"] == pytest.approx(expected)

    # A1's checkpoint validator must still reject A2 auxiliary keys.
    with pytest.raises(ValueError, match="extra layers"):
        stage_base.validate_checkpoint_params(
            ResidualAdapter(
                config.adapter,
                gpt2.architecture,
            ),
            saved,
        )

    assert gpt2.params_unchanged()
