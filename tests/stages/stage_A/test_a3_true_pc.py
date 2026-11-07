"""Stage A3 (FabricPC) tests on the 12-layer GPT-2 fixture, as in test_a1_training.py.

Every expectation is computed independently of FabricPC from the A3 equations and the
substrate's own ``compute_loss`` (the loss stage A1 optimises), then compared with what the
FabricPC engine produces:

    z^(0)  = sigma(h0 A)
    z     <- z - eta * (s * B^T delta_sum + precision * (z - f))        delta_sum = N * delta_mean
    dE/dB  = s * sum_t z_t delta_t^T        (gradient of A1's token-mean loss at the final z)
    dE/dA  = -precision * h0^T (z - f) / N,                             N = batch * (T - 1)

The module is skipped unless FabricPC >= 0.6 is importable (the shared pin may be older).
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

pytest.importorskip("fabricpc.graph_assembly", reason="A3 needs FabricPC >= 0.6")

# Keep FabricPC imports after importorskip so unavailable versions skip this module cleanly.
from fabricpc.core.inference import run_inference  # noqa: E402
from fabricpc.core.initializers import NormalInitializer  # noqa: E402
from fabricpc.graph_initialization.state_initializer import (  # noqa: E402
    initialize_graph_state,
)
from fabricpc.training.trainer import build_clamps  # noqa: E402

from frozenllm.substrate import identity_modify  # noqa: E402
from stages.stage_A.A3_model import A3Config, A3Model  # noqa: E402
from stages.stage_A.a3_tail import make_frozen_tail, residual_modify  # noqa: E402
from tests.frozenllm.conftest import (  # noqa: E402
    BATCH,
    NUM_TOKENS,
    SEQ,
    make_substrate,
)

LAYERS = (6, 7)
RANK = 4
K = 2
ETA = 0.05
SIGMA2 = 0.7
PREC = 1.0 / SIGMA2
N = BATCH * (SEQ - 1)
TOL = {"atol": 2e-4, "rtol": 2e-3}


def make_model(sub, k=K, lam=0.0):
    cfg = A3Config(
        infer_steps=k,
        eta_z=ETA,
        sigma2=SIGMA2,
        lambda_kl=lam,
        predictor_init_std=0.3,  # large enough that z^(0) and the gradients are visible
        projection_init_std=0.3,
    )
    return A3Model(sub, layers=LAYERS, rank=RANK, seq_len=SEQ, config=cfg)


def make_ref(sub, model, seed=0):
    params = model.init_params(jax.random.PRNGKey(seed))
    rng = np.random.default_rng(seed)
    ids = jnp.asarray(rng.integers(0, NUM_TOKENS, (BATCH, SEQ)), dtype=jnp.int32)
    h0, base = model.clean_pass(ids)
    return SimpleNamespace(
        sub=sub, model=model, params=params, ids=ids, h0=h0, base=base
    )


def b_of(ref, params=None):
    p = ref.params if params is None else params
    return {layer: ref.model.projection(p, layer) for layer in LAYERS}


def loss_at(ref, z, b=None, lam=0.0, probe=None):
    """Token-mean model loss at latents z, from the substrate's own loss API (as in A1)."""
    inject = residual_modify(z, b_of(ref) if b is None else b, ref.model.scale)

    def modify(h, layer):
        h = inject(h, layer)
        return h + probe[layer] if probe is not None and layer in probe else h

    logits = ref.sub.run_with_interception(
        ref.ids, modify_fn=modify, intercept_layers=LAYERS
    ).logits
    ce = ref.sub.compute_loss(logits, ref.ids, losses=("CE",))
    if lam <= 0:
        return ce
    kl = ref.sub.compute_loss(
        logits[:, :-1], None, losses=("KL",), base_logits=ref.base[:, :-1]
    )
    return ce + lam * kl


def delta_mean(ref, z, lam=0.0):
    """Gradient of the token-mean loss at h~_l (total derivative, via a zero probe)."""
    probe0 = {layer: jnp.zeros_like(ref.h0[layer]) for layer in LAYERS}
    return jax.grad(lambda p: loss_at(ref, z, lam=lam, probe=p))(probe0)


def settle_by_hand(ref, k, lam=0.0):
    f = ref.model.predict_latents(ref.params, ref.h0)
    z = dict(f)
    s = ref.model.scale
    for _ in range(k):
        d = delta_mean(ref, z, lam)
        z = {
            layer: z[layer]
            - ETA
            * (s * ((N * d[layer]) @ b_of(ref)[layer].T) + PREC * (z[layer] - f[layer]))
            for layer in LAYERS
        }
    return z, f


def engine_start(ref):
    batch, _, _ = ref.model.make_batch(ref.ids)
    clamps = build_clamps(batch, ref.model.structure, clamp_target=True)
    state = initialize_graph_state(
        ref.model.structure, BATCH, jax.random.PRNGKey(1), clamps, params=ref.params
    )
    return clamps, state


@pytest.fixture(scope="module")
def substrate():
    return make_substrate("gpt2")[1]


@pytest.fixture(scope="module")
def env(substrate):
    return make_ref(substrate, make_model(substrate))


@pytest.fixture(scope="module")
def trained(env):
    """One eager A3 batch (K inference steps) shared by the tests that read its outputs."""
    return env.model.train_grads(env.params, env.ids, jax.random.PRNGKey(2))


# --------------------------------------------------------------------- cached hidden states
def test_cached_hidden_states_come_from_the_substrate_cache(env):
    cached = env.sub.run_with_interception(
        env.ids, modify_fn=identity_modify, intercept_layers=LAYERS
    )
    seen = {}

    def spy(h, layer):
        seen[layer] = h
        return h

    env.sub.run_with_interception(env.ids, modify_fn=spy, intercept_layers=LAYERS)
    batch, _, _ = env.model.make_batch(env.ids)
    for layer in LAYERS:
        np.testing.assert_allclose(env.h0[layer], cached.hidden_state(layer), atol=1e-6)
        np.testing.assert_allclose(
            env.h0[layer], seen[layer], atol=1e-6
        )  # pristine: hook saw the same
        np.testing.assert_allclose(batch[f"h0_{layer}"], env.h0[layer], atol=1e-6)
    np.testing.assert_allclose(env.base, cached.logits, atol=1e-6)


def test_cached_hidden_states_survive_jit(env):
    h0, base = jax.jit(env.model.clean_pass)(env.ids)
    for layer in LAYERS:
        np.testing.assert_allclose(h0[layer], env.h0[layer], atol=1e-4)
    np.testing.assert_allclose(base, env.base, atol=1e-4)


def test_z0_is_the_predictor_applied_to_cached_h0(env):
    _, state0 = engine_start(env)
    f = env.model.predict_latents(env.params, env.h0)
    for layer in LAYERS:
        np.testing.assert_allclose(state0.nodes[f"z{layer}"].z_latent, f[layer], **TOL)
        np.testing.assert_allclose(state0.nodes[f"z{layer}"].energy, 0.0, atol=1e-6)


def test_parameters_use_the_a1_layout(env):
    d = env.model.hidden_size
    for layer in LAYERS:
        assert env.model.predictor_weight(env.params, layer).shape == (d, RANK)
        assert env.model.projection(env.params, layer).shape == (RANK, d)
        assert float(jnp.abs(env.model.projection(env.params, layer)).max()) > 0
    assert len(jax.tree_util.tree_leaves(env.params)) == 2 * len(LAYERS)
    adapter = env.model.adapter_params(env.params)
    assert set(adapter) == {f"layer_{layer}" for layer in LAYERS}
    assert all(set(v) == {"A", "B"} for v in adapter.values())


# --------------------------------------------------------------------- inference
def test_k_step_inference_matches_the_a3_equation(env):
    clamps, state0 = engine_start(env)
    state = run_inference(env.params, state0, clamps, env.model.structure)
    z_hand, f = settle_by_hand(env, K)
    for layer in LAYERS:
        np.testing.assert_allclose(
            state.nodes[f"z{layer}"].z_latent, z_hand[layer], **TOL
        )
    assert (
        max(float(jnp.abs(z_hand[layer] - f[layer]).max()) for layer in LAYERS) > 1e-5
    )  # z moved


def test_one_step_from_z0_has_no_spring_force(substrate):
    model = make_model(substrate, k=1)
    ref = make_ref(substrate, model)
    clamps, state0 = engine_start(ref)
    state = run_inference(ref.params, state0, clamps, model.structure)
    z_hand, f = settle_by_hand(ref, 1)
    for (
        layer
    ) in LAYERS:  # at z = f the spring gradient is 0: only the task gradient moves z
        np.testing.assert_allclose(
            state.nodes[f"z{layer}"].z_latent, z_hand[layer], **TOL
        )


# --------------------------------------------------------------------- final-state energy
def test_final_state_energy_is_exact(env, trained):
    _, aux = trained
    z, f = aux["z_star"], env.model.predict_latents(env.params, env.h0)
    spring = (
        0.5
        * PREC
        * sum(float(jnp.sum((z[layer] - f[layer]) ** 2)) for layer in LAYERS)
        / N
    )
    hand = spring + float(loss_at(env, z))
    np.testing.assert_allclose(aux["energy_after"], hand, **TOL)
    np.testing.assert_allclose(aux["spring"], spring, **TOL)

    curve = env.model.energy_curve(env.params, env.ids, jax.random.PRNGKey(2), steps=K)
    assert len(curve) == K + 1
    np.testing.assert_allclose(aux["energy_before"], curve[0], **TOL)
    np.testing.assert_allclose(aux["energy_after"], curve[-1], **TOL)
    assert aux["energy_after"] < aux["energy_before"]  # inference lowered the energy


def test_final_energy_is_not_the_stale_pre_update_value(substrate):
    """With one step the state's own energies are those of z^(0); the reported final energy must
    be that of z^(1), so it must differ from energy_before."""
    model = make_model(substrate, k=1)
    ref = make_ref(substrate, model)
    _, aux = model.train_grads(ref.params, ref.ids, jax.random.PRNGKey(2))
    assert float(aux["energy_after"]) < float(aux["energy_before"]) - 1e-7


# --------------------------------------------------------------------- gradient propagation
def test_local_gradients_match_closed_forms_and_autodiff(env, trained):
    grads, aux = trained
    z, f = aux["z_star"], aux["pred"]
    d = delta_mean(env, z)
    b = b_of(env)
    for layer in LAYERS:
        g_b = grads.nodes["tail"].weights[f"z{layer}->tail:z"]
        closed = env.model.scale * jnp.einsum("btr,btd->rd", z[layer], d[layer])
        np.testing.assert_allclose(g_b, closed, **TOL)  # closed form
        auto = jax.grad(lambda bl, layer=layer: loss_at(env, z, b={**b, layer: bl}))(
            b[layer]
        )
        np.testing.assert_allclose(
            g_b, auto, **TOL
        )  # = autodiff of A1's loss (substrate API)

        g_a = grads.nodes[f"z{layer}"].weights[f"h0_{layer}->z{layer}:in"]
        np.testing.assert_allclose(  # local rule: needs no pass through the frozen model
            g_a,
            -PREC
            * jnp.einsum(
                "btd,btr->dr",
                env.h0[layer],
                z[layer] - f[layer],
            )
            / N,
            **TOL,
        )


def test_latent_gradient_reaches_z_through_the_frozen_layers(env):
    """B^T delta must be non-zero at every adapted layer: the task gradient propagates
    through the frozen blocks above each injection point."""
    f = env.model.predict_latents(env.params, env.h0)
    d = delta_mean(env, f)
    for layer in LAYERS:
        assert float(jnp.abs(d[layer]).max()) > 0
        assert float(jnp.abs(d[layer] @ b_of(env)[layer].T).max()) > 0


# --------------------------------------------------------------------- KL and baseline passes
def test_kl_term_uses_cached_base_logits_and_matches_the_equation(substrate):
    lam = 0.5
    model = make_model(substrate, k=K, lam=lam)
    ref = make_ref(substrate, model)
    assert "base" in model.structure.nodes and "base" in model.structure.task_map

    clamps, state0 = engine_start(ref)
    state = run_inference(ref.params, state0, clamps, model.structure)
    z_kl, _ = settle_by_hand(ref, K, lam)
    z_plain, _ = settle_by_hand(ref, K, 0.0)
    for layer in LAYERS:
        np.testing.assert_allclose(
            state.nodes[f"z{layer}"].z_latent, z_kl[layer], **TOL
        )
    assert (
        max(float(jnp.abs(z_kl[layer] - z_plain[layer]).max()) for layer in LAYERS)
        > 1e-5
    )

    _, aux = model.train_grads(ref.params, ref.ids, jax.random.PRNGKey(2))
    assert float(aux["kl"]) >= 0
    np.testing.assert_allclose(aux["loss"], aux["task"] + lam * aux["kl"], **TOL)


def test_baseline_is_not_recomputed_inside_the_tail(substrate, monkeypatch):
    """The KL term must add no substrate pass: the base logits come from the clean pass."""
    counts = {}
    for lam in (0.0, 0.5):
        model = make_model(substrate, k=K, lam=lam)
        ref = make_ref(substrate, model)
        calls = {"n": 0}
        original = substrate.run_with_interception

        def counting(*args, _original=original, _calls=calls, **kwargs):
            _calls["n"] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(substrate, "run_with_interception", counting)
        model.train_grads(ref.params, ref.ids, jax.random.PRNGKey(2))
        monkeypatch.undo()
        counts[lam] = calls["n"]
    assert counts[0.5] == counts[0.0]


# --------------------------------------------------------------------- frozen parameters
def test_frozen_parameters_survive_a3_training(substrate):
    model = make_model(substrate, k=1)
    ref = make_ref(substrate, model)
    opt = optax.adam(1e-3)
    opt_state = opt.init(ref.params)
    assert substrate.verify_frozen()["params_unchanged"]

    @jax.jit
    def step(params, opt_state, ids, key):
        grads, aux = model.train_grads(params, ids, key)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, grads, aux

    params = ref.params
    for i in range(3):
        params, opt_state, grads, aux = step(
            params, opt_state, ref.ids, jax.random.PRNGKey(i)
        )

    assert substrate.verify_frozen()["params_unchanged"]
    # Only the adapter matrices are trainable: no substrate weight is in the optimised tree.
    assert len(jax.tree_util.tree_leaves(params)) == 2 * len(LAYERS)
    assert jax.tree_util.tree_structure(grads) == jax.tree_util.tree_structure(
        ref.params
    )
    assert all(bool(jnp.isfinite(g).all()) for g in jax.tree_util.tree_leaves(grads))
    assert all(getattr(p, "grad", None) is None for p in substrate.params.values())
    moved = max(
        float(
            jnp.abs(
                model.projection(params, layer) - model.projection(ref.params, layer)
            ).max()
        )
        for layer in LAYERS
    )
    assert moved > 0  # B trained


# --------------------------------------------------------------------- deployment
def test_deployed_model_uses_z_equal_predictor_of_cached_h0(env):
    logits, base = env.model.deploy_logits(env.params, env.ids)
    f = env.model.predict_latents(env.params, env.h0)
    expected = env.sub.run_with_interception(
        env.ids,
        modify_fn=residual_modify(f, b_of(env), env.model.scale),
        intercept_layers=LAYERS,
    ).logits
    np.testing.assert_allclose(logits, expected, **TOL)
    np.testing.assert_allclose(base, env.base, atol=1e-6)


# --------------------------------------------------------------------- validation
def test_configuration_and_input_validation(substrate, env):
    for bad in (
        {"layers": (), "rank": RANK, "seq_len": SEQ},
        {"layers": (99,), "rank": RANK, "seq_len": SEQ},
        {"layers": LAYERS, "rank": 0, "seq_len": SEQ},
        {"layers": LAYERS, "rank": RANK, "seq_len": 1},
        {"layers": LAYERS, "rank": RANK, "seq_len": SEQ, "activation": "swish"},
        {"layers": LAYERS, "rank": RANK, "seq_len": SEQ, "scale": 0.0},
    ):
        with pytest.raises(ValueError):
            A3Model(substrate, **bad)
    with pytest.raises(ValueError):
        A3Config(eta_z=0.0)
    with pytest.raises(ValueError):
        A3Config(lambda_kl=-1.0)
    with pytest.raises(ValueError):
        A3Config.from_dict({"not_a_key": 1})
    with pytest.raises(TypeError):
        env.model.make_batch(env.ids.astype(jnp.float32))
    with pytest.raises(ValueError):
        env.model.make_batch(env.ids[:, :-1])


def test_tail_wiring_validation(substrate, env):
    tail_cls = make_frozen_tail(substrate)
    d = env.model.hidden_size
    config = {"layers": LAYERS, "hidden_size": d, "lambda_kl": 0.5, "scale": 1.0}
    shapes = {
        f"z{LAYERS[0]}->tail:z": (SEQ, RANK),
        f"z{LAYERS[1]}->tail:z": (SEQ, RANK),
        "ids->tail:ids": (SEQ,),
    }
    key, init = jax.random.PRNGKey(0), NormalInitializer()
    with pytest.raises(ValueError, match="base"):  # KL needs the cached base logits
        tail_cls.initialize_params(key, (SEQ - 1, 1), shapes, init, config)
    with pytest.raises(ValueError, match="ids"):
        tail_cls.initialize_params(
            key,
            (SEQ - 1, 1),
            {k: v for k, v in shapes.items() if "ids" not in k},
            init,
            {**config, "lambda_kl": 0.0},
        )
    with pytest.raises(ValueError):  # wrong node shape
        tail_cls.initialize_params(
            key, (SEQ, 1), shapes, init, {**config, "lambda_kl": 0.0}
        )
    with pytest.raises(ValueError):
        tail_cls(shape=(SEQ - 1, 2), name="tail", layers=LAYERS, hidden_size=d)
    with pytest.raises(ValueError):
        tail_cls(shape=(SEQ - 1, 1), name="tail", layers=(), hidden_size=d)
