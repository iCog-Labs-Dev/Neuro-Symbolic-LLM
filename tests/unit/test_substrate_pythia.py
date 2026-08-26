"""Pythia/GPT-NeoX tests for the frozen JAX substrate.

Equivalent coverage to the GPT-2 suite: automatic architecture detection,
transformer-block count, hidden-state interception, multiple interception
points, activation caching, JIT execution, valid logits, original-vs-wrapper
numerical equivalence, KL divergence and memory diagnostics. The Pythia layer
count is discovered from the parameter tree, never hardcoded.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from substrate import (
    compute_kl_drift,
    detect_architecture,
    discover_layers,
)
from tests.conftest import (
    BATCH,
    NUM_TOKENS,
    SEQ,
    make_input_ids,
    make_substrate,
    torch_logits,
)

NEOX_LAYERS = 12
NEOX_HIDDEN = 32
INTERCEPT = [2, 5, 8, 11]
TOL = 5e-4


@pytest.fixture(scope="module")
def pythia_fixture():
    model, substrate = make_substrate("neox", intercept_layers=INTERCEPT)
    return model, substrate


def _assert_close(actual, expected, rtol=1e-3, atol=1e-3):
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=rtol, atol=atol)


# 1. Automatic architecture detection ────────────────────────────────────────
class TestDetection:
    def test_model_family(self, pythia_fixture):
        _, sub = pythia_fixture
        assert sub.architecture.model_family == "neox"

    def test_detect_architecture_function(self, pythia_fixture):
        _, sub = pythia_fixture
        arch = detect_architecture(sub.get_params(), None)
        assert arch.model_family == "neox"
        assert arch.hidden_size == NEOX_HIDDEN

    def test_layer_count_discovered_not_hardcoded(self, pythia_fixture):
        _, sub = pythia_fixture
        assert sub.architecture.num_layers == NEOX_LAYERS
        assert discover_layers(sub.get_params()) == NEOX_LAYERS


# 2. Correct transformer block count ─────────────────────────────────────────
class TestBlockCount:
    def test_block_count_matches_config(self, pythia_fixture):
        _, sub = pythia_fixture
        assert sub.architecture.num_layers == NEOX_LAYERS


# 3 & 4. Hidden-state interception, multiple points ──────────────────────────
class TestInterception:
    def test_intercept_layers_registered(self, pythia_fixture):
        _, sub = pythia_fixture
        assert sub.intercept_layers == tuple(sorted(INTERCEPT))

    def test_interception_happens(self, pythia_fixture):
        _, sub = pythia_fixture
        result = sub(jnp.asarray(make_input_ids().numpy()))
        assert result.layer_indices() == tuple(sorted(INTERCEPT))

    def test_multiple_interception_points(self, pythia_fixture):
        _, sub = pythia_fixture
        result = sub(jnp.asarray(make_input_ids().numpy()))
        assert len(result.intermediates) == len(INTERCEPT)


# 5. Activation caching ──────────────────────────────────────────────────────
class TestCaching:
    def test_cache_has_shape(self, pythia_fixture):
        _, sub = pythia_fixture
        result = sub(jnp.asarray(make_input_ids().numpy()))
        for layer in INTERCEPT:
            hidden = result.hidden_state(layer)
            assert tuple(hidden.shape) == (BATCH, SEQ, NEOX_HIDDEN)
        assert result.hidden_shapes()[INTERCEPT[0]] == (BATCH, SEQ, NEOX_HIDDEN)

    def test_cache_miss_raises(self, pythia_fixture):
        _, sub = pythia_fixture
        result = sub(jnp.asarray(make_input_ids().numpy()))
        with pytest.raises(KeyError):
            result.hidden_state(0)

    def test_cache_is_premodification(self):
        def dim_perturb(h, i):
            return h * (1.0 + 0.1 * jnp.arange(h.shape[-1], dtype=h.dtype))

        model, sub = make_substrate(
            "neox", intercept_layers=[3], modify_hook=dim_perturb
        )
        ids = make_input_ids()
        with_sub = sub(jnp.asarray(ids.numpy()))
        assert not np.allclose(
            np.asarray(with_sub.logits), torch_logits(model, ids), atol=1e-3
        )
        _, plain = make_substrate("neox", intercept_layers=[3])
        ref = plain(jnp.asarray(ids.numpy())).hidden_state(3)
        np.testing.assert_allclose(
            np.asarray(with_sub.hidden_state(3)), np.asarray(ref), atol=1e-4
        )


# 6. JIT execution ───────────────────────────────────────────────────────────
class TestJIT:
    def test_jit_matches_eager(self, pythia_fixture):
        _, sub = pythia_fixture
        ids = jnp.asarray(make_input_ids().numpy())
        eager = sub(ids)
        jitted = jax.jit(sub)(ids)
        _assert_close(jitted.logits, np.asarray(eager.logits))
        assert jitted.layer_indices() == eager.layer_indices()

    def test_jit_matches_torch(self, pythia_fixture):
        model, sub = pythia_fixture
        ids = make_input_ids()
        jitted = jax.jit(sub)(jnp.asarray(ids.numpy()))
        _assert_close(jitted.logits, torch_logits(model, ids))


# 7. Valid logits ────────────────────────────────────────────────────────────
class TestLogits:
    def test_logits_shape_and_finite(self, pythia_fixture):
        _, sub = pythia_fixture
        result = sub(jnp.asarray(make_input_ids().numpy()))
        assert tuple(result.logits.shape) == (BATCH, SEQ, NUM_TOKENS)
        assert bool(jnp.isfinite(result.logits).all())


# 8. Original vs wrapper equivalence ─────────────────────────────────────────
class TestEquivalence:
    def test_identity_interception_preserves_logits(self, pythia_fixture):
        model, sub = pythia_fixture
        ids = make_input_ids()
        _assert_close(sub(jnp.asarray(ids.numpy())).logits, torch_logits(model, ids))

    def test_multiple_intercept_sets_match(self):
        for intercept in ([], [1], [2, 5, 8, 11], [0, 6, 11]):
            model, sub = make_substrate("neox", intercept_layers=intercept)
            ids = make_input_ids()
            _assert_close(
                sub(jnp.asarray(ids.numpy())).logits, torch_logits(model, ids)
            )


# 9. KL divergence ───────────────────────────────────────────────────────────
class TestKL:
    def test_identity_kl_approx_zero(self, pythia_fixture):
        model, sub = pythia_fixture
        ids = make_input_ids()
        wrapped = sub(jnp.asarray(ids.numpy())).logits
        drift = compute_kl_drift(jnp.asarray(torch_logits(model, ids)), wrapped)
        assert drift["kl_divergence"] < 1e-6

    def test_perturbation_increases_kl(self):
        model, sub = make_substrate(
            "neox",
            intercept_layers=[4],
            modify_hook=lambda h, i: h + 0.5 * jnp.arange(h.shape[-1], dtype=h.dtype),
        )
        ids = make_input_ids()
        original = jnp.asarray(torch_logits(model, ids))
        wrapped = sub(jnp.asarray(ids.numpy())).logits
        drift = compute_kl_drift(original, wrapped)
        assert drift["kl_divergence"] > 1e-3


# 10. Memory diagnostics ─────────────────────────────────────────────────────
class TestMemory:
    def test_memory_status_reports(self, pythia_fixture):
        _, sub = pythia_fixture
        status = sub.memory_status()
        if any(d.platform == "gpu" for d in jax.devices()):
            # Accelerator visible: real memory statistics must be reported.
            assert status.available is True
            assert status.platform == "gpu"
        else:
            # CPU backend: diagnostic-only status, must not crash.
            assert status.available is False
            assert status.diagnostic

    def test_run_with_memory_guard(self, pythia_fixture):
        _, sub = pythia_fixture
        ids = jnp.asarray(make_input_ids().numpy())
        result, report = sub.run_with_memory_guard(ids, auto_reduce_batch_size=False)
        assert "memory_status" in report
        assert report["batch_size_reduced"] is False
        assert tuple(result.logits.shape) == (BATCH, SEQ, NUM_TOKENS)


# Freezing guarantee ─────────────────────────────────────────────────────────
class TestFreeze:
    def test_params_unchanged_after_run(self, pythia_fixture):
        _, sub = pythia_fixture
        sub(jnp.asarray(make_input_ids().numpy()))
        jax.jit(sub)(jnp.asarray(make_input_ids().numpy()))
        assert sub.params_unchanged() is True
        report = sub.verify_frozen()
        assert report["params_unchanged"] is True
        assert report["architecture"]["num_layers"] == NEOX_LAYERS
