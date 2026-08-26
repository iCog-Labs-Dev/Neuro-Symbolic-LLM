"""Invalid-input and robustness tests for the frozen JAX substrate."""

from __future__ import annotations

import numpy as np
import pytest

from substrate import FrozenJAXSubstrate, detect_architecture, state_dict_to_jax_pytree
from tests.conftest import make_substrate


def _dummy_gpt2_params():
    return state_dict_to_jax_pytree(
        {
            "transformer.wte.weight": np.zeros((64, 32), dtype=np.float32),
            "transformer.wpe.weight": np.zeros((32, 32), dtype=np.float32),
            "transformer.ln_f.weight": np.ones((32,), dtype=np.float32),
            "transformer.ln_f.bias": np.zeros((32,), dtype=np.float32),
            "transformer.h.0.ln_1.weight": np.ones((32,), dtype=np.float32),
            "transformer.h.0.ln_1.bias": np.zeros((32,), dtype=np.float32),
            "transformer.h.0.ln_2.weight": np.ones((32,), dtype=np.float32),
            "transformer.h.0.ln_2.bias": np.zeros((32,), dtype=np.float32),
            "transformer.h.0.attn.c_attn.weight": np.zeros((32, 96), dtype=np.float32),
            "transformer.h.0.attn.c_attn.bias": np.zeros((96,), dtype=np.float32),
            "transformer.h.0.attn.c_proj.weight": np.zeros((32, 32), dtype=np.float32),
            "transformer.h.0.attn.c_proj.bias": np.zeros((32,), dtype=np.float32),
            "transformer.h.0.mlp.c_fc.weight": np.zeros((32, 128), dtype=np.float32),
            "transformer.h.0.mlp.c_fc.bias": np.zeros((128,), dtype=np.float32),
            "transformer.h.0.mlp.c_proj.weight": np.zeros((128, 32), dtype=np.float32),
            "transformer.h.0.mlp.c_proj.bias": np.zeros((32,), dtype=np.float32),
            "lm_head.weight": np.zeros((64, 32), dtype=np.float32),
        }
    )


class TestInvalidLayerIndices:
    def test_negative_layer_index(self):
        with pytest.raises(ValueError, match="non-negative"):
            FrozenJAXSubstrate(_dummy_gpt2_params(), intercept_layers=[-1])

    def test_layer_index_out_of_range(self):
        with pytest.raises(ValueError, match="only 1 transformer layers"):
            FrozenJAXSubstrate(_dummy_gpt2_params(), intercept_layers=[1])

    def test_duplicate_layer_indices(self):
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            FrozenJAXSubstrate(_dummy_gpt2_params(), intercept_layers=[0, 0])

    def test_error_message_is_explicit(self):
        with pytest.raises(ValueError) as excinfo:
            FrozenJAXSubstrate(_dummy_gpt2_params(), intercept_layers=[7])
        msg = str(excinfo.value)
        assert "7" in msg and "zero-based" in msg


class TestEmptyInterception:
    def test_empty_list_is_valid(self):
        sub = FrozenJAXSubstrate(_dummy_gpt2_params(), intercept_layers=[])
        assert sub.intercept_layers == ()

    def test_none_is_valid(self):
        sub = FrozenJAXSubstrate(_dummy_gpt2_params())
        assert sub.intercept_layers == ()


class TestWrongInputShape:
    def test_1d_input_ids_rejected(self):
        _, sub = make_substrate("gpt2", intercept_layers=[0])
        import jax.numpy as jnp

        with pytest.raises(ValueError, match="2D array"):
            sub(jnp.zeros((5,), dtype=jnp.int32))

    def test_3d_input_ids_rejected(self):
        _, sub = make_substrate("gpt2", intercept_layers=[0])
        import jax.numpy as jnp

        with pytest.raises(ValueError, match="2D array"):
            sub(jnp.zeros((1, 5, 3), dtype=jnp.int32))

    def test_zero_sequence_rejected(self):
        _, sub = make_substrate("gpt2", intercept_layers=[0])
        import jax.numpy as jnp

        with pytest.raises(ValueError, match="at least one"):
            sub(jnp.zeros((1, 0), dtype=jnp.int32))


class TestUnsupportedArchitecture:
    def test_unknown_top_level_keys(self):
        params = {
            "bert": {"embeddings": {"word_embeddings": {"weight": np.zeros((1, 1))}}}
        }
        with pytest.raises(ValueError, match="Unsupported model architecture"):
            FrozenJAXSubstrate(params)

    def test_detect_architecture_raises(self):
        with pytest.raises(ValueError, match="Unsupported model architecture"):
            detect_architecture({"nonsense": {"x": np.zeros((1,))}})

    def test_gpt2_and_neox_params_confuse_nothing(self):
        # both families detected correctly by their container keys
        from tests.conftest import make_substrate

        _, sub = make_substrate("gpt2")
        assert sub.architecture.model_family == "gpt2"
        _, sub = make_substrate("neox")
        assert sub.architecture.model_family == "neox"
