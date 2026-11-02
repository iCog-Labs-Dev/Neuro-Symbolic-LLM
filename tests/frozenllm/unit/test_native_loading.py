"""Native checkpoint loading and TorchAX mode restoration."""

import numpy as np
import pytest
import torch
import torchax
from torch.overrides import _get_current_function_mode_stack

from frozenllm.substrate import torchax_backend
from frozenllm.substrate.loader import (
    _native_torch_loading,
    load_tokenizer,
    load_torchax_model,
)


def active():
    return any(
        mode is torchax.default_env()._function_mode
        for mode in _get_current_function_mode_stack()
    )


@pytest.fixture(autouse=True)
def clean_environment():
    assert not active()
    torchax_backend._torchax_enabled = False
    yield
    if active():
        torchax.disable_globally()
    torchax_backend._torchax_enabled = False


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_native_loading_restores_modes_even_on_failure(initial, raises):
    if initial:
        torchax_backend.enable_torchax()

    def load():
        with _native_torch_loading():
            assert not active()
            value = torch.ones(2)
            assert type(value) is torch.Tensor
            assert value.device.type == "cpu"
            with _native_torch_loading():
                assert not active()
            if raises:
                raise ValueError("checkpoint failure")

    if raises:
        with pytest.raises(ValueError, match="checkpoint failure"):
            load()
    else:
        load()
    assert active() is initial


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("loader", [load_tokenizer, load_torchax_model])
def test_missing_checkpoint_restores_modes(tmp_path, initial, loader):
    if initial:
        torchax_backend.enable_torchax()
    with pytest.raises((OSError, ValueError)):
        loader(str(tmp_path / "missing"))
    assert active() is initial


def test_nested_active_scope_is_rejected_without_corrupting_outer_scope():
    torchax_backend.enable_torchax()
    with (
        torchax.default_env(),
        pytest.raises(RuntimeError, match="at most one"),
        _native_torch_loading(),
    ):
        pytest.fail("Unsupported nesting was accepted")
    assert active()
    with _native_torch_loading():
        assert not active()
    assert active()


@pytest.mark.parametrize("family", ["gpt2", "gpt_neox"])
def test_real_local_checkpoint_values_and_existing_conversion_behavior(
    tmp_path, family
):
    with _native_torch_loading():
        from transformers import (
            GPT2Config,
            GPT2LMHeadModel,
            GPTNeoXConfig,
            GPTNeoXForCausalLM,
        )

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            if family == "gpt2":
                model = GPT2LMHeadModel(
                    GPT2Config(
                        n_layer=1, n_head=2, n_embd=16, vocab_size=32, n_positions=8
                    )
                )
            else:
                model = GPTNeoXForCausalLM(
                    GPTNeoXConfig(
                        num_hidden_layers=1,
                        num_attention_heads=2,
                        hidden_size=16,
                        intermediate_size=32,
                        vocab_size=32,
                        max_position_embeddings=8,
                        rotary_pct=1.0,
                        tie_word_embeddings=False,
                    )
                )
        expected = {
            name: value.detach().numpy().copy()
            for name, value in model.state_dict().items()
        }
        model.save_pretrained(tmp_path)

    torchax_backend.enable_torchax()
    converted, params = load_torchax_model(str(tmp_path), torch_dtype=torch.float32)
    assert converted.training is False
    assert all(not value.requires_grad for value in params.values())
    # Device conversion creates separate embedding and output-head parameters.
    assert (
        converted.get_input_embeddings().weight
        is not converted.get_output_embeddings().weight
    )
    for name, value in converted.state_dict().items():
        np.testing.assert_array_equal(np.asarray(value.jax()), expected[name])

    second, _ = load_torchax_model(str(tmp_path), torch_dtype=torch.float32)
    for name, value in second.state_dict().items():
        np.testing.assert_array_equal(np.asarray(value.jax()), expected[name])
