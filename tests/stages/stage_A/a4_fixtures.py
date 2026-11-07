"""Native-model fixtures for A4 integration tests."""

import torchax

from frozenllm.substrate import FrozenSubstrate
from frozenllm.substrate.torchax_backend import is_torchax_enabled
from tests.frozenllm.conftest import _torch_model


def small_substrate():
    enabled = is_torchax_enabled()
    if enabled:
        torchax.disable_globally()
    try:
        native = _torch_model("gpt2")
    finally:
        if enabled:
            torchax.enable_globally()
    return FrozenSubstrate(native)
