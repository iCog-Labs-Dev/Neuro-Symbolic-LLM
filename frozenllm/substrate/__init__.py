"""Frozen LLM substrate based on TorchAX.

A reusable wrapper around pretrained causal language models (GPT-2 and
Pythia/GPT-NeoX) that keeps the base model completely frozen while allowing
arbitrary per-layer hidden-state interception, activation caching, drift
monitoring and device memory monitoring.
"""

import torch

# Compatibility patch for torchax versions expecting FP4 dtype on PyTorch < 2.5
if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = getattr(torch, "float8_e4m3fn", torch.uint8)

from .architecture import (
    Architecture,
    detect_architecture,
    detect_architecture_from_config,
    discover_layers,
    discover_layers_from_config,
    get_block_accessor,
    get_embedding_module,
    get_head_modules,
    get_position_embedding_module,
    validate_interception_layers,
)
from .drift import compute_kl_drift
from .interception import (
    InterceptionContext,
    ModifyFn,
    identity_modify,
    run_with_hooks,
)
from .loader import (
    load_tokenizer,
    load_torchax_model,
)
from .memory import (
    MemoryStatus,
    check_memory_headroom,
    compute_memory_headroom,
    get_memory_status,
    maybe_reduce_batch_size,
)
from .substrate import ForwardResult, FrozenSubstrate, Substrate
from .torchax_backend import (
    call_jax_differentiable,
    enable_torchax,
    from_jax_array,
    is_on_torchax_device,
    is_torchax_enabled,
    to_jax_array,
    to_torchax_device,
)
from .torchax_models import (
    check_numerical_fidelity,
    functional_model,
)

__all__ = [
    # Architecture
    "Architecture",
    "detect_architecture",
    "detect_architecture_from_config",
    "discover_layers",
    "discover_layers_from_config",
    "get_block_accessor",
    "get_embedding_module",
    "get_head_modules",
    "get_position_embedding_module",
    "validate_interception_layers",
    # Substrate & Execution
    "ForwardResult",
    "FrozenSubstrate",
    "Substrate",
    # Interception
    "InterceptionContext",
    "ModifyFn",
    "identity_modify",
    "run_with_hooks",
    # Loader
    "load_tokenizer",
    "load_torchax_model",
    # TorchAX Backend & Interop
    "call_jax_differentiable",
    "enable_torchax",
    "from_jax_array",
    "is_on_torchax_device",
    "is_torchax_enabled",
    "to_jax_array",
    "to_torchax_device",
    # TorchAX Models
    "check_numerical_fidelity",
    "functional_model",
    # Drift & Memory
    "compute_kl_drift",
    "MemoryStatus",
    "check_memory_headroom",
    "compute_memory_headroom",
    "get_memory_status",
    "maybe_reduce_batch_size",
]

__version__ = "0.1.0"
