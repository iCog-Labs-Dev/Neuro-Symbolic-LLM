"""Frozen LLM substrate in JAX/Flax.

A reusable wrapper around pretrained causal language models (GPT-2 and
Pythia/GPT-NeoX) that keeps the base model completely frozen while allowing
arbitrary per-layer hidden-state interception, activation caching, drift
monitoring and device memory monitoring.
"""

from .architecture import (
    Architecture,
    detect_architecture,
    discover_layers,
    validate_interception_layers,
)
from .drift import compute_kl_drift
from .loader import (
    build_substrate_from_state_dict,
    load_substrate_from_hf,
    state_dict_to_jax_pytree,
)
from .memory import (
    MemoryStatus,
    check_memory_headroom,
    compute_memory_headroom,
    get_memory_status,
    maybe_reduce_batch_size,
)
from .substrate import ForwardResult, FrozenJAXSubstrate

__all__ = [
    "Architecture",
    "ForwardResult",
    "FrozenJAXSubstrate",
    "MemoryStatus",
    "build_substrate_from_state_dict",
    "check_memory_headroom",
    "compute_kl_drift",
    "compute_memory_headroom",
    "detect_architecture",
    "discover_layers",
    "get_memory_status",
    "load_substrate_from_hf",
    "maybe_reduce_batch_size",
    "state_dict_to_jax_pytree",
    "validate_interception_layers",
]

__version__ = "0.1.0"
