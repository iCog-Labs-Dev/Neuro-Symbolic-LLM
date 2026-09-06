"""Frozen LLM substrate in JAX/Flax.

A reusable wrapper around pretrained causal language models (GPT-2 and
Pythia/GPT-NeoX) that keeps the base model completely frozen while allowing
arbitrary per-layer hidden-state interception, activation caching, drift
monitoring and device memory monitoring.
"""

from .architecture import (
    Architecture)

from .drift import compute_kl_drift

from .memory import (
    MemoryStatus,
    check_memory_headroom,
    compute_memory_headroom,
    get_memory_status,
    maybe_reduce_batch_size,)

from .substrate import ForwardResult, Substrate

__all__ = [
    "Architecture",
    "ForwardResult",
    "MemoryStatus",
    "Substrate"
    "check_memory_headroom",
    "compute_kl_drift",
    "compute_memory_headroom",
    "get_memory_status",
    "maybe_reduce_batch_size",
]

__version__ = "0.1.0"
