"""Trainable residuals on top of the frozen substrate."""

from .adapter import (
    ACTIVATIONS,
    AdapterParams,
    ResidualAdapter,
    ResidualConfig,
    layer_key,
    resolve_late_layers,
)

__all__ = [
    "ACTIVATIONS",
    "AdapterParams",
    "ResidualAdapter",
    "ResidualConfig",
    "layer_key",
    "resolve_late_layers",
]
