"""Hidden-state residual adapter on frozen blocks.

    h~ = h + s * sigma(h @ A) @ B,   A: [d, r],  B: [r, d],  s = alpha / r (or 1.0)

``B = 0`` at init, so the adapted model starts equal to the base. Parameters are
a separate PyTree, applied through ``modify_fn`` hooks on the substrate.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp

from frozenllm.substrate.architecture import (
    Architecture,
    validate_interception_layers,
)

AdapterParams = dict[str, dict[str, jax.Array]]

ACTIVATIONS: Mapping[str, Callable[[jax.Array], jax.Array]] = MappingProxyType(
    {
        "identity": lambda x: x,
        "gelu": jax.nn.gelu,
        "relu": jax.nn.relu,
        "tanh": jnp.tanh,
    }
)

MIN_NUM_LAYERS = 5


@dataclass(frozen=True)
class ResidualConfig:
    """Adapter hyper-parameters; mirrors the ``adapter:`` YAML section."""

    rank: int = 16
    activation: str = "identity"
    alpha: float | None = None
    init_std: float | None = None
    late_start: int | None = None
    late_end: int | None = None
    # Explicit blocks; replaces the late range, exclusive with late_start/late_end.
    layers: list[int] | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError(f"rank must be an integer >= 1, got {self.rank!r}")
        if self.activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {self.activation!r}. "
                f"Expected one of {sorted(ACTIVATIONS)}."
            )
        if self.alpha is not None and not self.alpha > 0:
            raise ValueError(f"alpha must be > 0 or null, got {self.alpha!r}")
        if self.init_std is not None and not self.init_std > 0:
            raise ValueError(f"init_std must be > 0 or null, got {self.init_std!r}")
        for name in ("late_start", "late_end"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, int):
                raise ValueError(f"{name} must be an integer or null, got {value!r}")
        if self.layers is not None:
            if (
                not isinstance(self.layers, list | tuple)
                or not self.layers
                or not all(
                    isinstance(i, int) and not isinstance(i, bool) for i in self.layers
                )
            ):
                raise ValueError(
                    f"layers must be a non-empty list of integers or null, "
                    f"got {self.layers!r}"
                )
            if self.late_start is not None or self.late_end is not None:
                raise ValueError(
                    "Set either layers or late_start/late_end, not both "
                    f"(layers={self.layers!r}, late_start={self.late_start!r}, "
                    f"late_end={self.late_end!r})."
                )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResidualConfig:
        return cls(**_checked_kwargs(cls, data))

    @property
    def scale(self) -> float:
        return 1.0 if self.alpha is None else float(self.alpha) / self.rank


def _checked_kwargs(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(
            f"Unknown {cls.__name__} keys: {unknown}. Allowed: {sorted(known)}."
        )
    return dict(data)


def resolve_late_layers(
    num_layers: int,
    late_start: int | None = None,
    late_end: int | None = None,
) -> tuple[int, ...]:
    l_mid = num_layers // 2
    start = l_mid + 1 if late_start is None else late_start
    end = num_layers - 2 if late_end is None else late_end
    valid = f"valid: {l_mid} < late_start <= late_end <= {num_layers - 2}"
    if num_layers < MIN_NUM_LAYERS:
        raise ValueError(
            f"Late-layer adapter needs a model with >= {MIN_NUM_LAYERS} blocks; "
            f"model has {num_layers}."
        )
    if not l_mid < start <= end <= num_layers - 2:
        raise ValueError(
            f"Invalid late-layer bounds late_start={start}, late_end={end} for a "
            f"model with {num_layers} blocks (0..{num_layers - 1}); {valid}."
        )
    return tuple(range(start, end + 1))


def layer_key(layer_idx: int) -> str:
    return f"layer_{layer_idx}"


class ResidualAdapter:
    def __init__(self, config: ResidualConfig, architecture: Architecture) -> None:
        self.config = config
        self.architecture = architecture
        self.hidden_size = architecture.hidden_size
        # Both paths are validated by the substrate (in range, unique) and sorted.
        requested = (
            config.layers
            if config.layers is not None
            else resolve_late_layers(
                architecture.num_layers, config.late_start, config.late_end
            )
        )
        self.layers = validate_interception_layers(requested, architecture.num_layers)
        self.scale = config.scale
        self.init_std = (
            config.init_std
            if config.init_std is not None
            else 1.0 / math.sqrt(self.hidden_size)
        )
        self._activation = ACTIVATIONS[config.activation]

    def init_params(self) -> AdapterParams:
        base = jax.random.PRNGKey(self.config.seed)
        d, r = self.hidden_size, self.config.rank
        # Keys fold in the block index, so a block's A is stable if the range moves. B = 0 zeroes the residual while dL/dB stays non-zero.
        return {
            layer_key(idx): {
                "A": self.init_std
                * jax.random.normal(
                    jax.random.fold_in(base, idx), (d, r), dtype=jnp.float32
                ),
                "B": jnp.zeros((r, d), dtype=jnp.float32),
            }
            for idx in self.layers
        }

    def encode(self, layer_params: Mapping[str, jax.Array], h: jax.Array) -> jax.Array:
        # Upcast so the bottleneck keeps precision on bf16/fp16 substrates.
        h32 = h.astype(jnp.float32)
        return self._activation(h32 @ layer_params["A"])

    def decode(self, layer_params: Mapping[str, jax.Array], z: jax.Array) -> jax.Array:
        return self.scale * (z @ layer_params["B"])

    def residual(
        self, layer_params: Mapping[str, jax.Array], h: jax.Array
    ) -> jax.Array:
        return self.decode(layer_params, self.encode(layer_params, h))

    def apply(self, params: AdapterParams, h: jax.Array, layer_idx: int) -> jax.Array:
        layer_params = params.get(layer_key(layer_idx))
        if layer_params is None:
            return h
        # Cast back so downstream frozen blocks see their native dtype.
        return h + self.residual(layer_params, h).astype(h.dtype)

    def modify_fn(self, params: AdapterParams) -> Callable[[jax.Array, int], jax.Array]:
        # Closing over params (possibly tracers) lets jax.grad reach the adapter.
        def modify(hidden: jax.Array, layer_idx: int) -> jax.Array:
            return self.apply(params, hidden, layer_idx)

        return modify

    @staticmethod
    def num_params(params: AdapterParams) -> int:
        return sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(params))

    @staticmethod
    def l2_norm_sq(params: AdapterParams) -> jax.Array:
        # Array-valued start so an empty PyTree still yields a float32 scalar.
        return sum(
            (jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(params)),
            start=jnp.zeros((), dtype=jnp.float32),
        )


__all__ = [
    "ACTIVATIONS",
    "AdapterParams",
    "ResidualAdapter",
    "ResidualConfig",
    "layer_key",
    "resolve_late_layers",
]
