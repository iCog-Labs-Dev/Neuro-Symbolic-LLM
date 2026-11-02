"""JAX query projection, symbolic attention and hidden-state writeback."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp

from symbolic_memory.jax_bridge import RetrievedTemplates

HeadParams = dict[str, dict[str, jax.Array]]
Retriever = Callable[[jax.Array], RetrievedTemplates]
ModifyFn = Callable[[jax.Array, int], jax.Array]


class SymbolicHead:
    """Apply query projection, symbolic attention and residual writeback.

    Row-vector equations: q = h @ W_sym + b_sym, context = softmax(q K^T) V,
    and h_out = h + context @ U_out. W_sym has shape [d, k] and U_out [k, d].

    Inputs are settled hidden states and a retriever bound to a fixed bank.
    Model loading, latent inference and alignment losses are external.
    """

    def __init__(
        self,
        hidden_size: int,
        key_dim: int,
        layers: Sequence[int],
        retrieve: Retriever,
        *,
        query_init_std: float | None = None,
    ) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (hidden_size, key_dim)
        ):
            raise ValueError("Hidden and symbolic dimensions must be positive integers")
        selected = tuple(layers)
        if (
            not selected
            or any(
                isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
                for layer in selected
            )
            or len(set(selected)) != len(selected)
        ):
            raise ValueError("Head layers must be unique nonnegative integers")
        std = 1.0 / math.sqrt(hidden_size) if query_init_std is None else query_init_std
        if isinstance(std, bool) or not math.isfinite(std) or std <= 0:
            raise ValueError("Query initialization standard deviation must be positive")
        self.hidden_size = hidden_size
        self.key_dim = key_dim
        self.layers = tuple(sorted(selected))
        self.query_init_std = float(std)
        self.retrieve = retrieve

    def init_params(self, seed: int = 0) -> HeadParams:
        """Initialize query weights and an exactly zero writeback contribution."""
        key = jax.random.PRNGKey(seed)
        return {
            f"layer_{layer}": {
                "W_sym": self.query_init_std
                * jax.random.normal(
                    jax.random.fold_in(key, layer),
                    (self.hidden_size, self.key_dim),
                    dtype=jnp.float32,
                ),
                "b_sym": jnp.zeros((self.key_dim,), dtype=jnp.float32),
                "U_out": jnp.zeros((self.key_dim, self.hidden_size), dtype=jnp.float32),
            }
            for layer in self.layers
        }

    def project(self, params: Mapping[str, jax.Array], hidden: jax.Array) -> jax.Array:
        """Project hidden states without normalization or temperature scaling."""
        if hidden.ndim < 1 or hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Hidden state must have last dimension {self.hidden_size}"
            )
        expected = {
            "W_sym": (self.hidden_size, self.key_dim),
            "b_sym": (self.key_dim,),
            "U_out": (self.key_dim, self.hidden_size),
        }
        if set(params) != set(expected) or any(
            params[name].shape != shape for name, shape in expected.items()
        ):
            raise ValueError(
                "Head parameter keys or shapes do not match its dimensions"
            )
        return hidden.astype(jnp.float32) @ params["W_sym"] + params["b_sym"]

    def apply(self, params: HeadParams, hidden: jax.Array, layer_idx: int) -> jax.Array:
        if layer_idx not in self.layers:
            return hidden
        layer = params[f"layer_{layer_idx}"]
        queries = self.project(layer, hidden)
        flat = queries.reshape((-1, self.key_dim))
        templates = self.retrieve(flat)
        if (
            templates.keys.ndim != 3
            or templates.keys.shape[0] != flat.shape[0]
            or templates.keys.shape[-1] != self.key_dim
            or templates.values.shape != templates.keys.shape
            or templates.keys.shape[1] == 0
        ):
            raise ValueError(
                "Retrieved K/V must cover every query in the symbolic space"
            )
        # Attention gradients use the query before retrieval's stop_gradient.
        logits = jnp.einsum("bd,bmd->bm", flat, templates.keys)
        weights = jax.nn.softmax(logits, axis=-1)
        context = jnp.einsum("bm,bmd->bd", weights, templates.values)
        context = context.reshape(queries.shape)
        contribution = context @ layer["U_out"]
        return hidden + contribution.astype(hidden.dtype)

    def modify_fn(
        self, params: HeadParams, continuous_modify: ModifyFn | None = None
    ) -> ModifyFn:
        """Apply the optional continuous residual hook before symbolic attention."""

        def modify(hidden: jax.Array, layer_idx: int) -> jax.Array:
            settled = (
                hidden
                if continuous_modify is None
                else continuous_modify(hidden, layer_idx)
            )
            return self.apply(params, settled, layer_idx)

        return modify
