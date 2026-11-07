"""Highway error propagation (HEP) matrices ``V_{y -> i}`` for the stability loss."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TypeAlias

import jax
import jax.numpy as jnp

HEPParams: TypeAlias = dict[str, dict[str, jax.Array]]
Highway: TypeAlias = "jax.Array | tuple[jax.Array, jax.Array]"

HEP_PREFIX = "hep_"
KINDS = ("dense", "lowrank", "fixed_random", "tied")
TRAINABLE_KINDS = ("dense", "lowrank")

# Same stream root as the CPC heads, disjoint kind codes (CPC uses 1 and 2).
_STREAM = 0xA2
_KIND_DENSE = 3
_KIND_U = 4
_KIND_W = 5


def hep_key(layer_idx: int) -> str:
    return f"{HEP_PREFIX}{layer_idx}"


def _check(kind: str, rank: int) -> None:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {list(KINDS)}, got {kind!r}")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise ValueError(f"rank must be an integer >= 1, got {rank!r}")


def _site_key(seed: int, layer: int) -> jax.Array:
    return jax.random.fold_in(
        jax.random.fold_in(jax.random.PRNGKey(seed), _STREAM), layer
    )


def _factors(
    seed: int, layer: int, d_out: int, d_hidden: int, rank: int
) -> tuple[jax.Array, jax.Array]:
    site = _site_key(seed, layer)
    u = jax.random.normal(
        jax.random.fold_in(site, _KIND_U), (d_out, rank), dtype=jnp.float32
    ) / jnp.sqrt(jnp.float32(d_out))
    w = jax.random.normal(
        jax.random.fold_in(site, _KIND_W), (rank, d_hidden), dtype=jnp.float32
    ) / jnp.sqrt(jnp.float32(rank))
    return u, w


def init_hep(
    kind: str,
    seed: int,
    layers: Iterable[int],
    d_out: int,
    d_hidden: int,
    rank: int = 16,
) -> HEPParams:
    """Trainable HEP parameters for ``layers``; ``{}`` for the fixed kinds.

    Entries are ``N(0, 1/fan_in)``, so ``eps @ V`` has roughly unit variance per
    unit-variance error component; the gain ``alpha`` sets the real strength.
    """
    _check(kind, rank)
    if d_out < 1 or d_hidden < 1:
        raise ValueError(f"d_out and d_hidden must be >= 1, got {d_out}, {d_hidden}")
    if kind not in TRAINABLE_KINDS:
        return {}
    params: HEPParams = {}
    for layer in sorted(set(layers)):
        if kind == "dense":
            key = jax.random.fold_in(_site_key(seed, layer), _KIND_DENSE)
            params[hep_key(layer)] = {
                "V": jax.random.normal(key, (d_out, d_hidden), dtype=jnp.float32)
                / jnp.sqrt(jnp.float32(d_out))
            }
        else:
            u, w = _factors(seed, layer, d_out, d_hidden, rank)
            params[hep_key(layer)] = {"U": u, "W": w}
    return params


def highway_for(
    kind: str,
    layer: int,
    *,
    params: Mapping[str, Mapping[str, jax.Array]] | None = None,
    seed: int = 0,
    d_out: int,
    d_hidden: int,
    rank: int = 16,
    tied: jax.Array | None = None,
) -> Highway:
    """The matrix ``V_{y -> layer}``: from ``params`` (trainable kinds), generated
    from ``seed`` (``fixed_random``) or the supplied ``tied`` matrix."""
    _check(kind, rank)
    if kind in TRAINABLE_KINDS:
        if params is None or hep_key(layer) not in params:
            raise KeyError(
                f"missing HEP parameters {hep_key(layer)!r} for kind={kind!r}"
            )
        entry = params[hep_key(layer)]
        return entry["V"] if kind == "dense" else (entry["U"], entry["W"])
    if kind == "fixed_random":
        return _factors(seed, layer, d_out, d_hidden, rank)
    if tied is None:
        raise ValueError("kind='tied' needs the `tied` matrix [d_y, d_i].")
    if tied.shape != (d_out, d_hidden):
        raise ValueError(f"tied must have shape {(d_out, d_hidden)}, got {tied.shape}")
    return tied


def apply_highway(v: Highway, eps: jax.Array) -> jax.Array:
    """``eps @ V`` in float32; a low-rank ``V`` is never materialised."""
    eps = eps.astype(jnp.float32)
    if isinstance(v, tuple):
        u, w = v
        return (eps @ u) @ w
    return eps @ v


def split_hep(params: Mapping[str, Mapping[str, jax.Array]]) -> HEPParams:
    """The HEP entries of a full A2 parameter tree."""
    return {k: dict(v) for k, v in params.items() if k.startswith(HEP_PREFIX)}


__all__ = [
    "HEP_PREFIX",
    "HEPParams",
    "Highway",
    "KINDS",
    "TRAINABLE_KINDS",
    "apply_highway",
    "hep_key",
    "highway_for",
    "init_hep",
    "split_hep",
]
