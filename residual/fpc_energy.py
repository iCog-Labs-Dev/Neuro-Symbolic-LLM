"""Squared-error energies of the A2 stability terms, computed by FabricPC nodes.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import jax
import jax.numpy as jnp
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.types import GraphParams, NodeParams
from fabricpc.nodes import IdentityNode, Linear

from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_graph_state
from residual.hep import Highway

BACKENDS = ("fabricpc", "jax")

_INFERENCE = InferenceSGD(eta_infer=0.0, infer_steps=0)


def check_backend(backend: str) -> None:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {list(BACKENDS)}, got {backend!r}")


@cache
def _anchor_graph(d: int) -> Any:
    source = Linear(name="z", shape=(d,))
    anchor = IdentityNode(name="anch", shape=(d,))
    return graph(
        nodes=[source, anchor],
        edges=[Edge(source=source, target=anchor.slot("in"))],
        task_map=TaskMap(x=source, y=anchor),
        inference=_INFERENCE,
    )


@cache
def _highway_graph(d_y: int, d_i: int, rank: int | None) -> Any:
    source = Linear(name="eps", shape=(d_y,))
    out = Linear(name="hep", shape=(d_i,), use_bias=False)
    if rank is None:
        nodes = [source, out]
        edges = [Edge(source=source, target=out.slot("in"))]
    else:
        mid = Linear(name="mid", shape=(rank,), use_bias=False)
        nodes = [source, mid, out]
        edges = [
            Edge(source=source, target=mid.slot("in")),
            Edge(source=mid, target=out.slot("in")),
        ]
    return graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(x=source, y=out),
        inference=_INFERENCE,
    )


def _per_token_energy(
    structure: Any, params: GraphParams, x: jax.Array, y: jax.Array, node: str
) -> jax.Array:
    """Per-sample energy of ``node`` with ``x`` and ``y`` clamped, as a flat vector."""
    clamps = {structure.task_map["x"]: x, structure.task_map["y"]: y}
    state = initialize_graph_state(
        structure, x.shape[0], jax.random.PRNGKey(0), clamps=clamps, params=params
    )
    return state.nodes[node].energy


def anchor_sq_distance(z: jax.Array, anchor: jax.Array) -> jax.Array:
    """``||z - anchor||^2`` per token, ``[..., B, T]``, from an ``IdentityNode`` energy.

    ``anchor`` is clamped as the target, so it must already be under
    ``stop_gradient``; ``z`` carries the gradient.
    """
    lead = z.shape[:-1]
    d = z.shape[-1]
    flat_z = z.astype(jnp.float32).reshape(-1, d)
    flat_a = jnp.broadcast_to(anchor.astype(jnp.float32), z.shape).reshape(-1, d)
    params = GraphParams(nodes={"z": NodeParams({}, {}), "anch": NodeParams({}, {})})
    energy = _per_token_energy(_anchor_graph(d), params, flat_z, flat_a, "anch")
    return (2.0 * energy).reshape(lead)


def highway_sq_norm(eps: jax.Array, v: Highway) -> jax.Array:
    """``||eps V||^2`` per token, ``[..., B, T]``, from ``Linear`` node energies.

    ``v`` is a ``[d_y, d_i]`` matrix (one node) or a ``(U, W)`` pair (two nodes;
    the ``[.., d_y] x [d_y, r]`` intermediate is never materialised as ``V``).
    The target is clamped to zero.
    """
    lead = eps.shape[:-1]
    d_y = eps.shape[-1]
    flat = eps.astype(jnp.float32).reshape(-1, d_y)
    if isinstance(v, tuple):
        u, w = v
        if u.ndim != 2 or w.ndim != 2 or u.shape[1] != w.shape[0]:
            raise ValueError(
                f"low-rank pair must be [d_y, r] and [r, d_i], got {u.shape}, {w.shape}"
            )
        structure = _highway_graph(int(u.shape[0]), int(w.shape[1]), int(u.shape[1]))
        nodes = {
            "eps": NodeParams({}, {}),
            "mid": NodeParams({"eps->mid:in": u.astype(jnp.float32)}, {}),
            "hep": NodeParams({"mid->hep:in": w.astype(jnp.float32)}, {}),
        }
        d_i = int(w.shape[1])
    else:
        if v.ndim != 2:
            raise ValueError(f"V must be a matrix, got shape {v.shape}")
        structure = _highway_graph(int(v.shape[0]), int(v.shape[1]), None)
        nodes = {
            "eps": NodeParams({}, {}),
            "hep": NodeParams({"eps->hep:in": v.astype(jnp.float32)}, {}),
        }
        d_i = int(v.shape[1])
    zeros = jnp.zeros((flat.shape[0], d_i), jnp.float32)
    energy = _per_token_energy(structure, GraphParams(nodes=nodes), flat, zeros, "hep")
    return (2.0 * energy).reshape(lead)


__all__ = ["BACKENDS", "anchor_sq_distance", "check_backend", "highway_sq_norm"]