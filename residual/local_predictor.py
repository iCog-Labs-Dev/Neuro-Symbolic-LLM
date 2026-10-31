"""Local hidden-state predictors for Stage A2, built from FabricPC nodes.

For each adapted site ``l`` a small predictor ``P_l`` guesses the next hidden state
from the residual-corrected state ``u = h~_l``::

    P_l(u) = W2 GELU(W1 LN(u) + b1) + b2

It is one FabricPC graph::

    IdentityNode -> LnMlp1Node (LayerNorm -> Linear -> GELU) -> Linear (identity act.)

The graph is evaluated with ``FeedforwardStateInit`` and ``inference=None``: a single
feedforward pass, **no iterative predictive-coding settling**. Training is ordinary
backprop through ``jax.grad`` (the A2 objective), never ``algorithm="pc"``.

Parameters are exposed as a flat ``{outer: {inner: array}}`` PyTree (one ``outer`` key per
site, ``pred_layer_<l>``) so they can sit in the same dict as the adapter parameters
(``layer_<l>``) and reuse the A1 optimizer and checkpoint code unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import jax
import jax.numpy as jnp
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.types import GraphParams, GraphStructure, NodeParams
from fabricpc.graph import (
    FeedforwardStateInit,
    initialize_graph_state,
    initialize_params,
)
from fabricpc.nodes import IdentityNode, Linear, LnMlp1Node

from frozenllm.substrate.architecture import Architecture, validate_interception_layers
from residual.adapter import layer_key

PredictorParams = dict[str, dict[str, jax.Array]]

PREDICTOR_PREFIX = "pred_"
INPUT_NODE = "pred_input"
HIDDEN_NODE = "pred_hidden"
OUTPUT_NODE = "pred_output"
# Flat-PyTree name of the output projection weight (FabricPC keys it by edge).
_OUT_WEIGHT = "W"


def predictor_key(layer_idx: int) -> str:
    return f"{PREDICTOR_PREFIX}{layer_key(layer_idx)}"


@dataclass(frozen=True)
class PredictorConfig:
    """``objective.predictor`` section of the A2 YAML."""

    ff_dim: int = 256
    out_init_std: float | None = None
    predict_delta: bool = False
    detach_input: bool = True
    layers: list[int] | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.ff_dim, int)
            or isinstance(self.ff_dim, bool)
            or self.ff_dim < 1
        ):
            raise ValueError(f"ff_dim must be an integer >= 1, got {self.ff_dim!r}")
        if self.out_init_std is not None and (
            not math.isfinite(self.out_init_std) or not self.out_init_std > 0
        ):
            raise ValueError(
                f"out_init_std must be > 0 or null, got {self.out_init_std!r}"
            )
        for name in ("predict_delta", "detach_input"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(
                    f"{name} must be a boolean, got {getattr(self, name)!r}"
                )
        if self.layers is not None and (
            not isinstance(self.layers, list | tuple)
            or not self.layers
            or not all(
                isinstance(i, int) and not isinstance(i, bool) for i in self.layers
            )
        ):
            raise ValueError(
                f"layers must be a non-empty list of integers or null, got {self.layers!r}"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PredictorConfig:
        return cls(**_checked_kwargs(cls, data))


def _checked_kwargs(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(
            f"Unknown {cls.__name__} keys: {unknown}. Allowed: {sorted(known)}."
        )
    return dict(data)


def split_params(params: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a combined A2 PyTree into ``(adapter_params, predictor_params)``."""
    adapter = {k: v for k, v in params.items() if not k.startswith(PREDICTOR_PREFIX)}
    predictor = {k: v for k, v in params.items() if k.startswith(PREDICTOR_PREFIX)}
    return adapter, predictor


def merge_params(
    adapter_params: Mapping[str, Any], predictor_params: Mapping[str, Any]
) -> dict[str, Any]:
    clash = set(adapter_params) & set(predictor_params)
    if clash:
        raise ValueError(
            f"Adapter and predictor parameter keys overlap: {sorted(clash)}"
        )
    return {**adapter_params, **predictor_params}


class LocalPredictorBank:
    def __init__(
        self,
        config: PredictorConfig,
        architecture: Architecture,
        seq_len: int,
        default_layers: Sequence[int],
    ) -> None:
        self.config = config
        self.architecture = architecture
        self.seq_len = seq_len
        self.hidden_size = architecture.hidden_size
        requested = config.layers if config.layers is not None else default_layers
        layers = validate_interception_layers(requested, architecture.num_layers)
        # P_l predicts block l+1, which must exist.
        usable = tuple(i for i in layers if i + 1 < architecture.num_layers)
        if config.layers is not None and usable != layers:
            raise ValueError(
                f"Predictor layers {list(layers)} need a following block; "
                f"model has {architecture.num_layers} blocks."
            )
        if not usable:
            raise ValueError(
                "No predictor layers: every requested site is the last block."
            )
        self.layers = usable
        self.structure = self._build_structure()

    def _build_structure(self) -> GraphStructure:
        s, d, ff = self.seq_len, self.hidden_size, self.config.ff_dim
        pred_input = IdentityNode(shape=(s, d), name=INPUT_NODE)
        pred_hidden = LnMlp1Node(
            shape=(s, ff), name=HIDDEN_NODE, embed_dim=d, ff_dim=ff
        )
        out_kwargs: dict[str, Any] = {}
        if self.config.out_init_std is not None:
            out_kwargs["weight_init"] = NormalInitializer(
                mean=0.0, std=self.config.out_init_std
            )
        pred_output = Linear(
            shape=(s, d),
            name=OUTPUT_NODE,
            activation=IdentityActivation(),
            **out_kwargs,
        )
        return graph(
            nodes=[pred_input, pred_hidden, pred_output],
            edges=[
                Edge(source=pred_input, target=pred_hidden.slot("in")),
                Edge(source=pred_hidden, target=pred_output.slot("in")),
            ],
            task_map=TaskMap(x=pred_input, y=pred_output),
            inference=None,
            graph_state_initializer=FeedforwardStateInit(),
        )

    # parameters
    def _flatten(self, gp: GraphParams) -> dict[str, jax.Array]:
        flat: dict[str, jax.Array] = {}
        for node_name, node_params in gp.nodes.items():
            in_edges = self.structure.nodes[node_name].node_info.in_edges
            for wname, w in node_params.weights.items():
                flat[f"{node_name}.w.{_OUT_WEIGHT if wname in in_edges else wname}"] = w
            for bname, b in node_params.biases.items():
                flat[f"{node_name}.b.{bname}"] = b
        return flat

    def _unflatten(self, flat: Mapping[str, jax.Array]) -> GraphParams:
        nodes: dict[str, NodeParams] = {}
        for node_name, node in self.structure.nodes.items():
            info = node.node_info
            weights: dict[str, jax.Array] = {}
            biases: dict[str, jax.Array] = {}
            prefix_w, prefix_b = f"{node_name}.w.", f"{node_name}.b."
            for key, val in flat.items():
                if key.startswith(prefix_w):
                    name = key[len(prefix_w) :]
                    weights[info.in_edges[0] if name == _OUT_WEIGHT else name] = val
                elif key.startswith(prefix_b):
                    biases[key[len(prefix_b) :]] = val
            nodes[node_name] = NodeParams(weights=weights, biases=biases)
        return GraphParams(nodes=nodes)

    def init_params(self) -> PredictorParams:
        base = jax.random.PRNGKey(self.config.seed)
        # Fold in the block index so a site's weights are stable if the layer set changes.
        return {
            predictor_key(i): self._flatten(
                initialize_params(self.structure, jax.random.fold_in(base, i))
            )
            for i in self.layers
        }

    @staticmethod
    def num_params(params: Mapping[str, Any]) -> int:
        return sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(dict(params)))

    # forward

    def predict(
        self, params: Mapping[str, Any], layer_idx: int, u: jax.Array
    ) -> jax.Array:
        """Feedforward prediction ``P_l(u)`` (plus ``u`` when ``predict_delta``).

        ``u``: ``[B, S, D]`` with ``S == seq_len``. Returns float32 ``[B, S, D]``.
        """
        if u.ndim != 3 or u.shape[1] != self.seq_len or u.shape[2] != self.hidden_size:
            raise ValueError(
                f"predictor input must be [B, {self.seq_len}, {self.hidden_size}], "
                f"got {tuple(u.shape)}"
            )
        key = predictor_key(layer_idx)
        if key not in params:
            raise KeyError(f"No predictor parameters for layer {layer_idx} ({key!r}).")
        u32 = u.astype(jnp.float32)
        if self.config.detach_input:
            u32 = jax.lax.stop_gradient(u32)
        state = initialize_graph_state(
            self.structure,
            u32.shape[0],
            jax.random.PRNGKey(0),
            clamps={INPUT_NODE: u32},
            state_init=FeedforwardStateInit(),
            params=self._unflatten(params[key]),
        )
        out = state.nodes[OUTPUT_NODE].z_mu
        return u32 + out if self.config.predict_delta else out


__all__ = [
    "PREDICTOR_PREFIX",
    "LocalPredictorBank",
    "PredictorConfig",
    "PredictorParams",
    "merge_params",
    "predictor_key",
    "split_params",
]
