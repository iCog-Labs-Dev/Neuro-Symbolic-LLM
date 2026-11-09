"""FrozenTail: the one A3-specific node in the FabricPC graph (FabricPC 0.6.0 node contract).

    h0_l (clamped) --> z_l  Linear + GaussianEnergy      f_phi(h0) and the spring
    z_l ------------> tail  FrozenTail                   frozen model + residual, task energy
    ids (clamped) --> tail
    base_logp (clamped, only when lambda_kl > 0) --> tail

What the node does
  * parameters: one B_l [r, d] per incoming z edge (``z @ B``).
  * predict:    ONE forward of the frozen substrate. The residual is injected through the
                substrate's ``modify_fn``: h~_l = h_l + scale * z_l B_l on the LIVE stream, so the
                residual of an earlier adapted layer flows into the later ones (the same rule as
                stage A1). z_mu = next-token probabilities at positions 0..T-2.
  * energy:     -sum log p(target) + lambda_kl * KL(p_base || p), from log-softmax. FabricPC's
                stock CrossEntropyEnergy clips probabilities, which zeroes the gradient of the
                tokens the model gets most wrong; this does not.
  * baseline:   the base log-probabilities arrive as DATA (clamped ``base_logp`` node, computed
                once by the clean pass). The tail never runs a baseline forward of its own.
  * gradients:  differentiate the energy w.r.t. each z edge and the tail's own target latent.
                ids and base_logp are data and get zero gradients.

The substrate is bound to a subclass by make_frozen_tail(), so no object lives in the node config.
"""

from __future__ import annotations

import math
import re
from numbers import Integral, Real
from typing import Any

import jax
import jax.numpy as jnp
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.initializers import NormalInitializer, initialize
from fabricpc.core.types import NodeParams
from fabricpc.nodes.base import NodeBase, SlotSpec

_Z_EDGE = re.compile(r"(?:^|/)z(\d+)->")


# ---------------------------------------------------------------------- validation helpers
def _is_int(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _check_real(label: str, value: Any, *, allow_zero: bool) -> float:
    ok = (
        isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    )
    if not ok or value < 0 or (value == 0 and not allow_zero):
        relation = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{label} must be a finite number {relation}, got {value!r}")
    return float(value)


def _check_layers(layers: Any) -> tuple[int, ...]:
    try:
        raw = tuple(layers)
    except TypeError as exc:
        raise ValueError("layers must be a non-empty sequence of integers") from exc
    if not raw:
        raise ValueError("FrozenTail requires at least one intercepted layer")
    if not all(_is_int(x) for x in raw):
        raise ValueError(f"layer indices must be integers, got {raw}")
    out = tuple(int(x) for x in raw)
    if len(set(out)) != len(out) or any(x < 0 for x in out):
        raise ValueError(f"layers must be unique and non-negative, got {out}")
    return out


def layer_of(edge_key: str) -> int:
    """Intercepted-layer index from a z edge key such as 'z4->tail:z'."""
    match = _Z_EDGE.search(edge_key)
    if match is None:
        raise ValueError(f"cannot read a layer index from edge key {edge_key!r}")
    return int(match.group(1))


def _z_edges(keys) -> dict[int, str]:
    """{layer: edge_key} for the ':z' edges; rejects two edges for one layer."""
    edges: dict[int, str] = {}
    for key in keys:
        if key.endswith(":z"):
            layer = layer_of(key)
            if layer in edges:
                raise ValueError(
                    f"duplicate z edge for layer {layer}: {edges[layer]!r}, {key!r}"
                )
            edges[layer] = key
    return edges


def _split_inputs(inputs, layers, use_base):
    """Validate the incoming edges and return ({layer: (edge_key, z)}, ids, base_logp)."""
    z_keys = _z_edges(inputs)
    if set(z_keys) != set(layers):
        raise ValueError(
            f"z edges {sorted(z_keys)} do not match configured layers {sorted(layers)}"
        )
    ids = base_logp = None
    for key, value in inputs.items():
        if key.endswith(":ids"):
            ids = value
        elif key.endswith(":base_logp"):
            base_logp = value
        elif not key.endswith(":z"):
            raise ValueError(f"unexpected FrozenTail input edge {key!r}")
    if ids is None:
        raise ValueError("FrozenTail requires an incoming :ids edge")
    if ids.ndim != 2 or ids.shape[1] < 2:
        raise ValueError(
            f"ids must have shape [batch, seq >= 2], got {tuple(ids.shape)}"
        )
    if not jnp.issubdtype(ids.dtype, jnp.integer):
        raise TypeError(f"ids must have an integer dtype, got {ids.dtype}")
    for layer, key in z_keys.items():
        z = inputs[key]
        if z.ndim != 3 or z.shape[:2] != ids.shape:
            raise ValueError(
                f"z at layer {layer} must have shape [batch, seq, rank] matching ids "
                f"{tuple(ids.shape)}, got {tuple(z.shape)}"
            )
    if use_base and base_logp is None:
        raise ValueError(
            "lambda_kl > 0 requires base log-probabilities on the :base_logp edge"
        )
    z_by_layer = {layer: (key, inputs[key]) for layer, key in z_keys.items()}
    return z_by_layer, ids, (base_logp if use_base else None)


def residual_modify(z, projections, scale):
    """modify_fn for FrozenSubstrate.run_with_interception.

    Adds scale * z_l @ B_l to the LIVE hidden state of each layer in ``z`` (computed in float32,
    cast back to the hidden dtype). Layers not in ``z`` pass through unchanged. Used by the tail
    and by the model's deployment and exact-energy passes, so all three apply one rule.
    """
    scale = _check_real("scale", scale, allow_zero=False)

    def modify(hidden, layer_idx):
        if layer_idx not in z:
            return hidden
        if layer_idx not in projections:
            raise ValueError(f"missing projection B for intercepted layer {layer_idx}")
        latent, proj = z[layer_idx], projections[layer_idx]
        if proj.ndim != 2 or latent.ndim < 2:
            raise ValueError(
                f"layer {layer_idx}: need z [..., rank] and B [rank, hidden], "
                f"got z={tuple(latent.shape)}, B={tuple(proj.shape)}"
            )
        if latent.shape[-1] != proj.shape[0] or hidden.shape[-1] != proj.shape[1]:
            raise ValueError(
                f"layer {layer_idx}: shape mismatch z={tuple(latent.shape)}, "
                f"B={tuple(proj.shape)}, hidden={tuple(hidden.shape)}"
            )
        if hidden.shape[:-1] != latent.shape[:-1]:
            raise ValueError(
                f"layer {layer_idx}: batch/sequence mismatch hidden={tuple(hidden.shape)}, "
                f"z={tuple(latent.shape)}"
            )
        residual = scale * (latent.astype(jnp.float32) @ proj.astype(jnp.float32))
        return hidden + residual.astype(hidden.dtype)

    return modify


class FrozenTail(NodeBase):
    _substrate: Any = None  # bound by make_frozen_tail()

    def __init__(
        self,
        shape,
        name,
        *,
        layers,
        hidden_size,
        scale=1.0,
        projection_scale=0.01,
        lambda_kl=0.0,
        activation=None,
        energy=None,
        latent_init=None,
        weight_init=None,
    ):
        layers = _check_layers(layers)
        try:
            shape = tuple(shape)
        except TypeError as exc:
            raise ValueError("FrozenTail shape must be [sequence, vocabulary]") from exc
        if len(shape) != 2 or not all(_is_int(size) and size > 0 for size in shape):
            raise ValueError(
                f"FrozenTail shape must be [sequence, vocabulary], got {shape}"
            )
        sub = type(self)._substrate
        if sub is not None and shape[1] != sub.architecture.vocab_size:
            raise ValueError(
                f"FrozenTail vocabulary size {shape[1]} does not match substrate "
                f"vocabulary size {sub.architecture.vocab_size}"
            )
        if not _is_int(hidden_size) or hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be a positive integer, got {hidden_size!r}"
            )
        scale = _check_real("scale", scale, allow_zero=False)
        projection_scale = _check_real(
            "projection_scale", projection_scale, allow_zero=False
        )
        lambda_kl = _check_real("lambda_kl", lambda_kl, allow_zero=True)
        # `energy` is only the declared label: the node overrides energy() below.
        super().__init__(
            shape=shape,
            name=name,
            activation=activation or IdentityActivation(),
            energy=energy or CrossEntropyEnergy(),
            latent_init=latent_init or NormalInitializer(),
            weight_init=weight_init
            or NormalInitializer(mean=0.0, std=projection_scale),
            layers=layers,
            hidden_size=int(hidden_size),
            scale=scale,
            lambda_kl=lambda_kl,
        )

    @staticmethod
    def get_slots() -> dict[str, SlotSpec]:
        return {
            "z": SlotSpec(name="z", is_multi_input=True),
            "ids": SlotSpec(name="ids", is_multi_input=False),
            "base_logp": SlotSpec(name="base_logp", is_multi_input=False),
        }

    @staticmethod
    def initialize_params(
        key, node_shape, input_shapes, weight_init, config
    ) -> NodeParams:
        """One B_l [r, d] per z edge, small and random (never zero: with B = 0 the gradient on z
        is exactly zero and inference could not move z)."""
        d = int(config["hidden_size"])
        ids_keys = [key for key in input_shapes if key.endswith(":ids")]
        if len(ids_keys) != 1:
            raise ValueError("FrozenTail requires exactly one incoming :ids edge")
        ids_shape = tuple(input_shapes[ids_keys[0]])
        if len(ids_shape) != 1 or ids_shape[0] < 2:
            raise ValueError(f"ids input must have shape [seq >= 2], got {ids_shape}")
        if len(node_shape) != 2 or node_shape[0] != ids_shape[0] - 1:
            raise ValueError(
                f"FrozenTail output shape {tuple(node_shape)} must have "
                f"sequence length {ids_shape[0] - 1}"
            )

        base_keys = [key for key in input_shapes if key.endswith(":base_logp")]
        if float(config["lambda_kl"]) > 0.0 and len(base_keys) != 1:
            raise ValueError("lambda_kl > 0 requires an incoming :base_logp edge")
        if base_keys and tuple(input_shapes[base_keys[0]]) != tuple(node_shape):
            raise ValueError(
                f"base_logp shape {tuple(input_shapes[base_keys[0]])} must match "
                f"FrozenTail output shape {tuple(node_shape)}"
            )

        z_keys = _z_edges(input_shapes)
        if set(z_keys) != set(config["layers"]):
            raise ValueError(
                f"z edges {sorted(z_keys)} must match configured layers {sorted(config['layers'])}"
            )
        weights = {}
        subkeys = jax.random.split(key, len(z_keys))
        for subkey, layer in zip(subkeys, sorted(z_keys), strict=False):
            z_shape = tuple(input_shapes[z_keys[layer]])
            if len(z_shape) != 2 or z_shape[0] != ids_shape[0] or z_shape[-1] <= 0:
                raise ValueError(
                    f"z input for layer {layer} must be "
                    f"[{ids_shape[0]}, rank>0], got {z_shape}"
                )
            weights[z_keys[layer]] = initialize(subkey, (z_shape[-1], d), weight_init)
        return NodeParams(weights=weights, biases={})

    @staticmethod
    def predict(params, inputs, state, node_info):
        sub = node_info.node_class._substrate
        if sub is None:
            raise RuntimeError(
                "FrozenTail has no substrate; build it with make_frozen_tail()"
            )
        cfg = node_info.node_config
        layers = tuple(cfg["layers"])
        use_base = float(cfg["lambda_kl"]) > 0.0
        z_by_layer, ids, base_logp = _split_inputs(inputs, layers, use_base)
        ids = ids.astype(jnp.int32)

        modify = residual_modify(
            {layer: z for layer, (_, z) in z_by_layer.items()},
            {layer: params.weights[key] for layer, (key, _) in z_by_layer.items()},
            float(cfg["scale"]),
        )
        # The only substrate forward in this node: residuals enter through modify_fn.
        logits = sub.run_with_interception(
            ids, modify_fn=modify, intercept_layers=layers
        ).logits
        if logits.ndim != 3 or logits.shape[:2] != ids.shape:
            raise ValueError(
                f"substrate logits must be [batch, seq, vocab] matching ids {tuple(ids.shape)}, "
                f"got {tuple(logits.shape)}"
            )
        logp = jax.nn.log_softmax(logits[:, :-1, :].astype(jnp.float32), axis=-1)
        if base_logp is not None:
            if base_logp.shape != logp.shape:
                raise ValueError(
                    f"base_logp {tuple(base_logp.shape)} must match log-probs {tuple(logp.shape)}"
                )
            base_logp = jax.lax.stop_gradient(base_logp.astype(jnp.float32))
        return jnp.exp(logp), (logp, base_logp)  # (z_mu, aux)

    @staticmethod
    def energy(params, inputs, state, aux, node_info):
        """Per-sample  -sum_t log p(y_t) + lambda_kl * sum_t KL(p_base || p)  (z_latent is the
        one-hot target clamp). The KL term is absent when lambda_kl = 0."""
        if aux is None:
            return NodeBase.energy(params, inputs, state, aux, node_info)
        logp, base_logp = aux
        target = state.z_latent
        if target.shape != logp.shape:
            raise ValueError(
                f"target clamp must be one-hot with shape {tuple(logp.shape)}, "
                f"got {tuple(target.shape)}"
            )
        axes = tuple(range(1, logp.ndim))
        e = -jnp.sum(target.astype(jnp.float32) * logp, axis=axes)
        if base_logp is not None:
            lam = float(node_info.node_config["lambda_kl"])
            e = e + lam * jnp.sum(jnp.exp(base_logp) * (base_logp - logp), axis=axes)
        return e

    @staticmethod
    def forward_and_latent_grads(params, inputs, state, node_info, is_clamped):
        node_class = node_info.node_class
        z_inputs = {k: v for k, v in inputs.items() if k.endswith(":z")}
        fixed = {k: v for k, v in inputs.items() if not k.endswith(":z")}
        if not z_inputs:
            raise ValueError("FrozenTail received no differentiable z inputs")

        def energy_fn(z_in, z_latent):
            new_s = node_class.forward(
                params, {**fixed, **z_in}, state._replace(z_latent=z_latent), node_info
            )
            return jnp.sum(new_s.energy), new_s

        (_, new_state), (z_grads, self_grad) = jax.value_and_grad(
            energy_fn, argnums=(0, 1), has_aux=True
        )(z_inputs, state.z_latent)
        input_grads = dict(z_grads)
        for key, value in fixed.items():  # ids and base_logp are data: no gradient
            input_grads[key] = jnp.zeros(value.shape, jnp.float32)
        return new_state, input_grads, self_grad


def make_frozen_tail(substrate):
    """FrozenTail class bound to a substrate (anything with run_with_interception)."""
    if substrate is None or not callable(
        getattr(substrate, "run_with_interception", None)
    ):
        raise TypeError(
            "make_frozen_tail requires a substrate with run_with_interception(...)"
        )
    return type("FrozenTail", (FrozenTail,), {"_substrate": substrate})


__all__ = ["FrozenTail", "layer_of", "make_frozen_tail", "residual_modify"]
