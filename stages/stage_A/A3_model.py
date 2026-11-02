"""A3Model: true predictive-coding latent residual, on FabricPC 0.6.0 (plan section 7.5).

The graph (T tokens, d hidden, r latent, V vocabulary):

    h0_l (clamped, T x d) --> z_l   Linear + GaussianEnergy(precision = 1/sigma^2)
    h0_l ---------------------> tail
    z_l ----------------------> tail (T-1 x V)   FrozenTail: h~_l via modify_fn, model loss
    ids (clamped, T) ---------> tail
    base log-probs (clamped, T-1 x V; only when lambda_kl > 0) --> tail

Per batch (``train_grads``):
    1. one clean pass of the frozen substrate; pristine h0_l come from its own hidden-state
       cache (``ForwardResult.hidden_state``), not from a hand-written collector;
    2. z^(0) = f_phi(h0)             FabricPC feedforward state initialisation, then a refresh
                                     so the stored energies belong to z^(0);
    3. K steps  z <- z - eta_z * grad_z E       ``run_inference`` with ``InferenceSGD``;
    4. REFRESH at z^(K): ``inference_step`` evaluates the energy BEFORE it updates z, so after
       K steps the stored energies, z_mu and latent gradients belong to z^(K-1). One extra
       forward/gradient phase with no latent update makes the state consistent with z^(K);
    5. local gradients of f_phi and B from the refreshed state (``pc_weight_gradients``);
    6. the exact final loss at z^(K) through ``FrozenSubstrate.compute_loss`` as a cross-check.

Energies are sums over predictions; ``grad_denominator`` gives N = batch x (T - 1) for the
(B, T-1, V) target, and every reported energy is per prediction (E / N).

Residuals are injected into the live hidden-state stream, so an earlier adapted layer's residual
flows through later layers. Training and deployment use the same rule.

Optimisation and logging live in experiments/run_stage_a3.py.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import jax
import jax.numpy as jnp
from fabricpc.core.activations import (
    GeluActivation,
    IdentityActivation,
    ReLUActivation,
    TanhActivation,
)
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import IdentityNode, Linear
from fabricpc.training.trainer import (
    build_clamps,
    grad_denominator,
    pc_weight_gradients,
)

from frozenllm.substrate import FrozenSubstrate, identity_modify
from stages.stage_base import checked_kwargs

from .a3_tail import make_frozen_tail, residual_modify

_ACTIVATIONS = {
    "identity": IdentityActivation,
    "tanh": TanhActivation,
    "gelu": GeluActivation,
    "relu": ReLUActivation,
}


# ---------------------------------------------------------------------- validation helpers
def _is_int(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_real(value: Any) -> bool:
    return (
        isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    )


def _require_positive(name: str, value: Any) -> None:
    if not _is_real(value) or not value > 0:
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")


def _require_nonnegative(name: str, value: Any) -> None:
    if not _is_real(value) or value < 0:
        raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")


def _validate_layers(layers: Any, num_layers: int) -> tuple[int, ...]:
    try:
        raw = tuple(layers)
    except TypeError as exc:
        raise ValueError("layers must be a non-empty sequence of integers") from exc
    if not raw:
        raise ValueError("layers must not be empty")
    if not all(_is_int(layer) for layer in raw):
        raise ValueError(f"layers must contain integers, got {raw}")
    out = tuple(int(layer) for layer in raw)
    if len(set(out)) != len(out) or any(not 0 <= layer < num_layers for layer in out):
        raise ValueError(f"layers must be unique and in [0, {num_layers}), got {out}")
    return out


@dataclass(frozen=True)
class A3Config:
    """The ``objective:`` section of configs/stage_A/a3_true_pc.yaml."""

    infer_steps: int = 2  # K, the plan sweeps {1, 2, 4, 8}
    eta_z: float = 0.1  # per-token inference step (stable below 2 / lambda_max)
    sigma2: float = 1.0  # the spring has precision 1 / sigma^2
    lambda_kl: float = 0.0  # KL trust-region weight, as in A1; 0 = off
    predictor_init_std: float = (
        0.001  # small: z^(0) starts near 0, so the model starts near base
    )
    projection_init_std: float = (
        0.1  # B must not be 0: grad_z E = B^T delta would vanish
    )

    def __post_init__(self) -> None:
        if not _is_int(self.infer_steps) or self.infer_steps < 0:
            raise ValueError(
                f"infer_steps must be an integer >= 0, got {self.infer_steps!r}"
            )
        for name in ("eta_z", "sigma2", "predictor_init_std", "projection_init_std"):
            _require_positive(name, getattr(self, name))
        _require_nonnegative("lambda_kl", self.lambda_kl)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> A3Config:
        return cls(**checked_kwargs(cls, data))


class A3Model:
    def __init__(
        self,
        substrate: FrozenSubstrate,
        *,
        layers: Sequence[int],
        rank: int,
        seq_len: int,
        scale: float = 1.0,
        activation: str = "identity",
        config: A3Config | None = None,
    ) -> None:
        cfg = A3Config() if config is None else config
        if not isinstance(cfg, A3Config):
            raise TypeError(f"config must be an A3Config, got {type(cfg).__name__}")
        arch = substrate.architecture
        if min(arch.num_layers, arch.hidden_size, arch.vocab_size) <= 0:
            raise ValueError(
                "substrate architecture must have positive layer, hidden and vocab sizes"
            )
        layers = _validate_layers(layers, arch.num_layers)
        if not _is_int(rank) or rank < 1:
            raise ValueError(f"rank must be an integer >= 1, got {rank!r}")
        if not _is_int(seq_len) or seq_len < 2:
            raise ValueError(
                f"seq_len must be an integer >= 2 for next-token loss, got {seq_len!r}"
            )
        max_pos = getattr(arch, "max_position_embeddings", None)
        if max_pos is not None and seq_len > max_pos:
            raise ValueError(
                f"seq_len={seq_len} exceeds the model's {max_pos} positions"
            )
        _require_positive("scale", scale)
        if not isinstance(activation, str) or activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}"
            )

        self.sub, self.cfg, self.layers = substrate, cfg, layers
        self.rank, self.seq_len, self.scale = int(rank), int(seq_len), float(scale)
        self.hidden_size = int(arch.hidden_size)
        self.vocab_size = int(
            arch.vocab_size
        )  # from the architecture: no probe forward pass
        self._activation = _ACTIVATIONS[activation]()
        d, T, r, V = self.hidden_size, self.seq_len, self.rank, self.vocab_size

        use_base = cfg.lambda_kl > 0
        ids_node = IdentityNode(shape=(T,), name="ids")
        h0_nodes = {
            layer: IdentityNode(shape=(T, d), name=f"h0_{layer}") for layer in layers
        }
        z_nodes = {
            layer: Linear(
                shape=(T, r),
                name=f"z{layer}",
                activation=_ACTIVATIONS[activation](),
                energy=GaussianEnergy(precision=1.0 / cfg.sigma2),
                use_bias=False,
                weight_init=NormalInitializer(mean=0.0, std=cfg.predictor_init_std),
            )
            for layer in layers
        }
        tail_node = make_frozen_tail(substrate)(
            shape=(T - 1, V),
            name="tail",
            layers=layers,
            hidden_size=d,
            scale=self.scale,
            projection_scale=cfg.projection_init_std,
            lambda_kl=cfg.lambda_kl,
        )
        # Edge keys and parameter names use the node's own name, so lookups stay correct
        # whatever name the tail class ends up registering.
        self._tail_name = str(getattr(tail_node, "name", "tail"))

        nodes = [ids_node, *h0_nodes.values(), *z_nodes.values(), tail_node]
        edges = [Edge(source=ids_node, target=tail_node.slot("ids"))]
        for layer in layers:
            edges += [
                Edge(source=h0_nodes[layer], target=z_nodes[layer]),
                Edge(source=z_nodes[layer], target=tail_node.slot("z")),
            ]
        task = {
            "x": ids_node,
            "y": tail_node,
            **{f"h0_{layer}": h0_nodes[layer] for layer in layers},
        }
        if use_base:
            base_node = IdentityNode(shape=(T - 1, V), name="base")
            nodes.append(base_node)
            edges.append(Edge(source=base_node, target=tail_node.slot("base_logp")))
            task["base"] = base_node

        self.structure = graph(
            nodes=nodes,
            edges=edges,
            task_map=TaskMap(**task),
            inference=InferenceSGD(eta_infer=cfg.eta_z, infer_steps=cfg.infer_steps),
        )

    # ------------------------------------------------------------------ parameters
    def init_params(self, key: jax.Array):
        return initialize_params(self.structure, key)

    def predictor_weight(self, params, layer: int) -> jax.Array:
        """A_l [d, r]: the predictor matrix of layer ``layer``."""
        return params.nodes[f"z{layer}"].weights[f"h0_{layer}->z{layer}:in"]

    def projection(self, params, layer: int) -> jax.Array:
        """B_l [r, d]: the decoder of layer ``layer``."""
        return params.nodes[self._tail_name].weights[f"z{layer}->{self._tail_name}:z"]

    def adapter_params(self, params) -> dict[str, dict[str, jax.Array]]:
        """Parameters in stage A1's AdapterParams layout ({"layer_l": {"A", "B"}})."""
        return {
            f"layer_{layer}": {
                "A": self.predictor_weight(params, layer),
                "B": self.projection(params, layer),
            }
            for layer in self.layers
        }

    # ------------------------------------------------------------------ step 1: cached states
    def _check_ids(self, ids: jax.Array) -> jax.Array:
        ids = jnp.asarray(ids)
        if ids.ndim != 2 or ids.shape[1] != self.seq_len:
            raise ValueError(
                f"ids must have shape [batch, {self.seq_len}], got {tuple(ids.shape)}"
            )
        if ids.shape[0] < 1:
            raise ValueError("ids batch dimension must be non-empty")
        if not jnp.issubdtype(ids.dtype, jnp.integer):
            raise TypeError(f"ids must be an integer array, got dtype {ids.dtype}")
        return ids.astype(jnp.int32)

    def clean_pass(self, ids: jax.Array) -> tuple[dict[int, jax.Array], jax.Array]:
        """(pristine h0_l, base logits) from ONE untouched pass, read from the substrate's
        hidden-state cache (``ForwardResult.hidden_state``), not collected by hand."""
        clean = self.sub.run_with_interception(
            ids, modify_fn=identity_modify, intercept_layers=self.layers
        )
        h0 = {
            layer: jax.lax.stop_gradient(clean.hidden_state(layer))
            for layer in self.layers
        }
        return h0, jax.lax.stop_gradient(clean.logits)

    def make_batch(
        self, ids: jax.Array
    ) -> tuple[dict[str, jax.Array], dict[int, jax.Array], jax.Array]:
        """Batch dict keyed by the task map, plus the clean h0 and the base logits.

        ``y`` holds integer next-token targets; ``build_clamps`` one-hot encodes them to the
        tail's vocabulary axis."""
        ids = self._check_ids(ids)
        h0, base_logits = self.clean_pass(ids)
        batch = {
            "x": ids,
            "y": ids[:, 1:],
            **{f"h0_{layer}": h0[layer].astype(jnp.float32) for layer in self.layers},
        }
        if self.cfg.lambda_kl > 0:
            batch["base"] = jax.nn.log_softmax(
                base_logits[:, :-1].astype(jnp.float32), axis=-1
            )
        return batch, h0, base_logits

    # ------------------------------------------------------------------ stale-state fix
    def _refresh_state(self, params, state, clamps):
        """Recompute predictions, energies and latent gradients at the CURRENT z.

        ``inference_step`` is zero_grads -> forward_value_and_grad -> update_latents, so the
        stored energy is the one from before the last update. This runs the first two phases
        only, with no latent update."""
        inference = type(self.structure.config["inference"])
        state = inference.zero_grads(params, state, clamps, self.structure)
        return inference.forward_value_and_grad(params, state, clamps, self.structure)

    # ------------------------------------------------------------------ energies
    def total_energy(self, state) -> jax.Array:
        """Sum over nodes with in_degree > 0 of the node energies held in ``state``."""
        return sum(
            jnp.sum(state.nodes[n].energy)
            for n, node in self.structure.nodes.items()
            if node.node_info.in_degree > 0
        )

    def _final_terms(
        self, params, ids, base_logits, z_star, pred, n_pred
    ) -> dict[str, jax.Array]:
        """Exact terms at the final z through the substrate's loss API, per prediction.

        KL is taken over the same T - 1 positions as the tail, so ``energy`` is comparable
        with the refreshed graph energy."""
        b = {layer: self.projection(params, layer) for layer in self.layers}
        logits = self.sub.run_with_interception(
            ids,
            modify_fn=residual_modify(z_star, b, self.scale),
            intercept_layers=self.layers,
        ).logits
        ce = self.sub.compute_loss(logits, ids, losses=("CE",))
        terms = {"task": ce}
        loss = ce
        if self.cfg.lambda_kl > 0:
            kl = self.sub.compute_loss(
                logits[:, :-1], None, losses=("KL",), base_logits=base_logits[:, :-1]
            )
            terms["kl"] = kl
            loss = ce + self.cfg.lambda_kl * kl
        gauss = {"precision": 1.0 / self.cfg.sigma2}
        spring = (
            sum(
                jnp.sum(GaussianEnergy.energy(z_star[layer], pred[layer], gauss))
                for layer in self.layers
            )
            / n_pred
        )
        return {**terms, "loss": loss, "spring": spring, "energy": loss + spring}

    # ------------------------------------------------------------------ steps 2-6
    def train_grads(self, params, ids: jax.Array, key: jax.Array):
        """One A3 batch. Returns (grads, aux). Pure in its array arguments.

        grads are means per prediction (descend them). aux holds, per prediction:
        energy_before (graph energy at z^(0)), energy_after (graph energy at z^(K), from the
        refreshed state), energy_after_exact (substrate recomputation at z^(K); equals
        energy_after up to numerics), task, kl (when lambda_kl > 0), loss, spring, plus
        n_pred, z_star and pred."""
        batch, h0, base_logits = self.make_batch(ids)
        ids = batch["x"]
        clamps = build_clamps(batch, self.structure, clamp_target=True)
        n = grad_denominator(self.structure, clamps)

        state = initialize_graph_state(
            self.structure, ids.shape[0], key, clamps, params=params
        )
        state = self._refresh_state(params, state, clamps)  # energies at z^(0)
        energy_before = self.total_energy(state) / n

        state = run_inference(params, state, clamps, self.structure)  # K latent updates
        state = self._refresh_state(params, state, clamps)  # energies at z^(K)
        energy_after = self.total_energy(state) / n

        grads = pc_weight_gradients(params, state, self.structure, clamps)
        z_star = {layer: state.nodes[f"z{layer}"].z_latent for layer in self.layers}
        pred = {
            layer: state.nodes[f"z{layer}"].z_mu for layer in self.layers
        }  # f_phi(h0): independent of z

        exact = self._final_terms(params, ids, base_logits, z_star, pred, n)
        energy_after_exact = exact.pop("energy")
        aux = {
            "n_pred": n,
            "energy_before": energy_before,
            "energy_after": energy_after,
            "energy_after_exact": energy_after_exact,
            **exact,
            "z_star": z_star,
            "pred": pred,
        }
        return grads, aux

    def energy_curve(
        self, params, ids: jax.Array, key: jax.Array, steps: int | None = None
    ):
        """Graph energy per prediction at z^(0), z^(1), ..., z^(K). Eager diagnostic.

        Each point is refreshed at its own z, so the list has steps + 1 entries from exactly
        ``steps`` latent updates."""
        batch, _, _ = self.make_batch(ids)
        clamps = build_clamps(batch, self.structure, clamp_target=True)
        n = grad_denominator(self.structure, clamps)
        state = initialize_graph_state(
            self.structure, batch["x"].shape[0], key, clamps, params=params
        )
        inference = self.structure.config["inference"]
        if steps is None:
            steps = int(inference.config["infer_steps"])
        elif not _is_int(steps) or steps < 0:
            raise ValueError(f"steps must be an integer >= 0, got {steps!r}")

        state = self._refresh_state(params, state, clamps)
        curve = [float(self.total_energy(state)) / n]
        for _ in range(int(steps)):
            state = type(inference).inference_step(
                params, state, clamps, self.structure, inference.config
            )
            state = self._refresh_state(
                params, state, clamps
            )  # energy at the updated z
            curve.append(float(self.total_energy(state)) / n)
        return curve

    # ------------------------------------------------------------------ deployment
    def predict_latents(
        self, params, h0: Mapping[int, jax.Array]
    ) -> dict[int, jax.Array]:
        """z_l = sigma(h0_l A_l): the predictor applied to the PRISTINE cached states."""
        a = self._activation
        return {
            layer: type(a).forward(
                h0[layer] @ self.predictor_weight(params, layer), a.config
            )
            for layer in self.layers
        }

    def deploy_logits(self, params, ids: jax.Array) -> tuple[jax.Array, jax.Array]:
        """(adapted logits with z = f_phi(h0), base logits): the label-free model.
        Two passes: the clean pass (cache) and the modified pass."""
        ids = self._check_ids(ids)
        h0, base_logits = self.clean_pass(ids)
        z = self.predict_latents(params, h0)
        b = {layer: self.projection(params, layer) for layer in self.layers}
        logits = self.sub.run_with_interception(
            ids,
            modify_fn=residual_modify(z, b, self.scale),
            intercept_layers=self.layers,
        ).logits
        return logits, base_logits


__all__ = ["A3Config", "A3Model"]
