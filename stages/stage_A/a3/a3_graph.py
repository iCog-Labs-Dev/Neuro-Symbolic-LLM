from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import jax
import jax.numpy as jnp
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import IdentityNode, Linear
from fabricpc.training.trainer import (
    build_clamps,
    grad_denominator,
    pc_weight_gradients,
)

from .a3_tail import make_frozen_tail


@dataclass(frozen=True)
class A3GraphResult:
    grads: Any
    energy_before: jax.Array
    energy_after: jax.Array
    n_pred: Any
    z_star: dict[int, jax.Array]
    pred: dict[int, jax.Array]


def build_a3_graph(
    substrate,
    layers,
    rank,
    seq_len,
    scale,
    activation_class,
    config,
):
    d = int(substrate.architecture.hidden_size)
    vocab_size = int(substrate.architecture.vocab_size)

    ids_node = IdentityNode(
        shape=(seq_len,),
        name="ids",
    )

    h0_nodes = {
        layer: IdentityNode(
            shape=(seq_len, d),
            name=f"h0_{layer}",
        )
        for layer in layers
    }

    z_nodes = {
        layer: Linear(
            shape=(seq_len, rank),
            name=f"z{layer}",
            activation=activation_class(),
            energy=GaussianEnergy(
                precision=1.0 / config.sigma2,
            ),
            use_bias=False,
            weight_init=NormalInitializer(
                mean=0.0,
                std=config.predictor_init_std,
            ),
        )
        for layer in layers
    }

    tail_node = make_frozen_tail(substrate)(
        shape=(seq_len - 1, vocab_size),
        name="tail",
        layers=layers,
        hidden_size=d,
        scale=scale,
        projection_scale=config.projection_init_std,
        lambda_kl=config.lambda_kl,
    )

    tail_name = str(getattr(tail_node, "name", "tail"))

    nodes = [
        ids_node,
        *h0_nodes.values(),
        *z_nodes.values(),
        tail_node,
    ]

    edges = [
        Edge(
            source=ids_node,
            target=tail_node.slot("ids"),
        )
    ]

    for layer in layers:
        edges.extend(
            [
                Edge(
                    source=h0_nodes[layer],
                    target=z_nodes[layer],
                ),
                Edge(
                    source=z_nodes[layer],
                    target=tail_node.slot("z"),
                ),
            ]
        )

    task = {
        "x": ids_node,
        "y": tail_node,
        **{f"h0_{layer}": h0_nodes[layer] for layer in layers},
    }

    if config.lambda_kl > 0:
        base_node = IdentityNode(
            shape=(seq_len - 1, vocab_size),
            name="base",
        )

        nodes.append(base_node)

        edges.append(
            Edge(
                source=base_node,
                target=tail_node.slot("base_logp"),
            )
        )

        task["base"] = base_node

    structure = graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(**task),
        inference=InferenceSGD(
            eta_infer=config.eta_z,
            infer_steps=config.infer_steps,
        ),
    )

    return structure, tail_name


def refresh_state(
    structure,
    params,
    state,
    clamps,
):
    inference = type(structure.config["inference"])

    state = inference.zero_grads(
        params,
        state,
        clamps,
        structure,
    )

    return inference.forward_value_and_grad(
        params,
        state,
        clamps,
        structure,
    )


def total_energy(
    structure,
    state,
) -> jax.Array:
    return sum(
        jnp.sum(state.nodes[name].energy)
        for name, node in structure.nodes.items()
        if node.node_info.in_degree > 0
    )


def infer_and_compute_grads(
    structure,
    params,
    batch,
    key,
    layers,
) -> A3GraphResult:
    clamps = build_clamps(
        batch,
        structure,
        clamp_target=True,
    )

    n_pred = grad_denominator(
        structure,
        clamps,
    )

    state = initialize_graph_state(
        structure,
        batch["x"].shape[0],
        key,
        clamps,
        params=params,
    )

    state = refresh_state(
        structure,
        params,
        state,
        clamps,
    )

    energy_before = total_energy(structure, state) / n_pred

    state = run_inference(
        params,
        state,
        clamps,
        structure,
    )

    state = refresh_state(
        structure,
        params,
        state,
        clamps,
    )

    energy_after = total_energy(structure, state) / n_pred

    grads = pc_weight_gradients(
        params,
        state,
        structure,
        clamps,
    )

    z_star = {layer: state.nodes[f"z{layer}"].z_latent for layer in layers}

    pred = {layer: state.nodes[f"z{layer}"].z_mu for layer in layers}

    return A3GraphResult(
        grads=grads,
        energy_before=energy_before,
        energy_after=energy_after,
        n_pred=n_pred,
        z_star=z_star,
        pred=pred,
    )


def compute_energy_curve(
    structure,
    params,
    batch,
    key,
    steps: int | None = None,
) -> list[float]:
    clamps = build_clamps(
        batch,
        structure,
        clamp_target=True,
    )

    n_pred = grad_denominator(
        structure,
        clamps,
    )

    state = initialize_graph_state(
        structure,
        batch["x"].shape[0],
        key,
        clamps,
        params=params,
    )

    inference = structure.config["inference"]

    if steps is None:
        steps = int(inference.config["infer_steps"])
    elif isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 0:
        raise ValueError(f"steps must be an integer >= 0, got {steps!r}")

    state = refresh_state(
        structure,
        params,
        state,
        clamps,
    )

    curve = [float(total_energy(structure, state) / n_pred)]

    for _ in range(int(steps)):
        state = type(inference).inference_step(
            params,
            state,
            clamps,
            structure,
            inference.config,
        )

        state = refresh_state(
            structure,
            params,
            state,
            clamps,
        )

        curve.append(float(total_energy(structure, state) / n_pred))

    return curve


__all__ = [
    "A3GraphResult",
    "build_a3_graph",
    "refresh_state",
    "total_energy",
    "infer_and_compute_grads",
    "compute_energy_curve",
]
