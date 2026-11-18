"""Restore continuous residual weights from the shared adapter checkpoint layout."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from stages.stage_A.a3 import A3Model


def save_training_checkpoint(writer, tag, params, state, **metadata):
    """Atomically save all head/projection weights and optimizer leaves together."""
    return writer.save_checkpoint(
        tag,
        {
            f"{group}.{layer}": weights
            for group, layers in params.items()
            for layer, weights in layers.items()
        },
        opt_state=state,
        **metadata,
    )


def restore_training_checkpoint(path, template, optimizer, signature):
    """Restore arrays without unpickling the checkpoint's saved tree definition."""
    from stages.results import load_params

    params = {group: {} for group in template}
    for name, values in load_params(path).items():
        group, layer = name.split(".", 1)
        if group not in params:
            raise ValueError("Checkpoint has an unexpected parameter group")
        params[group][layer] = values
    if jax.tree.structure(params) != jax.tree.structure(template):
        raise ValueError("Checkpoint parameter structure differs from this run")

    def checked(value, reference):
        if (
            value.shape != reference.shape
            or value.dtype != reference.dtype
            or not np.isfinite(value).all()
        ):
            raise ValueError("Checkpoint shape, dtype or finite-value check failed")
        return jnp.asarray(value)

    params = jax.tree.map(checked, params, template)
    leaves, tree = jax.tree.flatten(optimizer.init(template))
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["__metadata__"]))
        if meta.get("resume_version") != 1 or meta.get("resume_signature") != signature:
            raise ValueError(
                "Resume requires the same data, bank, frozen A3 and training settings"
            )
        if {name for name in archive.files if name.startswith("opt/")} != {
            f"opt/leaf_{i}" for i in range(len(leaves))
        }:
            raise ValueError("Checkpoint optimizer structure differs from this run")
        saved = [checked(archive[f"opt/leaf_{i}"], ref) for i, ref in enumerate(leaves)]
    state = jax.tree.unflatten(tree, saved)
    return params, state, meta


def load_a3_config(path: Path):
    """Load A3 settings through the shared runtime configuration loader."""
    from stages.stage_A.a3 import A3HybridConfig
    from stages.stage_base import load_configs

    return load_configs(path, A3HybridConfig, objective_section="a3_hybrid")


def restore_a3_weights(
    model: A3Model,
    weights: Mapping[str, Mapping[str, Any]],
) -> Any:
    """Restore predictor A and projection B into a matching continuous model."""
    if set(weights) != {f"layer_{layer}" for layer in model.layers}:
        raise ValueError("Checkpoint layers do not match the continuous model")
    arrays: dict[int, dict[str, jax.Array]] = {}
    for layer in model.layers:
        layer_weights = weights[f"layer_{layer}"]
        if set(layer_weights) != {"A", "B"}:
            raise ValueError("Each checkpoint layer must contain both A and B")
        arrays[layer] = {}
        for name, shape in (
            ("A", (model.hidden_size, model.rank)),
            ("B", (model.rank, model.hidden_size)),
        ):
            value = np.asarray(layer_weights[name])
            if (
                value.shape != shape
                or not np.issubdtype(value.dtype, np.floating)
                or not np.isfinite(value).all()
            ):
                raise ValueError(
                    f"Checkpoint {name} shape, dtype or values are invalid"
                )
            converted = value.astype(np.float32)
            if not np.isfinite(converted).all():
                raise ValueError(f"Checkpoint {name} cannot be represented as float32")
            arrays[layer][name] = jnp.asarray(converted)

    params = model.init_params(jax.random.PRNGKey(0))
    nodes = dict(params.nodes)
    for layer in model.layers:
        predictor_name = f"z{layer}"
        predictor = nodes[predictor_name]
        predictor_weights = dict(predictor.weights)
        predictor_weights[f"h0_{layer}->z{layer}:in"] = arrays[layer]["A"]
        nodes[predictor_name] = predictor._replace(weights=predictor_weights)
        candidates = [
            (name, f"z{layer}->{name}:z")
            for name, node in nodes.items()
            if f"z{layer}->{name}:z" in node.weights
        ]
        if len(candidates) != 1:
            raise ValueError("Continuous graph must have one projection per layer")
        name, edge = candidates[0]
        projection_weights = dict(nodes[name].weights)
        projection_weights[edge] = arrays[layer]["B"]
        nodes[name] = nodes[name]._replace(weights=projection_weights)
    return params._replace(nodes=nodes)


__all__ = [
    "load_a3_config",
    "restore_a3_weights",
    "restore_training_checkpoint",
    "save_training_checkpoint",
]
