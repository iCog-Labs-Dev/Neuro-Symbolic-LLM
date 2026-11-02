"""Restore continuous residual weights from the shared adapter checkpoint layout."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from stages.stage_A.A3_model import A3Model


def restore_a3_weights(
    model: A3Model,
    weights: Mapping[str, Mapping[str, Any]],
) -> Any:
    """Restore both predictor A and projection B without modifying source weights.

    Accepts the mapping returned by stages.results.load_params. The caller must
    construct the model with the checkpoint's base-model identity, activation,
    scale and PC configuration. This restores weights, not optimizer state.
    """
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


__all__ = ["restore_a3_weights"]
