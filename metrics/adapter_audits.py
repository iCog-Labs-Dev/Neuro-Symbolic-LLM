"""Checkpoint audits with one deployed adapter layer as one component.

No semantic labels or intended task supports are inferred. Loss functions must
evaluate the deployed model on fixed, matched held-out examples.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


def measure_adapter_ablations(
    losses: Mapping[str, Callable[[Any], Any]], params: dict[str, Any]
) -> dict[str, Any]:
    """Disable each adapter by zeroing B; record signed CE changes by domain."""
    import jax.numpy as jnp

    components = sorted(params)
    if not components or not losses:
        raise ValueError("adapter ablations require components and held-out domains")
    baseline = {domain: float(loss(params)) for domain, loss in losses.items()}
    ablated: dict[str, dict[str, float]] = {}
    effects: dict[str, dict[str, float]] = {}
    for component in components:
        block = params[component]
        disabled = {
            **params,
            component: {**block, "B": jnp.zeros_like(block["B"])},
        }
        ablated[component] = {
            domain: float(loss(disabled)) for domain, loss in losses.items()
        }
        effects[component] = {
            domain: value - baseline[domain]
            for domain, value in ablated[component].items()
        }
    matrix = np.asarray([[effects[c][d] for d in losses] for c in components])
    if not np.isfinite(matrix).all() or not all(
        math.isfinite(value) for value in baseline.values()
    ):
        raise ValueError("non-finite adapter ablation loss or effect")
    return {
        "status": "measured",
        "component": "one whole deployed adapter at one intercepted layer",
        "intervention": "zero write matrix B; keep all other parameters fixed",
        "domains": list(losses),
        "components": components,
        "full_loss": baseline,
        "ablated_loss": ablated,
        "effects": effects,
        "mean_loss_increase": float(matrix.mean()),
        "fingerprint_rms": float(np.sqrt(np.mean(matrix**2))),
    }


def measure_cross_adapter_curvature(
    loss_fn: Callable[[Any], Any],
    params: dict[str, Any],
    *,
    samples: int = 2,
    seed: int = 0,
    relative_step: float = 1e-3,
) -> dict[str, Any]:
    """Estimate mixed full A/B block norms between different adapter layers.

    Central differences of reverse-mode gradients approximate H_ij v. This
    avoids requiring second derivatives through the frozen-model backend.
    Rademacher vectors have unit entries; their squared product norms estimate
    squared Frobenius norms. The returned norm is a sampled approximation.
    """
    import jax
    import jax.numpy as jnp

    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("cross-adapter curvature samples must be a positive integer")
    if not math.isfinite(relative_step) or relative_step <= 0:
        raise ValueError("cross-adapter curvature relative_step must be positive")
    names = sorted(params)
    if len(names) < 2:
        raise ValueError(
            "cross-adapter curvature requires at least two intercepted adapter layers; "
            "configure adapter.layers and train a matching checkpoint"
        )
    key = jax.random.PRNGKey(seed)
    pairs = {}
    steps = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            # Holding other blocks fixed excludes A2 training-only heads.
            def objective(a: Any, b: Any, left=left, right=right) -> Any:
                return loss_fn({**params, left: a, right: b})

            gradient = jax.grad(objective, argnums=0)
            block = params[right]
            leaves = jax.tree.leaves(block)
            size = sum(int(leaf.size) for leaf in leaves)
            norm = math.sqrt(sum(float(jnp.sum(leaf**2)) for leaf in leaves))
            step = relative_step * max(1.0, norm) / math.sqrt(size)
            squared = []
            for _ in range(samples):
                draws = jax.random.split(key, len(leaves) + 1)
                key = draws[0]
                directions = [
                    jax.random.rademacher(draw, leaf.shape).astype(leaf.dtype)
                    for draw, leaf in zip(draws[1:], leaves, strict=True)
                ]
                direction = jax.tree.unflatten(jax.tree.structure(block), directions)
                plus = jax.tree.map(
                    lambda b, v, step=step: b + step * v, block, direction
                )
                minus = jax.tree.map(
                    lambda b, v, step=step: b - step * v, block, direction
                )
                g_plus = gradient(params[left], plus)
                g_minus = gradient(params[left], minus)
                product = jax.tree.map(
                    lambda a, b, step=step: (
                        (a.astype(jnp.float32) - b.astype(jnp.float32)) / (2 * step)
                    ),
                    g_plus,
                    g_minus,
                )
                squared.append(
                    sum(float(jnp.sum(leaf**2)) for leaf in jax.tree.leaves(product))
                )
            pair = f"{left}:{right}"
            pairs[pair] = float(np.sqrt(np.mean(squared)))
            steps[pair] = step
    if not all(math.isfinite(value) for value in pairs.values()):
        raise ValueError("non-finite cross-adapter curvature estimate")
    return {
        "status": "measured",
        "component": "all deployed A/B parameters at one intercepted layer",
        "estimator": "central-difference gradient HVP, Rademacher Frobenius estimate",
        "samples_per_pair": samples,
        "seed": seed,
        "relative_step": relative_step,
        "steps": steps,
        "pairs": pairs,
        "value": float(np.sqrt(np.mean(np.square(list(pairs.values()))))),
    }
