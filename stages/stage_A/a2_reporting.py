"""Post-hoc internal metrics on a saved A1 or A2 adapter, without settling."""

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import numpy as np

from frozenllm.substrate import FrozenSubstrate
from metrics.representation import (
    gradient_conflict,
    residual_norm_ratio,
    residual_redundancy,
)
from residual import ResidualAdapter
from stages.stage_A.a2 import A2Config, a2_objective, adapter_params


def internal_metrics(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    params: Mapping[str, Any],
    config: A2Config,
    eval_batches: Sequence[jax.Array],
) -> dict[str, float]:
    """Held-out residual size, read/write redundancy and task/aux gradient conflict.

    Size is averaged across every supplied batch. Gradient conflict is measured
    on the first batch only and refers to adapter gradients (not head weights).
    Inputs are causal feedforward states; gradients are measured, never applied.
    """
    if not eval_batches:
        raise ValueError("Internal reporting requires at least one evaluation batch.")
    ap = adapter_params(params)
    adapter.validate_params(ap)
    result = {}
    for layer in adapter.layers:
        block = ap[f"layer_{layer}"]
        for name, value in residual_redundancy(block["A"], block["B"]).items():
            result[f"redundancy_{name}_l{layer}"] = float(value)
    ratios: dict[int, list[float]] = {i: [] for i in adapter.layers}
    for ids in eval_batches:
        run = substrate.run_with_interception(
            ids, modify_fn=adapter.modify_fn(ap), intercept_layers=adapter.layers
        )
        for i in adapter.layers:
            h = run.hidden_state(i)
            ratios[i].append(float(residual_norm_ratio(h, adapter.apply(ap, h, i))))
    for i, values in ratios.items():
        result[f"residual_norm_ratio_l{i}"] = float(np.mean(values))

    def terms(p):
        return a2_objective(
            p,
            substrate=substrate,
            adapter=adapter,
            config=config,
            input_ids=eval_batches[0],
        )[1]

    available = terms(params)
    task_grad = adapter_params(jax.grad(lambda p: terms(p)["task"])(params))
    for name in ("hid", "cpc", "stab"):
        if name in available:
            aux_grad = adapter_params(
                jax.grad(lambda p, name=name: terms(p)[name])(params)
            )
            result[f"gradient_conflict_task_{name}"] = float(
                gradient_conflict(task_grad, aux_grad)
            )
    result["gradient_conflict_batches"] = 1.0
    result["representation_batches"] = float(len(eval_batches))
    return result
