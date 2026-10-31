"""Stability diagnostics for Stage A2: residual smoothness and settling-step metrics.

Both are **measurement only**. They are logged during training and never enter the A2 loss or
change a gradient (everything is stop-gradient), mirroring how the A2 review suggests treating
stability quantities first: log them, and add them to the objective only if a problem shows up.

Residual smoothness (computable in A2)::

    L_res-smooth = sum_{t>1} || R(h_t) - R(h_{t-1}) ||^2        R(h) = adapter(h) - h

``h`` is the hidden state the adapter actually receives at the site (``hidden_state(l)`` of the
adapted pass, i.e. including upstream corrections), not a separate base trajectory. The input's
own token-to-token jitter ``||h_t - h_{t-1}||^2`` is logged next to it, and ``res_vs_input`` is
their ratio: below 1 the correction is smoother than its input, above 1 it amplifies jitter.

Step metrics (defined over settling iterations, so zero in A2)::

    step_loss        = mean_tokens sum_s ||z^(s+1) - z^(s)||^2
    step_noncontract = mean_tokens sum_s [ ||d^(s+1)|| / (||d^(s)|| + delta) - 1 ]_+
    step_last_norm   = mean_tokens ||z^(S) - z^(S-1)||           d^(s) = z^(s+1) - z^(s)

``step_metrics`` is a pure function of a trajectory ``[S+1, B, T, D]``. A2 has no settling, so the
objective passes the single feedforward state (``S = 0``): there are no differences and every
step metric is exactly zero. A3's settling loop passes its real trajectory to the same function.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from residual.adapter import AdapterParams, ResidualAdapter

STABILITY_MODES = ("off", "diagnostic")
_EPS = 1e-8
STEP_KEYS = ("step_loss", "step_noncontract", "step_last_norm")


@dataclass(frozen=True)
class StabilityConfig:
    mode: str = "off"
    layers: list[int] | None = None
    delta: float = 1e-6

    def __post_init__(self) -> None:
        if self.mode not in STABILITY_MODES:
            raise ValueError(
                f"stability mode must be one of {STABILITY_MODES}, got {self.mode!r}"
            )
        if not isinstance(self.delta, int | float) or not self.delta > 0:
            raise ValueError(f"delta must be > 0, got {self.delta!r}")
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
    def from_dict(cls, data: Mapping[str, Any]) -> StabilityConfig:
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"Unknown StabilityConfig keys: {unknown}. Allowed: {sorted(known)}."
            )
        kwargs = dict(data)
        # YAML 1.1 reads a bare ``off`` as False.
        if kwargs.get("mode") is False:
            kwargs["mode"] = "off"
        return cls(**kwargs)

    @property
    def enabled(self) -> bool:
        return self.mode != "off"


def _adjacent_stats(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    diff = x[:, 1:, :] - x[:, :-1, :]
    sq = jnp.sum(jnp.square(diff), axis=-1)  # [B, S-1]
    return jnp.sum(sq), jnp.asarray(sq.size, dtype=jnp.float32)


def residual_smoothness(
    correction: jax.Array, inputs: jax.Array
) -> dict[str, jax.Array]:
    correction = jax.lax.stop_gradient(correction.astype(jnp.float32))
    inputs = jax.lax.stop_gradient(inputs.astype(jnp.float32))
    r_sum, count = _adjacent_stats(correction)
    h_sum, _ = _adjacent_stats(inputs)
    denom = jnp.maximum(count, 1.0)
    res, inp = r_sum / denom, h_sum / denom
    return {
        "res_smooth": res,
        "input_smooth": inp,
        "res_vs_input": res / (inp + _EPS),
    }


def step_metrics(trajectory: jax.Array, delta: float = 1e-6) -> dict[str, jax.Array]:
    """Settling-step metrics of a trajectory ``[S+1, B, T, D]`` (``S`` settling steps).

    With ``S = 0`` (one state, no settling) there are no differences and all metrics are zero.
    With ``S = 1`` there is one difference, so non-contractivity (a ratio of consecutive
    differences) is zero as well.
    """
    if trajectory.ndim != 4:
        raise ValueError(
            f"trajectory must be [S+1, B, T, D], got {tuple(trajectory.shape)}"
        )
    z = jax.lax.stop_gradient(trajectory.astype(jnp.float32))
    n_tokens = float(z.shape[1] * z.shape[2])
    diffs = z[1:] - z[:-1]  # [S, B, T, D]
    zero = jnp.zeros((), dtype=jnp.float32)
    if diffs.shape[0] == 0:
        return {k: zero for k in STEP_KEYS}
    norms = jnp.linalg.norm(diffs, axis=-1)  # [S, B, T]
    step_loss = jnp.sum(jnp.square(norms)) / n_tokens
    last = jnp.sum(norms[-1]) / n_tokens
    if diffs.shape[0] < 2:
        noncontract = zero
    else:
        ratio = norms[1:] / (norms[:-1] + delta) - 1.0
        noncontract = jnp.sum(jnp.maximum(ratio, 0.0)) / n_tokens
    return {
        "step_loss": step_loss,
        "step_noncontract": noncontract,
        "step_last_norm": last,
    }


class StabilityMonitor:
    """Computes the stability metrics for the adapted sites of one forward pass."""

    def __init__(
        self,
        config: StabilityConfig,
        adapter_layers: Sequence[int],
    ) -> None:
        self.config = config
        layers = (
            tuple(sorted(config.layers))
            if config.layers is not None
            else tuple(adapter_layers)
        )
        stray = sorted(set(layers) - set(adapter_layers))
        if stray:
            raise ValueError(
                f"Stability sites must be adapted sites; {stray} are not in "
                f"{list(adapter_layers)}."
            )
        self.layers = layers

    def metrics(
        self,
        adapter: ResidualAdapter,
        adapter_params: AdapterParams,
        site_inputs: Mapping[int, jax.Array],
        corrected: Mapping[int, jax.Array] | None = None,
    ) -> dict[str, jax.Array]:
        """Metrics averaged over the sites; nothing here carries a gradient.

        ``site_inputs[i]`` is ``h_i`` as received by the adapter (``hidden_state(i)``).
        ``corrected[i]`` may supply ``h~_i = h_i + R(h_i)`` if the caller already has it.
        The step metrics use the one-state trajectory ``[h~_i]``: no settling, so they are
        exactly zero (see module docstring).
        """
        per_site: dict[str, list[jax.Array]] = {}
        for i in self.layers:
            h = site_inputs[i].astype(jnp.float32)
            out = (
                corrected[i].astype(jnp.float32)
                if corrected is not None and i in corrected
                else adapter.apply(adapter_params, h, i).astype(jnp.float32)
            )
            row = residual_smoothness(out - h, h)
            row.update(step_metrics(out[None], self.config.delta))
            for key, value in row.items():
                per_site.setdefault(key, []).append(value)
        metrics = {k: jnp.mean(jnp.stack(v)) for k, v in per_site.items()}
        metrics.update(
            {
                f"res_smooth_l{i}": v
                for i, v in zip(self.layers, per_site["res_smooth"], strict=True)
            }
        )
        return metrics


__all__ = [
    "STABILITY_MODES",
    "STEP_KEYS",
    "StabilityConfig",
    "StabilityMonitor",
    "residual_smoothness",
    "step_metrics",
]
