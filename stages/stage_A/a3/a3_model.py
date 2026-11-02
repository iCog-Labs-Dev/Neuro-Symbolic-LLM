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
from fabricpc.graph_initialization import initialize_params

from frozenllm.substrate import FrozenSubstrate, identity_modify
from stages.stage_base import checked_kwargs

from .a3_graph import (
    build_a3_graph,
    compute_energy_curve,
    infer_and_compute_grads,
)
from .a3_tail import residual_modify

_ACTIVATIONS = {
    "identity": IdentityActivation,
    "tanh": TanhActivation,
    "gelu": GeluActivation,
    "relu": ReLUActivation,
}


def _is_int(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_real(value: Any) -> bool:
    return (
        isinstance(value, Integral | Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_positive(name: str, value: Any) -> None:
    if not _is_real(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")


def _require_nonnegative(name: str, value: Any) -> None:
    if not _is_real(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number, got {value!r}")


def _validate_layers(layers: Any, num_layers: int) -> tuple[int, ...]:
    if isinstance(layers, str | bytes) or not isinstance(layers, Sequence):
        raise TypeError("layers must be a sequence of integer layer indices")

    result = tuple(layers)

    if not result:
        raise ValueError("layers must not be empty")

    if any(not _is_int(layer) for layer in result):
        raise TypeError("all layer indices must be integers")

    if any(layer < 0 or layer >= num_layers for layer in result):
        raise ValueError(f"layer indices must be between 0 and {num_layers - 1}")

    if len(set(result)) != len(result):
        raise ValueError("layers must not contain duplicates")

    return tuple(sorted(result))


@dataclass(frozen=True)
class A3HybridConfig:
    infer_steps: int
    eta_z: float
    sigma2: float
    lambda_kl: float
    predictor_init_std: float
    projection_init_std: float

    def __post_init__(self) -> None:
        if not _is_int(self.infer_steps) or self.infer_steps < 0:
            raise ValueError("infer_steps must be an integer >= 0")

        for name in (
            "eta_z",
            "sigma2",
            "predictor_init_std",
            "projection_init_std",
        ):
            _require_positive(name, getattr(self, name))

        _require_nonnegative("lambda_kl", self.lambda_kl)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> A3HybridConfig:
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
        config: A3HybridConfig,
    ) -> None:
        if not isinstance(config, A3HybridConfig):
            raise TypeError(
                f"config must be an A3HybridConfig, got {type(config).__name__}"
            )

        cfg = config
        arch = substrate.architecture

        if (
            min(
                arch.num_layers,
                arch.hidden_size,
                arch.vocab_size,
            )
            <= 0
        ):
            raise ValueError(
                "substrate architecture must have positive "
                "layer, hidden and vocab sizes"
            )

        layers = _validate_layers(layers, arch.num_layers)

        if not _is_int(rank) or rank < 1:
            raise ValueError(f"rank must be an integer >= 1, got {rank!r}")

        if not _is_int(seq_len) or seq_len < 2:
            raise ValueError(f"seq_len must be an integer >= 2, got {seq_len!r}")

        max_pos = getattr(
            arch,
            "max_position_embeddings",
            None,
        )

        if max_pos is not None and seq_len > max_pos:
            raise ValueError(
                f"seq_len={seq_len} exceeds the model's " f"{max_pos} positions"
            )

        _require_positive("scale", scale)

        if not isinstance(activation, str) or activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of "
                f"{sorted(_ACTIVATIONS)}, got {activation!r}"
            )

        self.sub = substrate
        self.cfg = cfg
        self.layers = layers
        self.rank = int(rank)
        self.seq_len = int(seq_len)
        self.scale = float(scale)

        self.hidden_size = int(arch.hidden_size)
        self.vocab_size = int(arch.vocab_size)
        self._activation = _ACTIVATIONS[activation]()

        self.structure, self._tail_name = build_a3_graph(
            substrate=self.sub,
            layers=self.layers,
            rank=self.rank,
            seq_len=self.seq_len,
            scale=self.scale,
            activation_class=_ACTIVATIONS[activation],
            config=self.cfg,
        )

    def init_params(self, key: jax.Array):
        return initialize_params(self.structure, key)

    def predictor_weight(
        self,
        params,
        layer: int,
    ) -> jax.Array:
        return params.nodes[f"z{layer}"].weights[f"h0_{layer}->z{layer}:in"]

    def projection(
        self,
        params,
        layer: int,
    ) -> jax.Array:
        return params.nodes[self._tail_name].weights[f"z{layer}->{self._tail_name}:z"]

    def adapter_params(
        self,
        params,
    ) -> dict[str, dict[str, jax.Array]]:
        return {
            f"layer_{layer}": {
                "A": self.predictor_weight(params, layer),
                "B": self.projection(params, layer),
            }
            for layer in self.layers
        }

    def _check_ids(
        self,
        ids: jax.Array,
    ) -> jax.Array:
        ids = jnp.asarray(ids)

        if ids.ndim != 2 or ids.shape[1] != self.seq_len:
            raise ValueError(
                f"ids must have shape [batch, {self.seq_len}], "
                f"got {tuple(ids.shape)}"
            )

        if ids.shape[0] < 1:
            raise ValueError("ids batch dimension must be non-empty")

        if not jnp.issubdtype(ids.dtype, jnp.integer):
            raise TypeError(f"ids must be an integer array, got dtype {ids.dtype}")

        return ids.astype(jnp.int32)

    def clean_pass(
        self,
        ids: jax.Array,
    ) -> tuple[dict[int, jax.Array], jax.Array]:
        clean = self.sub.run_with_interception(
            ids,
            modify_fn=identity_modify,
            intercept_layers=self.layers,
        )

        h0 = {
            layer: jax.lax.stop_gradient(clean.hidden_state(layer))
            for layer in self.layers
        }

        return h0, jax.lax.stop_gradient(clean.logits)

    def make_batch(
        self,
        ids: jax.Array,
    ) -> tuple[
        dict[str, jax.Array],
        dict[int, jax.Array],
        jax.Array,
    ]:
        ids = self._check_ids(ids)

        h0, base_logits = self.clean_pass(ids)

        batch = {
            "x": ids,
            "y": ids[:, 1:],
            **{f"h0_{layer}": h0[layer].astype(jnp.float32) for layer in self.layers},
        }

        if self.cfg.lambda_kl > 0:
            batch["base"] = jax.nn.log_softmax(
                base_logits[:, :-1].astype(jnp.float32),
                axis=-1,
            )

        return batch, h0, base_logits

    def _final_terms(
        self,
        params,
        ids,
        base_logits,
        z_star,
        pred,
        n_pred,
    ) -> dict[str, jax.Array]:
        b = {layer: self.projection(params, layer) for layer in self.layers}

        logits = self.sub.run_with_interception(
            ids,
            modify_fn=residual_modify(
                z_star,
                b,
                self.scale,
            ),
            intercept_layers=self.layers,
        ).logits

        ce = self.sub.compute_loss(
            logits,
            ids,
            losses=("CE",),
        )

        terms = {"task": ce}
        loss = ce

        if self.cfg.lambda_kl > 0:
            kl = self.sub.compute_loss(
                logits[:, :-1],
                None,
                losses=("KL",),
                base_logits=base_logits[:, :-1],
            )

            terms["kl"] = kl
            loss = ce + self.cfg.lambda_kl * kl

        gauss = {"precision": 1.0 / self.cfg.sigma2}

        spring = (
            sum(
                jnp.sum(
                    GaussianEnergy.energy(
                        z_star[layer],
                        pred[layer],
                        gauss,
                    )
                )
                for layer in self.layers
            )
            / n_pred
        )

        return {
            **terms,
            "loss": loss,
            "spring": spring,
            "energy": loss + spring,
        }

    def train_grads(
        self,
        params,
        ids: jax.Array,
        key: jax.Array,
    ):
        batch, _, base_logits = self.make_batch(ids)

        result = infer_and_compute_grads(
            structure=self.structure,
            params=params,
            batch=batch,
            key=key,
            layers=self.layers,
        )

        exact = self._final_terms(
            params,
            batch["x"],
            base_logits,
            result.z_star,
            result.pred,
            result.n_pred,
        )

        energy_after_exact = exact.pop("energy")

        aux = {
            "n_pred": result.n_pred,
            "energy_before": result.energy_before,
            "energy_after": result.energy_after,
            "energy_after_exact": energy_after_exact,
            **exact,
            "z_star": result.z_star,
            "pred": result.pred,
        }

        return result.grads, aux

    def energy_curve(
        self,
        params,
        ids: jax.Array,
        key: jax.Array,
        steps: int | None = None,
    ):
        batch, _, _ = self.make_batch(ids)

        return compute_energy_curve(
            structure=self.structure,
            params=params,
            batch=batch,
            key=key,
            steps=steps,
        )

    def predict_latents(
        self,
        params,
        h0: Mapping[int, jax.Array],
    ) -> dict[int, jax.Array]:
        a = self._activation

        return {
            layer: type(a).forward(
                h0[layer]
                @ self.predictor_weight(
                    params,
                    layer,
                ),
                a.config,
            )
            for layer in self.layers
        }

    def deploy_logits(
        self,
        params,
        ids: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        ids = self._check_ids(ids)

        h0, base_logits = self.clean_pass(ids)
        z = self.predict_latents(params, h0)

        b = {layer: self.projection(params, layer) for layer in self.layers}

        logits = self.sub.run_with_interception(
            ids,
            modify_fn=residual_modify(
                z,
                b,
                self.scale,
            ),
            intercept_layers=self.layers,
        ).logits

        return logits, base_logits


__all__ = ["A3HybridConfig", "A3Model"]
