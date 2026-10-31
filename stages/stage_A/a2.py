"""Stage A2: the PC-inspired auxiliary objective on top of the A1 residual adapter.

L_A2 = L_task + lambda_kl * KL(p_F0 || p_F0+R) + lambda_wd * ||phi||^2   (A1, unchanged)
     + lambda_cp   * L_cpc                                               (contrastive future)
     + lambda_stab * (w_anch * L_zanch + w_hep * L_hep-nudge
                      + w_smooth * L_res-smooth + w_step * L_step)    (stability)
     + lambda_hid  * L_hid                                               (hidden prediction)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from frozenllm.substrate import FrozenSubstrate, identity_modify
from residual import AdapterParams, ResidualAdapter
from residual.a2_losses import (
    _EPS,
    _VAR_FLOOR,
    _masked_mean,
    anchor_loss,
    collect_stability_metrics,
    contrastive_future_loss,
    hep_nudge_loss,
    output_error,
)
from residual.a2_losses import hidden_prediction_loss as _hidden_loss_unmasked
from residual.cpc_heads import BACKENDS as CPC_BACKENDS
from residual.cpc_heads import (
    CPCParams,
    a_key,
    init_cpc_heads,
    q_key,
    split_cpc,
)
from residual.fpc_energy import BACKENDS as ENERGY_BACKENDS
from residual.fpc_energy import anchor_sq_distance, check_backend
from residual.hep import (
    KINDS,
    TRAINABLE_KINDS,
    HEPParams,
    highway_for,
    init_hep,
    split_hep,
)
from residual.local_predictor import (
    LocalPredictorBank,
    PredictorConfig,
    PredictorParams,
    split_params,
)
from residual.stability import StabilityConfig as MonitorConfig
from residual.stability import StabilityMonitor
from stages.stage_A.a1 import A1Config, a1_objective
from stages.stage_base import base_logits, checked_kwargs

POSITIVES = ("base", "corrected")
HIDDEN_TARGETS = ("corrected_teacher", "base_anchor")


@dataclass(frozen=True)
class CPCConfig:
    """``objective.cpc`` section of the A2 YAML."""

    weight: float = 0.0
    offsets: tuple[int, ...] = (1,)
    dim: int = 32
    tau: float = 0.1
    positive: str = "base"
    max_anchors_per_seq: int | None = 256
    a_init_noise: float = 0.01
    heads: str = "fabricpc"

    def __post_init__(self) -> None:
        if not isinstance(self.weight, int | float) or self.weight < 0:
            raise ValueError(f"weight must be a number >= 0, got {self.weight!r}")
        offsets = self.offsets
        if isinstance(offsets, int) and not isinstance(offsets, bool):
            offsets = (offsets,)
        offsets = tuple(offsets) if isinstance(offsets, list | tuple) else offsets
        if (
            not isinstance(offsets, tuple)
            or not offsets
            or any(
                not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in offsets
            )
            or len(set(offsets)) != len(offsets)
        ):
            raise ValueError(
                f"offsets must be distinct integers >= 1, got {self.offsets!r}"
            )
        object.__setattr__(self, "offsets", offsets)
        if self.heads not in CPC_BACKENDS:
            raise ValueError(
                f"heads must be one of {list(CPC_BACKENDS)}, got {self.heads!r}"
            )
        if not isinstance(self.dim, int) or isinstance(self.dim, bool) or self.dim < 1:
            raise ValueError(f"dim must be an integer >= 1, got {self.dim!r}")
        if not self.tau > 0:
            raise ValueError(f"tau must be > 0, got {self.tau!r}")
        if self.positive not in POSITIVES:
            raise ValueError(
                f"positive must be one of {list(POSITIVES)}, got {self.positive!r}"
            )
        if self.max_anchors_per_seq is not None and (
            not isinstance(self.max_anchors_per_seq, int)
            or isinstance(self.max_anchors_per_seq, bool)
            or self.max_anchors_per_seq < 1
        ):
            raise ValueError(
                "max_anchors_per_seq must be an integer >= 1 or null, "
                f"got {self.max_anchors_per_seq!r}"
            )
        if not self.a_init_noise >= 0:
            raise ValueError(f"a_init_noise must be >= 0, got {self.a_init_noise!r}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CPCConfig:
        data = dict(data)
        if "offset" in data:  # single-offset spelling
            if "offsets" in data:
                raise ValueError("give either 'offset' or 'offsets', not both")
            data["offsets"] = (data.pop("offset"),)
        return cls(**checked_kwargs(cls, data))

    @property
    def enabled(self) -> bool:
        return self.weight > 0

    @property
    def needs_base_states(self) -> bool:
        return self.enabled and self.positive == "base"


NORMALIZATIONS = ("relative", "none")


def _sq_norm(x: jax.Array, backend: str) -> jax.Array:
    """``||x||^2`` per token, ``[..., B, T]``; on ``"fabricpc"`` the energy of an
    ``IdentityNode`` whose target is clamped to 0 (so ``x`` carries the gradient)."""
    if backend == "fabricpc":
        return anchor_sq_distance(x, jnp.zeros_like(x))
    return jnp.sum(jnp.square(x), axis=-1)


def residual_smoothness_loss(
    correction: jax.Array,
    inputs: jax.Array,
    *,
    normalize: str = "relative",
    mask: jax.Array | None = None,
    backend: str = "fabricpc",
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Residual-smoothness loss ``L_res-smooth``: neighbouring tokens get similar corrections.

    L = mean_{b, t>1} || R(h_t) - R(h_{t-1}) ||^2                         (``"none"``)
    L = that / ( mean_{b, t>1} || sg(h_t - h_{t-1}) ||^2 + eps )          (``"relative"``)

    """
    if normalize not in NORMALIZATIONS:
        raise ValueError(f"normalize must be 'relative' or 'none', got {normalize!r}")
    check_backend(backend)
    if correction.ndim < 3 or correction.shape != inputs.shape:
        raise ValueError(
            f"correction and inputs must share a shape [..., B, T, d], got "
            f"{correction.shape} and {inputs.shape}"
        )
    r = correction.astype(jnp.float32)
    h = jax.lax.stop_gradient(inputs.astype(jnp.float32))
    if r.shape[-2] < 2:
        zero = jnp.zeros((), jnp.float32) + 0.0 * jnp.sum(r)
        return zero, {"smooth_res": zero, "smooth_ratio": zero}
    pair = None if mask is None else (mask[..., 1:] & mask[..., :-1])
    res = _masked_mean(_sq_norm(r[..., 1:, :] - r[..., :-1, :], backend), pair)
    inp = _masked_mean(_sq_norm(h[..., 1:, :] - h[..., :-1, :], backend), pair)
    ratio = res / (inp + _EPS)
    loss = ratio if normalize == "relative" else res
    return loss, {"smooth_res": res, "smooth_ratio": ratio}


def step_loss(
    trajectory: jax.Array,
    *,
    normalize: str = "relative",
    mask: jax.Array | None = None,
    backend: str = "fabricpc",
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Settling-step loss ``L_step``: successive states of a trajectory stay close.

        L = mean_{b,t} sum_s || z^(s+1) - z^(s) ||^2                          (``"none"``)
        L = that / ( mean_{b,t} || sg(z^(0)) ||^2 + eps )                     (``"relative"``)

    ``trajectory`` is ``[S+1, B, T, d]``. A2 has no settling and passes the single state
    ``[1, B, T, d]``: no differences, so the loss is exactly 0 with zero gradient. Stage A3
    passes its real trajectory. Returns ``(loss, {"step_sq"})``.
    """
    if normalize not in NORMALIZATIONS:
        raise ValueError(f"normalize must be 'relative' or 'none', got {normalize!r}")
    check_backend(backend)
    if trajectory.ndim != 4:
        raise ValueError(
            f"trajectory must be [S+1, B, T, d], got {tuple(trajectory.shape)}"
        )
    z = trajectory.astype(jnp.float32)
    if z.shape[0] < 2:
        zero = jnp.zeros((), jnp.float32) + 0.0 * jnp.sum(z)
        return zero, {"step_sq": zero}
    sq = jnp.sum(_sq_norm(z[1:] - z[:-1], backend), axis=0)  # [B, T]
    raw = _masked_mean(sq, mask)
    scale = _masked_mean(jnp.sum(jnp.square(jax.lax.stop_gradient(z[0])), -1), mask)
    loss = raw / (scale + _EPS) if normalize == "relative" else raw
    return loss, {"step_sq": raw}


def hidden_prediction_loss(
    pred: jax.Array,
    target: jax.Array,
    source: jax.Array,
    config: Any,
    mask: jax.Array | None = None,
) -> dict[str, jax.Array]:
    """The teammate's ``hidden_prediction_loss``, plus an optional padding ``mask``.

    Without ``mask`` this *is* ``residual.a2_losses.hidden_prediction_loss``. With ``mask``
    (``[B, S]``, ``False`` = padding or an ignored label) every mean, and the target variance,
    runs over the valid tokens only.
    """
    if mask is None:
        return _hidden_loss_unmasked(pred, target, source, config)
    target = jax.lax.stop_gradient(target.astype(jnp.float32))
    source = source.astype(jnp.float32)
    if config.skip_first_token:
        pred, target, source = pred[:, 1:], target[:, 1:], source[:, 1:]
        mask = mask[:, 1:]
    sq = jnp.square(pred - target)
    copy_sq = jnp.square(jax.lax.stop_gradient(source) - target)
    w = mask.astype(jnp.float32)[..., None]  # [B, S, 1]
    n = jnp.maximum(jnp.sum(w), 1.0)
    d = target.shape[-1]
    mse = jnp.sum(w * sq) / (n * d)
    copy_mse = jnp.sum(w * copy_sq) / (n * d)
    mean = jnp.sum(w * target, axis=(0, 1)) / n  # [D]
    var = jnp.mean(jnp.sum(w * jnp.square(target - mean), axis=(0, 1)) / n) + _VAR_FLOOR
    rel = mse / var
    return {
        "loss": rel if config.hidden_norm == "target_var" else mse,
        "mse": mse,
        "rel": rel,
        "copy_rel": copy_mse / var,
    }


def _number(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise ValueError(f"{name} must be a number >= 0, got {value!r}")


@dataclass(frozen=True)
class HEPConfig:
    """``objective.stab.hep`` section of the A2 YAML."""

    # null switches HEP off (no nudge metric, no HEP loss).
    kind: str | None = None
    # Rank of the low-rank / fixed-random highway.
    rank: int = 16
    # alpha_y: gain of the nudge (the program sweeps it).
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.kind is not None and self.kind not in KINDS:
            raise ValueError(
                f"kind must be null or one of {list(KINDS)}, got {self.kind!r}"
            )
        if (
            not isinstance(self.rank, int)
            or isinstance(self.rank, bool)
            or self.rank < 1
        ):
            raise ValueError(f"rank must be an integer >= 1, got {self.rank!r}")
        _number("alpha", self.alpha)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HEPConfig:
        return cls(**checked_kwargs(cls, data))

    @property
    def trainable(self) -> bool:
        return self.kind in TRAINABLE_KINDS


@dataclass(frozen=True)
class StabLossConfig:
    """``objective.stab`` section of the A2 YAML: the stability *loss*.

    The stability loss has four sub-terms:

    =========== ============================= ================================
    Sub-term    Weight field                  Status in A2
    =========== ============================= ================================
    L_zanch     ``w_anch``                    Active
    L_hep-nudge ``w_hep``                     Active (needs trainable highway)
    L_res-smooth``w_smooth``                  Active
    L_step      ``w_step``                    **Inactive** in A2 — see below
    =========== ============================= ================================

    ``w_step`` and ``L_step`` in A2
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``L_step = E_{b,t} sum_{s=0}^{K-1} ||z^(s+1) - z^(s)||^2``.
    A2 provides only a single feed-forward state (no settling trajectory),
    so ``K = 0`` and the sum is empty: the loss is *identically zero* with
    *zero gradient*.  Enabling ``w_step > 0`` in A2 (as the ``stab_all``
    arm does) is therefore harmless but has **no training effect**.

    The field exists for API compatibility so that ``stab_all`` exercises
    the step-loss code path, and A3 can reuse ``StabLossConfig`` with a
    real settling trajectory without any schema changes.
    """

    weight: float = 0.0
    w_anch: float = 1.0
    w_hep: float = 0.0
    w_smooth: float = 0.0
    # Settling-step loss: structurally inactive in A2 (no settling trajectory,
    # so L_step ≡ 0 with zero gradient).  Retained for A3 API compatibility.
    w_step: float = 0.0
    normalize: str = "relative"
    backend: str = "fabricpc"
    hep: HEPConfig = field(default_factory=HEPConfig)

    def __post_init__(self) -> None:
        for name in ("weight", "w_anch", "w_hep", "w_smooth", "w_step"):
            _number(name, getattr(self, name))
        if self.normalize not in NORMALIZATIONS:
            raise ValueError(
                f"normalize must be one of {list(NORMALIZATIONS)}, got {self.normalize!r}"
            )
        if self.backend not in ENERGY_BACKENDS:
            raise ValueError(
                f"backend must be one of {list(ENERGY_BACKENDS)}, got {self.backend!r}"
            )
        if self.w_hep > 0 and self.hep.kind is None:
            raise ValueError("w_hep > 0 needs hep.kind (dense or lowrank).")
        if self.w_hep > 0 and not self.hep.trainable:
            raise ValueError(
                f"w_hep > 0 with hep.kind={self.hep.kind!r}: the highway is fixed and "
                "eps is under stop-gradient, so the term has no gradient. Set w_hep = 0 "
                "(the nudge norm is still logged) or use kind dense / lowrank."
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StabLossConfig:
        raw = dict(checked_kwargs(cls, data))
        raw["hep"] = HEPConfig.from_dict(raw.get("hep") or {})
        return cls(**raw)

    @property
    def enabled(self) -> bool:
        return self.weight > 0

    @property
    def needs_base_states(self) -> bool:
        return self.enabled and self.w_anch > 0

    @property
    def needs_output_error(self) -> bool:
        return self.enabled and self.hep.kind is not None


HIDDEN_TARGETS = ("corrected_teacher", "base_anchor")
HIDDEN_NORMS = ("target_var", "none")


@dataclass(frozen=True)
class HiddenConfig:
    """``objective.hid`` section of the A2 YAML: the hidden-prediction loss ``L_hid``.

    Hidden-target semantics (``hidden_target`` field)
    --------------------------------------------------
    Two target definitions are supported, corresponding to two interpretations
    of "predict the next frozen hidden state":

    ``corrected_teacher`` **(Option B — default)**
        y^hid_{l,t} = F^0_{l→l+1}(h̃_{l,t})  — the next *frozen* block
        applied to the *corrected* (adapter-modified) input.
        L_hid = Σ ‖sg(h̃^teacher_{l+1,t}) − P_l(h̃_{l,t})‖²
        This trains P_l as a local surrogate for the frozen block around the
        states the residual layer actually visits.

    ``base_anchor`` **(Option A)**
        y^hid_{l,t} = h^0_{l+1,t}  — the next-block state from a clean,
        unmodified base forward pass.
        This anchors the predictor to the original frozen trajectory and
        requires an additional base forward (sets ``needs_base_targets``).

    **Why ``corrected_teacher`` is the default:**  The paper's recommendation
    for the A2 hidden-prediction auxiliary loss is Option B
    (``corrected_teacher``).  Option A (``base_anchor``) is appropriate only
    when the intended term is explicitly a hidden-state anchor rather than a
    local predictor.  In both cases, the target is stop-gradiented and the
    frozen base parameters remain frozen.

    To switch to the base-anchor interpretation::

        hidden_target: base_anchor   # in the YAML
        --hidden-target base_anchor  # on the CLI
    """

    weight: float = 0.0
    lambda_hid: float | None = None
    hidden_target: str = "corrected_teacher"
    predictor_lr_mult: float = 1.0
    hidden_norm: str = "target_var"
    skip_first_token: bool = True
    predictor: PredictorConfig = field(default_factory=PredictorConfig)

    def __post_init__(self) -> None:
        if self.lambda_hid is not None:
            object.__setattr__(self, "weight", self.lambda_hid)
            object.__setattr__(self, "lambda_hid", None)
        _number("weight", self.weight)
        if self.hidden_target not in HIDDEN_TARGETS:
            raise ValueError(
                f"hidden_target must be one of {list(HIDDEN_TARGETS)}, "
                f"got {self.hidden_target!r}"
            )
        if (
            isinstance(self.predictor_lr_mult, bool)
            or not isinstance(self.predictor_lr_mult, int | float)
            or not self.predictor_lr_mult > 0
        ):
            raise ValueError(
                f"predictor_lr_mult must be a number > 0, got {self.predictor_lr_mult!r}"
            )
        if self.hidden_norm not in HIDDEN_NORMS:
            raise ValueError(
                f"hidden_norm must be one of {list(HIDDEN_NORMS)}, got {self.hidden_norm!r}"
            )
        if not isinstance(self.skip_first_token, bool):
            raise ValueError(
                f"skip_first_token must be a boolean, got {self.skip_first_token!r}"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HiddenConfig:
        raw = dict(checked_kwargs(cls, data))
        alias = raw.pop("lambda_hid", None)
        if alias is not None:
            if "weight" in raw and raw["weight"] != alias:
                raise ValueError("hid.weight and hid.lambda_hid must match")
            raw["weight"] = alias
        raw["predictor"] = PredictorConfig.from_dict(raw.get("predictor") or {})
        return cls(**raw)

    @property
    def enabled(self) -> bool:
        return self.weight > 0

    @property
    def coupled(self) -> bool:
        """Whether the predictor error reaches the adapter (``detach_input`` false)."""
        return self.enabled and not self.predictor.detach_input


_LEGACY_HID_KEYS = (
    "lambda_hid",
    "hidden_target",
    "hidden_norm",
    "skip_first_token",
    "predictor_lr_mult",
    "predictor",
)


@dataclass(frozen=True)
class A2Config:
    """``objective:`` section of the A2 YAML."""

    lambda_kl: float = 0.0
    lambda_wd: float = 0.0
    seed: int = 0
    cpc: CPCConfig = field(default_factory=CPCConfig)
    stab: StabLossConfig = field(default_factory=StabLossConfig)
    hid: HiddenConfig = field(default_factory=HiddenConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)

    def __post_init__(self) -> None:
        _number("lambda_kl", self.lambda_kl)
        _number("lambda_wd", self.lambda_wd)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an integer, got {self.seed!r}")
        if self.stab.hep.kind == "tied":
            raise ValueError(
                "hep.kind='tied' is not wired into a2_objective yet (it needs the frozen "
                "LM-head matrix); use dense, lowrank or fixed_random."
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> A2Config:
        raw = dict(data)
        legacy = {k: raw.pop(k) for k in _LEGACY_HID_KEYS if k in raw}
        if legacy:
            hid = dict(raw.get("hid") or {})
            overlap = sorted(set(hid) & set(legacy))
            if overlap:
                raise ValueError(
                    f"keys given both at the top level and in objective.hid: {overlap}"
                )
            raw["hid"] = {**hid, **legacy}
        if "stability" in raw:
            if "monitor" in raw:
                raise ValueError("give either 'stability' or 'monitor', not both")
            raw["monitor"] = raw.pop("stability")
        raw = dict(checked_kwargs(cls, raw))
        raw["cpc"] = CPCConfig.from_dict(raw.get("cpc") or {})
        raw["stab"] = StabLossConfig.from_dict(raw.get("stab") or {})
        raw["hid"] = HiddenConfig.from_dict(raw.get("hid") or {})
        raw["monitor"] = MonitorConfig.from_dict(raw.get("monitor") or {})
        return cls(**raw)

    @property
    def lambda_hid(self) -> float:
        return self.hid.weight

    @property
    def hidden_target(self) -> str:
        return self.hid.hidden_target

    @property
    def hidden_norm(self) -> str:
        return self.hid.hidden_norm

    @property
    def skip_first_token(self) -> bool:
        return self.hid.skip_first_token

    @property
    def predictor_lr_mult(self) -> float:
        return self.hid.predictor_lr_mult

    @property
    def predictor(self) -> PredictorConfig:
        return self.hid.predictor

    @property
    def stability(self) -> MonitorConfig:
        return self.monitor

    # --- what the objective needs -----------------------------------------------------------------

    @property
    def uses_hidden_pred(self) -> bool:
        """Whether the hidden-prediction loss (and so the predictor bank) is on."""
        return self.hid.enabled

    @property
    def hid_coupled(self) -> bool:
        """Whether the predictor error reaches the adapter (``detach_input`` false)."""
        return self.hid.coupled

    @property
    def needs_base_targets(self) -> bool:
        """Whether the hidden-prediction target comes from the base model (Option A)."""
        return self.hid.enabled and self.hid.hidden_target == "base_anchor"

    @property
    def needs_base_states(self) -> bool:
        """Whether the objective needs the base model's hidden states ``h^base``."""
        return (
            self.cpc.needs_base_states
            or self.stab.needs_base_states
            or self.needs_base_targets
        )

    @property
    def aux_enabled(self) -> bool:
        """Whether any auxiliary *loss* is on."""
        return self.cpc.enabled or self.stab.enabled or self.hid.enabled

    @property
    def needs_corrected(self) -> bool:
        """Whether the corrected states ``h~_l`` are needed (a loss or the metrics)."""
        return self.aux_enabled or self.monitor.enabled


#  Parameters
ADAPTER_PREFIX = "layer_"  # layer_key(l) == f"{ADAPTER_PREFIX}{l}"


def adapter_params(params: Mapping[str, Mapping[str, jax.Array]]) -> AdapterParams:
    """The ``layer_<l>`` entries: what ``ResidualAdapter`` and the A1 weight decay see."""
    return {k: dict(v) for k, v in params.items() if k.startswith(ADAPTER_PREFIX)}


_INIT_SEQ_LEN = 1


def init_a2_params(
    config: A2Config, adapter: ResidualAdapter, seq_len: int | None = None
) -> dict[str, Any]:
    """The adapter parameters plus the heads and predictors the enabled terms need."""
    d = adapter.hidden_size
    layers = tuple(adapter.layers)
    params: dict[str, Any] = dict(adapter.init_params())
    heads = {
        **init_cpc_params(config.cpc, config.seed, layers, d),
        **init_hep_params(
            config.stab, config.seed, layers, adapter.architecture.vocab_size, d
        ),
    }
    clash = sorted(set(heads) & set(params))
    if clash:
        raise ValueError(f"head parameters clash with the adapter keys: {clash}")
    params.update(heads)
    if config.uses_hidden_pred:
        bank = predictor_bank(config, adapter, seq_len or _INIT_SEQ_LEN)
        pred = bank.init_params()
        clash = sorted(set(pred) & set(params))
        if clash:
            raise ValueError(f"predictor parameters clash with the other keys: {clash}")
        params.update(pred)
    return params


_BANKS: dict[tuple[Any, ...], LocalPredictorBank] = {}


def predictor_bank(
    config: A2Config, adapter: ResidualAdapter, seq_len: int
) -> LocalPredictorBank:
    """The predictors ``P_l`` of ``config.predictor`` for ``adapter``'s sites (built once)."""
    p = config.predictor
    arch = adapter.architecture
    key = (
        p.ff_dim,
        p.out_init_std,
        p.predict_delta,
        p.detach_input,
        None if p.layers is None else tuple(p.layers),
        p.seed,
        arch.hidden_size,
        arch.num_layers,
        seq_len,
        tuple(adapter.layers),
    )
    if key not in _BANKS:
        bank = LocalPredictorBank(p, arch, seq_len, adapter.layers)
        stray = sorted(set(bank.layers) - set(adapter.layers))
        if stray:
            raise ValueError(
                f"Predictor sites must be adapted layers; {stray} are not in "
                f"{list(adapter.layers)}."
            )
        _BANKS[key] = bank
    return _BANKS[key]


def predictor_params(params: Mapping[str, Mapping[str, jax.Array]]) -> PredictorParams:
    """The ``pred_layer_<l>`` entries of a full A2 parameter tree."""
    return split_params(params)[1]


def param_counts(
    params: Mapping[str, Mapping[str, jax.Array]],
) -> dict[str, int]:
    """Trainable parameters by group: ``adapter``, ``cpc``, ``hep``, ``pred`` and ``total``."""

    def count(tree: Mapping[str, Mapping[str, jax.Array]]) -> int:
        return sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(tree))

    groups = {
        "adapter": count(adapter_params(params)),
        "cpc": count(split_cpc(params)),
        "hep": count(split_hep(params)),
        "pred": count(predictor_params(params)),
    }
    other = count(params) - sum(groups.values())
    if other:
        raise ValueError(f"{other} parameters belong to no known group.")
    return {**groups, "total": sum(groups.values())}


def init_cpc_params(
    config: CPCConfig, seed: int, layers: tuple[int, ...], hidden_size: int
) -> CPCParams:
    """Heads for ``layers``; empty when the term is disabled."""
    if not config.enabled:
        return {}
    return init_cpc_heads(
        seed,
        layers,
        hidden_size,
        config.dim,
        config.offsets,
        config.a_init_noise,
    )


def init_hep_params(
    config: StabLossConfig,
    seed: int,
    layers: tuple[int, ...],
    vocab_size: int,
    hidden_size: int,
) -> HEPParams:
    """Trainable highway parameters; empty unless HEP is enabled with a trainable kind."""
    if not (config.enabled and config.hep.trainable):
        return {}
    assert config.hep.kind is not None
    return init_hep(
        config.hep.kind, seed, layers, vocab_size, hidden_size, config.hep.rank
    )


# =============================================================================
# 3. Terms (pure functions of hidden-state dictionaries ``{layer: [B, T, d]}``)
# =============================================================================


def token_mask(labels: jax.Array | None, ignore_index: int = -100) -> jax.Array | None:
    """``[B, T]`` validity of the tokens from the labels: ``labels != ignore_index``."""
    if labels is None:
        return None
    return labels != ignore_index


def cpc_term(
    params: Mapping[str, Mapping[str, jax.Array]],
    *,
    config: CPCConfig,
    corrected: Mapping[int, jax.Array],
    base: Mapping[int, jax.Array] | None = None,
    mask: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Unweighted ``L_cpc`` (mean over sites ``(l, k)``) and its diagnostics."""
    if not corrected:
        raise ValueError("cpc_term needs at least one site.")
    if config.positive == "base" and base is None:
        raise ValueError("positive='base' requires the base hidden states.")
    losses = []
    metrics: dict[str, list[jax.Array]] = {}
    for layer, anchors in corrected.items():
        if config.positive == "base":
            assert base is not None
            if layer not in base:
                raise KeyError(f"missing base hidden state for layer {layer}.")
            targets = base[layer]
        else:
            targets = anchors
        for offset in config.offsets:
            loss, m = contrastive_future_loss(
                params[q_key(layer)],
                params[a_key(layer, offset)],
                anchors,
                targets,
                offset=offset,
                tau=config.tau,
                max_anchors_per_seq=config.max_anchors_per_seq,
                mask=mask,
                backend=config.heads,
            )
            losses.append(loss)
            for name, value in m.items():
                metrics.setdefault(name, []).append(value)
    reduced = {name: jnp.mean(jnp.stack(vals)) for name, vals in metrics.items()}
    return jnp.mean(jnp.stack(losses)), reduced


def stability_term(
    params: Mapping[str, Mapping[str, jax.Array]],
    *,
    config: StabLossConfig,
    corrected: Mapping[int, jax.Array],
    base: Mapping[int, jax.Array] | None = None,
    logits: jax.Array | None = None,
    labels: jax.Array | None = None,
    seed: int = 0,
    tied: jax.Array | None = None,
    mask: jax.Array | None = None,
    sites: Mapping[int, jax.Array] | None = None,
    trajectories: Mapping[int, jax.Array] | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """The stability loss without ``lambda_stab``, and diagnostics::

    w_anch * L_zanch + w_hep * L_hep-nudge + w_smooth * L_res-smooth + w_step * L_step
    """
    if not corrected:
        raise ValueError("stability_term needs at least one site.")
    total = jnp.zeros((), jnp.float32)
    metrics: dict[str, jax.Array] = {}

    if config.w_anch > 0:
        if base is None:
            raise ValueError("w_anch > 0 requires the base hidden states.")
        losses, parts = [], []
        for layer, h in corrected.items():
            if layer not in base:
                raise KeyError(f"missing base hidden state for layer {layer}.")
            loss, m = anchor_loss(
                h,
                base[layer],
                normalize=config.normalize,
                mask=mask,
                backend=config.backend,
            )
            losses.append(loss)
            parts.append(m)
        l_anch = jnp.mean(jnp.stack(losses))
        total = total + config.w_anch * l_anch
        metrics["stab_anch"] = l_anch
        for name in parts[0]:
            metrics[name] = jnp.mean(jnp.stack([m[name] for m in parts]))

    if config.w_smooth > 0:
        if sites is None:
            raise ValueError("w_smooth > 0 requires the adapter input states (sites).")
        losses, parts = [], []
        for layer, h in corrected.items():
            if layer not in sites:
                raise KeyError(f"missing adapter input state for layer {layer}.")
            loss, m = residual_smoothness_loss(
                h - sites[layer],
                sites[layer],
                normalize=config.normalize,
                mask=mask,
                backend=config.backend,
            )
            losses.append(loss)
            parts.append(m)
        l_smooth = jnp.mean(jnp.stack(losses))
        total = total + config.w_smooth * l_smooth
        metrics["stab_smooth"] = l_smooth
        for name in parts[0]:
            metrics[name] = jnp.mean(jnp.stack([m[name] for m in parts]))

    if config.w_step > 0:
        # In A2, trajectories is None → traj = h[None] with shape [1,B,T,d].
        # step_loss returns exactly 0 (no successive states to compare).
        # This branch is exercised for code-path coverage (stab_all arm)
        # but has no training effect until A3 supplies a real trajectory.
        losses, parts = [], []
        for layer, h in corrected.items():
            traj = h[None] if trajectories is None else trajectories[layer]
            loss, m = step_loss(
                traj, normalize=config.normalize, mask=mask, backend=config.backend
            )
            losses.append(loss)
            parts.append(m)
        l_step = jnp.mean(jnp.stack(losses))
        total = total + config.w_step * l_step
        metrics["stab_step"] = l_step
        for name in parts[0]:
            metrics[name] = jnp.mean(jnp.stack([m[name] for m in parts]))

    if config.hep.kind is not None:
        if logits is None or labels is None:
            raise ValueError("HEP needs the adapted logits and the labels.")
        eps, eps_mask = output_error(logits, labels)
        n_pos = eps.shape[-2]
        losses, parts = [], []
        for layer, h in corrected.items():
            v = highway_for(
                config.hep.kind,
                layer,
                params=params,
                seed=seed,
                d_out=eps.shape[-1],
                d_hidden=h.shape[-1],
                rank=config.hep.rank,
                tied=tied,
            )

            loss, m = hep_nudge_loss(
                eps,
                v,
                config.hep.alpha,
                reference=h[..., :n_pos, :],
                mask=eps_mask,
                backend=config.backend,
            )
            losses.append(loss)
            parts.append(m)
        l_hep = jnp.mean(jnp.stack(losses))
        if config.w_hep > 0:
            total = total + config.w_hep * l_hep
        metrics["stab_hep"] = l_hep
        for name in parts[0]:
            metrics[name] = jnp.mean(jnp.stack([m[name] for m in parts]))

    return total, metrics


def hid_term(
    params: Mapping[str, Mapping[str, jax.Array]],
    *,
    config: A2Config,
    bank: LocalPredictorBank,
    corrected: Mapping[int, jax.Array],
    targets: Mapping[int, jax.Array],
    mask: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Unweighted ``L_hid`` (mean over the predictor sites) and its diagnostics."""
    losses = []
    parts: list[dict[str, jax.Array]] = []
    for layer in bank.layers:
        if layer not in corrected:
            raise KeyError(f"missing corrected hidden state for layer {layer}.")
        if layer not in targets:
            raise KeyError(f"missing next-block hidden state for layer {layer}.")
        source = corrected[layer]
        pred = bank.predict(params, layer, source)
        out = hidden_prediction_loss(pred, targets[layer], source, config, mask=mask)
        losses.append(out["loss"])
        parts.append(out)
    metrics = {
        f"hid_{name}": jnp.mean(jnp.stack([p[name] for p in parts]))
        for name in ("mse", "rel", "copy_rel")
    }
    return jnp.mean(jnp.stack(losses)), metrics


#  Objective


def compose_a2_loss(
    model_loss: jax.Array,
    *,
    config: A2Config,
    weight_decay: jax.Array,
    task: jax.Array | None = None,
    kl: jax.Array | None = None,
    hidden: jax.Array | None = None,
    cpc: jax.Array | None = None,
    stab: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Combine the loss terms and return ``(total, terms)``."""
    total = model_loss + config.lambda_wd * weight_decay
    terms: dict[str, jax.Array] = {
        "task": model_loss if task is None else task,
        "wd": weight_decay,
    }
    if kl is not None:
        terms["kl"] = kl
    for name, value, weight in (
        ("cpc", cpc, config.cpc.weight),
        ("stab", stab, config.stab.weight),
        ("hid", hidden, config.lambda_hid),
    ):
        if value is not None:
            total = total + weight * value
            terms[name] = value
    return total, terms


def _intercept_sites(
    adapter: ResidualAdapter, pred_layers: Sequence[int]
) -> tuple[int, ...]:
    """Adapted sites, the predictor input sites, and the blocks that follow them."""
    return tuple(
        sorted(set(adapter.layers) | set(pred_layers) | {i + 1 for i in pred_layers})
    )


def _a2_objective(
    params: Mapping[str, Mapping[str, jax.Array]],
    *,
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    config: A2Config,
    input_ids: jax.Array,
    labels: jax.Array | None,
    base: jax.Array | None,
    bank: LocalPredictorBank | None,
    monitor: StabilityMonitor | None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    ap = adapter_params(params)
    use_hid = config.uses_hidden_pred
    if use_hid and bank is None:
        bank = predictor_bank(config, adapter, int(input_ids.shape[1]))
    pred_layers: tuple[int, ...] = tuple(bank.layers) if (use_hid and bank) else ()
    if monitor is None and config.stability.enabled:
        monitor = StabilityMonitor(config.stability, adapter.layers)
    monitor_on = monitor is not None and monitor.config.enabled

    run = substrate.run_with_interception(
        input_ids,
        modify_fn=adapter.modify_fn(ap),
        intercept_layers=_intercept_sites(adapter, pred_layers),
    )
    adapted = run.logits
    target_labels = input_ids if labels is None else labels
    wd = ResidualAdapter.l2_norm_sq(ap)

    need_states = config.cpc.needs_base_states or config.stab.needs_base_states
    need_targets = config.needs_base_targets
    need_kl = config.lambda_kl > 0 or base is not None
    base_states: dict[int, jax.Array] | None = None
    base_targets: dict[int, jax.Array] | None = None
    base_lg: jax.Array | None = None
    if need_states or need_targets:
        wanted = set(adapter.layers) if need_states else set()
        if need_targets:
            wanted |= {i + 1 for i in pred_layers}
        base_run = substrate.run_with_interception(
            input_ids,
            modify_fn=identity_modify,
            intercept_layers=tuple(sorted(wanted)),
        )
        if need_states:
            base_states = {
                layer: jax.lax.stop_gradient(base_run.hidden_state(layer))
                for layer in adapter.layers
            }
        if need_targets:
            base_targets = {
                i: jax.lax.stop_gradient(base_run.hidden_state(i + 1))
                for i in pred_layers
            }
        base_lg = jax.lax.stop_gradient(base_run.logits if base is None else base)
    elif need_kl:
        base_lg = (
            base_logits(substrate, input_ids)
            if base is None
            else jax.lax.stop_gradient(base)
        )

    if base_lg is None:  # no KL: skip it rather than log NaN (strict JSON rejects it)
        model_loss = task = substrate.compute_loss(
            adapted, target_labels, losses=("CE",)
        )
        kl = None
    else:
        model_loss, components = substrate.compute_loss(
            adapted,
            target_labels,
            losses=("CE", "KL"),
            base_logits=base_lg,
            weights={"KL": config.lambda_kl},
            return_components=True,
        )
        task, kl = components["ce"], components["kl"]

    l_cpc = l_stab = l_hid = None
    extra: dict[str, jax.Array] = {}
    if config.needs_corrected or monitor_on:
        sites = {layer: run.hidden_state(layer) for layer in adapter.layers}
        corrected = {layer: adapter.apply(ap, h, layer) for layer, h in sites.items()}
        mask = token_mask(labels)
        if monitor_on:
            assert monitor is not None

            extra.update(
                collect_stability_metrics(
                    monitor,
                    adapter,
                    ap,
                    {
                        i: sites[i] if i in sites else run.hidden_state(i)
                        for i in monitor.layers
                    },
                    corrected=corrected,
                )
            )
        if config.cpc.enabled:
            l_cpc, m = cpc_term(
                params,
                config=config.cpc,
                corrected=corrected,
                base=base_states,
                mask=mask,
            )
            extra.update(m)
        if config.stab.enabled:
            l_stab, m = stability_term(
                params,
                config=config.stab,
                corrected=corrected,
                base=base_states,
                logits=adapted,
                labels=target_labels,
                seed=config.seed,
                mask=mask,
                sites=sites,
            )
            extra.update(m)
        if use_hid:
            assert bank is not None
            inputs = {
                i: (
                    corrected[i]
                    if i in corrected
                    else adapter.apply(ap, run.hidden_state(i), i)
                )
                for i in pred_layers
            }
            # base_anchor: clean-base next-block states (base_targets, built above);
            # corrected_teacher: the frozen next block on the actual corrected input.
            if config.hid.hidden_target == "base_anchor":
                assert base_targets is not None
                targets = base_targets
            else:
                targets = {
                    i: jax.lax.stop_gradient(run.hidden_state(i + 1))
                    for i in pred_layers
                }
            l_hid, m = hid_term(
                params,
                config=config,
                bank=bank,
                corrected=inputs,
                targets=targets,
                mask=mask,
            )
            extra.update(m)
            for i in pred_layers:  # per-site relative error, as the predictor sees it
                extra[f"hid_rel_l{i}"] = _site_rel(
                    params, config, bank, inputs[i], targets[i], i, mask
                )

    total, terms = compose_a2_loss(
        model_loss,
        config=config,
        weight_decay=wd,
        task=task,
        kl=kl,
        hidden=l_hid,
        cpc=l_cpc,
        stab=l_stab,
    )
    return total, {"total": total, **terms, **extra}


def _site_rel(
    params: Mapping[str, Mapping[str, jax.Array]],
    config: A2Config,
    bank: LocalPredictorBank,
    source: jax.Array,
    target: jax.Array,
    layer: int,
    mask: jax.Array | None,
) -> jax.Array:
    out = hidden_prediction_loss(
        bank.predict(params, layer, source), target, source, config, mask=mask
    )
    return out["rel"]


def make_a2_objective(
    bank: LocalPredictorBank | None = None, monitor: StabilityMonitor | None = None
) -> Callable[..., tuple[jax.Array, dict[str, jax.Array]]]:
    """Bind the predictor bank and the stability monitor into a stage objective."""

    def a2_objective(
        params: Mapping[str, Mapping[str, jax.Array]],
        *,
        substrate: FrozenSubstrate,
        adapter: ResidualAdapter,
        config: A2Config,
        input_ids: jax.Array,
        labels: jax.Array | None = None,
        base: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        return _a2_objective(
            params,
            substrate=substrate,
            adapter=adapter,
            config=config,
            input_ids=input_ids,
            labels=labels,
            base=base,
            bank=bank,
            monitor=monitor,
        )

    return a2_objective


a2_objective = make_a2_objective()


def make_a2_optimizer(
    config: A2Config, learning_rate: float
) -> optax.GradientTransformation:
    """Adam over the combined PyTree; the predictor group can use its own step size."""
    if config.predictor_lr_mult == 1.0:
        return optax.adam(learning_rate)

    def labels(params: Mapping[str, Any]) -> dict[str, Any]:
        _, pred = split_params(params)
        return {
            k: jax.tree_util.tree_map(
                lambda _, group="pred" if k in pred else "adapter": group, v
            )
            for k, v in params.items()
        }

    return optax.multi_transform(
        {
            "adapter": optax.adam(learning_rate),
            "pred": optax.adam(learning_rate * config.predictor_lr_mult),
        },
        labels,
    )


def a2_diagnostics(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    bank: LocalPredictorBank,
    config: A2Config,
    params: Mapping[str, Any],
    eval_batches: Sequence[jax.Array],
    *,
    jit: bool = True,
) -> dict[str, float]:
    """Held-out predictor diagnostics on the current adapted model."""
    adapter_p = adapter_params(params)
    pred_p = predictor_params(params)
    sites = _intercept_sites(adapter, bank.layers)
    next_blocks = tuple(sorted({i + 1 for i in bank.layers}))

    def step(p_adapter: AdapterParams, p_pred: Mapping[str, Any], ids: jax.Array):
        res = substrate.run_with_interception(
            ids, modify_fn=adapter.modify_fn(p_adapter), intercept_layers=sites
        )
        base_res = substrate.run_with_interception(
            ids, modify_fn=identity_modify, intercept_layers=next_blocks
        )
        out: dict[str, jax.Array] = {}
        for i in bank.layers:
            u = adapter.apply(p_adapter, res.hidden_state(i), i)
            guess = bank.predict(p_pred, i, u)
            teacher = res.hidden_state(i + 1)
            base_next = base_res.hidden_state(i + 1)
            on_teacher = hidden_prediction_loss(guess, teacher, u, config)
            on_base = hidden_prediction_loss(guess, base_next, u, config)
            shift = hidden_prediction_loss(teacher, base_next, teacher, config)
            out[f"pred_rel_l{i}"] = on_teacher["rel"]
            out[f"copy_rel_l{i}"] = on_teacher["copy_rel"]
            out[f"pred_rel_base_l{i}"] = on_base["rel"]
            out[f"teacher_shift_l{i}"] = shift["rel"]
        return out

    fn = jax.jit(step) if jit else step
    rows = [
        {k: float(v) for k, v in fn(adapter_p, pred_p, ids).items()}
        for ids in eval_batches
    ]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    for name in ("pred_rel", "copy_rel"):
        summary[f"{name}_mean"] = float(
            np.mean([summary[f"{name}_l{i}"] for i in bank.layers])
        )
    return summary


# Ablation arms

ARM_DEFAULT_WEIGHTS = {"cpc": 0.05, "stab": 0.1, "hid": 0.1}

ARMS = (
    "a1_mode",  # no auxiliary term: the A1 objective
    "stab_anch",  # anchor only
    "stab_hep",  # anchor + trainable low-rank HEP highway
    "stab_smooth",  # anchor + residual smoothness
    "stab_all",  # all four: anchor, HEP nudge, residual smoothness, step(step inactive in A2)
    "cpc_base",  # contrastive future, positive from the base state
    "cpc_corrected",  # contrastive future, positive from the corrected state
    "cpc_multi",  # contrastive future, offsets 1, 2, 4
    "hid_detached",  # hidden prediction, predictor input detached: trains P_l only (control)
    "hid_coupled",  # hidden prediction, predictor error reaches the adapter
    "full",  # cpc_base + stab_anch + hid_coupled
)


def _weight(section_weight: float, term: str) -> float:
    return section_weight if section_weight > 0 else ARM_DEFAULT_WEIGHTS[term]


def apply_arm(config: A2Config, arm: str) -> A2Config:
    """``config`` with the terms of ablation arm ``arm`` on and all others off."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose one of {list(ARMS)}")
    off_cpc = replace(config.cpc, weight=0.0)
    off_stab = replace(
        config.stab, weight=0.0, w_hep=0.0, w_smooth=0.0, w_step=0.0, hep=HEPConfig()
    )

    def cpc_on(**over: Any) -> CPCConfig:
        return replace(config.cpc, weight=_weight(config.cpc.weight, "cpc"), **over)

    def anchor_on() -> StabLossConfig:
        hep = config.stab.hep if not config.stab.hep.trainable else HEPConfig()
        return replace(
            config.stab,
            weight=_weight(config.stab.weight, "stab"),
            w_anch=1.0,
            w_hep=0.0,
            w_smooth=0.0,
            w_step=0.0,
            hep=hep,
        )

    def hid_on(detach_input: bool) -> HiddenConfig:
        return replace(
            config.hid,
            weight=_weight(config.hid.weight, "hid"),
            predictor=replace(config.hid.predictor, detach_input=detach_input),
        )

    cpc, stab, hid = off_cpc, off_stab, replace(config.hid, weight=0.0)
    if arm == "stab_anch":
        stab = anchor_on()
    elif arm == "stab_hep":
        base = anchor_on()
        kind = config.stab.hep.kind if config.stab.hep.trainable else "lowrank"
        w_hep = config.stab.w_hep if config.stab.w_hep > 0 else 0.1
        stab = replace(base, w_hep=w_hep, hep=replace(config.stab.hep, kind=kind))
    elif arm == "stab_smooth":
        stab = replace(
            anchor_on(),
            w_smooth=config.stab.w_smooth if config.stab.w_smooth > 0 else 0.1,
        )
    elif arm == "stab_all":
        kind = config.stab.hep.kind if config.stab.hep.trainable else "lowrank"
        stab = replace(
            anchor_on(),
            w_hep=config.stab.w_hep if config.stab.w_hep > 0 else 0.1,
            w_smooth=config.stab.w_smooth if config.stab.w_smooth > 0 else 0.1,
            w_step=config.stab.w_step if config.stab.w_step > 0 else 1.0,
            hep=replace(config.stab.hep, kind=kind),
        )
    elif arm == "cpc_base":
        cpc = cpc_on(positive="base")
    elif arm == "cpc_corrected":
        cpc = cpc_on(positive="corrected")
    elif arm == "cpc_multi":
        cpc = cpc_on(positive="base", offsets=(1, 2, 4))
    elif arm == "hid_detached":
        hid = hid_on(detach_input=True)
    elif arm == "hid_coupled":
        hid = hid_on(detach_input=False)
    elif arm == "full":
        cpc, stab, hid = (
            cpc_on(positive="base"),
            anchor_on(),
            hid_on(detach_input=False),
        )
    return replace(config, cpc=cpc, stab=stab, hid=hid)


#  Audit: checkpoints you can inspect


@dataclass(frozen=True)
class Check:
    """One audit checkpoint. ``ok`` is ``None`` for information that has no pass/fail."""

    name: str
    ok: bool | None
    value: Any
    detail: str = ""


def _f(x: Any) -> float:
    return float(x)


def _group_norms(tree: Mapping[str, Mapping[str, jax.Array]]) -> dict[str, float]:
    """L2 norm of ``tree`` per parameter group (``adapter`` / ``cpc`` / ``hep`` / ``pred``)."""
    out = {}
    for name, sub in (
        ("adapter", adapter_params(tree)),
        ("cpc", split_cpc(tree)),
        ("hep", split_hep(tree)),
        ("pred", predictor_params(tree)),
    ):
        sq = sum(
            (jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(sub)),
            start=jnp.zeros((), jnp.float32),
        )
        out[name] = _f(jnp.sqrt(sq))
    return out


def a2_audit(
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    config: A2Config,
    input_ids: jax.Array,
    *,
    probe_scale: float = 0.05,
) -> list[Check]:
    """Run the A2 checkpoints once on ``input_ids`` and return them."""
    checks: list[Check] = []

    def add(name: str, ok: bool | None, value: Any, detail: str = "") -> None:
        checks.append(Check(name, ok, value, detail))

    params0 = init_a2_params(config, adapter, seq_len=int(input_ids.shape[1]))
    key = jax.random.PRNGKey(config.seed + 1)
    probe = {k: dict(v) for k, v in params0.items()}
    for i, k in enumerate(adapter_params(params0)):
        shape = probe[k]["B"].shape
        probe[k]["B"] = probe_scale * jax.random.normal(
            jax.random.fold_in(key, i), shape, jnp.float32
        )

    def run(p: Any, cfg: A2Config = config, **kw: Any):
        return a2_objective(
            p,
            substrate=substrate,
            adapter=adapter,
            config=cfg,
            input_ids=input_ids,
            **kw,
        )

    counts = param_counts(params0)
    want_cpc = config.cpc.enabled
    want_hep = config.stab.enabled and config.stab.hep.trainable
    want_pred = config.uses_hidden_pred
    add(
        "param_groups",
        (counts["cpc"] > 0) == want_cpc
        and (counts["hep"] > 0) == want_hep
        and (counts["pred"] > 0) == want_pred,
        counts,
        f"expected cpc heads: {want_cpc}, trainable highway: {want_hep}, "
        f"hidden predictors: {want_pred}",
    )

    add("frozen_before", bool(substrate.params_unchanged()), None)

    # 3. At init the adapted model equals the base model.
    base = base_logits(substrate, input_ids)
    adapted0 = substrate.run_with_interception(
        input_ids,
        modify_fn=adapter.modify_fn(adapter_params(params0)),
        intercept_layers=adapter.layers,
    ).logits
    diff = _f(jnp.max(jnp.abs(adapted0 - base)))
    add("init_equals_base", diff < 1e-4, diff, "max |adapted logits - base logits|")

    for label, p in (("init", params0), ("probe", probe)):
        _, terms = run(p)
        bad = sorted(k for k, v in terms.items() if not bool(jnp.isfinite(v)))
        add(f"finite_{label}", not bad, {k: _f(v) for k, v in terms.items()}, str(bad))

    total, t = run(probe)
    expected = (
        t["task"]
        + config.lambda_kl * t.get("kl", 0.0)
        + config.lambda_wd * t["wd"]
        + config.cpc.weight * t.get("cpc", 0.0)
        + config.stab.weight * t.get("stab", 0.0)
        + config.lambda_hid * t.get("hid", 0.0)
    )
    err = abs(_f(total) - _f(expected)) / max(abs(_f(total)), 1e-12)
    add("total_decomposition", err < 1e-4, err, "relative error of total vs the sum")

    off = replace(
        config,
        cpc=replace(config.cpc, weight=0.0),
        stab=replace(config.stab, weight=0.0),
        hid=replace(config.hid, weight=0.0),
    )
    a1_cfg = A1Config(lambda_kl=config.lambda_kl, lambda_wd=config.lambda_wd)
    ap = adapter_params(probe)

    def a1_total(p: Any) -> Any:
        return a1_objective(
            p,
            substrate=substrate,
            adapter=adapter,
            config=a1_cfg,
            input_ids=input_ids,
        )[0]

    t_a1, t_a2 = a1_total(ap), run(probe, off)[0]
    g_a1 = jax.grad(a1_total)(ap)
    g_a2 = adapter_params(jax.grad(lambda p: run(p, off)[0])(probe))
    gdiff = max(
        _f(jnp.max(jnp.abs(x - y)))
        for x, y in zip(
            jax.tree_util.tree_leaves(g_a1),
            jax.tree_util.tree_leaves(g_a2),
            strict=True,
        )
    )
    add(
        "a1_parity",
        abs(_f(t_a1) - _f(t_a2)) < 1e-5 * max(1.0, abs(_f(t_a1))) and gdiff < 1e-6,
        {"total_diff": abs(_f(t_a1) - _f(t_a2)), "grad_max_diff": gdiff},
        "aux weights 0: A2 total and adapter gradient vs a1_objective",
    )

    if config.stability.enabled:
        quiet = replace(config, monitor=replace(config.monitor, mode="off"))
        t_on, t_off = run(probe)[0], run(probe, quiet)[0]
        g_on = jax.grad(lambda p: run(p)[0])(probe)
        g_off = jax.grad(lambda p: run(p, quiet)[0])(probe)
        mdiff = max(
            _f(jnp.max(jnp.abs(x - y)))
            for x, y in zip(
                jax.tree_util.tree_leaves(g_on),
                jax.tree_util.tree_leaves(g_off),
                strict=True,
            )
        )
        add(
            "monitor_measurement_only",
            abs(_f(t_on) - _f(t_off)) < 1e-6 * max(1.0, abs(_f(t_off)))
            and mdiff < 1e-6,
            {"total_diff": abs(_f(t_on) - _f(t_off)), "grad_max_diff": mdiff},
            "monitor on vs off: total and full gradient",
        )

    _, probe_terms = run(probe)
    routed = {"task": ("task", {"adapter"})}
    if config.cpc.enabled:
        routed["cpc"] = ("cpc", {"adapter", "cpc"})
    if config.uses_hidden_pred:
        # detach_input: only P_l learns (negative control); otherwise the adapter does too.
        routed["hid"] = ("hid", {"adapter", "pred"} if config.hid_coupled else {"pred"})
    if "stab_anch" in probe_terms:
        routed["stab_anch"] = ("stab_anch", {"adapter"})
    if "stab_smooth" in probe_terms:
        routed["stab_smooth"] = ("stab_smooth", {"adapter"})
    if (
        "stab_step" in probe_terms
    ):  # no settling in A2: one state, no steps, no gradient
        routed["stab_step"] = ("stab_step", set())
    if "stab_hep" in probe_terms:
        routed["stab_hep"] = (
            "stab_hep",
            {"hep"} if config.stab.hep.trainable else set(),
        )
    norms: dict[str, dict[str, float]] = {}
    for label, (term, expected_groups) in routed.items():
        grad = jax.grad(lambda p, term=term: run(p)[1][term])(probe)
        norms[label] = _group_norms(grad)
        got = {g for g, v in norms[label].items() if v > 0.0}
        add(
            f"grad_routing_{label}",
            got == expected_groups,
            norms[label],
            f"gradient reaches {sorted(got)}, expected {sorted(expected_groups)}",
        )

    task_norm = norms["task"]["adapter"]
    weights = {
        "cpc": config.cpc.weight,
        "stab_anch": config.stab.weight * config.stab.w_anch,
        "stab_smooth": config.stab.weight * config.stab.w_smooth,
        "hid": config.lambda_hid if config.hid_coupled else 0.0,
    }
    for label, weight in weights.items():
        if label in norms and task_norm > 0 and weight > 0:
            ratio = weight * norms[label]["adapter"] / task_norm
            add(
                f"grad_ratio_{label}",
                None,
                ratio,
                "lambda * |grad_adapter(term)| / |grad_adapter(task)|; target ~0.1-1",
            )

    if config.uses_hidden_pred:
        add(
            "hid_vs_copy",
            None,
            {
                "hid_rel": _f(probe_terms["hid_rel"]),
                "copy_rel": _f(probe_terms["hid_copy_rel"]),
            },
            "target-variance-normalised error of P_l(h~) vs the copy baseline h~ (untrained P_l)",
        )

    eager = _f(run(probe)[0])
    jitted = _f(jax.jit(lambda p: run(p)[0])(probe))
    add(
        "jit_equals_eager",
        abs(eager - jitted) <= 1e-4 * max(1.0, abs(eager)),
        {"eager": eager, "jit": jitted},
    )

    if config.needs_corrected:
        half = input_ids.shape[1] // 2
        labels = input_ids.at[:, half:].set(-100)
        _, masked = run(probe, labels=labels)
        bad = sorted(k for k, v in masked.items() if not bool(jnp.isfinite(v)))
        add("finite_with_ignored_labels", not bad, None, str(bad))

    add("frozen_after", bool(substrate.params_unchanged()), None)
    return checks


def audit_ok(checks: list[Check]) -> bool:
    """``True`` when no checkpoint failed (information-only checks never fail)."""
    return all(c.ok is not False for c in checks)


def audit_to_dict(checks: list[Check]) -> dict[str, Any]:
    """JSON-serialisable form of an audit, for ``audit.json``."""
    return {
        "ok": audit_ok(checks),
        "checks": [
            {"name": c.name, "ok": c.ok, "value": c.value, "detail": c.detail}
            for c in checks
        ],
    }


def format_audit(checks: list[Check]) -> str:
    """A readable table of the audit: ``PASS`` / ``FAIL`` / ``info`` per checkpoint."""
    lines = []
    for c in checks:
        status = "info" if c.ok is None else ("PASS" if c.ok else "FAIL")
        value = ""
        if isinstance(c.value, float):
            value = f"{c.value:.4g}"
        elif isinstance(c.value, dict):
            value = " ".join(
                f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in c.value.items()
                if not isinstance(v, dict)
            )
        lines.append(f"[{status:4s}] {c.name:28s} {value}")
        if c.ok is False and c.detail:
            lines.append(f"         -> {c.detail}")
    return "\n".join(lines)


__all__ = [
    "ARMS",
    "ARM_DEFAULT_WEIGHTS",
    "A2Config",
    "ADAPTER_PREFIX",
    "CPCConfig",
    "Check",
    "HEPConfig",
    "HIDDEN_NORMS",
    "HIDDEN_TARGETS",
    "POSITIVES",
    "StabLossConfig",
    "a2_audit",
    "a2_diagnostics",
    "a2_objective",
    "adapter_params",
    "apply_arm",
    "audit_ok",
    "audit_to_dict",
    "collect_stability_metrics",
    "compose_a2_loss",
    "cpc_term",
    "format_audit",
    "hid_term",
    "hidden_prediction_loss",
    "residual_smoothness_loss",
    "step_loss",
    "init_a2_params",
    "init_cpc_params",
    "init_hep_params",
    "make_a2_objective",
    "make_a2_optimizer",
    "param_counts",
    "predictor_bank",
    "predictor_params",
    "stability_term",
    "token_mask",
]
