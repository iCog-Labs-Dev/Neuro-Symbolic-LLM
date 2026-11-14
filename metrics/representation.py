"""Representation metrics and A3 settling penalties.

A2 reporting uses residual size, compressibility, redundancy and gradient conflict.
The settling and HEP helpers remain differentiable for future A3 objectives;
relocating them here does not add them to A2 training. Settling references are
detached and fixed, with reductions over valid tokens, features, steps and sites.
Low-rank adapter metrics operate on ``A`` and ``B`` without constructing ``A @ B``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from residual.hep import Highway

_EPS = 1e-8


def loss_improvement(
    base_loss: jax.Array | float, loss: jax.Array | float
) -> jax.Array:
    """Return ``L_base - L_adapted``; positive when the residual helps."""
    return jnp.asarray(base_loss, jnp.float32) - jnp.asarray(loss, jnp.float32)


def residual_compressibility(
    delta_small: jax.Array | float,
    delta_large: jax.Array | float,
    eps: float = 1e-8,
) -> jax.Array:
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps!r}")
    return jnp.asarray(delta_small, jnp.float32) / (
        jnp.asarray(delta_large, jnp.float32) + eps
    )


def compressibility_curve(
    deltas: Mapping[int, jax.Array | float], eps: float = 1e-8
) -> dict[int, jax.Array]:
    if not deltas:
        raise ValueError("deltas is empty; expected at least one rank.")
    ranks = sorted(deltas)
    if ranks[0] < 1:
        raise ValueError(f"Ranks must be >= 1, got {ranks[0]}.")
    reference = deltas[ranks[-1]]
    return {r: residual_compressibility(deltas[r], reference, eps) for r in ranks}


MIN_REFERENCE_IMPROVEMENT = 0.02


def guarded_compressibility(
    deltas: Mapping[int, float],
    min_reference: float = MIN_REFERENCE_IMPROVEMENT,
) -> dict:
    if min_reference <= 0:
        raise ValueError(f"min_reference must be > 0, got {min_reference!r}")
    clean = {
        int(r): float(v)
        for r, v in deltas.items()
        if v is not None and float(v) == float(v) and abs(float(v)) != float("inf")
    }
    if any(r < 1 for r in clean):
        raise ValueError("Ranks must be >= 1.")
    out: dict = {
        "status": "insufficient_ranks",
        "reference_rank": None,
        "reference_improvement": None,
        "min_reference": min_reference,
        "negative_ranks": sorted(r for r, v in clean.items() if v < 0),
        "curve": None,
    }
    if len(clean) < 2:
        return out
    ref_rank = max(clean)
    ref = clean[ref_rank]
    out["reference_rank"] = ref_rank
    out["reference_improvement"] = ref
    if ref < min_reference:
        out["status"] = "unreliable_reference"
        return out
    out["status"] = "ok"
    out["curve"] = {r: clean[r] / ref for r in sorted(clean)}
    return out


def residual_norm_ratio(
    h: jax.Array,
    h_corrected: jax.Array,
    mask: jax.Array | None = None,
    eps: float = 1e-8,
) -> jax.Array:
    """Mean over tokens of ``||h~ - h|| / ||h||``: how large the residual ``R`` is."""
    if h.shape != h_corrected.shape:
        raise ValueError(f"shape mismatch: {h.shape} vs {h_corrected.shape}")
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps!r}")
    h32 = h.astype(jnp.float32)
    r_norm = jnp.linalg.norm(h_corrected.astype(jnp.float32) - h32, axis=-1)
    ratio = r_norm / (jnp.linalg.norm(h32, axis=-1) + eps)
    if mask is None:
        return jnp.mean(ratio)
    if mask.shape != ratio.shape:
        raise ValueError(f"mask shape {mask.shape} must be {ratio.shape}")
    w = mask.astype(jnp.float32)
    return jnp.sum(ratio * w) / jnp.maximum(jnp.sum(w), 1.0)


def direction_redundancy(vectors: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Mean ``|cos|`` over the pairs of distinct rows of ``vectors`` ``[n, d]``."""
    if vectors.ndim != 2:
        raise ValueError(f"expected a 2-D [n, d] matrix, got shape {vectors.shape}")
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps!r}")
    n = vectors.shape[0]
    if n < 2:
        return jnp.zeros((), jnp.float32)
    v = vectors.astype(jnp.float32)
    unit = v / jnp.maximum(jnp.linalg.norm(v, axis=-1, keepdims=True), eps)
    cos = jnp.abs(unit @ unit.T)
    off_diagonal = cos.sum() - jnp.trace(cos)
    return off_diagonal / (n * (n - 1))


def residual_redundancy(a: jax.Array, b: jax.Array) -> dict[str, jax.Array]:
    """Redundancy of one adapter block ``h~ = h + s * sigma(h A) B``."""
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape[::-1]:
        raise ValueError(f"expected A [d, r] and B [r, d], got {a.shape} and {b.shape}")
    return {"read": direction_redundancy(a.T), "write": direction_redundancy(b)}


def effective_rank(matrix: jax.Array, eps: float = _EPS) -> jax.Array:
    """Entropy effective rank, invariant to low-rank basis changes."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-D")
    singular = jnp.linalg.svd(matrix.astype(jnp.float32), compute_uv=False)
    weights = singular / jnp.maximum(jnp.sum(singular), eps)
    return jnp.exp(-jnp.sum(jnp.where(weights > 0, weights * jnp.log(weights), 0.0)))


def canonical_low_rank_svd(
    a: jax.Array, b: jax.Array, eps: float = _EPS
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return a sign-canonical thin SVD of ``A @ B`` without forming it.

    For ``A[d, r]`` and ``B[r, d]``, reduced QR decompositions turn the problem
    into the SVD of an ``r x r`` core. The returned factors satisfy
    ``A @ B == U @ diag(s) @ Vh`` up to numerical precision. Singular-vector
    signs are fixed from the largest-magnitude entry of each left vector, making
    the result invariant to invertible adapter gauge changes when singular values
    are distinct. Null factors are represented by zero singular values.
    """
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps!r}")
    a_arr = jnp.asarray(a)
    b_arr = jnp.asarray(b)
    if (
        a_arr.ndim != 2
        or b_arr.ndim != 2
        or a_arr.shape[1] != b_arr.shape[0]
        or a_arr.shape[0] != b_arr.shape[1]
    ):
        raise ValueError(
            f"expected A [d, r] and B [r, d], got {a_arr.shape} and {b_arr.shape}"
        )
    d, rank = a_arr.shape
    if rank > d:
        raise ValueError(
            f"expected a low-rank adapter with r <= d, got d={d}, r={rank}"
        )

    dtype = jnp.result_type(a_arr.dtype, b_arr.dtype, jnp.float32)
    q_a, r_a = jnp.linalg.qr(a_arr.astype(dtype), mode="reduced")
    q_b, r_b = jnp.linalg.qr(b_arr.T.astype(dtype), mode="reduced")
    u_core, singular, vh_core = jnp.linalg.svd(r_a @ r_b.T, full_matrices=False)
    u = q_a @ u_core
    vh = vh_core @ q_b.T

    cutoff = eps * jnp.max(singular)
    singular = jnp.where(singular > cutoff, singular, 0.0)

    # SVD vectors are rotationally ambiguous inside a repeated-singular-value
    # subspace. Diagonalizing a fixed coordinate anchor restricted to each such
    # subspace chooses the same basis for every adapter gauge. All matrices here
    # are at most d-by-r or r-by-r.
    relative_tolerance = max(eps, 8 * jnp.finfo(dtype).eps)
    scale = jnp.maximum(singular[0], jnp.finfo(dtype).tiny)
    boundaries = jnp.abs(jnp.diff(singular)) > relative_tolerance * scale
    group_ids = jnp.concatenate(
        (jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(boundaries))
    )
    same_group = group_ids[:, None] == group_ids[None, :]
    coordinate_anchor = jnp.arange(d, dtype=dtype) / max(d - 1, 1)
    restricted_anchor = u.T @ (coordinate_anchor[:, None] * u)
    anchored_blocks = jnp.where(same_group, restricted_anchor, 0.0)
    anchored_blocks += jnp.diag(2.0 * group_ids.astype(dtype))
    _, rotation = jnp.linalg.eigh(anchored_blocks)
    u = u @ rotation
    vh = rotation.T @ vh

    membership = jax.nn.one_hot(group_ids, rank, dtype=dtype)
    group_counts = jnp.sum(membership, axis=0)
    group_sums = membership.T @ singular
    group_values = group_sums / jnp.maximum(group_counts, 1.0)
    singular = group_values[group_ids]

    pivots = jnp.argmax(jnp.abs(u), axis=0)
    pivot_values = u[pivots, jnp.arange(rank)]
    signs = jnp.where(pivot_values < 0, -1.0, 1.0).astype(dtype)
    u = u * signs
    vh = vh * signs[:, None]
    return u, singular, vh


def canonical_low_rank_factors(
    a: jax.Array, b: jax.Array, eps: float = _EPS
) -> tuple[jax.Array, jax.Array]:
    """Return balanced canonical factors whose product is ``A @ B``.

    The factors are ``U sqrt(S)`` and ``sqrt(S) Vh`` from
    :func:`canonical_low_rank_svd`, and retain the input ``[d, r]``/``[r, d]``
    shapes.
    """
    u, singular, vh = canonical_low_rank_svd(a, b, eps)
    root = jnp.sqrt(jnp.maximum(singular, 0.0))
    return u * root, root[:, None] * vh


def canonicalize_adapter_params(
    params: Mapping[str, Any],
    eps: float = _EPS,
    *,
    activation: str = "identity",
) -> dict[str, Any]:
    """Return a copy with each adapter ``A``/``B`` block canonicalized.

    Canonicalization preserves an adapter's function only for an identity
    bottleneck activation. Non-adapter entries are retained unchanged, allowing
    this helper to operate on combined training PyTrees.
    """
    if not isinstance(params, Mapping):
        raise ValueError("params must be a mapping")
    if activation != "identity":
        raise ValueError("adapter gauge canonicalization requires identity activation")

    def canonicalize_block(block: Mapping[str, Any]) -> dict[str, Any]:
        copied = dict(block)
        if "A" in block and "B" in block:
            copied["A"], copied["B"] = canonical_low_rank_factors(
                block["A"], block["B"], eps
            )
        return copied

    if "A" in params and "B" in params:
        return canonicalize_block(params)
    return {
        name: canonicalize_block(value) if isinstance(value, Mapping) else value
        for name, value in params.items()
    }


# The explicit alias makes the decomposition API easy to discover while keeping
# ``canonical_low_rank_factors`` reserved for balanced adapter parameters.
canonical_svd_factors = canonical_low_rank_svd


def invariant_redundancy(a: jax.Array, b: jax.Array, eps: float = _EPS) -> jax.Array:
    """Function-level redundancy based on the effective rank of ``A @ B``."""
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape[::-1]:
        raise ValueError("expected A [d,r] and B [r,d]")
    _, singular, _ = canonical_low_rank_svd(a, b, eps)
    weights = singular / jnp.maximum(jnp.sum(singular), eps)
    rank = jnp.exp(-jnp.sum(jnp.where(weights > 0, weights * jnp.log(weights), 0.0)))
    return 1.0 - rank / max(1, min(a.shape))


def gradient_conflict(grad_a: Any, grad_b: Any, eps: float = 1e-12) -> jax.Array:
    """``[-cos(g_a, g_b)]_+`` of two gradient PyTrees with the same structure."""
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps!r}")
    leaves_a = jax.tree_util.tree_leaves(grad_a)
    leaves_b = jax.tree_util.tree_leaves(grad_b)
    if jax.tree_util.tree_structure(grad_a) != jax.tree_util.tree_structure(grad_b):
        raise ValueError("the two gradients must have the same structure")
    if not leaves_a:
        raise ValueError("gradients are empty")
    for x, y in zip(leaves_a, leaves_b, strict=True):
        if x.shape != y.shape:
            raise ValueError(f"leaf shape mismatch: {x.shape} vs {y.shape}")
    dot = sum(
        jnp.sum(x.astype(jnp.float32) * y.astype(jnp.float32))
        for x, y in zip(leaves_a, leaves_b, strict=True)
    )
    na = jnp.sqrt(sum(jnp.sum(jnp.square(x.astype(jnp.float32))) for x in leaves_a))
    nb = jnp.sqrt(sum(jnp.sum(jnp.square(y.astype(jnp.float32))) for y in leaves_b))
    cos = dot / jnp.maximum(na * nb, eps)
    return jnp.maximum(-cos, 0.0)


def _flatten_tree(tree: Any, name: str) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError(f"{name} is empty")
    return jnp.concatenate(
        [jnp.ravel(jnp.asarray(leaf, dtype=jnp.float32)) for leaf in leaves]
    )


def gradient_conflict_summary(
    gradients: Mapping[Any, Any] | Sequence[Any] | jax.Array,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Summarize pairwise gradient conflict over tasks.

    ``gradients`` may be a mapping from independently named tasks to gradient
    PyTrees, a sequence of PyTrees, or an array stacked on its first axis. Pairs
    containing a zero-norm gradient are excluded rather than scored as agreement.
    """
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps!r}")
    if isinstance(gradients, Mapping):
        labels = tuple(gradients)
        trees = tuple(gradients.values())
    elif hasattr(gradients, "shape"):
        array = jnp.asarray(gradients)
        if array.ndim < 1:
            raise ValueError("gradients must have a task axis")
        labels = tuple(range(array.shape[0]))
        trees = tuple(array[index] for index in range(array.shape[0]))
    elif isinstance(gradients, Sequence) and not isinstance(gradients, str | bytes):
        labels = tuple(range(len(gradients)))
        trees = tuple(gradients)
    else:
        raise ValueError("gradients must be a mapping, sequence, or stacked array")

    n_tasks = len(trees)
    if n_tasks == 0:
        raise ValueError("gradients are empty")
    structure = jax.tree_util.tree_structure(trees[0])
    if any(jax.tree_util.tree_structure(tree) != structure for tree in trees[1:]):
        raise ValueError("all task gradients must have the same PyTree structure")
    vectors = [_flatten_tree(tree, "gradient") for tree in trees]
    if any(vector.shape != vectors[0].shape for vector in vectors[1:]):
        raise ValueError("all task gradients must have equal leaf shapes")
    stacked = jnp.stack(vectors)
    norms = jnp.linalg.norm(stacked, axis=1)
    valid_tasks = norms > eps
    denominators = norms[:, None] * norms[None, :]
    valid_matrix = valid_tasks[:, None] & valid_tasks[None, :]
    cosine = jnp.where(
        valid_matrix,
        (stacked @ stacked.T) / jnp.maximum(denominators, eps),
        jnp.nan,
    )
    cosine = jnp.clip(cosine, -1.0, 1.0)
    conflict = jnp.maximum(-cosine, 0.0)
    diagonal = jnp.eye(n_tasks, dtype=bool)
    conflict = jnp.where(diagonal & valid_matrix, 0.0, conflict)

    pair_mask = jnp.triu(jnp.ones((n_tasks, n_tasks), dtype=bool), k=1)
    pair_mask &= valid_matrix
    pair_count = jnp.sum(pair_mask)
    conflict_sum = jnp.sum(jnp.where(pair_mask, conflict, 0.0))
    mean_conflict = jnp.where(pair_count > 0, conflict_sum / pair_count, jnp.nan)
    max_conflict = jnp.where(
        pair_count > 0,
        jnp.max(jnp.where(pair_mask, conflict, -jnp.inf)),
        jnp.nan,
    )
    fraction_conflicting = jnp.where(
        pair_count > 0,
        jnp.sum(pair_mask & (cosine < 0)) / pair_count,
        jnp.nan,
    )
    pair_count_host = int(jax.device_get(pair_count))
    return {
        "status": "ok" if pair_count_host else "unavailable",
        "reason": None if pair_count_host else "fewer than two nonzero gradients",
        "labels": labels,
        "n_tasks": n_tasks,
        "pair_count": pair_count,
        "valid_tasks": valid_tasks,
        "cosine": cosine,
        "conflict": conflict,
        "mean": mean_conflict,
        "mean_conflict": mean_conflict,
        "max": max_conflict,
        "max_conflict": max_conflict,
        "fraction_conflicting": fraction_conflicting,
    }


summarize_gradient_conflicts = gradient_conflict_summary


def parameter_commutator(
    params_i_then_j: Any,
    params_j_then_i: Any,
    reference_params: Any | None = None,
    eps: float = _EPS,
) -> jax.Array:
    """Normalized finite-order parameter commutator for two update orders."""
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps!r}")
    structure = jax.tree_util.tree_structure(params_i_then_j)
    if jax.tree_util.tree_structure(params_j_then_i) != structure:
        raise ValueError("ordered parameter PyTrees must have the same structure")
    left = _flatten_tree(params_i_then_j, "params_i_then_j")
    right = _flatten_tree(params_j_then_i, "params_j_then_i")
    if left.shape != right.shape:
        raise ValueError("ordered parameter PyTrees must have equal leaf shapes")
    numerator = jnp.linalg.norm(left - right)
    if reference_params is None:
        denominator = 0.5 * (jnp.linalg.norm(left) + jnp.linalg.norm(right))
    else:
        if jax.tree_util.tree_structure(reference_params) != structure:
            raise ValueError("reference_params must match the ordered PyTree structure")
        reference = _flatten_tree(reference_params, "reference_params")
        if reference.shape != left.shape:
            raise ValueError("reference_params must have equal leaf shapes")
        denominator = jnp.linalg.norm(reference)
    return numerator / (denominator + eps)


def operator_commutator_matrix(
    operator_i: jax.Array, operator_j: jax.Array
) -> jax.Array:
    """Return the algebraic commutator ``operator_i operator_j - operator_j operator_i``."""
    left = jnp.asarray(operator_i, dtype=jnp.float32)
    right = jnp.asarray(operator_j, dtype=jnp.float32)
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape != right.shape
        or left.shape[0] != left.shape[1]
    ):
        raise ValueError("operators must be square matrices with equal shapes")
    return left @ right - right @ left


def operator_commutator(
    operator_i: jax.Array,
    operator_j: jax.Array,
    eps: float = _EPS,
) -> jax.Array:
    """Frobenius-normalized operator commutator magnitude."""
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps!r}")
    left = jnp.asarray(operator_i, dtype=jnp.float32)
    right = jnp.asarray(operator_j, dtype=jnp.float32)
    commutator = operator_commutator_matrix(left, right)
    scale = jnp.linalg.norm(left) * jnp.linalg.norm(right)
    return jnp.linalg.norm(commutator) / (scale + eps)


def _masked_mean(x, mask=None):
    if x.size == 0:
        return jnp.sum(x)
    w = (
        jnp.ones_like(x, dtype=bool)
        if mask is None
        else jnp.broadcast_to(mask.astype(bool), x.shape)
    )
    return jnp.sum(jnp.where(w, x, 0.0)) / jnp.maximum(jnp.sum(w), 1.0)


def v_hidden_size(v):
    return v.shape[-1] if hasattr(v, "shape") else v[1].shape[-1]


def settling_loss(
    trajectories: Mapping[int, jax.Array],
    feedforward: Mapping[int, jax.Array],
    *,
    mask=None,
    anchor_weight=1.0,
    step_weight=0.0,
    nudges: Mapping[int, jax.Array] | None = None,
    nudge_weight=0.0,
):
    """Trajectories are [K+1,B,T,D]; nudges are [K,B,T,D].

    ``feedforward[layer]`` is [B,T,D], captured before inference, never
    recomputed from a settled state. Padding, D, K and site counts are averaged.
    Nudge penalties alone do not define how inference applies a nudge.
    """
    if not trajectories or set(trajectories) != set(feedforward):
        raise ValueError(
            "trajectories and feedforward must have the same nonempty sites"
        )
    if any(
        not math.isfinite(w) or w < 0
        for w in (anchor_weight, step_weight, nudge_weight)
    ):
        raise ValueError("settling weights must be finite and nonnegative")
    parts = []
    for layer, trajectory in trajectories.items():
        z = trajectory.astype(jnp.float32)
        reference = jax.lax.stop_gradient(feedforward[layer].astype(jnp.float32))
        if z.ndim != 4 or z.shape[0] < 1 or z.shape[1:] != reference.shape:
            raise ValueError("expected [K+1,B,T,D] and a fixed [B,T,D] reference")
        if mask is not None and mask.shape != reference.shape[:-1]:
            raise ValueError("mask must be [B,T]")
        anchor = _masked_mean(jnp.mean((z[1:] - reference) ** 2, axis=-1), mask)
        step = _masked_mean(jnp.mean((z[1:] - z[:-1]) ** 2, axis=-1), mask)
        nudge = jnp.zeros((), jnp.float32)
        if nudges is not None:
            value = nudges[layer]
            if value.shape != z[1:].shape:
                raise ValueError("nudges must have shape [K,B,T,D]")
            nudge = _masked_mean(jnp.mean(value**2, axis=-1), mask)
        elif nudge_weight > 0:
            raise ValueError("nudge_weight > 0 requires nudges")
        parts.append(jnp.stack((anchor, step, nudge)))
    anchor, step, nudge = jnp.mean(jnp.stack(parts), axis=0)
    total = anchor_weight * anchor + step_weight * step + nudge_weight * nudge
    return total, {
        "settling_anchor": anchor,
        "settling_step": step,
        "settling_nudge": nudge,
    }


def output_error(logits: jax.Array, labels: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Output error ``eps_y = softmax(logits) - onehot(next token)``."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError(
            f"expected logits [B, T, V] and labels [B, T], got {logits.shape}, {labels.shape}"
        )
    p = jax.nn.softmax(
        jax.lax.stop_gradient(logits[:, :-1]).astype(jnp.float32), axis=-1
    )
    target = labels[:, 1:]
    mask = target != -100
    onehot = jax.nn.one_hot(
        jnp.where(mask, target, 0), logits.shape[-1], dtype=jnp.float32
    )
    eps = (p - onehot) * mask[..., None]
    return eps, mask


def hep_nudge(eps: jax.Array, v: Highway, alpha: float = 1.0) -> jax.Array:
    """Projected output error, for use by an A3 inference implementation."""
    from residual.hep import apply_highway

    return alpha * apply_highway(v, jax.lax.stop_gradient(eps))


def hep_nudge_loss(
    eps: jax.Array,
    v: Highway,
    alpha: float = 1.0,
    *,
    reference: jax.Array | None = None,
    mask: jax.Array | None = None,
    backend: str = "fabricpc",
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Projected-error norm penalty for A3.

    Only trainable highway weights receive a gradient. A zero highway minimizes
    this penalty, so reducing it does not demonstrate useful HEP learning.

    L = mean_{...,b,t} || alpha * sg(eps) @ V ||^2
    L = that / ( mean || sg(reference) ||^2 + eps )
    """

    from residual.fpc_energy import check_backend, highway_sq_norm

    if eps.ndim < 3:
        raise ValueError(f"expected eps [..., B, T, d_y], got shape {eps.shape}")
    if not alpha >= 0:
        raise ValueError(f"alpha must be >= 0, got {alpha!r}")
    check_backend(backend)
    if reference is not None and reference.shape[:-1] != eps.shape[:-1]:
        raise ValueError(
            f"reference {reference.shape} must match the nudge positions {eps.shape[:-1]}"
        )
    if backend == "fabricpc":
        sq = highway_sq_norm(alpha * jax.lax.stop_gradient(eps), v)
    else:
        sq = jnp.sum(jnp.square(hep_nudge(eps, v, alpha)), axis=-1)
    mean_sq = _masked_mean(sq, mask) / v_hidden_size(v)
    metrics = {
        "hep_nudge_norm": _masked_mean(jnp.sqrt(sq + _EPS) - jnp.sqrt(_EPS), mask)
    }
    loss = mean_sq
    if reference is not None:
        ref = jax.lax.stop_gradient(reference.astype(jnp.float32))
        ref_sq = _masked_mean(jnp.mean(jnp.square(ref), axis=-1), mask)
        loss = mean_sq / (ref_sq + _EPS)
        metrics["hep_nudge_ratio"] = jnp.sqrt(mean_sq / (ref_sq + _EPS))
    return loss, metrics


__all__ = [
    "canonical_low_rank_factors",
    "canonical_low_rank_svd",
    "canonical_svd_factors",
    "canonicalize_adapter_params",
    "compressibility_curve",
    "direction_redundancy",
    "effective_rank",
    "gradient_conflict",
    "gradient_conflict_summary",
    "guarded_compressibility",
    "hep_nudge",
    "hep_nudge_loss",
    "invariant_redundancy",
    "loss_improvement",
    "MIN_REFERENCE_IMPROVEMENT",
    "operator_commutator",
    "operator_commutator_matrix",
    "output_error",
    "parameter_commutator",
    "residual_compressibility",
    "residual_norm_ratio",
    "residual_redundancy",
    "settling_loss",
    "summarize_gradient_conflicts",
    "v_hidden_size",
]
