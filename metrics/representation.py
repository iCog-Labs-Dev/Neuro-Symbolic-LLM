"""Representation metrics and A3 settling penalties.

A2 reporting uses residual size, compressibility, redundancy and gradient conflict.
The settling and HEP helpers remain differentiable for future A3 objectives;
relocating them here does not add them to A2 training. Settling references are
detached and fixed, with reductions over valid tokens, features, steps and sites.

The audit helpers in this module are pure numerical kernels. In particular,
low-rank adapter audits operate on ``A`` and ``B`` without constructing ``A @ B``.
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


def effective_support(contributions: jax.Array, eps: float = _EPS) -> jax.Array:
    """Entropy effective support size of non-negative component contributions."""
    values = jnp.asarray(contributions, dtype=jnp.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("contributions must be a non-empty vector")
    values = jnp.maximum(values, 0.0)
    probabilities = values / jnp.maximum(jnp.sum(values), eps)
    return jnp.exp(
        -jnp.sum(
            jnp.where(probabilities > 0, probabilities * jnp.log(probabilities), 0.0)
        )
    )


def fracture_index(contributions: jax.Array, eps: float = _EPS) -> jax.Array:
    """Normalized effective support; lower values are more localized."""
    values = jnp.asarray(contributions)
    return effective_support(values, eps) / values.size


def overlap(
    contributions_a: jax.Array, contributions_b: jax.Array, eps: float = _EPS
) -> jax.Array:
    """Overlap of two component-contribution distributions."""
    a = jnp.maximum(jnp.asarray(contributions_a, jnp.float32), 0)
    b = jnp.maximum(jnp.asarray(contributions_b, jnp.float32), 0)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError("contributions must be equal-length vectors")
    a = a / jnp.maximum(jnp.sum(a), eps)
    b = b / jnp.maximum(jnp.sum(b), eps)
    return jnp.sum(jnp.minimum(a, b))


def factor_importance(signed_effects: jax.Array) -> jax.Array:
    """Convert signed component-by-factor ablation effects to importance.

    An ablation that raises loss has positive causal importance. Effects that are
    zero or negative do not provide positive evidence that a component supports
    the factor, so ``I(u; q) = max(0, effect[u, q])``.
    """
    effects = jnp.asarray(signed_effects, dtype=jnp.float32)
    if effects.ndim not in (1, 2) or effects.size == 0:
        raise ValueError("signed_effects must be a non-empty vector or matrix")
    return jnp.maximum(effects, 0.0)


def factor_positive_mass(signed_effects: jax.Array) -> jax.Array:
    """Total positive component importance for each factor."""
    importance = factor_importance(signed_effects)
    return jnp.sum(importance, axis=0)


def semantic_factor_metrics(
    signed_effects: jax.Array,
) -> dict[str, Any]:
    """Fracture and overlap from signed effects ``[components, factors]``.

    Negative effects are clipped before normalization. A factor with zero
    positive mass is unavailable: its fracture and every overlap involving it
    are ``NaN`` and its per-factor status is ``"unavailable"``. This avoids the
    misleading ``1 / n_components`` fracture produced by normalizing an all-zero
    vector.
    """
    effects = jnp.asarray(signed_effects, dtype=jnp.float32)
    if effects.ndim != 2 or not all(effects.shape):
        raise ValueError(
            "signed_effects must be a non-empty [components, factors] matrix"
        )

    importance = factor_importance(effects)
    positive_mass = jnp.sum(importance, axis=0)
    available = positive_mass > 0
    probabilities = jnp.where(
        available[None, :],
        importance / jnp.where(available, positive_mass, 1.0)[None, :],
        0.0,
    )
    entropy = -jnp.sum(
        jnp.where(
            probabilities > 0,
            probabilities * jnp.log(probabilities),
            0.0,
        ),
        axis=0,
    )
    effective = jnp.exp(entropy)
    unavailable = jnp.asarray(jnp.nan, dtype=effects.dtype)
    fracture = jnp.where(available, effective / effects.shape[0], unavailable)

    pair_overlap = jnp.sum(
        jnp.minimum(probabilities[:, :, None], probabilities[:, None, :]),
        axis=0,
    )
    pair_available = available[:, None] & available[None, :]
    pair_overlap = jnp.where(pair_available, pair_overlap, unavailable)

    available_host = tuple(bool(value) for value in jax.device_get(available))
    available_count = sum(available_host)
    if available_count == 0:
        status = "unavailable"
        reason = "zero positive mass for every factor"
    elif available_count < len(available_host):
        status = "partial"
        reason = "zero positive mass for one or more factors"
    else:
        status = "ok"
        reason = None
    return {
        "status": status,
        "reason": reason,
        "factor_status": tuple(
            "ok" if value else "unavailable" for value in available_host
        ),
        "importance": importance,
        "positive_mass": positive_mass,
        "available": available,
        "probabilities": probabilities,
        "effective_support": jnp.where(available, effective, unavailable),
        "fracture": fracture,
        "overlap": pair_overlap,
        "overlap_available": pair_available,
    }


semantic_factor_fracture_overlap = semantic_factor_metrics


def taylor_delta(
    gradient: jax.Array, fisher_diag: jax.Array, delta: jax.Array
) -> jax.Array:
    """Diagonal-Fisher second-order loss-change approximation."""
    if gradient.shape != fisher_diag.shape or gradient.shape != delta.shape:
        raise ValueError("gradient, fisher_diag, and delta must have equal shapes")
    return jnp.sum(gradient * delta) + 0.5 * jnp.sum(fisher_diag * delta**2)


def synergy_proxy(
    delta_a: jax.Array, delta_b: jax.Array, metric_g: jax.Array
) -> jax.Array:
    """Metric-weighted interaction proxy: delta_a^T G delta_b.

    Directly follows the research plan: Syn_proxy(i,j) = delta_h_i^T G delta_h_j.
    Supports either a 1-D diagonal (such as empirical Fisher diagonal) or a 2-D metric tensor G.
    """
    da = jnp.asarray(delta_a, dtype=jnp.float32)
    db = jnp.asarray(delta_b, dtype=jnp.float32)
    g = jnp.asarray(metric_g, dtype=jnp.float32)
    if da.shape != db.shape:
        raise ValueError(
            f"delta_a shape {da.shape} must match delta_b shape {db.shape}"
        )
    if g.ndim == 1:
        if g.shape != da.shape:
            raise ValueError(
                f"metric_g shape {g.shape} must match delta shape {da.shape}"
            )
        return jnp.sum(g * da * db)
    if g.ndim == 2:
        if g.shape != (da.shape[-1], db.shape[-1]):
            raise ValueError(
                f"metric_g shape {g.shape} must be ({da.shape[-1]}, {db.shape[-1]})"
            )
        return da @ g @ db
    raise ValueError(f"metric_g must be 1-D or 2-D, got ndim={g.ndim}")


def synergy(
    delta_a: jax.Array, delta_b: jax.Array, fisher_diag: jax.Array
) -> jax.Array:
    """Diagonal-Fisher interaction estimate between two residual changes."""
    return synergy_proxy(delta_a, delta_b, fisher_diag)


def pairwise_ablation_interaction(
    loss_full: jax.Array | float,
    loss_ablate_i: jax.Array | float,
    loss_ablate_j: jax.Array | float,
    loss_ablate_ij: jax.Array | float,
) -> jax.Array:
    """Measured pairwise ablation interaction between atoms i and j.

    S_ij = L(ablate i,j) - L(ablate i) - L(ablate j) + L(full).
    Measures non-linear interaction / synergy beyond additive individual contributions.
    """
    return (
        jnp.asarray(loss_ablate_ij, jnp.float32)
        - jnp.asarray(loss_ablate_i, jnp.float32)
        - jnp.asarray(loss_ablate_j, jnp.float32)
        + jnp.asarray(loss_full, jnp.float32)
    )


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


def _sample_matrix(score_gradients: Any, name: str) -> jax.Array:
    if hasattr(score_gradients, "shape"):
        value = jnp.asarray(score_gradients, dtype=jnp.float32)
        if value.ndim == 0 or value.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one sample")
        return value.reshape((value.shape[0], -1))
    leaves = jax.tree_util.tree_leaves(score_gradients)
    if not leaves:
        raise ValueError(f"{name} is empty")
    arrays = [jnp.asarray(leaf, dtype=jnp.float32) for leaf in leaves]
    if any(array.ndim == 0 for array in arrays):
        raise ValueError(f"{name} leaves must have a sample axis")
    n_samples = arrays[0].shape[0]
    if n_samples == 0 or any(array.shape[0] != n_samples for array in arrays):
        raise ValueError(f"{name} leaves must have the same non-empty sample axis")
    return jnp.concatenate([array.reshape((n_samples, -1)) for array in arrays], axis=1)


def mixed_curvature_coupling(
    score_gradients_a: Any,
    score_gradients_b: Any,
    eps: float = _EPS,
) -> jax.Array:
    """Normalized empirical-Fisher coupling between parameter groups.

    For per-example score-gradient matrices ``G_a`` and ``G_b``, this computes
    ``||F_ab||_F / sqrt(||F_aa||_F ||F_bb||_F)``. Sample Gram matrices are used,
    avoiding materialization of parameter-by-parameter Fisher blocks. The value
    is scale invariant and lies in ``[0, 1]`` up to numerical precision.
    """
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps!r}")
    grad_a = _sample_matrix(score_gradients_a, "score_gradients_a")
    grad_b = _sample_matrix(score_gradients_b, "score_gradients_b")
    if grad_a.shape[0] != grad_b.shape[0]:
        raise ValueError("score-gradient groups must have the same sample count")
    n_samples = grad_a.shape[0]
    gram_a = (grad_a @ grad_a.T) / n_samples
    gram_b = (grad_b @ grad_b.T) / n_samples
    mixed_sq = jnp.maximum(jnp.sum(gram_a * gram_b), 0.0)
    self_a_sq = jnp.sum(jnp.square(gram_a))
    self_b_sq = jnp.sum(jnp.square(gram_b))
    denominator = jnp.sqrt(jnp.sqrt(self_a_sq * self_b_sq))
    value = jnp.where(denominator > eps, jnp.sqrt(mixed_sq) / denominator, 0.0)
    return jnp.clip(value, 0.0, 1.0)


empirical_fisher_coupling = mixed_curvature_coupling


def infer_support(
    support_scores: jax.Array,
    *,
    threshold: float = 0.0,
    top_k: int | None = None,
) -> jax.Array:
    """Infer component support from positive scores along axis zero."""
    scores = factor_importance(support_scores)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and >= 0")
    if top_k is None:
        return scores > threshold
    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or not 1 <= top_k <= scores.shape[0]
    ):
        raise ValueError(
            f"top_k must be an integer in [1, {scores.shape[0]}], got {top_k!r}"
        )
    order = jnp.argsort(-scores, axis=0, stable=True)
    ranks = jnp.argsort(order, axis=0, stable=True)
    return (ranks < top_k) & (scores > threshold)


def off_support_leakage(
    component_gradients: jax.Array,
    support_scores: jax.Array | None = None,
    *,
    support: jax.Array | None = None,
    threshold: float = 0.0,
    top_k: int | None = None,
    eps: float = _EPS,
) -> jax.Array:
    """Fraction of gradient L1 mass outside an inferred or explicit support.

    Numeric ``support_scores`` are converted to a support with
    :func:`infer_support`; a boolean array is treated as an explicit mask. For a
    component-by-factor support matrix, one leakage value is returned per factor.
    """
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps!r}")
    if (support_scores is None) == (support is None):
        raise ValueError("provide exactly one of support_scores or support")
    source = jnp.asarray(support if support is not None else support_scores)
    support_mask = (
        source.astype(bool)
        if source.dtype == jnp.bool_
        else infer_support(source, threshold=threshold, top_k=top_k)
    )
    if support_mask.ndim not in (1, 2) or support_mask.shape[0] == 0:
        raise ValueError("support must be a non-empty component vector or matrix")

    gradients = jnp.asarray(component_gradients, dtype=jnp.float32)
    if gradients.ndim == 0 or gradients.shape[0] != support_mask.shape[0]:
        raise ValueError("component_gradients and support must share component axis")
    if support_mask.ndim == 1:
        axes = tuple(range(1, gradients.ndim))
        magnitudes = (
            jnp.sum(jnp.abs(gradients), axis=axes) if axes else jnp.abs(gradients)
        )
    elif gradients.shape[:2] == support_mask.shape:
        axes = tuple(range(2, gradients.ndim))
        magnitudes = (
            jnp.sum(jnp.abs(gradients), axis=axes) if axes else jnp.abs(gradients)
        )
    else:
        axes = tuple(range(1, gradients.ndim))
        component_mass = (
            jnp.sum(jnp.abs(gradients), axis=axes) if axes else jnp.abs(gradients)
        )
        magnitudes = jnp.broadcast_to(component_mass[:, None], support_mask.shape)

    total = jnp.sum(magnitudes, axis=0)
    leaked = jnp.sum(jnp.where(support_mask, 0.0, magnitudes), axis=0)
    return jnp.where(total > 0, leaked / (total + eps), 0.0)


def _exploratory_kmeans(
    fingerprints: jax.Array, n_clusters: int, max_iter: int
) -> jax.Array:
    n_samples = fingerprints.shape[0]
    initial_indices = jnp.linspace(0, n_samples - 1, n_clusters).astype(jnp.int32)
    initial = fingerprints[initial_indices]

    def update(_iteration, centroids):
        distances = jnp.sum(
            jnp.square(fingerprints[:, None, :] - centroids[None, :, :]), axis=-1
        )
        assignments = jnp.argmin(distances, axis=1)
        membership = jax.nn.one_hot(assignments, n_clusters, dtype=fingerprints.dtype)
        counts = jnp.sum(membership, axis=0)
        updated = (membership.T @ fingerprints) / jnp.maximum(counts[:, None], 1.0)
        return jnp.where(counts[:, None] > 0, updated, centroids)

    centroids = jax.lax.fori_loop(0, max_iter, update, initial)
    distances = jnp.sum(
        jnp.square(fingerprints[:, None, :] - centroids[None, :, :]), axis=-1
    )
    return jnp.argmin(distances, axis=1)


def _fingerprint_variances(
    fingerprints: jax.Array, labels: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    same_label = labels[:, None] == labels[None, :]
    counts = jnp.sum(same_label, axis=1)
    pairwise_sq = jnp.sum(
        jnp.square(fingerprints[:, None, :] - fingerprints[None, :, :]), axis=-1
    )
    within_sum = 0.5 * jnp.sum(
        jnp.where(same_label, pairwise_sq / counts[:, None], 0.0)
    )
    within = within_sum / fingerprints.shape[0]
    centered = fingerprints - jnp.mean(fingerprints, axis=0, keepdims=True)
    total = jnp.mean(jnp.sum(jnp.square(centered), axis=1))
    between = jnp.maximum(total - within, 0.0)

    seen_before = jnp.any(
        same_label & jnp.tril(jnp.ones_like(same_label, dtype=bool), k=-1),
        axis=1,
    )
    n_labels = jnp.sum(~seen_before)
    return between, within, n_labels


def causal_fingerprint_separability(
    fingerprints: jax.Array,
    labels: jax.Array | None = None,
    *,
    n_clusters: int | None = None,
    exploratory: bool = False,
    max_iter: int = 32,
    eps: float = _EPS,
) -> dict[str, Any]:
    """Between/within variance ratio for causal fingerprints.

    Independently supplied labels produce an ``"ok"`` confirmatory result.
    Labels inferred by the optional deterministic JAX k-means path are explicitly
    marked ``"exploratory"`` so clustering the fingerprints cannot be mistaken
    for validation against external semantic roles.
    """
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps!r}")
    values = jnp.asarray(fingerprints, dtype=jnp.float32)
    if values.ndim != 2 or not all(values.shape):
        raise ValueError("fingerprints must be a non-empty [components, probes] matrix")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError("max_iter must be an integer >= 1")

    inferred = labels is None
    if inferred:
        if n_clusters is None:
            nan = jnp.asarray(jnp.nan, dtype=values.dtype)
            return {
                "status": "unavailable",
                "reason": "independent labels were not supplied",
                "score": nan,
                "between_variance": nan,
                "within_variance": nan,
                "n_labels": jnp.asarray(0),
                "labels": None,
                "labels_independent": False,
            }
        if (
            not isinstance(n_clusters, int)
            or isinstance(n_clusters, bool)
            or not 2 <= n_clusters <= values.shape[0]
        ):
            raise ValueError(
                f"n_clusters must be in [2, {values.shape[0]}], got {n_clusters!r}"
            )
        label_values = _exploratory_kmeans(values, n_clusters, max_iter)
        exploratory = True
    else:
        if n_clusters is not None:
            raise ValueError("n_clusters is only used when labels are not supplied")
        label_values = jnp.asarray(labels)
        if label_values.ndim != 1 or label_values.shape[0] != values.shape[0]:
            raise ValueError("labels must be a vector matching the component count")

    between, within, n_labels = _fingerprint_variances(values, label_values)
    score = between / (within + eps)
    n_labels_host = int(jax.device_get(n_labels))
    if n_labels_host < 2:
        status = "unavailable"
        reason = "at least two distinct labels are required"
        score = jnp.asarray(jnp.nan, dtype=values.dtype)
    elif exploratory:
        status = "exploratory"
        reason = "labels were inferred from the evaluated fingerprints"
    else:
        status = "ok"
        reason = None
    return {
        "status": status,
        "reason": reason,
        "score": score,
        "separability": score,
        "between_variance": between,
        "within_variance": within,
        "n_labels": n_labels,
        "labels": label_values,
        "labels_independent": not exploratory,
    }


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


def naturality_error(
    category_of_transformed: jax.Array,
    transformed_category: jax.Array,
    *,
    mask: jax.Array | None = None,
) -> jax.Array:
    """Mean squared category-space norm between the two naturality paths."""
    left = jnp.asarray(category_of_transformed, dtype=jnp.float32)
    right = jnp.asarray(transformed_category, dtype=jnp.float32)
    if left.shape != right.shape or left.ndim < 1:
        raise ValueError("naturality paths must have the same non-scalar shape")
    per_item = jnp.sum(jnp.square(left - right), axis=-1)
    if mask is not None and mask.shape != per_item.shape:
        raise ValueError(f"mask shape {mask.shape} must be {per_item.shape}")
    return _masked_mean(per_item, mask)


def _apply_operator(value: jax.Array, operator: Any, name: str) -> jax.Array:
    if callable(operator):
        return jnp.asarray(operator(value), dtype=jnp.float32)
    matrix = jnp.asarray(operator, dtype=jnp.float32)
    if matrix.ndim != 2 or matrix.shape[1] != value.shape[-1]:
        raise ValueError(
            f"{name} must have shape [output, {value.shape[-1]}], got {matrix.shape}"
        )
    return jnp.einsum("...i,oi->...o", value, matrix)


def category_naturality_error(
    states: jax.Array,
    hidden_operator: Any,
    category_map: Any | None = None,
    category_operator: Any | None = None,
    *,
    mask: jax.Array | None = None,
) -> jax.Array:
    """Compute ``||Lambda(U h) - T Lambda(h)||^2``.

    Operators may be callables or matrices acting on column vectors. Passing only
    the first two arguments treats them as already-computed category-space paths,
    which is useful for nonlinear category maps.
    """
    if category_map is None and category_operator is None:
        return naturality_error(states, hidden_operator, mask=mask)
    if category_map is None or category_operator is None:
        raise ValueError("category_map and category_operator must be supplied together")
    values = jnp.asarray(states, dtype=jnp.float32)
    if values.ndim < 1:
        raise ValueError("states must have a feature axis")
    transformed = _apply_operator(values, hidden_operator, "hidden_operator")
    category_of_transformed = _apply_operator(transformed, category_map, "category_map")
    category = _apply_operator(values, category_map, "category_map")
    transformed_category = _apply_operator(
        category, category_operator, "category_operator"
    )
    return naturality_error(category_of_transformed, transformed_category, mask=mask)


def _average_ranks(values: jax.Array) -> jax.Array:
    less = jnp.sum(values[None, :] < values[:, None], axis=1)
    equal = jnp.sum(values[None, :] == values[:, None], axis=1)
    return less.astype(jnp.float32) + 0.5 * (equal.astype(jnp.float32) - 1.0)


def _correlation(
    left: jax.Array, right: jax.Array, eps: float
) -> tuple[jax.Array, jax.Array]:
    left_centered = left - jnp.mean(left)
    right_centered = right - jnp.mean(right)
    denominator = jnp.linalg.norm(left_centered) * jnp.linalg.norm(right_centered)
    available = denominator > eps
    value = jnp.where(
        available,
        jnp.sum(left_centered * right_centered) / jnp.maximum(denominator, eps),
        jnp.nan,
    )
    return value, available


def ablation_fidelity(
    predicted: jax.Array,
    actual: jax.Array,
    *,
    top_k: int = 5,
    eps: float = _EPS,
) -> dict[str, Any]:
    """Compare predicted and measured signed ablation effects.

    Spearman correlation uses average ranks for ties. NRMSE is normalized by the
    range of measured effects, and top-k overlap ranks effects from largest to
    smallest (rather than by absolute magnitude).
    """
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps!r}")
    prediction = jnp.asarray(predicted, dtype=jnp.float32)
    measurement = jnp.asarray(actual, dtype=jnp.float32)
    if (
        prediction.ndim != 1
        or prediction.shape != measurement.shape
        or prediction.size == 0
    ):
        raise ValueError("predicted and actual must be equal non-empty vectors")
    if not bool(
        jax.device_get(
            jnp.all(jnp.isfinite(prediction)) & jnp.all(jnp.isfinite(measurement))
        )
    ):
        raise ValueError("predicted and actual must contain only finite values")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be an integer >= 1")
    k = min(top_k, prediction.size)

    pearson, pearson_available = _correlation(prediction, measurement, eps)
    predicted_ranks = _average_ranks(prediction)
    actual_ranks = _average_ranks(measurement)
    spearman, spearman_available = _correlation(predicted_ranks, actual_ranks, eps)
    rmse = jnp.sqrt(jnp.mean(jnp.square(prediction - measurement)))
    actual_range = jnp.max(measurement) - jnp.min(measurement)
    nrmse_available = actual_range > eps
    nrmse = jnp.where(
        nrmse_available,
        rmse / jnp.maximum(actual_range, eps),
        jnp.where(rmse <= eps, 0.0, jnp.nan),
    )

    predicted_sign = jnp.where(jnp.abs(prediction) <= eps, 0.0, jnp.sign(prediction))
    actual_sign = jnp.where(jnp.abs(measurement) <= eps, 0.0, jnp.sign(measurement))
    sign_accuracy = jnp.mean(predicted_sign == actual_sign)
    predicted_top = jnp.argsort(-prediction, stable=True)[:k]
    actual_top = jnp.argsort(-measurement, stable=True)[:k]
    top_k_overlap = (
        jnp.sum(jnp.any(predicted_top[:, None] == actual_top[None, :], axis=1)) / k
    )

    complete = bool(
        jax.device_get(
            pearson_available & spearman_available & (nrmse_available | (rmse <= eps))
        )
    )
    return {
        "status": "ok" if complete else "partial",
        "n": prediction.size,
        "top_k": k,
        "spearman": spearman,
        "spearman_rho": spearman,
        "pearson": pearson,
        "pearson_r": pearson,
        "rmse": rmse,
        "nrmse": nrmse,
        "sign_accuracy": sign_accuracy,
        "top_k_overlap": top_k_overlap,
        "available": {
            "spearman": spearman_available,
            "pearson": pearson_available,
            "nrmse": nrmse_available | (rmse <= eps),
        },
    }


predicted_vs_actual_ablation_fidelity = ablation_fidelity


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
    "ablation_fidelity",
    "canonical_low_rank_factors",
    "canonical_low_rank_svd",
    "canonical_svd_factors",
    "canonicalize_adapter_params",
    "category_naturality_error",
    "causal_fingerprint_separability",
    "compressibility_curve",
    "empirical_fisher_coupling",
    "factor_importance",
    "factor_positive_mass",
    "guarded_compressibility",
    "MIN_REFERENCE_IMPROVEMENT",
    "direction_redundancy",
    "effective_rank",
    "effective_support",
    "fracture_index",
    "gradient_conflict",
    "gradient_conflict_summary",
    "hep_nudge",
    "hep_nudge_loss",
    "infer_support",
    "loss_improvement",
    "mixed_curvature_coupling",
    "naturality_error",
    "off_support_leakage",
    "operator_commutator",
    "operator_commutator_matrix",
    "output_error",
    "pairwise_ablation_interaction",
    "parameter_commutator",
    "predicted_vs_actual_ablation_fidelity",
    "residual_compressibility",
    "residual_norm_ratio",
    "residual_redundancy",
    "invariant_redundancy",
    "overlap",
    "semantic_factor_fracture_overlap",
    "semantic_factor_metrics",
    "settling_loss",
    "summarize_gradient_conflicts",
    "synergy",
    "synergy_proxy",
    "taylor_delta",
    "v_hidden_size",
]
