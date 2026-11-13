"""Pure NumPy representation diagnostics for A4 and later symbolic stages.

The functions in this module score explicit observations.  They do not load a
model, run a stage, or mutate a report, which keeps future-stage diagnostics
usable before their training and reporting pipelines exist.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _matrix(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite nonempty matrix") from error
    if array.ndim != 2 or min(array.shape, default=0) < 1:
        raise ValueError(f"{name} must be a finite nonempty matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty matrix")
    return array


def _vector(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite nonempty vector") from error
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty vector")
    return array


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def key_log_probabilities(queries: Any, keys: Any) -> np.ndarray:
    """Return full-bank raw-dot-product log probabilities for key NLL."""
    queries, keys = _matrix(queries, "queries"), _matrix(keys, "keys")
    if queries.shape[1] != keys.shape[1]:
        raise ValueError("Query and key dimensions differ")
    with np.errstate(over="ignore", invalid="ignore"):
        scores = queries @ keys.T
    if not np.isfinite(scores).all():
        raise ValueError("Query/key scores must be finite")
    scores -= scores.max(axis=1, keepdims=True)
    return scores - np.log(np.exp(scores).sum(axis=1, keepdims=True))


def factor_organization(contributions: Any) -> dict[str, Any]:
    """Compute fracture and overlap from nonnegative [factor, component] effects.

    Zero-effect factors have no identifiable contribution distribution.  They
    are excluded from overlap and receive no fracture score, rather than zero.
    """
    effects = _matrix(contributions, "contributions")
    if np.any(effects < 0):
        raise ValueError("Contributions must be nonnegative")
    totals = effects.sum(axis=1)
    valid = totals > 0
    probabilities = np.zeros_like(effects)
    probabilities[valid] = effects[valid] / totals[valid, None]
    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, np.finfo(float).tiny)),
        axis=1,
    )
    selected = probabilities[valid]
    overlap = np.minimum(selected[:, None, :], selected[None, :, :]).sum(axis=-1)
    pairs = overlap[np.triu_indices(len(selected), k=1)]
    return {
        "factors": len(effects),
        "components": effects.shape[1],
        "valid_factor_indices": np.flatnonzero(valid).tolist(),
        "zero_effect_factor_indices": np.flatnonzero(~valid).tolist(),
        "normalized_contributions": probabilities.tolist(),
        "fracture": [
            float(np.exp(entropy[index]) / effects.shape[1]) if valid[index] else None
            for index in range(len(effects))
        ],
        "overlap_matrix": overlap.tolist(),
        "mean_off_diagonal_overlap": float(pairs.mean()) if pairs.size else None,
    }


def cosine_geometry(vectors: Any, *, block_size: int = 256) -> dict[str, Any]:
    """Summarize exact unordered-pair cosines without allocating an N by N matrix."""
    vectors = _matrix(vectors, "vectors")
    block_size = _positive_integer(block_size, "block_size")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(norms).all():
        raise ValueError("Vector norms must be finite")
    if np.any(norms == 0):
        raise ValueError("Cosine similarity is undefined for zero vectors")
    unit = vectors / norms[:, None]
    count, total, absolute, squared, maximum = 0, 0.0, 0.0, 0.0, -1.0
    for start in range(0, len(unit), block_size):
        scores = np.clip(unit[start : start + block_size] @ unit.T, -1, 1)
        for offset, row in enumerate(scores):
            pairs = row[start + offset + 1 :]
            count += pairs.size
            total += float(pairs.sum())
            absolute += float(np.abs(pairs).sum())
            squared += float(np.square(pairs).sum())
            if pairs.size:
                maximum = max(maximum, float(pairs.max()))
    return {
        "vectors": len(unit),
        "dimensions": unit.shape[1],
        "pairs": count,
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "norm_mean": float(norms.mean()),
        "mean_cosine": total / count if count else None,
        "mean_absolute_cosine": absolute / count if count else None,
        "mean_squared_cosine": squared / count if count else None,
        "max_cosine": maximum if count else None,
    }


def query_silhouette(queries: Any, memberships: list[list[str]]) -> dict[str, Any]:
    """Compute Euclidean silhouette on expanded reference-template memberships.

    A multilabel query occurs in each labelled group, rather than receiving an
    arbitrary single label.  Singleton membership scores follow the zero rule.
    """
    queries = _matrix(queries, "queries")
    if len(memberships) != len(queries) or any(
        len(set(row)) != len(row) for row in memberships
    ):
        raise ValueError("Unique template memberships must align with queries")
    indices, labels = [], []
    for index, names in enumerate(memberships):
        indices.extend([index] * len(names))
        labels.extend(names)
    groups = sorted(set(labels))
    result: dict[str, Any] = {
        "queries": len(queries),
        "memberships": len(indices),
        "groups": len(groups),
        "metric": "euclidean",
        "multilabel_policy": "one membership per query/template pair",
        "silhouette": None,
    }
    if len(groups) < 2 or len(groups) >= len(indices):
        result["reason"] = "Requires at least two groups and a nonsingleton group"
        return result
    points = queries[indices]
    labels_array = np.asarray(labels)
    scores = []
    squared_norms = np.sum(points * points, axis=1)
    for start in range(0, len(points), 128):
        block = points[start : start + 128]
        squared = (
            np.sum(block * block, axis=1)[:, None]
            + squared_norms[None]
            - 2 * block @ points.T
        )
        distance = np.sqrt(np.maximum(squared, 0))
        for offset, row in enumerate(distance):
            index = start + offset
            row[index] = 0
            same = labels_array == labels[index]
            count = int(same.sum())
            if count == 1:
                scores.append(0.0)
                continue
            a = float(row[same].sum() / (count - 1))
            b = min(
                float(row[labels_array == group].mean())
                for group in groups
                if group != labels[index]
            )
            scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    result["silhouette"] = float(np.mean(scores))
    return result


def select_factors(
    probes: list[dict[str, Any]],
    *,
    minimum_documents: int = 2,
    maximum_factors: int = 32,
) -> tuple[list[str], dict[str, list[int]]]:
    """Select factors by reference support only, never by model performance."""
    minimum_documents = _positive_integer(minimum_documents, "minimum_documents")
    maximum_factors = _positive_integer(maximum_factors, "maximum_factors")
    support: dict[str, list[int]] = {}
    for index, row in enumerate(probes):
        for name in row["relevant_ids"]:
            support.setdefault(name, []).append(index)
    eligible = {
        name: indices
        for name, indices in support.items()
        if len(
            {
                (probes[index]["corpus_id"], probes[index]["document_id"])
                for index in indices
            }
        )
        >= minimum_documents
    }
    chosen = sorted(eligible, key=lambda name: (-len(eligible[name]), name))[
        :maximum_factors
    ]
    return chosen, {name: eligible[name] for name in chosen}


def capacity_sensitivity(
    queries: Any,
    keys: Any,
    template_ids: list[str],
    relevance: list[list[str]],
    projection: Any,
    *,
    head_rankings: list[list[str]] | None = None,
) -> dict[str, Any]:
    """Measure fixed-checkpoint query/key dimension and bank-size controls.

    Every control uses exact cosine search.  The shared orthonormal basis comes
    from trained projection weights, not held-out labels, and bank subsets are
    nested ID-sorted prefixes independent of relevance labels.
    """
    queries, keys = _matrix(queries, "queries"), _matrix(keys, "keys")
    projection = _matrix(projection, "projection")
    if queries.shape[1] != keys.shape[1] or projection.shape[1] != keys.shape[1]:
        raise ValueError("Capacity-control dimensions differ")
    if len(template_ids) != len(keys) or len(set(template_ids)) != len(keys):
        raise ValueError("Unique template IDs must cover the bank")
    if len(relevance) != len(queries) or not any(relevance):
        raise ValueError("Labelled queries must align and include positive references")
    known = set(template_ids)
    if any(len(set(row)) != len(row) or not set(row) <= known for row in relevance):
        raise ValueError("Unknown or duplicate reference IDs")
    _, singular_values, basis = np.linalg.svd(projection, full_matrices=False)
    dimensions = sorted(
        {min(rank, len(singular_values)) for rank in (16, 64, 128, keys.shape[1])}
    )
    order = np.argsort(np.asarray(template_ids), kind="stable")
    sizes = sorted({min(size, len(keys)) for size in (1000, 5000, len(keys))})
    controls = [("dimensions", rank, len(keys)) for rank in dimensions]
    controls += [
        ("bank_size", keys.shape[1], size) for size in sizes if size < len(keys)
    ]
    reports = []
    exact_full = None
    for kind, rank, size in controls:
        if kind == "dimensions":
            selected_keys, selected_queries = (
                keys @ basis[:rank].T,
                queries @ basis[:rank].T,
            )
        else:
            selected_keys, selected_queries = keys, queries
        selected_keys = selected_keys[order[:size]]
        norms = np.linalg.norm(selected_keys, axis=1)
        query_norms = np.linalg.norm(selected_queries, axis=1)
        if np.any(norms == 0) or np.any(query_norms == 0):
            reports.append(
                {
                    "control": kind,
                    "dimensions": rank,
                    "bank_size": size,
                    "status": "undefined_zero_vector",
                }
            )
            continue
        similarities = (selected_queries / query_norms[:, None]) @ (
            selected_keys / norms[:, None]
        ).T
        depth = min(16, size)
        indices = np.argsort(-similarities, axis=1, kind="stable")[:, :depth]
        names = np.asarray(template_ids)[order[:size]]
        rankings = [names[row].tolist() for row in indices]
        scores = []
        for cutoff in (1, 4, 8, 16):
            if cutoff > depth:
                continue
            hits = [
                len(set(row[:cutoff]) & set(truth))
                for row, truth in zip(rankings, relevance, strict=True)
            ]
            recalls = [
                hit / len(truth)
                for hit, truth in zip(hits, relevance, strict=True)
                if truth
            ]
            scores.append(
                {
                    "top_k": cutoff,
                    "all_probe_precision": sum(hits) / (len(queries) * cutoff),
                    "positive_macro_recall": float(np.mean(recalls)),
                }
            )
        available = set(names.tolist())
        reports.append(
            {
                "control": kind,
                "dimensions": rank,
                "bank_size": size,
                "status": "measured",
                "available_reference_pair_fraction": sum(
                    len(set(row) & available) for row in relevance
                )
                / sum(map(len, relevance)),
                "scores": scores,
            }
        )
        if rank == keys.shape[1] and size == len(keys):
            exact_full = rankings
    agreement = None
    if head_rankings is not None and exact_full is not None:
        if len(head_rankings) != len(queries) or any(
            len(row) < 8 for row in head_rankings
        ):
            raise ValueError("Head rankings must cover all queries at depth eight")
        agreement = float(
            np.mean(
                [
                    len(set(left[:8]) & set(right[:8])) / 8
                    for left, right in zip(head_rankings, exact_full, strict=True)
                ]
            )
        )
    return {
        "search": "exact cosine; deterministic ID tie break",
        "projection_singular_values": singular_values.tolist(),
        "controls": reports,
        "ann_exact_id_agreement_at_8": agreement,
        "limitations": "Post-training sensitivity, not training smaller heads; "
        "dimensional controls project queries and keys only, not values. ANN "
        "agreement is ID-based and includes tie sensitivity, not semantic accuracy.",
    }


def bank_template_mutual_information(joint_counts: Any) -> dict[str, Any]:
    """Measure bank-template mutual information from [bank, template] counts.

    Counts may be integer event counts or nonnegative weighted co-activations.
    Mutual information and marginal entropies use natural logarithms.
    """
    counts = _matrix(joint_counts, "joint_counts")
    if np.any(counts < 0):
        raise ValueError("joint_counts must be nonnegative")
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("joint_counts must contain positive mass")
    joint = counts / total
    bank_probabilities = joint.sum(axis=1)
    template_probabilities = joint.sum(axis=0)
    expected = bank_probabilities[:, None] * template_probabilities[None, :]
    positive = joint > 0
    mutual_information = float(
        np.sum(joint[positive] * np.log(joint[positive] / expected[positive]))
    )
    mutual_information = max(mutual_information, 0.0)

    def entropy(probabilities: np.ndarray) -> float:
        nonzero = probabilities > 0
        return -float(np.sum(probabilities[nonzero] * np.log(probabilities[nonzero])))

    return {
        "banks": counts.shape[0],
        "templates": counts.shape[1],
        "total_count": total,
        "units": "nats",
        "mutual_information": mutual_information,
        "bank_entropy": entropy(bank_probabilities),
        "template_entropy": entropy(template_probabilities),
        "bank_probabilities": bank_probabilities.tolist(),
        "template_probabilities": template_probabilities.tolist(),
        "joint_probabilities": joint.tolist(),
    }


def column_template_binding_stability(binding_matrices: Any) -> dict[str, Any]:
    """Correlate aligned [template, column] binding matrices across runs.

    Inputs must already use the same template and column ordering.  Constant
    matrices have undefined Pearson correlation and are excluded from aggregate
    summaries rather than being reported as stable.
    """
    try:
        bindings = np.asarray(binding_matrices, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "binding_matrices must be finite aligned probability matrices"
        ) from error
    if (
        bindings.ndim != 3
        or bindings.shape[0] < 2
        or min(bindings.shape[1:], default=0) < 1
        or not np.isfinite(bindings).all()
        or np.any((bindings < 0) | (bindings > 1))
    ):
        raise ValueError("binding_matrices must be finite aligned probability matrices")

    flattened = bindings.reshape(bindings.shape[0], -1)
    centered = flattened - flattened.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    pairwise: list[list[float | None]] = [
        [None for _ in range(len(bindings))] for _ in range(len(bindings))
    ]
    values = []
    for left in range(len(bindings)):
        if norms[left] > 0:
            pairwise[left][left] = 1.0
        for right in range(left + 1, len(bindings)):
            if norms[left] == 0 or norms[right] == 0:
                continue
            correlation = float(
                np.clip(
                    np.dot(centered[left], centered[right])
                    / (norms[left] * norms[right]),
                    -1,
                    1,
                )
            )
            pairwise[left][right] = pairwise[right][left] = correlation
            values.append(correlation)
    return {
        "matrices": len(bindings),
        "templates": bindings.shape[1],
        "columns": bindings.shape[2],
        "pairs": len(bindings) * (len(bindings) - 1) // 2,
        "defined_pairs": len(values),
        "pairwise_correlations": pairwise,
        "mean_pairwise_correlation": float(np.mean(values)) if values else None,
        "min_pairwise_correlation": min(values) if values else None,
        "max_pairwise_correlation": max(values) if values else None,
    }


def symbolic_alignment_preservation(
    before_scores: Any,
    after_scores: Any,
    *,
    top_k: int = 8,
) -> dict[str, Any]:
    """Measure top-k template agreement on aligned probes across base updates.

    Rows are the same probes and columns are the same templates before and after
    the update.  Rankings use descending scores and stable column-index tie
    breaking; agreement is set overlap divided by ``top_k``.
    """
    before = _matrix(before_scores, "before_scores")
    after = _matrix(after_scores, "after_scores")
    if before.shape != after.shape:
        raise ValueError("Before/after score matrices must have equal shapes")
    top_k = _positive_integer(top_k, "top_k")
    if top_k > before.shape[1]:
        raise ValueError("top_k cannot exceed the number of templates")
    before_rankings = np.argsort(-before, axis=1, kind="stable")[:, :top_k]
    after_rankings = np.argsort(-after, axis=1, kind="stable")[:, :top_k]
    agreements = np.asarray(
        [
            len(set(left.tolist()) & set(right.tolist())) / top_k
            for left, right in zip(before_rankings, after_rankings, strict=True)
        ],
        dtype=np.float64,
    )
    ordered = np.mean(before_rankings == after_rankings, axis=1)
    return {
        "probes": before.shape[0],
        "templates": before.shape[1],
        "top_k": top_k,
        "tie_break": "stable aligned template index",
        "top_k_agreement": float(agreements.mean()),
        "per_probe_top_k_agreement": agreements.tolist(),
        "ordered_top_k_agreement": float(ordered.mean()),
        "exact_top_k_set_agreement_rate": float(np.mean(agreements == 1)),
        "top_1_agreement": float(
            np.mean(before_rankings[:, 0] == after_rankings[:, 0])
        ),
    }


def symbolic_key_entanglement(keys: Any) -> dict[str, Any]:
    """Return the full symbolic-key cosine matrix and off-diagonal summaries."""
    keys = _matrix(keys, "keys")
    norms = np.linalg.norm(keys, axis=1)
    if not np.isfinite(norms).all():
        raise ValueError("Key norms must be finite")
    if np.any(norms == 0):
        raise ValueError("Symbolic-key cosine is undefined for zero vectors")
    unit = keys / norms[:, None]
    cosine = np.clip(unit @ unit.T, -1, 1)
    pairs = cosine[np.triu_indices(len(keys), k=1)]
    return {
        "keys": len(keys),
        "dimensions": keys.shape[1],
        "pairs": pairs.size,
        "cosine_matrix": cosine.tolist(),
        "off_diagonal_absolute_mass": float(np.abs(pairs).sum()),
        "mean_off_diagonal_cosine": float(pairs.mean()) if pairs.size else None,
        "mean_absolute_off_diagonal_cosine": (
            float(np.abs(pairs).mean()) if pairs.size else None
        ),
        "mean_squared_off_diagonal_cosine": (
            float(np.square(pairs).mean()) if pairs.size else None
        ),
        "max_off_diagonal_cosine": float(pairs.max()) if pairs.size else None,
        "max_absolute_off_diagonal_cosine": (
            float(np.abs(pairs).max()) if pairs.size else None
        ),
    }


def symbolic_commutator(
    parameters_i_then_j: Any,
    parameters_j_then_i: Any,
    outputs_i_then_j: Any,
    outputs_j_then_i: Any,
    *,
    reference_parameters: Any | None = None,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Measure symbolic parameter- and function-space update commutators.

    Parameter inputs are flattened vectors for the two task orders.  Output
    inputs have samples on axis zero; all remaining axes are flattened per
    sample.  The parameter difference is normalized by a supplied reference
    norm, or by the mean norm of the two ordered vectors.  The function metric
    is the research-plan mean per-sample Euclidean output difference.
    """
    if isinstance(eps, bool) or not np.isscalar(eps):
        raise ValueError("eps must be finite and nonnegative")
    eps = float(eps)
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")
    left = _vector(parameters_i_then_j, "parameters_i_then_j")
    right = _vector(parameters_j_then_i, "parameters_j_then_i")
    if left.shape != right.shape:
        raise ValueError("Ordered parameter vectors must have equal shapes")

    if reference_parameters is None:
        parameter_scale = 0.5 * (
            float(np.linalg.norm(left)) + float(np.linalg.norm(right))
        )
        scale_source = "mean ordered parameter norm"
    else:
        reference = _vector(reference_parameters, "reference_parameters")
        if reference.shape != left.shape:
            raise ValueError("Reference and ordered parameter vectors must align")
        parameter_scale = float(np.linalg.norm(reference))
        scale_source = "reference parameter norm"
    parameter_difference = float(np.linalg.norm(left - right))
    denominator = parameter_scale + eps
    parameter_commutator = (
        parameter_difference / denominator
        if denominator > 0
        else 0.0 if parameter_difference == 0 else None
    )

    def outputs(value: Any, name: str) -> np.ndarray:
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain finite nonempty outputs") from error
        if array.ndim < 1 or array.size == 0 or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain finite nonempty outputs")
        return array

    left_outputs = outputs(outputs_i_then_j, "outputs_i_then_j")
    right_outputs = outputs(outputs_j_then_i, "outputs_j_then_i")
    if left_outputs.shape != right_outputs.shape:
        raise ValueError("Ordered output arrays must have equal shapes")
    output_difference = (left_outputs - right_outputs).reshape(
        left_outputs.shape[0], -1
    )
    distances = np.linalg.norm(output_difference, axis=1)
    return {
        "parameters": left.size,
        "parameter_scale_source": scale_source,
        "parameter_scale": parameter_scale,
        "parameter_difference_norm": parameter_difference,
        "parameter_commutator": parameter_commutator,
        "samples": left_outputs.shape[0],
        "output_dimensions_per_sample": output_difference.shape[1],
        "function_commutator": float(distances.mean()),
        "function_commutator_rms": float(np.sqrt(np.mean(np.square(distances)))),
        "function_commutator_max": float(distances.max()),
    }


def value_alignment_error(predicted: Any, target: Any) -> float:
    """Return the mean per-example squared L2 value-alignment error.

    A one-dimensional input is one value vector.  For higher-dimensional
    inputs, axis zero indexes examples and every remaining axis is flattened
    into that example's value vector.
    """
    try:
        prediction = np.asarray(predicted, dtype=np.float64)
        reference = np.asarray(target, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "predicted and target must be finite nonempty arrays"
        ) from error
    if (
        prediction.ndim < 1
        or prediction.size == 0
        or not np.isfinite(prediction).all()
        or reference.ndim < 1
        or reference.size == 0
        or not np.isfinite(reference).all()
    ):
        raise ValueError("predicted and target must be finite nonempty arrays")
    if prediction.shape != reference.shape:
        raise ValueError("predicted and target must have equal shapes")
    difference = prediction - reference
    examples = (
        difference.reshape(1, -1)
        if difference.ndim == 1
        else difference.reshape(difference.shape[0], -1)
    )
    with np.errstate(over="ignore", invalid="ignore"):
        squared_errors = np.einsum("ij,ij->i", examples, examples)
    if not np.isfinite(squared_errors).all():
        raise ValueError("value-alignment errors must be finite")
    return float(squared_errors.mean())


def symbolic_gradient_conflict(gradients: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize ``[-cos(g_i, g_j)]_+`` over named gradient vectors.

    Zero vectors are retained in the output but excluded from pair summaries,
    because cosine similarity is undefined for them.
    """
    if not isinstance(gradients, Mapping) or not gradients:
        raise ValueError("gradients must be a nonempty mapping")
    names = list(gradients)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("gradient names must be nonempty strings")
    vectors = [_vector(gradients[name], f"gradient {name!r}") for name in names]
    if any(vector.shape != vectors[0].shape for vector in vectors[1:]):
        raise ValueError("all gradient vectors must have equal shapes")

    stacked = np.stack(vectors)
    norms = np.linalg.norm(stacked, axis=1)
    if not np.isfinite(norms).all():
        raise ValueError("gradient norms must be finite")
    valid = norms > 0
    unit = np.zeros_like(stacked)
    unit[valid] = stacked[valid] / norms[valid, None]
    cosine: list[list[float | None]] = [[None for _ in names] for _ in names]
    conflict: list[list[float | None]] = [[None for _ in names] for _ in names]
    for index in np.flatnonzero(valid):
        cosine[index][index] = 1.0
        conflict[index][index] = 0.0

    pair_summaries = []
    values = []
    conflicting_pairs = 0
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            if not (valid[left] and valid[right]):
                continue
            pair_cosine = float(np.clip(np.dot(unit[left], unit[right]), -1, 1))
            pair_conflict = max(-pair_cosine, 0.0)
            cosine[left][right] = cosine[right][left] = pair_cosine
            conflict[left][right] = conflict[right][left] = pair_conflict
            values.append(pair_conflict)
            conflicting_pairs += int(pair_cosine < 0)
            pair_summaries.append(
                {
                    "left": names[left],
                    "right": names[right],
                    "cosine": pair_cosine,
                    "conflict": pair_conflict,
                }
            )

    reason = None if values else "fewer than two nonzero gradients"
    return {
        "status": "ok" if values else "unavailable",
        "reason": reason,
        "gradient_names": names,
        "gradients": len(names),
        "dimensions": vectors[0].size,
        "valid_gradient_names": [
            name for name, is_valid in zip(names, valid, strict=True) if is_valid
        ],
        "zero_gradient_names": [
            name for name, is_valid in zip(names, valid, strict=True) if not is_valid
        ],
        "pairs": len(names) * (len(names) - 1) // 2,
        "defined_pairs": len(values),
        "pair_count": len(values),
        "cosine_matrix": cosine,
        "conflict_matrix": conflict,
        "pair_summaries": pair_summaries,
        "mean_conflict": float(np.mean(values)) if values else None,
        "max_conflict": max(values) if values else None,
        "fraction_conflicting": (conflicting_pairs / len(values) if values else None),
    }


def symbolic_mixed_fisher_coupling(
    score_gradients: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize normalized empirical-Fisher coupling between named groups.

    Every group is an ``[example, parameter]`` score-gradient matrix with the
    same example ordering.  For groups A and B, the reported coupling is
    ``||F_AB||_F / sqrt(||F_AA||_F ||F_BB||_F)`` where
    ``F_AB = G_A.T @ G_B / N``.  The exact Frobenius norms are evaluated through
    example-space Gram matrices to avoid materializing parameter-sized blocks.
    """
    if not isinstance(score_gradients, Mapping) or not score_gradients:
        raise ValueError("score_gradients must be a nonempty mapping")
    names = list(score_gradients)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("score-gradient group names must be nonempty strings")
    groups = [
        _matrix(score_gradients[name], f"score-gradient group {name!r}")
        for name in names
    ]
    examples = groups[0].shape[0]
    if any(group.shape[0] != examples for group in groups[1:]):
        raise ValueError("score-gradient groups must align on examples")

    scales = [float(np.max(np.abs(group))) for group in groups]
    valid = np.asarray([scale > 0 for scale in scales])
    grams = []
    for group, scale in zip(groups, scales, strict=True):
        scaled = group / scale if scale > 0 else np.zeros_like(group)
        with np.errstate(over="ignore", invalid="ignore"):
            gram = (scaled @ scaled.T) / examples
        if not np.isfinite(gram).all():
            raise ValueError("score-gradient Fisher products must be finite")
        grams.append(gram)
    self_norms = [float(np.linalg.norm(gram)) for gram in grams]

    coupling: list[list[float | None]] = [[None for _ in names] for _ in names]
    for index in np.flatnonzero(valid):
        coupling[index][index] = 1.0
    values = []
    pair_summaries = []
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            if not (valid[left] and valid[right]):
                continue
            cross_norm_squared = float(np.sum(grams[left] * grams[right]))
            denominator = np.sqrt(self_norms[left] * self_norms[right])
            value = float(
                np.clip(np.sqrt(max(cross_norm_squared, 0.0)) / denominator, 0, 1)
            )
            coupling[left][right] = coupling[right][left] = value
            values.append(value)
            pair_summaries.append(
                {"left": names[left], "right": names[right], "coupling": value}
            )

    reason = None if values else "fewer than two nonzero score-gradient groups"
    return {
        "status": "ok" if values else "unavailable",
        "reason": reason,
        "group_names": names,
        "groups": len(names),
        "examples": examples,
        "group_dimensions": {
            name: group.shape[1] for name, group in zip(names, groups, strict=True)
        },
        "zero_score_gradient_groups": [
            name for name, is_valid in zip(names, valid, strict=True) if not is_valid
        ],
        "pairs": len(names) * (len(names) - 1) // 2,
        "defined_pairs": len(values),
        "coupling_matrix": coupling,
        "pair_summaries": pair_summaries,
        "mean_coupling": float(np.mean(values)) if values else None,
        "max_coupling": max(values) if values else None,
        "estimator": "uncentered empirical Fisher from per-example score gradients",
        "normalization": "||F_AB||_F / sqrt(||F_AA||_F ||F_BB||_F)",
    }


def symbolic_attention_contribution(
    head_on: Mapping[str, Any],
    head_off: Mapping[str, Any],
    higher_is_better: bool | Mapping[str, bool],
) -> dict[str, Any]:
    """Compare named scalar metrics with symbolic attention enabled and ablated.

    ``contribution`` is oriented so that a positive value always means the
    symbolic head helped.  Metric direction is required explicitly, either once
    for every metric or as an exactly aligned name-to-boolean mapping.
    """
    if not isinstance(head_on, Mapping) or not head_on:
        raise ValueError("head_on must be a nonempty mapping")
    if not isinstance(head_off, Mapping):
        raise ValueError("head_off must be a mapping")
    names = list(head_on)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("metric names must be nonempty strings")
    if set(head_off) != set(names):
        raise ValueError("head_on and head_off metric names must match exactly")

    if isinstance(higher_is_better, (bool, np.bool_)):
        directions = {name: bool(higher_is_better) for name in names}
    elif isinstance(higher_is_better, Mapping):
        if set(higher_is_better) != set(names) or any(
            not isinstance(value, (bool, np.bool_))
            for value in higher_is_better.values()
        ):
            raise ValueError("higher_is_better must provide a boolean for every metric")
        directions = {name: bool(higher_is_better[name]) for name in names}
    else:
        raise ValueError("higher_is_better must be a boolean or metric mapping")

    def scalar(value: Any, name: str) -> float:
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
            raise ValueError(f"{name} must be a finite scalar")
        try:
            result = float(value)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a finite scalar") from error
        if not np.isfinite(result):
            raise ValueError(f"{name} must be a finite scalar")
        return result

    metrics = {}
    improved = unchanged = 0
    for name in names:
        enabled = scalar(head_on[name], f"head_on[{name!r}]")
        ablated = scalar(head_off[name], f"head_off[{name!r}]")
        difference = enabled - ablated
        contribution = difference if directions[name] else -difference
        improved += int(contribution > 0)
        unchanged += int(contribution == 0)
        metrics[name] = {
            "head_on": enabled,
            "head_off": ablated,
            "higher_is_better": directions[name],
            "difference": difference,
            "contribution": contribution,
            "relative_contribution": (
                contribution / abs(ablated) if ablated != 0 else None
            ),
        }
    return {
        "status": "ok",
        "metric_count": len(names),
        "improved_count": improved,
        "unchanged_count": unchanged,
        "degraded_count": len(names) - improved - unchanged,
        "metrics": metrics,
    }


def human_template_review_summary(
    relevance: Any,
    readability: Any,
    reviewer_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Summarize boolean human reviews and inter-reviewer agreement when known.

    One-dimensional inputs contain one rating per item and cannot establish
    agreement.  Two-dimensional inputs use ``[item, reviewer]`` layout and can
    report pairwise agreement and Fleiss' kappa.  Reviewer IDs label columns for
    two-dimensional inputs and individual ratings for one-dimensional inputs.
    """

    def ratings(value: Any, name: str) -> np.ndarray:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a nonempty boolean array") from error
        if (
            array.ndim not in {1, 2}
            or array.size == 0
            or min(array.shape, default=0) < 1
            or array.dtype != np.dtype(bool)
        ):
            raise ValueError(f"{name} must be a nonempty boolean array")
        return array

    relevant = ratings(relevance, "relevance")
    readable = ratings(readability, "readability")
    if relevant.shape != readable.shape:
        raise ValueError("relevance and readability ratings must have equal shapes")

    labels = None
    if reviewer_ids is not None:
        if isinstance(reviewer_ids, (str, bytes)) or not isinstance(
            reviewer_ids, Sequence
        ):
            raise ValueError("reviewer_ids must be a sequence of nonempty strings")
        labels = list(reviewer_ids)
        expected = relevant.shape[1] if relevant.ndim == 2 else relevant.shape[0]
        if len(labels) != expected or any(
            not isinstance(label, str) or not label.strip() for label in labels
        ):
            raise ValueError("reviewer_ids must align and contain nonempty strings")
        if relevant.ndim == 2 and len(set(labels)) != len(labels):
            raise ValueError("reviewer_ids must be unique for reviewer columns")

    reviewer_count = (
        relevant.shape[1]
        if relevant.ndim == 2
        else len(dict.fromkeys(labels)) if labels is not None else None
    )
    if relevant.ndim == 1 or relevant.shape[1] < 2:
        agreement: dict[str, Any] = {
            "status": "unavailable",
            "reason": "at least two aligned ratings per item are required",
        }
    else:
        raters = relevant.shape[1]
        pairs_per_item = raters * (raters - 1) // 2

        def categorical_agreement(
            values: np.ndarray, categories: Sequence[int]
        ) -> dict[str, float | None]:
            counts = np.stack(
                [np.sum(values == category, axis=1) for category in categories],
                axis=1,
            )
            observed = float(
                np.mean(np.sum(counts * (counts - 1), axis=1) / (raters * (raters - 1)))
            )
            probabilities = counts.sum(axis=0) / counts.sum()
            expected_agreement = float(np.sum(np.square(probabilities)))
            kappa = (
                (observed - expected_agreement) / (1 - expected_agreement)
                if expected_agreement < 1
                else None
            )
            return {
                "pairwise_agreement": observed,
                "expected_agreement": expected_agreement,
                "fleiss_kappa": kappa,
            }

        joint = relevant.astype(np.int8) * 2 + readable.astype(np.int8)
        agreement = {
            "status": "ok",
            "pairs_per_item": pairs_per_item,
            "relevance": categorical_agreement(relevant, [0, 1]),
            "readability": categorical_agreement(readable, [0, 1]),
            "joint": categorical_agreement(joint, [0, 1, 2, 3]),
        }

    return {
        "status": "ok",
        "items": relevant.shape[0],
        "ratings": relevant.size,
        "ratings_per_item": relevant.shape[1] if relevant.ndim == 2 else 1,
        "reviewers": reviewer_count,
        "reviewer_ids": labels,
        "relevance_rate": float(relevant.mean()),
        "readability_rate": float(readable.mean()),
        "relevant_and_readable_rate": float(np.mean(relevant & readable)),
        "agreement": agreement,
    }


__all__ = [
    "bank_template_mutual_information",
    "capacity_sensitivity",
    "column_template_binding_stability",
    "cosine_geometry",
    "factor_organization",
    "human_template_review_summary",
    "key_log_probabilities",
    "query_silhouette",
    "select_factors",
    "symbolic_alignment_preservation",
    "symbolic_attention_contribution",
    "symbolic_commutator",
    "symbolic_gradient_conflict",
    "symbolic_key_entanglement",
    "symbolic_mixed_fisher_coupling",
    "value_alignment_error",
]
