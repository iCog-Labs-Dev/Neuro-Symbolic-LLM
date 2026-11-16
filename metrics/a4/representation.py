"""Representation diagnostics on fixed-bank symbolic queries."""

from __future__ import annotations

from typing import Any

import numpy as np


def _matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty matrix")
    return array


def key_log_probabilities(queries: Any, keys: Any) -> np.ndarray:
    """Full-bank raw-dot-product log probabilities, as in A4 key alignment."""
    queries, keys = _matrix(queries, "queries"), _matrix(keys, "keys")
    if queries.shape[1] != keys.shape[1]:
        raise ValueError("Query and key dimensions differ")
    scores = queries @ keys.T
    scores -= scores.max(axis=1, keepdims=True)
    return scores - np.log(np.exp(scores).sum(axis=1, keepdims=True))


def factor_organization(contributions: Any) -> dict[str, Any]:
    """Paper sections 14.3/14.4, given nonnegative [factor, component] effects.

    Zero-effect factors have no identifiable contribution distribution. They
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
        probabilities * np.log(np.maximum(probabilities, np.finfo(float).tiny)), axis=1
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
            float(np.exp(entropy[i]) / effects.shape[1]) if valid[i] else None
            for i in range(len(effects))
        ],
        "overlap_matrix": overlap.tolist(),
        "mean_off_diagonal_overlap": float(pairs.mean()) if pairs.size else None,
    }


def cosine_geometry(vectors: Any, *, block_size: int = 256) -> dict[str, Any]:
    """Exact unordered-pair cosine summary without allocating an N by N matrix."""
    vectors = _matrix(vectors, "vectors")
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size < 1
    ):
        raise ValueError("block_size must be positive")
    norms = np.linalg.norm(vectors, axis=1)
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
    """Euclidean silhouette on expanded reference-template memberships.

    A multilabel query occurs in each labelled group, rather than receiving an
    arbitrary single label. Singleton membership scores follow the zero rule.
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
    for start in range(0, len(points), 128):
        block = points[start : start + 128]
        squared = (
            np.sum(block * block, axis=1)[:, None]
            + np.sum(points * points, axis=1)[None]
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


def select_factors(probes, *, minimum_documents=2, maximum_factors=32):
    """Select by reference support only, never by measured model performance."""
    support: dict[str, list[int]] = {}
    for index, row in enumerate(probes):
        for name in row["relevant_ids"]:
            support.setdefault(name, []).append(index)
    eligible = {
        name: indices
        for name, indices in support.items()
        if len({(probes[i]["corpus_id"], probes[i]["document_id"]) for i in indices})
        >= minimum_documents
    }
    chosen = sorted(eligible, key=lambda name: (-len(eligible[name]), name))[
        :maximum_factors
    ]
    return chosen, {name: eligible[name] for name in chosen}


def capacity_sensitivity(
    queries, keys, template_ids, relevance, projection, *, head_rankings=None
):
    """Fixed-checkpoint query/key dimension and bank-size controls.

    All controls use exact cosine search to avoid confusing capacity with ANN
    approximation. The shared orthonormal basis comes from trained projection
    weights, not held-out labels. Values/writeback are not retrained or resized.
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
    # Bank subsets are nested ID-sorted prefixes, independent of relevance labels.
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
        "limitations": "Post-training sensitivity, not training smaller heads; dimensional controls project queries and keys only, not values. ANN agreement is ID-based and includes tie sensitivity, not semantic accuracy.",
    }
