"""Evidence-driven Stage A research metrics for continuous residual models.

Arrays passed here are measurements from separately specified probes and
interventions, not inferred semantic labels. See docs/benchmark_metrics.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _array(value: Any, name: str, ndim: int) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != ndim or not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite {ndim}-D array")
    return result


def factor_organization(importance: Any) -> dict[str, Any]:
    """Fracture and overlap, with rows=factors and columns=components."""
    values = _array(importance, "factor_importance", 2)
    if (values < 0).any():
        raise ValueError("factor importance must be nonnegative")
    totals = values.sum(axis=1)
    valid = totals > 0
    p = np.divide(
        values, totals[:, None], out=np.zeros_like(values), where=valid[:, None]
    )
    entropy = -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0), axis=1)
    fracture = np.exp(entropy) / values.shape[1]
    selected = p[valid]
    overlap = np.minimum(selected[:, None, :], selected[None, :, :]).sum(axis=2)
    pairs = overlap[np.triu_indices(len(selected), 1)]
    return {
        "fracture": [
            float(value) if is_valid else None
            for value, is_valid in zip(fracture, valid, strict=True)
        ],
        "valid_factor_indices": np.flatnonzero(valid).tolist(),
        "zero_effect_factor_indices": np.flatnonzero(~valid).tolist(),
        "mean_fracture": float(fracture[valid].mean()) if valid.any() else None,
        "overlap": overlap.tolist(),
        "mean_overlap": float(pairs.mean()) if pairs.size else None,
    }


def curvature_coupling(block: Any) -> float:
    """Frobenius magnitude of a declared mixed Hessian block."""
    return float(np.linalg.norm(_array(block, "mixed_hessian", 2)))


def off_support_leakage(gradients: Any, support: Any) -> float | None:
    """Fraction of total absolute component gradient outside declared support."""
    g = _array(gradients, "component_gradients", 1)
    if (
        not isinstance(support, list)
        or len(support) != len(g)
        or not all(isinstance(x, bool) for x in support)
    ):
        raise ValueError("support must contain one boolean per component")
    total = np.abs(g).sum()
    return float(np.abs(g)[~np.asarray(support)].sum() / total) if total else None


def fingerprint_separability(fingerprints: Any, groups: Any) -> float | None:
    """Between/within variance for independently specified component groups."""
    x = _array(fingerprints, "fingerprints", 2)
    if (
        not isinstance(groups, list)
        or len(groups) != len(x)
        or any(not isinstance(g, str) or not g for g in groups)
    ):
        raise ValueError(
            "fingerprint groups must provide one nonempty label per component"
        )
    unique = sorted(set(groups))
    if len(unique) < 2 or len(unique) == len(x):
        return None
    labels = np.asarray(groups)
    mean = x.mean(axis=0)
    between = sum(
        len(x[labels == label]) * np.sum((x[labels == label].mean(axis=0) - mean) ** 2)
        for label in unique
    )
    within = sum(
        np.sum((x[labels == label] - x[labels == label].mean(axis=0)) ** 2)
        for label in unique
    )
    return float(between / within) if within > 0 else None


def squared_error(left: Any, right: Any, name: str) -> float:
    a = _array(left, name + " left", 2)
    b = _array(right, name + " right", 2)
    if a.shape != b.shape:
        raise ValueError(f"{name} arrays must have equal shapes")
    return float(np.mean(np.sum((a - b) ** 2, axis=1)))


def functional_commutator(first: Any, second: Any) -> float:
    """Mean output-space Euclidean difference on matched probe examples."""
    a = _array(first, "first-order outputs", 2)
    b = _array(second, "reverse-order outputs", 2)
    if a.shape != b.shape:
        raise ValueError("order outputs must have equal shapes")
    return float(np.linalg.norm(a - b, axis=1).mean())


def fingerprint_stability(first: Any, second: Any) -> float | None:
    """Correlation of explicitly aligned component fingerprints across runs."""
    a = _array(first, "first fingerprints", 2)
    b = _array(second, "second fingerprints", 2)
    if a.shape != b.shape:
        raise ValueError("aligned fingerprint arrays must have equal shapes")
    x, y = a.ravel(), b.ravel()
    return (
        float(np.corrcoef(x, y)[0, 1])
        if x.size > 1 and np.std(x) > 0 and np.std(y) > 0
        else None
    )


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def ablation_fidelity(predicted: Any, actual: Any) -> dict[str, float | None]:
    p = _array(predicted, "predicted_ablation", 1)
    a = _array(actual, "actual_ablation", 1)
    if p.shape != a.shape:
        raise ValueError("ablation arrays must have equal lengths")
    rp, ra = _ranks(p), _ranks(a)
    correlation = (
        float(np.corrcoef(rp, ra)[0, 1])
        if len(p) >= 2 and np.std(rp) > 0 and np.std(ra) > 0
        else None
    )
    return {"mae": float(np.mean(np.abs(p - a))), "spearman": correlation}


def accuracy_forgetting(matrix: Any) -> dict[str, Any]:
    """Accuracy forgetting after each checkpoint, excluding pre-acquisition rows.

    Use null for unmeasured entries. Values must be task-level scores in [0,1].
    """
    if (
        not isinstance(matrix, list)
        or not matrix
        or any(
            not isinstance(row, list) or len(row) != len(matrix[0]) for row in matrix
        )
    ):
        raise ValueError("accuracy_matrix must be a nonempty rectangular list")
    x = np.asarray(
        [[np.nan if value is None else value for value in row] for row in matrix],
        dtype=float,
    )
    finite = x[np.isfinite(x)]
    if np.isinf(x).any() or ((finite < 0) | (finite > 1)).any():
        raise ValueError("task scores must be in [0,1] or null")
    result: list[float | None] = [None]
    for t in range(1, len(x)):
        changes = []
        for j in range(min(t, x.shape[1])):
            previous = x[j:t, j]
            previous = previous[np.isfinite(previous)]
            if previous.size and np.isfinite(x[t, j]):
                changes.append(float(previous.max() - x[t, j]))
        result.append(float(np.mean(changes)) if changes else None)
    return {"trajectory": result, "final": result[-1]}


def classification_metrics(
    targets: Any, predictions: Any, *, multilabel: bool = False
) -> dict[str, float]:
    """Compute accuracy, macro-F1 and micro-F1 from declared task predictions."""
    if multilabel:
        y = _array(targets, "targets", 2)
        p = _array(predictions, "predictions", 2)
        if (
            y.shape != p.shape
            or not np.isin(y, [0, 1]).all()
            or not np.isin(p, [0, 1]).all()
        ):
            raise ValueError(
                "multilabel targets and predictions must be equal binary matrices"
            )
        tp = ((y == 1) & (p == 1)).sum(axis=0)
        fp = ((y == 0) & (p == 1)).sum(axis=0)
        fn = ((y == 1) & (p == 0)).sum(axis=0)
        accuracy = float(np.mean(np.all(y == p, axis=1)))
    else:
        y = _array(targets, "targets", 1)
        p = _array(predictions, "predictions", 1)
        if (
            y.shape != p.shape
            or (y < 0).any()
            or (p < 0).any()
            or not np.equal(y, np.floor(y)).all()
            or not np.equal(p, np.floor(p)).all()
        ):
            raise ValueError(
                "single-label targets and predictions must be equal nonnegative integer arrays"
            )
        classes = np.union1d(y, p)
        tp = np.asarray([np.sum((y == c) & (p == c)) for c in classes])
        fp = np.asarray([np.sum((y != c) & (p == c)) for c in classes])
        fn = np.asarray([np.sum((y == c) & (p != c)) for c in classes])
        accuracy = float(np.mean(y == p))
    denom = 2 * tp + fp + fn
    f1 = np.divide(
        2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom != 0
    )
    return {
        "accuracy": accuracy,
        "macro_f1": float(f1.mean()),
        "micro_f1": float(2 * tp.sum() / denom.sum()) if denom.sum() else 0.0,
    }


def calibration_error(confidence: Any, correct: Any, bins: int = 15) -> float:
    c = _array(confidence, "confidence", 1)
    y = _array(correct, "correct", 1)
    if (
        c.shape != y.shape
        or (c < 0).any()
        or (c > 1).any()
        or not np.isin(y, [0, 1]).all()
        or not isinstance(bins, int)
        or bins < 1
    ):
        raise ValueError(
            "calibration requires matching confidence [0,1], binary outcomes, and positive bins"
        )
    indices = np.minimum((c * bins).astype(int), bins - 1)
    return float(
        sum(abs(c[indices == b].sum() - y[indices == b].sum()) for b in range(bins))
        / len(c)
    )


def evaluate_audits(inputs: dict[str, Any]) -> dict[str, Any]:
    """Calculate only metrics for which explicit measurement evidence is supplied."""
    if inputs.get("schema") != "a1-a3-research-audit/v1":
        raise ValueError("audit input must use schema a1-a3-research-audit/v1")
    out: dict[str, Any] = {}
    if "factor_importance" in inputs:
        out["factor_organization"] = factor_organization(inputs["factor_importance"])
    if "mixed_hessian" in inputs:
        out["mixed_curvature"] = curvature_coupling(inputs["mixed_hessian"])
    if "cross_support_hessians" in inputs:
        blocks = inputs["cross_support_hessians"]
        if not isinstance(blocks, list) or not blocks:
            raise ValueError("cross_support_hessians must be a nonempty list of blocks")
        out["cross_support_coupling"] = max(
            curvature_coupling(block) for block in blocks
        )
    if "component_gradients" in inputs:
        out["off_support_leakage"] = off_support_leakage(
            inputs["component_gradients"], inputs.get("support")
        )
    if "fingerprints" in inputs:
        out["fingerprint_separability"] = fingerprint_separability(
            inputs["fingerprints"], inputs.get("fingerprint_groups")
        )
    if "fingerprint_stability" in inputs:
        out["fingerprint_stability"] = fingerprint_stability(
            inputs["fingerprint_stability"]["first"],
            inputs["fingerprint_stability"]["second"],
        )
    for name in ("naturality", "closure"):
        if name in inputs:
            pair = inputs[name]
            out[name + "_error"] = squared_error(pair["left"], pair["right"], name)
    if "ablation" in inputs:
        out["ablation_fidelity"] = ablation_fidelity(
            inputs["ablation"]["predicted"], inputs["ablation"]["actual"]
        )
    if "ordered_outputs" in inputs:
        out["functional_commutator"] = functional_commutator(
            inputs["ordered_outputs"]["first"], inputs["ordered_outputs"]["reverse"]
        )
    if "accuracy_matrix" in inputs:
        out["accuracy_forgetting"] = accuracy_forgetting(inputs["accuracy_matrix"])
    if "classification" in inputs:
        task = inputs["classification"]
        out["classification"] = classification_metrics(
            task["targets"],
            task["predictions"],
            multilabel=task.get("multilabel", False),
        )
        if "confidence" in task:
            if task.get("multilabel", False):
                raise ValueError(
                    "declare a separate binary calibration protocol for multilabel tasks"
                )
            out["classification"]["ece"] = calibration_error(
                task["confidence"],
                (np.asarray(task["targets"]) == np.asarray(task["predictions"])).astype(
                    int
                ),
            )
    for name in ("category_roles", "support_explanations"):
        if name in inputs:
            value = inputs[name]
            out[name + "_accuracy"] = classification_metrics(
                value["targets"], value["predictions"]
            )["accuracy"]
    if "instruction_scores" in inputs:
        scores = _array(inputs["instruction_scores"], "instruction_scores", 1)
        if (scores < 0).any() or (scores > 1).any():
            raise ValueError("instruction scores must be in [0,1]")
        out["instruction_following"] = float(scores.mean())
    return out
