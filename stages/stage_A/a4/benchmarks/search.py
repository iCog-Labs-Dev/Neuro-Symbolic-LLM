"""HNSW search effort versus exact cosine neighbors, not semantic relevance."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .common import paired_samples


def neighbor_agreement(actual: Any, exact: Any, tolerance: float = 1e-5) -> dict:
    depth = exact.scores.shape[1]
    strict = np.mean(
        [
            len(set(a) & set(b)) / depth
            for a, b in zip(actual.template_ids, exact.template_ids, strict=True)
        ]
    )
    return {
        "strict_id_agreement": float(strict),
        "tie_aware_score_agreement": float(
            np.mean(actual.scores >= exact.scores[:, -1:] - tolerance)
        ),
        "score_tolerance": tolerance,
    }


def measure(
    args: Any, index: Any, queries: np.ndarray, result: dict, samples: list
) -> None:
    import faiss

    from residual.symbolic_head.contracts.template_record import TemplateRecord
    from residual.symbolic_head.retrieval.faiss_index import FaissTemplateIndex

    if index.index_backend != "hnsw":
        raise ValueError("Search sweep requires an HNSW MORK-derived index")
    faiss.omp_set_num_threads(args.threads)
    records = [
        TemplateRecord(*row)
        for row in zip(
            index.id_store,
            index.metta_store,
            index.key_store,
            index.val_store,
            strict=True,
        )
    ]
    exact = FaissTemplateIndex(key_dim=index.key_dim, backend="flat")
    exact.rebuild(records)
    reference = exact.query_top_k(queries, args.top_m)
    agreement = []
    previous = index._faiss_index.hnsw.efSearch
    try:
        calls: dict[str, Callable[[], dict[str, float]]] = {}
        for effort in dict.fromkeys(args.ef_search):

            def call(effort: int = effort) -> dict[str, float]:
                # Only this private benchmark index is changed; MORK is untouched.
                index._faiss_index.hnsw.efSearch = effort
                index.query_top_k(queries, args.top_m)
                return {}

            call()
            agreement.append(
                dict(
                    ef_search=effort,
                    **neighbor_agreement(
                        index.query_top_k(queries, args.top_m), reference
                    ),
                )
            )
            calls[f"hnsw-ef{effort}"] = call

        def flat() -> dict[str, float]:
            exact.query_top_k(queries, args.top_m)
            return {}

        calls["exact-flat"] = flat
        paired_samples(calls, args.samples, args.warmups, samples)
    finally:
        index._faiss_index.hnsw.efSearch = previous
    result.update(
        ann_quality=agreement,
        scope="CPU normalization/search/gather; exact-neighbor agreement, not template-label precision/recall",
    )
