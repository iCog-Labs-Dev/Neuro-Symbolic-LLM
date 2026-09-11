import numpy as np
import pytest

from Residual.symbolic_head.mork_client import (
    TemplateRecord,
    _LocalFAISSIndex,
)


def _records() -> list[TemplateRecord]:
    return [
        TemplateRecord(
            template_id="dog",
            metta_expr="(Inheritance dog mammal)",
            key_vector=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            val_vector=np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ),
        TemplateRecord(
            template_id="cat",
            metta_expr="(Inheritance cat mammal)",
            key_vector=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            val_vector=np.array([0.0, 2.0, 0.0, 0.0], dtype=np.float32),
        ),
    ]


@pytest.mark.parametrize("backend", ["flat", "hnsw"])
def test_rebuild_restores_retrieval_state(backend):
    index = _LocalFAISSIndex(key_dim=4, backend=backend)
    if backend == "hnsw" and index.index_backend != "hnsw":
        pytest.skip("FAISS HNSW is unavailable")

    records = _records()
    index.rebuild(records)
    before = index.query_top_k(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), top_m=2)
    index.clear()
    index.rebuild(records)
    after = index.query_top_k(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), top_m=2)

    assert before.template_ids == after.template_ids
    np.testing.assert_allclose(before.keys, after.keys)
    np.testing.assert_allclose(before.values, after.values)
    np.testing.assert_allclose(before.scores, after.scores)


def test_hnsw_backend_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        _LocalFAISSIndex(key_dim=4, backend="unknown")