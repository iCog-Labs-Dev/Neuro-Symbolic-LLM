import numpy as np
import pytest

from Residual.symbolic_head import mork_client as mork_module
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
    before = index.query_top_k(
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), top_m=2
    )
    index.clear()
    index.rebuild(records)
    after = index.query_top_k(
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), top_m=2
    )

    assert before.template_ids == after.template_ids
    np.testing.assert_allclose(before.keys, after.keys)
    np.testing.assert_allclose(before.values, after.values)
    np.testing.assert_allclose(before.scores, after.scores)


def test_hnsw_backend_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        _LocalFAISSIndex(key_dim=4, backend="unknown")


def test_faiss_is_required(monkeypatch):
    monkeypatch.setattr(mork_module, "faiss", None)

    with pytest.raises(RuntimeError, match="FAISS is required"):
        _LocalFAISSIndex(key_dim=4)


def test_empty_index_returns_no_synthetic_templates():
    index = _LocalFAISSIndex(key_dim=4)

    result = index.query_top_k(np.ones((2, 4), dtype=np.float32), top_m=3)

    assert result.keys.shape == (2, 0, 4)
    assert result.values.shape == (2, 0, 4)
    assert result.scores.shape == (2, 0)
    assert result.template_ids == [[], []]


def test_short_index_returns_only_real_templates():
    index = _LocalFAISSIndex(key_dim=4)
    index.rebuild(_records()[:1])

    result = index.query_top_k(np.ones((1, 4), dtype=np.float32), top_m=3)

    assert result.keys.shape == (1, 1, 4)
    assert result.values.shape == (1, 1, 4)
    assert result.scores.shape == (1, 1)
    assert result.template_ids == [["dog"]]


def test_rejects_duplicate_template_identifiers():
    index = _LocalFAISSIndex(key_dim=4)
    record = _records()[0]
    index.add_template(
        record.template_id,
        record.metta_expr,
        record.key_vector,
        record.val_vector,
    )

    with pytest.raises(ValueError, match="Duplicate template id"):
        index.add_template(
            record.template_id,
            record.metta_expr,
            record.key_vector,
            record.val_vector,
        )


@pytest.mark.parametrize("top_m", [0, -1])
def test_rejects_nonpositive_top_m(top_m):
    index = _LocalFAISSIndex(key_dim=4)

    with pytest.raises(ValueError, match="top_m"):
        index.query_top_k(np.ones((1, 4), dtype=np.float32), top_m=top_m)
