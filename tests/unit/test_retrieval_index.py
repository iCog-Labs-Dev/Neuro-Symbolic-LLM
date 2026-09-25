import numpy as np
import pytest

from Residual.symbolic_head.contracts.template_record import TemplateRecord
from Residual.symbolic_head.mork_client import DockerMorkClient
from Residual.symbolic_head.retrieval import (
    FaissTemplateIndex,
    load_retrieval_config,
)
from Residual.symbolic_head.retrieval import faiss_index as index_module


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
    index = FaissTemplateIndex(key_dim=4, backend=backend)

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
        FaissTemplateIndex(key_dim=4, backend="unknown")


def test_faiss_is_required(monkeypatch):
    monkeypatch.setattr(index_module, "faiss", None)

    with pytest.raises(RuntimeError, match="FAISS is required"):
        FaissTemplateIndex(key_dim=4)


def test_empty_index_returns_no_synthetic_templates():
    index = FaissTemplateIndex(key_dim=4)

    result = index.query_top_k(np.ones((2, 4), dtype=np.float32), top_m=3)

    assert result.keys.shape == (2, 0, 4)
    assert result.values.shape == (2, 0, 4)
    assert result.scores.shape == (2, 0)
    assert result.template_ids == [[], []]


def test_short_index_returns_only_real_templates():
    index = FaissTemplateIndex(key_dim=4)
    index.rebuild(_records()[:1])

    result = index.query_top_k(np.ones((1, 4), dtype=np.float32), top_m=3)

    assert result.keys.shape == (1, 1, 4)
    assert result.values.shape == (1, 1, 4)
    assert result.scores.shape == (1, 1)
    assert result.template_ids == [["dog"]]


def test_rejects_duplicate_template_identifiers():
    index = FaissTemplateIndex(key_dim=4)
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
    index = FaissTemplateIndex(key_dim=4)

    with pytest.raises(ValueError, match="top_m"):
        index.query_top_k(np.ones((1, 4), dtype=np.float32), top_m=top_m)


def test_q1_config_selects_hnsw_for_mork_client():
    config = load_retrieval_config()
    client = DockerMorkClient(key_dim=4)

    assert config.backend == "hnsw"
    assert client.index_backend == "hnsw"
    assert client.vector_index._faiss_index.hnsw.efSearch == config.hnsw_ef_search


def test_flat_backend_remains_explicit_reference():
    client = DockerMorkClient(key_dim=4, index_backend="flat")

    assert client.index_backend == "flat"


def test_rebuild_rejects_duplicate_without_losing_current_index():
    index = FaissTemplateIndex(key_dim=4, backend="hnsw")
    index.rebuild(_records()[:1])
    original = index.query_top_k(np.array([1.0, 0.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="Duplicate template id"):
        index.rebuild([_records()[0], _records()[0]])

    after = index.query_top_k(np.array([1.0, 0.0, 0.0, 0.0]))
    assert after.template_ids == original.template_ids == [["dog"]]


def test_rebuild_rejects_invalid_vector_without_losing_current_index():
    index = FaissTemplateIndex(key_dim=4, backend="hnsw")
    index.rebuild(_records()[:1])
    invalid = TemplateRecord(
        template_id="bad",
        metta_expr="(Bad)",
        key_vector=np.zeros(4, dtype=np.float32),
        val_vector=np.ones(4, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="non-zero norm"):
        index.rebuild([invalid])

    assert index.query_top_k(np.array([1.0, 0.0, 0.0, 0.0])).template_ids == [["dog"]]
