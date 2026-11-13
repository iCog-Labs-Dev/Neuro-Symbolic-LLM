import numpy as np
import pytest

from residual.symbolic_head.contracts.template_record import TemplateRecord
from residual.symbolic_head.mork_client import DockerMorkClient
from residual.symbolic_head.retrieval import (
    FaissTemplateIndex,
    load_retrieval_config,
)
from residual.symbolic_head.retrieval import faiss_index as index_module


def test_native_snapshot_preserves_graph_and_has_independent_vector_storage():
    import faiss

    index = FaissTemplateIndex(key_dim=4, backend="hnsw", hnsw_ef_search=37)
    index.rebuild(_records())
    generation, ids, serialized, keys, values = index.native_snapshot()
    restored = faiss.deserialize_index(serialized)
    assert ids == ("dog", "cat") and generation == index._generation
    assert restored.hnsw.efSearch == 37
    query = np.array([[1, 0, 0, 0]], dtype=np.float32)
    expected_scores, expected_labels = index._faiss_index.search(query, 2)
    scores, labels = restored.search(query, 2)
    np.testing.assert_array_equal(labels, expected_labels)
    np.testing.assert_array_equal(scores, expected_scores)
    keys[:] = 99
    values[:] = -99
    np.testing.assert_array_equal(index.key_store[0], [1, 0, 0, 0])
    np.testing.assert_array_equal(index.val_store[0], [2, 0, 0, 0])


def test_generation_watch_rejects_setup_race_and_can_be_removed():
    index = FaissTemplateIndex(key_dim=4, backend="flat")
    index.rebuild(_records())
    generation = index._generation
    calls = []
    token = index.watch_native_generation(generation, lambda: calls.append(1))
    index.clear()
    assert calls == [1]
    with pytest.raises(RuntimeError, match="generation changed"):
        index.watch_native_generation(generation, lambda: None)
    index.unwatch_native_generation(token)
    index.rebuild(_records())
    assert calls == [1]


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


@pytest.mark.parametrize("backend", ["flat", "hnsw"])
def test_batched_gather_is_invalidated_by_insert_and_rebuild(backend):
    index = FaissTemplateIndex(key_dim=4, backend=backend)
    dog, cat = _records()
    index.rebuild([dog])
    queries = np.tile(cat.key_vector, (32, 1))
    assert index.query_top_k(queries).template_ids == [["dog"]] * 32
    index.add_template(cat.template_id, cat.metta_expr, cat.key_vector, cat.val_vector)
    result = index.query_top_k(queries, top_m=1)
    assert result.template_ids == [["cat"]] * 32
    np.testing.assert_array_equal(result.values[:, 0], np.tile(cat.val_vector, (32, 1)))
    result.values[:] = -99
    assert np.all(index.query_top_k(queries, top_m=1).values >= 0)
    index.rebuild([dog])
    assert index.query_top_k(queries).template_ids == [["dog"]] * 32


def test_index_snapshot_owns_its_input_vectors():
    index = FaissTemplateIndex(key_dim=4, backend="flat")
    records = _records()
    index.rebuild(records)
    index.query_top_k(np.eye(4, dtype=np.float32)[:2])
    records[0].val_vector[:] = -99
    result = index.query_top_k(np.array([1, 0, 0, 0], dtype=np.float32), top_m=1)
    np.testing.assert_array_equal(result.values[0, 0], [2, 0, 0, 0])


@pytest.mark.parametrize("backend", ["flat", "hnsw"])
@pytest.mark.parametrize("count", [1, 5])
def test_numeric_into_matches_existing_query_without_changing_vectors(backend, count):
    index = FaissTemplateIndex(key_dim=4, backend=backend)
    index.rebuild(_records())
    generation, identifiers = index.numeric_binding()
    queries = np.tile(np.array([[0.1, 1, 0, 0]], dtype=np.float32), (count, 1))
    expected = index.query_top_k(queries, top_m=8)
    packed = np.full(count * 2 * (2 * 4 + 2), np.nan, dtype=np.float32)
    elapsed = index.query_top_k_into(queries, 8, packed, expected_generation=generation)
    vectors, metadata = count * 2 * 4, count * 2
    np.testing.assert_array_equal(packed[:vectors].reshape(count, 2, 4), expected.keys)
    np.testing.assert_array_equal(
        packed[vectors : 2 * vectors].reshape(count, 2, 4), expected.values
    )
    np.testing.assert_array_equal(
        packed[2 * vectors : 2 * vectors + metadata].reshape(count, 2), expected.scores
    )
    labels = packed.view(np.int32)[-metadata:].reshape(count, 2)
    assert [
        [identifiers[label] for label in row] for row in labels
    ] == expected.template_ids
    assert elapsed > 0
    packed[:] = -999
    np.testing.assert_array_equal(index.query_top_k(queries, 8).values, expected.values)


@pytest.mark.parametrize("change", ["add", "rebuild", "clear"])
def test_numeric_binding_rejects_stale_generation(change):
    index = FaissTemplateIndex(key_dim=4, backend="flat")
    dog, cat = _records()
    index.rebuild([dog])
    generation, _ = index.numeric_binding()
    if change == "add":
        index.add_template(
            cat.template_id, cat.metta_expr, cat.key_vector, cat.val_vector
        )
    elif change == "rebuild":
        index.rebuild([dog])
    else:
        index.clear()
    with pytest.raises(RuntimeError, match="generation changed"):
        index.query_top_k_into(
            np.ones((1, 4), dtype=np.float32),
            1,
            np.empty(10, dtype=np.float32),
            expected_generation=generation,
        )


@pytest.mark.parametrize("invalid", ["shape", "dtype", "readonly", "strided"])
def test_numeric_into_rejects_incompatible_output(invalid):
    index = FaissTemplateIndex(key_dim=4, backend="flat")
    index.rebuild(_records()[:1])
    generation, _ = index.numeric_binding()
    output = np.empty(10, dtype=np.float32)
    if invalid == "shape":
        output = output[:-1]
    elif invalid == "dtype":
        output = output.astype(np.float64)
    elif invalid == "strided":
        output = np.empty(20, dtype=np.float32)[::2]
    else:
        output.setflags(write=False)
    with pytest.raises(ValueError, match="Packed result"):
        index.query_top_k_into(
            np.ones((1, 4), dtype=np.float32), 1, output, expected_generation=generation
        )


def test_empty_index_cannot_be_native_bound():
    with pytest.raises(ValueError, match="nonempty bank"):
        FaissTemplateIndex(key_dim=4).numeric_binding()
