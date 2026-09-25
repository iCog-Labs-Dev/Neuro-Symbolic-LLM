"""MORK integration tests requiring a live Docker container.

Start MORK before running:
    docker compose up -d --build

Tests will FAIL (not skip) if MORK is unreachable.
"""

import concurrent.futures

import numpy as np
import pytest

from Residual.symbolic_head.mork_client import (
    DockerMorkClient,
    get_mork_client,
    template_record_sexpr,
)


def test_template_record_sexpr_contains_key_and_val():
    rec = template_record_sexpr(
        "tpl_1",
        "(Inheritance dog mammal)",
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([3.0, 4.0], dtype=np.float32),
    )
    assert rec.startswith("(record tpl_1 ")
    assert "(key 1 2)" in rec
    assert "(val 3 4)" in rec


def test_mork_client_operations(mork_client):
    """Test DockerMorkClient template insertion and vector retrieval via MORK."""
    key1 = np.ones(16, dtype=np.float32)
    val1 = np.ones(16, dtype=np.float32) * 5.0
    mork_client.add_template("tpl_001", "(And (NodeA) (NodeB))", key1, val1)

    key2 = -np.ones(16, dtype=np.float32)
    val2 = np.ones(16, dtype=np.float32) * -2.0
    mork_client.add_template("tpl_002", "(Or (NodeC) (NodeD))", key2, val2)

    query = np.ones((1, 16), dtype=np.float32)
    res = mork_client.query_top_k(query, top_m=2)

    assert res.keys.shape == (1, 2, 16)
    assert res.values.shape == (1, 2, 16)
    assert res.template_ids[0][0] == "tpl_001"
    assert res.scores[0][0] > res.scores[0][1]


def test_empty_index_query(mork_client):
    """An unpopulated index returns an explicit zero-result response."""
    # Use a fresh client (the fixture creates a new one each call with empty index)
    fresh_client = get_mork_client(key_dim=16)
    query = np.ones((4, 16), dtype=np.float32)
    res = fresh_client.query_top_k(query, top_m=8)

    assert res.keys.shape == (4, 0, 16)
    assert res.values.shape == (4, 0, 16)
    assert res.template_ids == [[] for _ in range(4)]
    assert res.scores.shape == (4, 0)


def test_boundary_values_top_m(mork_client):
    """Requesting top_m greater than index item count."""
    key_vec = np.ones(16, dtype=np.float32)
    val_vec = np.ones(16, dtype=np.float32) * 2.0
    mork_client.add_template("tpl_single", "(Single)", key_vec, val_vec)

    query = np.ones((1, 16), dtype=np.float32)
    res = mork_client.query_top_k(query, top_m=10)

    assert res.keys.shape == (1, 1, 16)
    assert res.values.shape == (1, 1, 16)
    assert res.scores.shape == (1, 1)
    assert res.template_ids[0][0] == "tpl_single"


def test_dimension_mismatch(mork_client):
    """ValueError raised on key/query dimension mismatch."""
    wrong_key = np.ones(32, dtype=np.float32)
    val_vec = np.ones(16, dtype=np.float32)
    with pytest.raises(ValueError, match="Key vector dim mismatch"):
        mork_client.add_template("tpl_bad", "(Bad)", wrong_key, val_vec)

    wrong_query = np.ones((2, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="Query vector dim mismatch"):
        mork_client.query_top_k(wrong_query, top_m=4)


def test_zero_vector_query(mork_client):
    """Zero-norm query vector does not produce NaN scores."""
    key_vec = np.ones(16, dtype=np.float32)
    val_vec = np.ones(16, dtype=np.float32)
    mork_client.add_template("tpl_norm", "(Norm)", key_vec, val_vec)

    zero_query = np.zeros((1, 16), dtype=np.float32)
    res = mork_client.query_top_k(zero_query, top_m=1)

    assert not np.isnan(res.scores).any()
    assert res.keys.shape == (1, 1, 16)


def test_concurrent_multithreaded_access(mork_client):
    """Concurrent thread-safe reads and writes to DockerMorkClient."""
    num_threads = 10
    items_per_thread = 20

    def worker_add(thread_id: int):
        for i in range(items_per_thread):
            k = np.random.normal(size=16).astype(np.float32)
            v = np.random.normal(size=16).astype(np.float32)
            mork_client.add_template(f"tpl_{thread_id}_{i}", f"(Test {i})", k, v)

    def worker_query():
        for _ in range(20):
            q = np.random.normal(size=(2, 16)).astype(np.float32)
            res = mork_client.query_top_k(q, top_m=4)
            assert res.keys.ndim == 3

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for tid in range(5):
            futures.append(executor.submit(worker_add, tid))
        for _ in range(5):
            futures.append(executor.submit(worker_query))
        concurrent.futures.wait(futures)

    assert len(mork_client.vector_index.key_store) >= 100


def test_duplicate_template_ingestion(mork_client):
    """Duplicate template identifiers are rejected before a second upload."""
    key_vec = np.ones(16, dtype=np.float32)
    val_vec = np.ones(16, dtype=np.float32) * 4.0

    mork_client.add_template("tpl_dup", "(Dup)", key_vec, val_vec)
    with pytest.raises(ValueError, match="Duplicate template id"):
        mork_client.add_template("tpl_dup", "(Dup)", key_vec, val_vec)

    query = np.ones((1, 16), dtype=np.float32)
    res = mork_client.query_top_k(query, top_m=2)

    assert len(mork_client.vector_index.key_store) == 1
    assert res.template_ids[0][0] == "tpl_dup"


def test_mork_unreachable_raises_connection_error():
    """ConnectionError raised when MORK server is unreachable."""
    client = DockerMorkClient(server_url="http://localhost:9999", key_dim=16)
    query = np.ones((1, 16), dtype=np.float32)
    with pytest.raises(ConnectionError, match="MORK server unreachable"):
        client.query_top_k(query, top_m=1)


def test_mork_unreachable_add_template_raises():
    """ConnectionError raised on add_template when MORK is unreachable."""
    client = DockerMorkClient(server_url="http://localhost:9999", key_dim=16)
    with pytest.raises(ConnectionError, match="MORK server unreachable"):
        client.add_template(
            "tpl_x",
            "(Test)",
            np.ones(16, dtype=np.float32),
            np.ones(16, dtype=np.float32),
        )


def test_get_mork_client_returns_docker_client():
    """get_mork_client() always returns a DockerMorkClient."""
    client = get_mork_client(key_dim=8)
    assert isinstance(client, DockerMorkClient)
