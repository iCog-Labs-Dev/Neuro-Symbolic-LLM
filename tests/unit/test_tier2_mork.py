"""Tier 2 MORK integration tests — all tests require a live MORK Docker container.

Start MORK before running:
    docker compose up -d --build

Tests will FAIL (not skip) if MORK is unreachable.
"""

import concurrent.futures

import numpy as np
import pytest

from symbolic_pipeline.head import SymbolicHead
from tier2_mork.retrieval_bridge import Tier2Retrieve
from tier2_mork.client import (
    DockerMorkClient,
    get_mork_client,
    template_record_sexpr,
)
from tier2_mork.communication import decode_query


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


def test_symbolic_head_bridge_tier1_contract(mork_client_256):
    """Full Tier 1 ↔ Tier 2 contract: project, retrieve, SAtt, integrate."""
    key_vec = np.ones(256, dtype=np.float32)
    val_vec = np.ones(256, dtype=np.float32) * 2.5
    mork_client_256.add_template(
        "tpl_test", "(Concept (Member X Y))", key_vec, val_vec
    )
    hop = Tier2Retrieve(mork_client=mork_client_256, key_dim=256)
    q = np.ones((16, 256), dtype=np.float32)
    retrieved, q_pkt, t_pkt = hop.retrieve(q, top_m=4)
    assert retrieved.keys.shape == (16, 4, 256)
    assert retrieved.template_ids[0][0] == "tpl_test"
    assert q_pkt.nbytes > 0 and t_pkt.nbytes > 0
    q_rt = decode_query(q_pkt)
    assert q_rt.shape == (16, 256)

    head = SymbolicHead(
        mork_client=mork_client_256,
        d_model=768, k_dim=256, top_m=4, temperature=1.0,
    )
    h_tier1 = np.ones((2, 8, 768), dtype=np.float32)
    h_integrated, info = head.forward(h_tier1)
    assert h_integrated.shape == (2, 8, 768)
    assert info["q_sym"].shape == (2, 8, 256)
    assert info["s_att"].shape == (2, 8, 256)


def test_empty_index_query(mork_client):
    """Querying an unpopulated index returns zero arrays and empty template IDs."""
    # Use a fresh client (the fixture creates a new one each call with empty index)
    fresh_client = get_mork_client(key_dim=16)
    query = np.ones((4, 16), dtype=np.float32)
    res = fresh_client.query_top_k(query, top_m=8)

    assert res.keys.shape == (4, 8, 16)
    assert res.values.shape == (4, 8, 16)
    assert res.template_ids == [["empty"] * 8 for _ in range(4)]
    assert np.all(res.scores == 0.0)


def test_boundary_values_top_m(mork_client):
    """Requesting top_m greater than index item count."""
    key_vec = np.ones(16, dtype=np.float32)
    val_vec = np.ones(16, dtype=np.float32) * 2.0
    mork_client.add_template("tpl_single", "(Single)", key_vec, val_vec)

    query = np.ones((1, 16), dtype=np.float32)
    res = mork_client.query_top_k(query, top_m=10)

    assert res.keys.shape == (1, 10, 16)
    assert res.template_ids[0][0] == "tpl_single"
    assert res.template_ids[0][1] == "empty"


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
    """Duplicate template registration handles stored keys cleanly."""
    key_vec = np.ones(16, dtype=np.float32)
    val_vec = np.ones(16, dtype=np.float32) * 4.0

    mork_client.add_template("tpl_dup", "(Dup)", key_vec, val_vec)
    mork_client.add_template("tpl_dup", "(Dup)", key_vec, val_vec)

    query = np.ones((1, 16), dtype=np.float32)
    res = mork_client.query_top_k(query, top_m=2)

    assert len(mork_client.vector_index.key_store) >= 2
    assert res.template_ids[0][0] == "tpl_dup"
    assert res.template_ids[0][1] == "tpl_dup"


def test_mork_unreachable_raises_connection_error():
    """ConnectionError raised when MORK server is unreachable."""
    client = DockerMorkClient(
        server_url="http://localhost:9999", key_dim=16
    )
    query = np.ones((1, 16), dtype=np.float32)
    with pytest.raises(ConnectionError, match="MORK server unreachable"):
        client.query_top_k(query, top_m=1)


def test_mork_unreachable_add_template_raises():
    """ConnectionError raised on add_template when MORK is unreachable."""
    client = DockerMorkClient(
        server_url="http://localhost:9999", key_dim=16
    )
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


def test_symbolic_head_batched_forward_matches_bridge(mork_client):
    """SymbolicHead forward pass through MORK-backed bridge."""
    key_vec = np.ones(16, dtype=np.float32)
    val_vec = np.ones(16, dtype=np.float32) * 2.0
    mork_client.add_template("tpl_a", "(Concept A)", key_vec, val_vec)

    head = SymbolicHead(
        mork_client=mork_client,
        d_model=32, k_dim=16, top_m=1, temperature=1.0,
    )
    h_in = np.ones((2, 3, 32), dtype=np.float32)
    h_out, info = head.forward(h_in)

    assert h_out.shape == (2, 3, 32)
    assert info["q_sym"].shape == (2, 3, 16)
    assert info["s_att"].shape == (2, 3, 16)
    assert info["matched_ids"][0] == "tpl_a"
