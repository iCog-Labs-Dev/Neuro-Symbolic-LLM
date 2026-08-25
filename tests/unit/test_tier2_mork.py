import concurrent.futures
import numpy as np
import pytest

from tier2_mork.bridge import SymbolicHeadBridge
from tier2_mork.client import DockerMorkClient, LocalHNSWClient


def test_local_hnsw_client_operations():
    """Test LocalHNSWClient template insertion and vector retrieval."""
    client = LocalHNSWClient(key_dim=16)

    key1 = np.ones(16, dtype=np.float32)
    val1 = np.ones(16, dtype=np.float32) * 5.0
    client.add_template("tpl_001", "(And (NodeA) (NodeB))", key1, val1)

    key2 = -np.ones(16, dtype=np.float32)
    val2 = np.ones(16, dtype=np.float32) * -2.0
    client.add_template("tpl_002", "(Or (NodeC) (NodeD))", key2, val2)

    query = np.ones((1, 16), dtype=np.float32)
    res = client.query_top_k(query, top_m=2)

    assert res.keys.shape == (1, 2, 16)
    assert res.values.shape == (1, 2, 16)
    assert res.template_ids[0][0] == "tpl_001"
    assert res.scores[0][0] > res.scores[0][1]


def test_docker_mork_client_fallback():
    """Test DockerMorkClient local fallback when strict_mork is disabled."""
    client = DockerMorkClient(
        server_url="http://localhost:9999", key_dim=16, strict_mork=False
    )
    key_vec = np.ones(16, dtype=np.float32)
    val_vec = np.ones(16, dtype=np.float32) * 3.0
    client.add_template("tpl_fallback", "(Test)", key_vec, val_vec)

    query = np.ones((2, 16), dtype=np.float32)
    res = client.query_top_k(query, top_m=1)

    assert res.keys.shape == (2, 1, 16)
    assert res.template_ids[0][0] == "tpl_fallback"


def test_symbolic_head_bridge_tier1_contract():
    """Test Tier 1 -> Tier 2 SymbolicHeadBridge tensor forward pass."""
    local_client = LocalHNSWClient(key_dim=256)
    bridge = SymbolicHeadBridge(hidden_dim=768, key_dim=256, top_m=4, mork_client=local_client)

    key_vec = np.ones(256, dtype=np.float32)
    val_vec = np.ones(256, dtype=np.float32) * 2.5
    bridge.mork_client.add_template(
        "tpl_test", "(Concept (Member X Y))", key_vec, val_vec
    )

    h_tier1 = np.ones((2, 8, 768), dtype=np.float32)

    h_integrated, q_sym, s_att, mork_res = bridge.forward(h_tier1)

    assert h_integrated.shape == (2, 8, 768)
    assert q_sym.shape == (2, 8, 256)
    assert s_att.shape == (2, 8, 256)
    assert mork_res.keys.shape == (16, 4, 256)
    assert mork_res.template_ids[0][0] == "tpl_test"


def test_empty_index_query():
    """Test querying an unpopulated index returns zero arrays and empty template IDs."""
    client = LocalHNSWClient(key_dim=16)
    query = np.ones((4, 16), dtype=np.float32)
    res = client.query_top_k(query, top_m=8)

    assert res.keys.shape == (4, 8, 16)
    assert res.values.shape == (4, 8, 16)
    assert res.template_ids == [["empty"] * 8 for _ in range(4)]
    assert np.all(res.scores == 0.0)


def test_boundary_values_top_m():
    """Test requesting top_m greater than index item count."""
    client = LocalHNSWClient(key_dim=16)
    key_vec = np.ones(16, dtype=np.float32)
    val_vec = np.ones(16, dtype=np.float32) * 2.0
    client.add_template("tpl_single", "(Single)", key_vec, val_vec)

    query = np.ones((1, 16), dtype=np.float32)
    res = client.query_top_k(query, top_m=10)

    assert res.keys.shape == (1, 10, 16)
    assert res.template_ids[0][0] == "tpl_single"
    assert res.template_ids[0][1] == "empty"


def test_dimension_mismatch():
    """Test ValueError raised on key/query dimension mismatch."""
    client = LocalHNSWClient(key_dim=16)

    wrong_key = np.ones(32, dtype=np.float32)
    val_vec = np.ones(16, dtype=np.float32)
    with pytest.raises(ValueError, match="Key vector dim mismatch"):
        client.add_template("tpl_bad", "(Bad)", wrong_key, val_vec)

    wrong_query = np.ones((2, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="Query vector dim mismatch"):
        client.query_top_k(wrong_query, top_m=4)


def test_zero_vector_query():
    """Test zero-norm query vector does not produce NaN scores."""
    client = LocalHNSWClient(key_dim=16)
    key_vec = np.ones(16, dtype=np.float32)
    val_vec = np.ones(16, dtype=np.float32)
    client.add_template("tpl_norm", "(Norm)", key_vec, val_vec)

    zero_query = np.zeros((1, 16), dtype=np.float32)
    res = client.query_top_k(zero_query, top_m=1)

    assert not np.isnan(res.scores).any()
    assert res.keys.shape == (1, 1, 16)


def test_concurrent_multithreaded_access():
    """Test concurrent thread-safe reads and writes to LocalHNSWClient."""
    client = LocalHNSWClient(key_dim=16)
    num_threads = 10
    items_per_thread = 20

    def worker_add(thread_id: int):
        for i in range(items_per_thread):
            k = np.random.normal(size=16).astype(np.float32)
            v = np.random.normal(size=16).astype(np.float32)
            client.add_template(f"tpl_{thread_id}_{i}", f"(Test {i})", k, v)

    def worker_query():
        for _ in range(20):
            q = np.random.normal(size=(2, 16)).astype(np.float32)
            res = client.query_top_k(q, top_m=4)
            assert res.keys.ndim == 3

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for tid in range(5):
            futures.append(executor.submit(worker_add, tid))
        for _ in range(5):
            futures.append(executor.submit(worker_query))
        concurrent.futures.wait(futures)

    assert len(client.key_store) == 100


def test_duplicate_template_ingestion():
    """Test duplicate template registration handles stored keys cleanly."""
    client = LocalHNSWClient(key_dim=16)
    key_vec = np.ones(16, dtype=np.float32)
    val_vec = np.ones(16, dtype=np.float32) * 4.0

    client.add_template("tpl_dup", "(Dup)", key_vec, val_vec)
    client.add_template("tpl_dup", "(Dup)", key_vec, val_vec)

    query = np.ones((1, 16), dtype=np.float32)
    res = client.query_top_k(query, top_m=2)

    assert len(client.key_store) == 2
    assert res.template_ids[0][0] == "tpl_dup"
    assert res.template_ids[0][1] == "tpl_dup"


def test_docker_mork_strict_mode_enforcement():
    """Test strict_mork=True raises ConnectionError when server is unreachable."""
    client = DockerMorkClient(
        server_url="http://localhost:9999", key_dim=16, strict_mork=True
    )
    query = np.ones((1, 16), dtype=np.float32)
    with pytest.raises(ConnectionError, match="Tier 2 MORK Docker connection failed"):
        client.query_top_k(query, top_m=1)
