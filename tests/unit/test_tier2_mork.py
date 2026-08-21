"""Dedicated Tier 2 Unit Test Suite.

Tests:
1. LocalHNSWClient template insertion and vector similarity search.
2. DockerMorkClient network fallback handling.
3. SymbolicHeadBridge Tier 1 integration contract.
"""

import numpy as np

from tier2_mork.bridge import SymbolicHeadBridge
from tier2_mork.client import DockerMorkClient, LocalHNSWClient


def test_local_hnsw_client_operations():
    client = LocalHNSWClient(key_dim=16)

    # Insert test templates
    key1 = np.ones(16, dtype=np.float32)
    val1 = np.ones(16, dtype=np.float32) * 5.0
    client.add_template("tpl_001", "(And (NodeA) (NodeB))", key1, val1)

    key2 = -np.ones(16, dtype=np.float32)
    val2 = np.ones(16, dtype=np.float32) * -2.0
    client.add_template("tpl_002", "(Or (NodeC) (NodeD))", key2, val2)

    # Query with matching key
    query = np.ones((1, 16), dtype=np.float32)
    res = client.query_top_k(query, top_m=2)

    assert res.keys.shape == (1, 2, 16)
    assert res.values.shape == (1, 2, 16)
    assert res.template_ids[0][0] == "tpl_001"
    assert res.scores[0][0] > res.scores[0][1]


def test_docker_mork_client_fallback():
    client = DockerMorkClient(server_url="http://localhost:9999", key_dim=16)
    key_vec = np.ones(16, dtype=np.float32)
    val_vec = np.ones(16, dtype=np.float32) * 3.0
    client.add_template("tpl_fallback", "(Test)", key_vec, val_vec)

    query = np.ones((2, 16), dtype=np.float32)
    res = client.query_top_k(query, top_m=1)

    assert res.keys.shape == (2, 1, 16)
    assert res.template_ids[0][0] == "tpl_fallback"


def test_symbolic_head_bridge_tier1_contract():
    # Instantiate Tier 2 Bridge for a 768-dim model (e.g. GPT-2 117M)
    bridge = SymbolicHeadBridge(hidden_dim=768, key_dim=256, top_m=4)

    # Seed MORK client with a sample template
    key_vec = np.ones(256, dtype=np.float32)
    val_vec = np.ones(256, dtype=np.float32) * 2.5
    bridge.mork_client.add_template("tpl_test", "(Concept (Member X Y))", key_vec, val_vec)

    # Simulate Tier 1 hidden states input from colleagues' model pass: shape (batch=2, seq=8, d=768)
    h_tier1 = np.ones((2, 8, 768), dtype=np.float32)

    # Execute Tier 1 -> Tier 2 bridge pass
    h_integrated, q_sym, s_att, mork_res = bridge.forward(h_tier1)

    # Verify shapes and types
    assert h_integrated.shape == (2, 8, 768)
    assert q_sym.shape == (2, 8, 256)
    assert s_att.shape == (2, 8, 256)
    assert mork_res.keys.shape == (16, 4, 256)  # flattened 2*8=16 tokens
    assert mork_res.template_ids[0][0] == "tpl_test"
