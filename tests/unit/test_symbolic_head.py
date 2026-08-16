"""
Unit tests for Stage A4 SymbolicHead & Alignment Losses.
"""

import numpy as np
import pytest
from residual.symbolic_head.head import SymbolicHead
from residual.symbolic_head.losses import combined_symbolic_loss
from tier2_mork.store import MORKTemplateStore, TemplateRecord


def test_symbolic_head_forward():
    d_model = 512
    k_dim = 64
    store = MORKTemplateStore(dim=k_dim, max_capacity=100)

    for i in range(1, 11):
        rec = TemplateRecord(
            template_id=i,
            metta_ast=f"(Ast_{i})",
            key_embedding=np.random.randn(k_dim).astype(np.float32),
            value_embedding=np.random.randn(k_dim).astype(np.float32),
        )
        store.insert_template(rec)

    head = SymbolicHead(d_model=d_model, k_dim=k_dim, top_m=4, mork_store=store)
    h_in = np.random.randn(d_model).astype(np.float32)

    h_out, info = head.forward(h_in)
    assert h_out.shape == (d_model,)
    assert info["q_sym"].shape == (k_dim,)
    assert info["s_att"].shape == (k_dim,)
    assert len(info["matched_ids"]) == 4


def test_combined_symbolic_loss():
    k_dim = 32
    d_model = 64
    q_sym = np.random.randn(k_dim).astype(np.float32)
    target_key = np.random.randn(k_dim).astype(np.float32)
    all_keys = np.random.randn(20, k_dim).astype(np.float32)
    u_satt = np.random.randn(d_model).astype(np.float32)
    target_val = np.random.randn(d_model).astype(np.float32)

    losses = combined_symbolic_loss(q_sym, target_key, all_keys, u_satt, target_val)
    assert "l_sym_total" in losses
    assert losses["l_sym_key"] > 0
    assert losses["l_sym_value"] >= 0
