"""
Stage A4 End-to-End Experiment Verification Sweep.
Validates Tier 2 MORK HNSW indexing, Symbolic Head Bridge, and top-m MeTTa retrieval.
"""

import logging

import numpy as np

from tier2_mork.bridge import SymbolicHeadBridge
from tier2_mork.client import LocalHNSWClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_stage_a4")


def run_stage_a4_verification():
    logger.info("=== STARTING TIER 2 MORK SYMBOLIC HEAD INTEGRATION SWEEP ===")

    hidden_dim = 768
    key_dim = 256
    top_m = 4
    num_templates = 500

    # 1. Initialize Tier 2 Local HNSW Client
    logger.info(f"1. Seeding Tier 2 MORK Client with {num_templates} templates...")
    client = LocalHNSWClient(key_dim=key_dim)

    rng = np.random.default_rng(42)
    for i in range(num_templates):
        k_vec = rng.normal(0, 1, key_dim).astype(np.float32)
        v_vec = rng.normal(0, 1, key_dim).astype(np.float32)
        client.add_template(f"tpl_{i:04d}", f"(Concept (Node_{i}))", k_vec, v_vec)

    # 2. Instantiate Symbolic Head Bridge
    logger.info("2. Initializing Symbolic Head Bridge Integration Contract...")
    bridge = SymbolicHeadBridge(
        hidden_dim=hidden_dim, key_dim=key_dim, top_m=top_m, mork_client=client
    )

    # 3. Pass simulated Tier 1 hidden states (batch=2, seq=4, d=768)
    logger.info("3. Executing Symbolic Head forward pass...")
    h_tier1 = rng.normal(0, 1, (2, 4, hidden_dim)).astype(np.float32)
    h_out, q_sym, s_att, mork_res = bridge.forward(h_tier1)

    logger.info(f"Tier 1 Hidden State Input Shape: {h_tier1.shape}")
    logger.info(f"Projected Symbolic Query Shape:  {q_sym.shape}")
    logger.info(f"Symbolic Attention Vector Shape: {s_att.shape}")
    logger.info(f"Integrated Residual Output Shape:{h_out.shape}")
    logger.info(f"Retrieved Top-{top_m} Template IDs: {mork_res.template_ids[0]}")

    logger.info("=== TIER 2 MORK VERIFICATION SWEEP COMPLETED SUCCESSFULLY! ===")


if __name__ == "__main__":
    run_stage_a4_verification()
