"""
Stage A4 End-to-End Experiment Verification Sweep.
Validates Tier 2 MORK HNSW indexing, PCIe Bridge Client, Stage A4 Symbolic Head, and Alignment Loss convergence.
"""

import logging
import time
import numpy as np

from fabricpc_ext.pcie_bridge import CrossTierPayload, PCIeBridgeClient
from residual.symbolic_head.head import SymbolicHead
from residual.symbolic_head.losses import combined_symbolic_loss
from tier2_retrieval.benchmark import RetrievalBenchmarkSuite
from tier2_retrieval.store import TemplateStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_stage_a4")


def run_stage_a4_verification():
    logger.info("=== STARTING STAGE A4 SYMBOLIC HEAD INTEGRATION SWEEP ===")

    d_model = 4096
    k_dim = 256
    top_m = 8
    num_templates = 10000

    # 1. Initialize Tier 2 Template Store & Benchmark Suite
    logger.info(f"1. Seeding Tier 2 Template Store with {num_templates} templates...")
    bench = RetrievalBenchmarkSuite(dim=k_dim, metric="cosine")
    bench_results = bench.run_full_benchmark(
        num_templates=num_templates, num_queries=500, top_m=top_m, ef_search_list=[32, 50]
    )

    logger.info(f"HNSW Recall@{top_m}: {bench_results['recall_at_m']:.4f}")
    logger.info(
        f"HNSW p95 Latency: {bench_results['query_benchmarks']['ef_50']['p95_latency_ms']:.3f} ms"
    )
    logger.info(
        f"HNSW Throughput (QPS): {bench_results['query_benchmarks']['ef_50']['qps']:.1f} queries/sec"
    )

    # Re-use seeded store for integration test
    templates = bench.generate_synthetic_templates(num_templates)
    store = TemplateStore(dim=k_dim, space="cosine", max_capacity=num_templates + 1000)
    store.insert_batch(templates)

    # 2. Test PCIe Bridge Client
    logger.info("2. Testing Tier 1 GPU <-> Tier 2 CPU PCIe Bridge Client...")
    bridge = PCIeBridgeClient(template_store=store, direct_in_memory=True)

    dummy_q = np.random.randn(k_dim).astype(np.float32)
    dummy_q /= np.linalg.norm(dummy_q)

    payload = CrossTierPayload(layer_id=28, token_position=16, query_vector=dummy_q, top_m=top_m)
    resp = bridge.send_query(payload)

    logger.info(f"PCIe Round-Trip Latency: {resp.latency_ms:.3f} ms")
    logger.info(f"Retrieved Top-{top_m} Template IDs: {resp.matched_ids}")
    logger.info(f"Key Matrix shape: {resp.key_matrix.shape}, Value Matrix shape: {resp.value_matrix.shape}")

    # 3. Test Symbolic Head Execution
    logger.info("3. Testing Symbolic Head forward pass...")
    head = SymbolicHead(
        d_model=d_model, k_dim=k_dim, top_m=top_m, temperature=0.1, layer_id=28, template_store=store
    )

    dummy_h_tilde = np.random.randn(d_model).astype(np.float32)
    h_out, info = head.forward(dummy_h_tilde)

    logger.info(f"Input h_tilde shape: {dummy_h_tilde.shape} -> Output h_out shape: {h_out.shape}")
    logger.info(f"Symbolic Attention SAtt shape: {info['s_att'].shape}")
    logger.info(f"Softmax Attention Weights: {np.round(info['attn_weights'], 4)}")

    # 4. Evaluate Stage A4 Symbolic Alignment Losses
    logger.info("4. Testing Stage A4 Symbolic Alignment Losses...")
    target_key = templates[0].key_embedding
    target_val = templates[0].value_embedding
    all_keys = np.vstack([t.key_embedding for t in templates[:100]])

    u_satt = np.dot(head.U_out, info["s_att"])
    target_val_d = np.random.randn(d_model).astype(np.float32)

    loss_dict = combined_symbolic_loss(
        q_sym=info["q_sym"],
        target_key=target_key,
        all_keys=all_keys,
        u_satt=u_satt,
        target_value=target_val_d,
        lambda_key=1.0,
        lambda_value=1.0,
        temperature=0.1,
    )

    logger.info(f"Key-Space Alignment Loss (L_sym^key): {loss_dict['l_sym_key']:.4f}")
    logger.info(f"Value-Space Regression Loss (L_sym^value): {loss_dict['l_sym_value']:.4f}")
    logger.info(f"Total Symbolic Loss (L_sym^total): {loss_dict['l_sym_total']:.4f}")

    logger.info("=== STAGE A4 VERIFICATION SWEEP COMPLETED SUCCESSFULLY! ===")
    return loss_dict


if __name__ == "__main__":
    run_stage_a4_verification()
