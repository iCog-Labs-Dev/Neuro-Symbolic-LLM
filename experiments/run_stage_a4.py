"""A4 path: T1 q=Wh+b, T2 wire retrieve, T1 SAtt and h+U SAtt, key/value losses.

Requires a live MORK Docker container:
    docker compose up -d --build
"""

import logging

import numpy as np

from Residual.symbolic_head.head import SymbolicHead
from Residual.symbolic_head.losses import combined_symbolic_loss
from Residual.symbolic_head.mork_client import get_mork_client

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_stage_a4")


def run_stage_a4_verification() -> None:
    hidden_dim = 768
    key_dim = 256
    top_m = 4
    num_templates = 10
    tau = 0.1

    client = get_mork_client(key_dim=key_dim)
    rng = np.random.default_rng(42)
    target_key = None
    target_val = None
    all_keys = []
    for i in range(num_templates):
        k_vec = rng.normal(0, 1, key_dim).astype(np.float32)
        v_vec = rng.normal(0, 1, key_dim).astype(np.float32)
        client.add_template(f"tpl_{i:04d}", f"(Concept (Node_{i}))", k_vec, v_vec)
        all_keys.append(k_vec)
        if i == 0:
            target_key, target_val = k_vec, v_vec
    all_keys_mat = np.stack(all_keys, axis=0)

    head = SymbolicHead(
        mork_client=client,
        d_model=hidden_dim,
        k_dim=key_dim,
        top_m=top_m,
        temperature=tau,
    )
    h = rng.normal(0, 1, (2, 4, hidden_dim)).astype(np.float32)
    h_out, info = head.forward(h)

    q0 = info["q_sym"].reshape(-1, key_dim)[0]
    u_satt = np.matmul(info["s_att"].reshape(-1, key_dim)[0], head.U_out.T)
    losses = combined_symbolic_loss(
        q_sym=q0,
        target_key=target_key,
        all_keys=all_keys_mat,
        u_satt=u_satt,
        target_value=np.matmul(target_val, head.U_out.T),
        temperature=tau,
    )

    logger.info("index=%s", client.index_backend)
    logger.info("h %s -> q_sym %s", h.shape, info["q_sym"].shape)
    logger.info("query packet bytes=%s template packet bytes=%s", info["query_nbytes"], info["template_nbytes"])
    logger.info("SAtt %s h_out %s", info["s_att"].shape, h_out.shape)
    logger.info("top-%s ids[0]=%s", top_m, info["matched_ids"])
    logger.info("L_key=%.6f L_value=%.6f L_total=%.6f", losses["l_sym_key"], losses["l_sym_value"], losses["l_sym_total"])
    logger.info("A4 hops: T1 project | T2 retrieve | T1 SAtt+U | losses")


if __name__ == "__main__":
    run_stage_a4_verification()
