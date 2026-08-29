"""Symbolic head projection and retrieval update."""

from typing import Any

import numpy as np

from tier2_mork.bridge import Tier2Retrieve
from tier2_mork.client import MorkClient
from tier2_mork.comm import QueryPacket, TemplatePacket


class SymbolicHead:
    def __init__(
        self,
        mork_client: MorkClient,
        d_model: int = 768,
        k_dim: int = 256,
        top_m: int = 8,
        temperature: float = 0.1,
        layer_id: int = 0,
    ) -> None:
        self.d_model = d_model
        self.k_dim = k_dim
        self.top_m = top_m
        self.temperature = temperature
        self.layer_id = layer_id
        self.tier2 = Tier2Retrieve(mork_client=mork_client, key_dim=k_dim)
        self.mork_client = self.tier2.mork_client
        rng = np.random.default_rng(42 + layer_id)
        self.W_sym = rng.normal(0.0, 0.02, (k_dim, d_model)).astype(np.float32)
        self.b_sym = np.zeros(k_dim, dtype=np.float32)
        self.U_out = rng.normal(0.0, 0.02, (d_model, k_dim)).astype(np.float32)
        self.last_attn_weights: np.ndarray | None = None
        self.last_query_packet: QueryPacket | None = None
        self.last_template_packet: TemplatePacket | None = None

    def project_query(self, h_tilde: np.ndarray) -> np.ndarray:
        h_arr = np.asarray(h_tilde, dtype=np.float32)
        original_shape = h_arr.shape
        flat = h_arr.reshape(-1, self.d_model)
        q_flat = np.matmul(flat, self.W_sym.T) + self.b_sym
        if h_arr.ndim == 1:
            return q_flat.reshape(self.k_dim)
        return q_flat.reshape(original_shape[:-1] + (self.k_dim,))

    def compute_symbolic_attention(
        self, q_sym: np.ndarray, key_matrix: np.ndarray, value_matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        q_arr = np.asarray(q_sym, dtype=np.float32)
        keys = np.asarray(key_matrix, dtype=np.float32)
        values = np.asarray(value_matrix, dtype=np.float32)
        if keys.size == 0:
            return np.zeros(q_arr.shape, dtype=np.float32), np.zeros((0,), dtype=np.float32)

        squeeze = q_arr.ndim == 1
        if squeeze:
            q_arr = q_arr.reshape(1, -1)
            if keys.ndim == 2:
                keys = keys.reshape(1, keys.shape[0], keys.shape[1])
                values = values.reshape(1, values.shape[0], values.shape[1])

        q_expanded = np.expand_dims(q_arr, axis=-2)
        logits = np.sum(q_expanded * keys, axis=-1) / self.temperature
        max_logits = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        weights = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-12)
        s_att = np.sum(np.expand_dims(weights, axis=-1) * values, axis=-2)
        self.last_attn_weights = weights
        if squeeze:
            return s_att.reshape(-1), weights.reshape(-1)
        return s_att, weights

    def forward(self, h_tilde: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        h_arr = np.asarray(h_tilde, dtype=np.float32)
        squeeze_1d = h_arr.ndim == 1
        original_shape = h_arr.shape
        flat_h = h_arr.reshape(-1, self.d_model)

        q_flat = np.matmul(flat_h, self.W_sym.T) + self.b_sym
        retrieved, q_pkt, t_pkt = self.tier2.retrieve(q_flat, top_m=self.top_m)
        self.last_query_packet = q_pkt
        self.last_template_packet = t_pkt

        s_att_flat, weights = self.compute_symbolic_attention(
            q_flat, retrieved.keys, retrieved.values
        )
        delta = np.matmul(s_att_flat, self.U_out.T)
        h_out = (flat_h + delta).reshape(original_shape)
        q_sym = q_flat.reshape(original_shape[:-1] + (self.k_dim,))
        s_att = s_att_flat.reshape(original_shape[:-1] + (self.k_dim,))
        if squeeze_1d:
            h_out = h_out.reshape(self.d_model)
            q_sym = q_sym.reshape(self.k_dim)
            s_att = s_att.reshape(self.k_dim)
        info = {
            "q_sym": q_sym,
            "s_att": s_att,
            "attn_weights": weights,
            "matched_ids": retrieved.template_ids[0],
            "scores": retrieved.scores[0],
            "query_nbytes": q_pkt.nbytes,
            "template_nbytes": t_pkt.nbytes,
        }
        return h_out, info
