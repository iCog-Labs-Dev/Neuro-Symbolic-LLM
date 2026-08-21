"""
Stage A4 Symbolic Head Implementation.
Projects continuous residual hidden states into k-dimensional key-space, retrieves top-m Atomese templates,
computes softmax symbolic attention, and integrates semantic value vectors back into the continuous stream.
"""

import logging
from typing import Any

import numpy as np

from tier2_mork.client import MorkClient, get_mork_client

logger = logging.getLogger(__name__)


class SymbolicHead:
    """
    Stage A4 Basic Symbolic Head Module.
    Connects Tier 1 GPU continuous activations with Tier 2 CPU hypergraph templates.
    """

    def __init__(
        self,
        d_model: int = 4096,
        k_dim: int = 256,
        top_m: int = 8,
        temperature: float = 0.1,
        layer_id: int = 0,
        mork_client: MorkClient | None = None,
    ) -> None:
        self.d_model = d_model
        self.k_dim = k_dim
        self.top_m = top_m
        self.temperature = temperature
        self.layer_id = layer_id
        self.mork_client = (
            mork_client if mork_client is not None else get_mork_client(key_dim=k_dim)
        )

        # Projection weights: W_sym in R^{k x d}, b_sym in R^k
        np.random.seed(42 + layer_id)
        self.W_sym = (np.random.randn(k_dim, d_model) * 0.01).astype(np.float32)
        self.b_sym = np.zeros((k_dim,), dtype=np.float32)

        # Output integration matrix: U in R^{d x k}
        self.U_out = (np.random.randn(d_model, k_dim) * 0.01).astype(np.float32)

    def project_query(self, h_tilde: np.ndarray) -> np.ndarray:
        """
        Project settled hidden state h_tilde in R^d to symbolic query q_sym in R^k.
        q_sym = W_sym * h_tilde + b_sym
        """
        h_arr = np.asarray(h_tilde, dtype=np.float32)
        if h_arr.ndim == 1:
            q_sym = np.dot(self.W_sym, h_arr) + self.b_sym
        else:
            q_sym = np.dot(h_arr, self.W_sym.T) + self.b_sym
        return np.asarray(q_sym, dtype=np.float32)

    def compute_symbolic_attention(
        self, q_sym: np.ndarray, key_matrix: np.ndarray, value_matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute softmax symbolic attention summary SAtt in R^k over retrieved templates.

        :param q_sym: Query vector of shape (k,) or (batch, k)
        :param key_matrix: Matrix of shape (m, k) containing top-m template keys.
        :param value_matrix: Matrix of shape (m, k) containing top-m template values.
        :return: Tuple of (SAtt, attn_weights)
        """
        if key_matrix.shape[0] == 0:
            shape = q_sym.shape
            return np.zeros(shape, dtype=np.float32), np.zeros((0,), dtype=np.float32)

        if q_sym.ndim == 1:
            scores = np.dot(key_matrix, q_sym) / self.temperature
            attn_weights = np.exp(scores - np.max(scores))
            attn_weights /= np.sum(attn_weights) + 1e-12
            s_att = np.dot(attn_weights, value_matrix)
        else:
            scores = np.dot(q_sym, key_matrix.T) / self.temperature
            scores_max = np.max(scores, axis=-1, keepdims=True)
            attn_weights = np.exp(scores - scores_max)
            attn_weights /= np.sum(attn_weights, axis=-1, keepdims=True) + 1e-12
            s_att = np.dot(attn_weights, value_matrix)

        return s_att, attn_weights

    def forward(self, h_tilde: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Full Stage A4 Symbolic Head forward pass."""
        q_sym = self.project_query(h_tilde)
        q_sym_2d = np.atleast_2d(q_sym)

        mork_res = self.mork_client.query_top_k(q_sym_2d, top_m=self.top_m)
        keys = mork_res.keys[0] if mork_res.keys.ndim == 3 else mork_res.keys
        values = mork_res.values[0] if mork_res.values.ndim == 3 else mork_res.values

        s_att, attn_weights = self.compute_symbolic_attention(q_sym, keys, values)

        if s_att.ndim == 1:
            residual_delta = np.dot(self.U_out, s_att)
        else:
            residual_delta = np.dot(s_att, self.U_out.T)

        h_out = h_tilde + residual_delta

        info = {
            "q_sym": q_sym,
            "s_att": s_att,
            "attn_weights": attn_weights,
            "matched_ids": mork_res.template_ids[0],
            "scores": mork_res.scores[0],
        }
        return h_out, info
