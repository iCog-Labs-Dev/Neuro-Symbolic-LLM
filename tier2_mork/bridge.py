"""Tier 1 -> Tier 2 Integration Bridge (SymbolicHeadBridge).

Provides a clean, stable interface for Tier 1 colleagues to integrate Tier 2 MORK symbolic
retrieval and symbolic attention seamlessly into their base LLM forward loop.

Research Traceability:
- Projects hidden states h_tilde to k-dim query: q_sym = W_sym * h_tilde + b_sym
- Queries Tier 2 MORK client for top-m templates: P_i = TopK_MORK(q_sym)
- Computes symbolic attention over retrieved keys p_j and values v_j: SAtt
- Back-projects to hidden dimension d: U * SAtt
"""

import numpy as np

from tier2_mork.client import MorkClient, MorkQueryResult, get_mork_client


class SymbolicHeadBridge:
    """Stable integration bridge for Tier 1 model forward pass."""

    def __init__(
        self,
        hidden_dim: int,
        key_dim: int = 256,
        top_m: int = 8,
        mork_client: MorkClient | None = None,
        tau: float = 1.0,
    ) -> None:
        """Initialize SymbolicHeadBridge.

        Args:
            hidden_dim: Model hidden state dimension d (e.g. 768 for GPT-2 117M)
            key_dim: Symbolic projection dimension k (e.g. 256)
            top_m: Number of MORK template keys to retrieve per token position
            mork_client: Active MorkClient instance (defaults to auto-detected client)
            tau: Softmax temperature parameter for symbolic attention
        """
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.top_m = top_m
        self.tau = tau
        self.mork_client = mork_client if mork_client is not None else get_mork_client(key_dim=key_dim)

        # Projection parameters (W_sym: k x d, b_sym: k, U_read: d x k)
        # Seeded for reproducible initial behavior; updated during fine-tuning
        rng = np.random.default_rng(42)
        self.w_sym = (rng.normal(0.0, 0.02, (key_dim, hidden_dim))).astype(np.float32)
        self.b_sym = np.zeros(key_dim, dtype=np.float32)
        self.u_read = (rng.normal(0.0, 0.02, (hidden_dim, key_dim))).astype(np.float32)

    def project_query(self, hidden_states: np.ndarray) -> np.ndarray:
        """Project continuous hidden states to k-dim query space q_sym.

        Args:
            hidden_states: Array of shape (..., d), float32

        Returns:
            q_sym Array of shape (..., k), float32
        """
        h_arr = np.asarray(hidden_states, dtype=np.float32)
        q_sym = np.matmul(h_arr, self.w_sym.T) + self.b_sym
        return q_sym

    def compute_symbolic_attention(
        self,
        q_sym: np.ndarray,
        p_keys: np.ndarray,
        v_vals: np.ndarray,
    ) -> np.ndarray:
        """Compute symbolic attention SAtt over retrieved template keys and values.

        Args:
            q_sym: Query array of shape (N, k) or (..., k)
            p_keys: Matched key array of shape (N, m, k) or (..., m, k)
            v_vals: Matched value array of shape (N, m, k) or (..., m, k)

        Returns:
            SAtt array of shape (N, k) or (..., k)
        """
        # q_sym: (..., 1, k) * p_keys: (..., m, k) -> sum over k -> logits: (..., m)
        q_expanded = np.expand_dims(q_sym, axis=-2)
        logits = np.sum(q_expanded * p_keys, axis=-1) / self.tau  # (..., m)

        # Stable softmax
        max_logits = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        weights = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)  # (..., m)

        # Weighted sum over values
        weights_expanded = np.expand_dims(weights, axis=-1)  # (..., m, 1)
        s_att = np.sum(weights_expanded * v_vals, axis=-2)  # (..., k)

        return s_att

    def forward(self, hidden_states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, MorkQueryResult]:
        """Execute full Tier 2 symbolic head pass given Tier 1 hidden states.

        Args:
            hidden_states: Continuous hidden states from Tier 1 of shape (batch, seq, d) or (N, d)

        Returns:
            Tuple of:
            - integrated_h: Updated hidden states of shape matching input (..., d)
            - q_sym: Projected queries of shape (..., k)
            - s_att: Symbolic attention vectors of shape (..., k)
            - mork_result: MorkQueryResult containing retrieved keys, values, IDs, and scores
        """
        h_arr = np.asarray(hidden_states, dtype=np.float32)
        original_shape = h_arr.shape
        flat_h = h_arr.reshape(-1, self.hidden_dim)  # (N, d)

        # 1. Project query: q_sym = W_sym * h + b_sym
        q_sym_flat = self.project_query(flat_h)  # (N, k)

        # 2. Query Tier 2 MORK Sparse Symbolic Engine
        mork_result = self.mork_client.query_top_k(q_sym_flat, top_m=self.top_m)

        # 3. Compute Symbolic Attention SAtt over keys and values
        s_att_flat = self.compute_symbolic_attention(
            q_sym_flat,
            mork_result.keys,
            mork_result.values,
        )  # (N, k)

        # 4. Back-project to hidden dimension d: U * SAtt
        delta_h_flat = np.matmul(s_att_flat, self.u_read.T)  # (N, d)

        # 5. Integrate into continuous hidden stream: h_out = h + U * SAtt
        integrated_flat = flat_h + delta_h_flat

        # Reshape to original dimensions
        integrated_h = integrated_flat.reshape(original_shape)
        q_sym = q_sym_flat.reshape(original_shape[:-1] + (self.key_dim,))
        s_att = s_att_flat.reshape(original_shape[:-1] + (self.key_dim,))

        return integrated_h, q_sym, s_att, mork_result
