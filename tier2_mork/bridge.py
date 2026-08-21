import numpy as np

from tier2_mork.client import MorkClient, MorkQueryResult, get_mork_client


class SymbolicHeadBridge:
    """Tier 1 -> Tier 2 Symbolic Head Integration Bridge."""

    def __init__(
        self,
        hidden_dim: int,
        key_dim: int = 256,
        top_m: int = 8,
        mork_client: MorkClient | None = None,
        tau: float = 1.0,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.top_m = top_m
        self.tau = tau
        self.mork_client = (
            mork_client if mork_client is not None else get_mork_client(key_dim=key_dim)
        )

        rng = np.random.default_rng(42)
        self.w_sym = (rng.normal(0.0, 0.02, (key_dim, hidden_dim))).astype(np.float32)
        self.b_sym = np.zeros(key_dim, dtype=np.float32)
        self.u_read = (rng.normal(0.0, 0.02, (hidden_dim, key_dim))).astype(np.float32)

    def project_query(self, hidden_states: np.ndarray) -> np.ndarray:
        """Project hidden states to k-dim query space q_sym = W_sym * h + b_sym."""
        h_arr = np.asarray(hidden_states, dtype=np.float32)
        return np.matmul(h_arr, self.w_sym.T) + self.b_sym

    def compute_symbolic_attention(
        self,
        q_sym: np.ndarray,
        p_keys: np.ndarray,
        v_vals: np.ndarray,
    ) -> np.ndarray:
        """Compute softmax symbolic attention SAtt over keys p_j and values v_j."""
        q_expanded = np.expand_dims(q_sym, axis=-2)
        logits = np.sum(q_expanded * p_keys, axis=-1) / self.tau

        max_logits = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        weights = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        weights_expanded = np.expand_dims(weights, axis=-1)
        return np.sum(weights_expanded * v_vals, axis=-2)

    def forward(
        self, hidden_states: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, MorkQueryResult]:
        """Execute symbolic head pass given Tier 1 hidden states."""
        h_arr = np.asarray(hidden_states, dtype=np.float32)
        original_shape = h_arr.shape
        flat_h = h_arr.reshape(-1, self.hidden_dim)

        q_sym_flat = self.project_query(flat_h)
        mork_result = self.mork_client.query_top_k(q_sym_flat, top_m=self.top_m)
        s_att_flat = self.compute_symbolic_attention(
            q_sym_flat,
            mork_result.keys,
            mork_result.values,
        )
        delta_h_flat = np.matmul(s_att_flat, self.u_read.T)
        integrated_flat = flat_h + delta_h_flat

        integrated_h = integrated_flat.reshape(original_shape)
        q_sym = q_sym_flat.reshape(original_shape[:-1] + (self.key_dim,))
        s_att = s_att_flat.reshape(original_shape[:-1] + (self.key_dim,))

        return integrated_h, q_sym, s_att, mork_result
