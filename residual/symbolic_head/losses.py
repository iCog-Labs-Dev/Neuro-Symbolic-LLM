"""
Stage A4 Symbolic Head Alignment Loss Functions.
Implements Key-Space Cross-Entropy Alignment and Value-Space Regression Loss.
"""

import numpy as np


def key_space_alignment_loss(
    q_sym: np.ndarray,
    target_key: np.ndarray,
    all_keys: np.ndarray,
    temperature: float = 0.1,
) -> float:
    """
    Compute Key-Space Alignment Loss (Log-Softmax Cross Entropy over template keys).

    :param q_sym: Projected query vector of shape (k,) or (batch, k).
    :param target_key: True target template key vector of shape (k,) or (batch, k).
    :param all_keys: Matrix of candidate template keys of shape (N, k).
    :param temperature: Softmax temperature scaling parameter tau.
    :return: Scalar alignment loss value.
    """
    q_sym = np.asarray(q_sym, dtype=np.float32)
    target_key = np.asarray(target_key, dtype=np.float32)
    all_keys = np.asarray(all_keys, dtype=np.float32)

    if q_sym.ndim == 1:
        pos_score = np.dot(q_sym, target_key) / temperature
        all_scores = np.dot(all_keys, q_sym) / temperature
        max_score = np.max(all_scores)
        log_sum_exp = max_score + np.log(np.sum(np.exp(all_scores - max_score)) + 1e-12)
        loss = float(log_sum_exp - pos_score)
    else:
        pos_scores = np.sum(q_sym * target_key, axis=-1) / temperature
        all_scores = np.dot(q_sym, all_keys.T) / temperature
        max_scores = np.max(all_scores, axis=-1, keepdims=True)
        log_sum_exp = np.squeeze(max_scores, axis=-1) + np.log(
            np.sum(np.exp(all_scores - max_scores), axis=-1) + 1e-12
        )
        loss = float(np.mean(log_sum_exp - pos_scores))

    return loss


def value_space_regression_loss(u_satt: np.ndarray, target_value: np.ndarray) -> float:
    """
    Compute Value-Space Regression Loss || U(h) - v_target ||^2.

    :param u_satt: Projected symbolic attention output in R^d.
    :param target_value: Ground-truth target semantic value vector in R^d.
    :return: Scalar MSE loss value.
    """
    u_satt = np.asarray(u_satt, dtype=np.float32)
    target_value = np.asarray(target_value, dtype=np.float32)
    diff = u_satt - target_value
    return float(np.mean(np.square(diff)))


def combined_symbolic_loss(
    q_sym: np.ndarray,
    target_key: np.ndarray,
    all_keys: np.ndarray,
    u_satt: np.ndarray,
    target_value: np.ndarray,
    lambda_key: float = 1.0,
    lambda_value: float = 1.0,
    temperature: float = 0.1,
) -> dict[str, float]:
    """
    Compute total Stage A4 Symbolic Head Loss.
    L_sym^total = lambda_key * L_sym^key + lambda_value * L_sym^value
    """
    l_key = key_space_alignment_loss(
        q_sym, target_key, all_keys, temperature=temperature
    )
    l_val = value_space_regression_loss(u_satt, target_value)
    l_total = lambda_key * l_key + lambda_value * l_val

    return {
        "l_sym_total": l_total,
        "l_sym_key": l_key,
        "l_sym_value": l_val,
    }
