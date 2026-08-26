"""Pure JAX forward implementations for GPT-2 and GPT-NeoX/Pythia.

The forward passes are written as pure functions over a Flax-convention
parameter PyTree (a nested mapping of ``jax.Array`` leaves). Parameter names
match HuggingFace checkpoint names so that weights can be converted without
renaming. No parameters are mutated anywhere: ``hidden + 0.0`` style identity
interception is fully JIT-trace-safe.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp

from .architecture import Architecture

# ── shared primitives ──────────────────────────────────────────────────────


def layer_norm(
    x: jax.Array,
    weight: jax.Array,
    bias: jax.Array | None,
    eps: float,
) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    x = (x - mean) * jax.lax.rsqrt(variance + eps)
    x = x * weight
    if bias is not None:
        x = x + bias
    return x


def _mlp(
    hidden: jax.Array,
    fc_w: jax.Array,
    fc_b: jax.Array,
    proj_w: jax.Array,
    proj_b: jax.Array,
    linear_transpose: bool = True,
    gelu_approximate: bool = True,
) -> jax.Array:
    # GPT-2 uses "gelu_new" (tanh approximation); GPT-NeoX/Pythia uses the
    # exact erf-based gelu. Using the wrong one diverges on real checkpoints.
    def act(x: jax.Array) -> jax.Array:
        return jax.nn.gelu(x, approximate=gelu_approximate)

    if linear_transpose:
        # standard nn.Linear layout: weight is [output, input]
        intermediate = act(hidden @ fc_w.T + fc_b)
        return intermediate @ proj_w.T + proj_b
    # GPT-2 Conv1D layout: weight is [input, output], forward is x @ w
    intermediate = act(hidden @ fc_w + fc_b)
    return intermediate @ proj_w + proj_b


def _causal_mask(seq_len: int) -> jax.Array:
    return jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))[None, None, :, :]


# ── GPT-2 ──────────────────────────────────────────────────────────────────


def gpt2_embed(params: Mapping[str, Any], input_ids: jax.Array) -> jax.Array:
    wte = params["wte"]["weight"]
    wpe = params["wpe"]["weight"]
    positions = jnp.arange(input_ids.shape[1])[None, :]
    return wte[input_ids] + wpe[positions]


def gpt2_attention(
    hidden: jax.Array,
    c_attn_w: jax.Array,
    c_attn_b: jax.Array,
    c_proj_w: jax.Array,
    c_proj_b: jax.Array,
    num_heads: int,
) -> jax.Array:
    batch, seq_len, _ = hidden.shape
    head_dim = hidden.shape[-1] // num_heads
    qkv = hidden @ c_attn_w + c_attn_b
    qkv = qkv.reshape(batch, seq_len, 3, num_heads, head_dim)
    qkv = qkv.transpose(2, 0, 3, 1, 4)  # [3, B, H, T, d]
    q, k, v = qkv[0], qkv[1], qkv[2]
    scores = jnp.einsum("bhtd,bhsd->bhts", q, k) * (head_dim**-0.5)
    scores = jnp.where(_causal_mask(seq_len), scores, jnp.finfo(scores.dtype).min)
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhts,bhsd->bhtd", weights, v)
    out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
    return out @ c_proj_w + c_proj_b


def gpt2_block(
    params: Mapping[str, Any], idx: int, hidden: jax.Array, arch: Architecture
) -> jax.Array:
    layer = params["h"][idx]
    ln_1 = layer["ln_1"]
    normed = layer_norm(hidden, ln_1["weight"], ln_1["bias"], arch.layer_norm_eps)
    attn = gpt2_attention(
        normed,
        layer["attn"]["c_attn"]["weight"],
        layer["attn"]["c_attn"]["bias"],
        layer["attn"]["c_proj"]["weight"],
        layer["attn"]["c_proj"]["bias"],
        arch.num_heads,
    )
    hidden = hidden + attn
    ln_2 = layer["ln_2"]
    normed = layer_norm(hidden, ln_2["weight"], ln_2["bias"], arch.layer_norm_eps)
    mlp = _mlp(
        normed,
        layer["mlp"]["c_fc"]["weight"],
        layer["mlp"]["c_fc"]["bias"],
        layer["mlp"]["c_proj"]["weight"],
        layer["mlp"]["c_proj"]["bias"],
        linear_transpose=False,
    )
    return hidden + mlp


def gpt2_lm_head(
    params: Mapping[str, Any], hidden: jax.Array, arch: Architecture
) -> jax.Array:
    ln_f = params["ln_f"]
    hidden = layer_norm(hidden, ln_f["weight"], ln_f["bias"], arch.layer_norm_eps)
    if "lm_head" in params and "weight" in params["lm_head"]:
        head_w = params["lm_head"]["weight"]
    else:
        head_w = params["wte"]["weight"]
    return hidden @ head_w.T


# ── GPT-NeoX / Pythia ──────────────────────────────────────────────────────


def neox_embed(params: Mapping[str, Any], input_ids: jax.Array) -> jax.Array:
    return params["embed_in"]["weight"][input_ids]


def _neox_rope(
    q: jax.Array,
    k: jax.Array,
    position_ids: jax.Array,
    arch: Architecture,
) -> tuple[jax.Array, jax.Array]:
    rotary_dim = int(arch.head_dim * arch.rotary_pct)
    inv_freq = 1.0 / (
        arch.rope_theta
        ** (jnp.arange(0, rotary_dim, 2).astype(jnp.float32) / rotary_dim)
    )
    pos = position_ids.astype(jnp.float32)
    freqs = pos[:, :, None] * inv_freq[None, None, :]  # [B, T, rotary_dim/2]
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # [B, T, rotary_dim]
    cos = jnp.cos(emb)[:, None, :, :]  # [B, 1, T, rotary_dim]
    sin = jnp.sin(emb)[:, None, :, :]

    def _rotate(x: jax.Array) -> jax.Array:
        rot = x[..., :rotary_dim]
        rest = x[..., rotary_dim:]
        half = rotary_dim // 2
        x1 = rot[..., :half]
        x2 = rot[..., half:]
        rotated = jnp.concatenate([-x2, x1], axis=-1)
        return jnp.concatenate([rot * cos + rotated * sin, rest], axis=-1)

    return _rotate(q), _rotate(k)


def neox_attention(
    hidden: jax.Array,
    params: Mapping[str, Any],
    position_ids: jax.Array,
    arch: Architecture,
) -> jax.Array:
    batch, seq_len, _ = hidden.shape
    qkv_w = params["query_key_value"]["weight"]
    qkv_b = params["query_key_value"]["bias"]
    dense_w = params["dense"]["weight"]
    dense_b = params["dense"]["bias"]
    head_dim = arch.head_dim

    qkv = hidden @ qkv_w.T + qkv_b  # [B, T, 3D]
    qkv = qkv.reshape(batch, seq_len, arch.num_heads, 3 * head_dim).transpose(
        0, 2, 1, 3
    )
    q, k, v = (
        qkv[..., :head_dim],
        qkv[..., head_dim : 2 * head_dim],
        qkv[..., 2 * head_dim :],
    )
    q, k = _neox_rope(q, k, position_ids, arch)
    scores = jnp.einsum("bhtd,bhsd->bhts", q, k) * (head_dim**-0.5)
    scores = jnp.where(_causal_mask(seq_len), scores, jnp.finfo(scores.dtype).min)
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhts,bhsd->bhtd", weights, v)
    out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
    return out @ dense_w.T + dense_b


def neox_block(
    params: Mapping[str, Any],
    idx: int,
    hidden: jax.Array,
    position_ids: jax.Array,
    arch: Architecture,
) -> jax.Array:
    layer = params["layers"][idx]
    ln_in = layer["input_layernorm"]
    ln_post = layer["post_attention_layernorm"]

    # NOTE: GPT-NeoX LayerNorms have affine weight AND bias; real checkpoints
    # carry trained non-zero biases, so they must never be dropped.
    attn_in = layer_norm(
        hidden, ln_in["weight"], ln_in.get("bias"), arch.layer_norm_eps
    )
    attn_out = neox_attention(attn_in, layer["attention"], position_ids, arch)
    if arch.use_parallel_residual:
        mlp_in = layer_norm(
            hidden, ln_post["weight"], ln_post.get("bias"), arch.layer_norm_eps
        )
        mlp_out = _mlp(
            mlp_in,
            layer["mlp"]["dense_h_to_4h"]["weight"],
            layer["mlp"]["dense_h_to_4h"]["bias"],
            layer["mlp"]["dense_4h_to_h"]["weight"],
            layer["mlp"]["dense_4h_to_h"]["bias"],
            gelu_approximate=False,
        )
        return hidden + attn_out + mlp_out
    hidden = hidden + attn_out
    mlp_in = layer_norm(
        hidden, ln_post["weight"], ln_post.get("bias"), arch.layer_norm_eps
    )
    mlp_out = _mlp(
        mlp_in,
        layer["mlp"]["dense_h_to_4h"]["weight"],
        layer["mlp"]["dense_h_to_4h"]["bias"],
        layer["mlp"]["dense_4h_to_h"]["weight"],
        layer["mlp"]["dense_4h_to_h"]["bias"],
        gelu_approximate=False,
    )
    return hidden + mlp_out


def neox_lm_head(
    params: Mapping[str, Any], hidden: jax.Array, arch: Architecture
) -> jax.Array:
    gpt_neox = params["gpt_neox"]
    final_norm = gpt_neox["final_layer_norm"]
    hidden = layer_norm(
        hidden,
        final_norm["weight"],
        final_norm.get("bias"),
        arch.layer_norm_eps,
    )
    if "lm_head" in params and "weight" in params["lm_head"]:
        head_w = params["lm_head"]["weight"]
    elif "embed_out" in gpt_neox and "weight" in gpt_neox["embed_out"]:
        head_w = gpt_neox["embed_out"]["weight"]
    else:
        head_w = gpt_neox["embed_in"]["weight"]
    return hidden @ head_w.T


# ── generic driver ─────────────────────────────────────────────────────────


def run_embeddings(
    params: Mapping[str, Any], arch: Architecture, input_ids: jax.Array
) -> jax.Array:
    if arch.model_family == "gpt2":
        return gpt2_embed(params["transformer"], input_ids)
    if arch.model_family == "neox":
        return neox_embed(params["gpt_neox"], input_ids)
    raise ValueError(f"Unsupported model family: {arch.model_family}")


def _run_block(
    params: Mapping[str, Any],
    arch: Architecture,
    idx: int,
    hidden: jax.Array,
    position_ids: jax.Array,
) -> jax.Array:
    if arch.model_family == "gpt2":
        return gpt2_block(params["transformer"], idx, hidden, arch)
    if arch.model_family == "neox":
        return neox_block(params["gpt_neox"], idx, hidden, position_ids, arch)
    raise ValueError(f"Unsupported model family: {arch.model_family}")


def run_transformer_blocks(
    params: Mapping[str, Any],
    arch: Architecture,
    hidden: jax.Array,
    intercept_layers: Sequence[int],
    hook: Callable[[jax.Array, int], jax.Array],
    position_ids: jax.Array,
) -> tuple[jax.Array, dict[int, jax.Array]]:
    """Run the transformer blocks sequentially, caching and optionally
    modifying hidden states at the requested zero-based layer indices."""
    intercept_set = set(intercept_layers)
    intermediates: dict[int, jax.Array] = {}
    for idx in range(arch.num_layers):
        hidden = _run_block(params, arch, idx, hidden, position_ids)
        if idx in intercept_set:
            cached = hidden
            hidden = hook(cached, idx)
            intermediates[idx] = cached
    return hidden, intermediates


def run_lm_head(
    params: Mapping[str, Any], arch: Architecture, hidden: jax.Array
) -> jax.Array:
    if arch.model_family == "gpt2":
        return gpt2_lm_head(params["transformer"], hidden, arch)
    if arch.model_family == "neox":
        return neox_lm_head(params, hidden, arch)
    raise ValueError(f"Unsupported model family: {arch.model_family}")
