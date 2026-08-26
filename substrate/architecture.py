"""Automatic architecture detection for frozen LLM substrates.

Detects the model family, transformer-block count and hidden size directly
from the loaded parameter PyTree, so that GPT-2 and Pythia/GPT-NeoX models of
any size are supported without hardcoding layer counts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Architecture:
    """Normalized description of a frozen causal-LM substrate."""

    model_family: str
    num_layers: int
    hidden_size: int
    num_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int | None = None
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rotary_pct: float = 1.0
    use_parallel_residual: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)


def _top_keys(params: Any) -> set[str]:
    if isinstance(params, Mapping):
        return set(params.keys())
    return set()


def _family_from_keys(keys: set[str]) -> str | None:
    # Family is determined by the presence of the family container keys;
    # other top-level keys (e.g. a shared 'lm_head') may be present in both.
    if "transformer" in keys:
        return "gpt2"
    if "gpt_neox" in keys:
        return "neox"
    return None


def _get_path(params: Any, *parts: str) -> Any:
    node: Any = params
    for part in parts:
        node = node[part]
    return node


def _config_value(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    return getattr(config, name, default)


def detect_architecture(params: Any, config: Any = None) -> Architecture:
    """Inspect the parameter tree (and optional HF config) to auto-detect the
    architecture.

    Raises ``ValueError`` for unsupported parameter layouts.
    """
    keys = _top_keys(params)
    family = _family_from_keys(keys)
    if family is None:
        raise ValueError(
            "Unsupported model architecture. Expected GPT-2 (params with "
            f"'transformer'/'lm_head' keys) or GPT-NeoX/Pythia (params with "
            f"'gpt_neox'/'embed_out' keys). Found top-level keys: {sorted(keys)}"
        )

    if family == "gpt2":
        wte_weight = _get_path(params, "transformer", "wte", "weight")
        wpe_weight = _get_path(params, "transformer", "wpe", "weight")
        blocks = _get_path(params, "transformer", "h")
        vocab_size = wte_weight.shape[0]
        hidden_size = wte_weight.shape[1]
        max_positions = wpe_weight.shape[0]
        num_heads = int(_config_value(config, "n_head", max(1, hidden_size // 64)))
    else:  # neox
        embed_weight = _get_path(params, "gpt_neox", "embed_in", "weight")
        blocks = _get_path(params, "gpt_neox", "layers")
        vocab_size = embed_weight.shape[0]
        hidden_size = embed_weight.shape[1]
        max_positions = int(_config_value(config, "max_position_embeddings", 0)) or None
        num_heads = int(
            _config_value(config, "num_attention_heads", max(1, hidden_size // 64))
        )

    if not isinstance(blocks, Mapping) and not isinstance(blocks, Sequence):
        raise ValueError(
            f"Cannot discover transformer blocks: unexpected {type(blocks)}"
        )

    num_layers = len(blocks)
    head_dim = hidden_size // num_heads
    if head_dim * num_heads != hidden_size:
        raise ValueError(
            f"hidden_size={hidden_size} is not divisible by num_heads={num_heads}"
        )

    return Architecture(
        model_family=family,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        max_position_embeddings=max_positions,
        layer_norm_eps=float(_config_value(config, "layer_norm_eps", 1e-5)),
        rope_theta=float(_config_value(config, "rope_theta", 10000.0)),
        rotary_pct=float(_config_value(config, "rotary_pct", 1.0)),
        use_parallel_residual=bool(
            _config_value(config, "use_parallel_residual", False)
        ),
    )


def discover_layers(params: Any, config: Any = None) -> int:
    """Return the number of transformer blocks discovered from the params."""
    return detect_architecture(params, config).num_layers


def validate_interception_layers(
    intercept_layers: Sequence[int] | None, num_layers: int
) -> tuple[int, ...]:
    """Validate a list of zero-based layer indices.

    Rejects negative indices, out-of-range indices and duplicates. ``None`` or
    an empty sequence is allowed and means "no interception".
    """
    if intercept_layers is None:
        return ()

    layers = tuple(int(i) for i in intercept_layers)
    if not layers:
        return ()

    seen: set[int] = set()
    for i in layers:
        if i < 0:
            raise ValueError(
                f"Invalid interception layer {i}: layer indices must be "
                f"non-negative (zero-based)."
            )
        if i >= num_layers:
            raise ValueError(
                f"Invalid interception layer {i}: model has only "
                f"{num_layers} transformer layers (zero-based indices "
                f"0..{num_layers - 1})."
            )
        if i in seen:
            raise ValueError(
                f"Duplicate interception layer {i}: each layer index may "
                f"appear at most once."
            )
        seen.add(i)

    return tuple(sorted(layers))
