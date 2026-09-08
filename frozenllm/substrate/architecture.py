"""Automatic architecture detection for frozen LLM substrates.

Detects the model family, transformer-block count and hidden size directly
from the loaded parameter PyTree, so that GPT-2 and Pythia/GPT-NeoX models of
any size are supported without hardcoding layer counts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


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


def _family_from_config(config: Any) -> str | None:
    """Detect model family from an HF config object or mapping."""
    if config is None:
        return None

    model_type = _config_value(config, "model_type", None)
    if isinstance(model_type, str) and model_type:
        model_type_lower = model_type.lower()
        if model_type_lower == "gpt2":
            return "gpt2"
        if model_type_lower in ("gpt_neox", "neox", "pythia"):
            return "neox"

    architectures = _config_value(config, "architectures", None)
    if architectures and isinstance(architectures, list | tuple):
        for arch_name in architectures:
            if not isinstance(arch_name, str):
                continue
            arch_lower = arch_name.lower()
            if "gpt2" in arch_lower:
                return "gpt2"
            if "neox" in arch_lower or "pythia" in arch_lower:
                return "neox"

    # Key/attribute heuristics
    if (
        _config_value(config, "n_layer", None) is not None
        and _config_value(config, "n_embd", None) is not None
    ):
        return "gpt2"
    if (
        _config_value(config, "use_parallel_residual", None) is not None
        or _config_value(config, "rotary_pct", None) is not None
        or _config_value(config, "rope_parameters", None) is not None
    ):
        return "neox"

    return None


def _extract_rope_params(config: Any) -> tuple[float, float]:
    """Extract rope_theta and rotary_pct respecting HF's rope_parameters schema."""
    rope_params = _config_value(config, "rope_parameters", None)
    if not isinstance(rope_params, Mapping):
        rope_params = {}

    theta_val = _config_value(
        config,
        "rope_theta",
        _config_value(config, "rotary_emb_base", rope_params.get("base", 10000.0)),
    )
    rope_theta = float(theta_val if theta_val is not None else 10000.0)

    pct_val = _config_value(
        config,
        "rotary_pct",
        rope_params.get("partial_rotary_factor", 1.0),
    )
    rotary_pct = float(pct_val if pct_val is not None else 1.0)

    return rope_theta, rotary_pct


def _get_path(params: Any, *parts: str) -> Any:
    node: Any = params
    for part in parts:
        node = node[part]
    return node


def _config_value(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _detect_architecture_from_params(params: Any, config: Any = None) -> Architecture:
    """Inspect parameter tree to auto-detect architecture."""
    keys = _top_keys(params)
    family = _family_from_keys(keys)
    if family is None:
        if config is not None:
            return detect_architecture_from_config(config)
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
        rope_theta=_extract_rope_params(config)[0],
        rotary_pct=_extract_rope_params(config)[1],
        use_parallel_residual=bool(
            _config_value(config, "use_parallel_residual", False)
        ),
    )


def detect_architecture_from_config(config: Any) -> Architecture:
    """Detect architecture directly from an HF config object or dictionary.

    Args:
        config: HuggingFace PretrainedConfig, dict, or duck-typed config object.

    Returns:
        Architecture dataclass describing the model layout.

    Raises:
        ValueError: If config is invalid or unsupported.
    """
    family = _family_from_config(config)
    if family is None:
        model_type = _config_value(config, "model_type", None)
        raise ValueError(
            f"Unsupported or unrecognized model configuration. Expected GPT-2 or "
            f"GPT-NeoX/Pythia config, but got model_type={model_type!r}."
        )

    if family == "gpt2":
        num_layers_raw = _config_value(
            config, "n_layer", _config_value(config, "num_hidden_layers", None)
        )
        hidden_size_raw = _config_value(
            config, "n_embd", _config_value(config, "hidden_size", None)
        )
        num_heads_raw = _config_value(
            config, "n_head", _config_value(config, "num_attention_heads", None)
        )
        vocab_size = int(_config_value(config, "vocab_size", 50257))
        max_positions = _config_value(
            config,
            "n_positions",
            _config_value(config, "max_position_embeddings", 1024),
        )
        layer_norm_eps = float(
            _config_value(
                config,
                "layer_norm_epsilon",
                _config_value(config, "layer_norm_eps", 1e-5),
            )
        )
        rope_theta = float(_config_value(config, "rope_theta", 10000.0))
        rotary_pct = float(_config_value(config, "rotary_pct", 1.0))
        use_parallel_residual = bool(
            _config_value(config, "use_parallel_residual", False)
        )
    else:  # neox
        num_layers_raw = _config_value(
            config, "num_hidden_layers", _config_value(config, "n_layer", None)
        )
        hidden_size_raw = _config_value(
            config, "hidden_size", _config_value(config, "n_embd", None)
        )
        num_heads_raw = _config_value(
            config, "num_attention_heads", _config_value(config, "n_head", None)
        )
        vocab_size = int(_config_value(config, "vocab_size", 50304))
        max_positions = _config_value(
            config,
            "max_position_embeddings",
            _config_value(config, "n_positions", 2048),
        )
        layer_norm_eps = float(
            _config_value(
                config,
                "layer_norm_eps",
                _config_value(config, "layer_norm_epsilon", 1e-5),
            )
        )
        rope_theta, rotary_pct = _extract_rope_params(config)
        use_parallel_residual = bool(
            _config_value(config, "use_parallel_residual", True)
        )

    if num_layers_raw is None:
        raise ValueError(
            f"Cannot discover layer count from config for family {family!r}."
        )
    num_layers = int(num_layers_raw)
    if num_layers <= 0:
        raise ValueError(f"Invalid num_layers in config: {num_layers}")

    if hidden_size_raw is None:
        raise ValueError(
            f"Cannot discover hidden size from config for family {family!r}."
        )
    hidden_size = int(hidden_size_raw)
    if hidden_size <= 0:
        raise ValueError(f"Invalid hidden_size in config: {hidden_size}")

    if num_heads_raw is not None:
        num_heads = int(num_heads_raw)
    else:
        num_heads = max(1, hidden_size // 64)

    if num_heads <= 0:
        raise ValueError(f"Invalid num_heads in config: {num_heads}")

    head_dim = hidden_size // num_heads
    if head_dim * num_heads != hidden_size:
        raise ValueError(
            f"hidden_size={hidden_size} is not divisible by num_heads={num_heads}"
        )

    max_position_embeddings = int(max_positions) if max_positions is not None else None

    return Architecture(
        model_family=family,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        layer_norm_eps=layer_norm_eps,
        rope_theta=rope_theta,
        rotary_pct=rotary_pct,
        use_parallel_residual=use_parallel_residual,
    )


def detect_architecture(target: Any, config: Any = None) -> Architecture:
    """Unified detector: auto-detects architecture from HF config, nn.Module, or parameter mapping.

    Args:
        target: HuggingFace PretrainedConfig, dict, torch.nn.Module, or parameter mapping.
        config: Optional HuggingFace configuration when target is a parameter mapping or module.

    Returns:
        Architecture dataclass describing the model layout.

    Raises:
        ValueError: If architecture is unsupported or unrecognized.
    """
    if config is not None and not isinstance(target, Mapping):
        return detect_architecture_from_config(config)
    if hasattr(target, "config") and target.config is not None:
        return detect_architecture_from_config(target.config)
    if _family_from_config(target) is not None:
        return detect_architecture_from_config(target)
    return _detect_architecture_from_params(target, config=config)


def discover_layers(target: Any, config: Any = None) -> int:
    """Return the number of transformer blocks discovered from target (config, model, or params)."""
    return detect_architecture(target, config).num_layers


# Backward compatibility alias
discover_layers_from_config = discover_layers


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


def _unwrap_model(model: Any) -> Any:
    """Unwrap wrapper layers (e.g. DistributedDataParallel) if present."""
    while hasattr(model, "module"):
        model = model.module
    return model


def get_block_accessor(
    model: torch.nn.Module, arch: Architecture
) -> Callable[[int], torch.nn.Module]:
    """Return a function that retrieves block `i` from `model`.

    For GPT-2: `lambda i: model.transformer.h[i]`
    For NeoX:  `lambda i: model.gpt_neox.layers[i]`

    Args:
        model: PyTorch causal LM instance.
        arch: Architecture metadata for the model.

    Returns:
        Callable taking an integer index `i` (0-based) and returning the submodule.
    """
    unwrapped = _unwrap_model(model)
    if arch.model_family == "gpt2":
        transformer = getattr(unwrapped, "transformer", unwrapped)
        if hasattr(transformer, "h"):
            blocks = transformer.h
        elif hasattr(unwrapped, "h"):
            blocks = unwrapped.h
        else:
            raise AttributeError(
                f"Could not locate transformer blocks on GPT-2 model {type(model).__name__}."
            )
        return lambda i: blocks[i]
    elif arch.model_family == "neox":
        neox = getattr(unwrapped, "gpt_neox", unwrapped)
        if hasattr(neox, "layers"):
            blocks = neox.layers
        elif hasattr(unwrapped, "layers"):
            blocks = unwrapped.layers
        else:
            raise AttributeError(
                f"Could not locate transformer blocks on NeoX model {type(model).__name__}."
            )
        return lambda i: blocks[i]
    else:
        raise ValueError(f"Unsupported model family: {arch.model_family!r}")


def get_embedding_module(model: torch.nn.Module, arch: Architecture) -> torch.nn.Module:
    """Return the primary token embedding submodule.

    For GPT-2: `model.transformer.wte`
    For NeoX:  `model.gpt_neox.embed_in`

    Args:
        model: PyTorch causal LM instance.
        arch: Architecture metadata for the model.

    Returns:
        Submodule responsible for token embedding lookup.
    """
    unwrapped = _unwrap_model(model)
    if arch.model_family == "gpt2":
        transformer = getattr(unwrapped, "transformer", unwrapped)
        if hasattr(transformer, "wte"):
            return transformer.wte
        if hasattr(unwrapped, "wte"):
            return unwrapped.wte
        raise AttributeError("Could not locate wte token embedding on GPT-2 model.")
    elif arch.model_family == "neox":
        neox = getattr(unwrapped, "gpt_neox", unwrapped)
        if hasattr(neox, "embed_in"):
            return neox.embed_in
        if hasattr(unwrapped, "embed_in"):
            return unwrapped.embed_in
        raise AttributeError("Could not locate embed_in embedding on NeoX model.")
    else:
        raise ValueError(f"Unsupported model family: {arch.model_family!r}")


def get_position_embedding_module(
    model: torch.nn.Module, arch: Architecture
) -> torch.nn.Module | None:
    """Return the position embedding submodule if present (e.g. GPT-2 wpe), or None.

    Args:
        model: PyTorch causal LM instance.
        arch: Architecture metadata for the model.

    Returns:
        Position embedding submodule for GPT-2, or None for architectures using RoPE.
    """
    unwrapped = _unwrap_model(model)
    if arch.model_family == "gpt2":
        transformer = getattr(unwrapped, "transformer", unwrapped)
        return getattr(transformer, "wpe", getattr(unwrapped, "wpe", None))
    return None


def get_head_modules(
    model: torch.nn.Module, arch: Architecture
) -> list[torch.nn.Module]:
    """Return the final LayerNorm and LM head submodules in execution order.

    For GPT-2: `[model.transformer.ln_f, model.lm_head]`
    For NeoX:  `[model.gpt_neox.final_layer_norm, model.embed_out]`

    Args:
        model: PyTorch causal LM instance.
        arch: Architecture metadata for the model.

    Returns:
        List of head submodules to execute after transformer blocks.
    """
    unwrapped = _unwrap_model(model)
    modules: list[torch.nn.Module] = []

    if arch.model_family == "gpt2":
        transformer = getattr(unwrapped, "transformer", unwrapped)
        ln_f = getattr(transformer, "ln_f", getattr(unwrapped, "ln_f", None))
        if ln_f is not None:
            modules.append(ln_f)
        lm_head = getattr(unwrapped, "lm_head", None)
        if lm_head is not None:
            modules.append(lm_head)
        return modules
    elif arch.model_family == "neox":
        neox = getattr(unwrapped, "gpt_neox", unwrapped)
        ln_f = getattr(
            neox, "final_layer_norm", getattr(unwrapped, "final_layer_norm", None)
        )
        if ln_f is not None:
            modules.append(ln_f)
        lm_head = getattr(unwrapped, "embed_out", getattr(unwrapped, "lm_head", None))
        if lm_head is not None:
            modules.append(lm_head)
        return modules
    else:
        raise ValueError(f"Unsupported model family: {arch.model_family!r}")


__all__ = [
    "Architecture",
    "detect_architecture",
    "detect_architecture_from_config",
    "discover_layers",
    "discover_layers_from_config",
    "get_block_accessor",
    "get_embedding_module",
    "get_head_modules",
    "get_position_embedding_module",
    "validate_interception_layers",
]
