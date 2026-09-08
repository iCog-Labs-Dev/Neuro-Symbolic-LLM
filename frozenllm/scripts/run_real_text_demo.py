#!/usr/bin/env python
"""End-to-end real-text demo for the frozen LLM substrate using TorchAX.

Feeds real text (Shakespeare by default, or any ``.txt`` file you supply)
through a pretrained causal LM wrapped in ``FrozenSubstrate``:

1. tokenizes it with the model's real HuggingFace tokenizer,
2. runs the monolithic TorchAX forward pass with every requested layer intercepted,
3. proves the default hook is a pure identity (+0.0) passthrough per
   layer by recording what goes in and what comes out,
4. compares wrapper logits against the untouched native HuggingFace torch model,
5. optionally applies a steering hook and measures the KL drift it causes,
6. reports device memory status and applies the 50% headroom rule,
7. verifies the parameters were never modified.

Usage:
    python frozenllm/scripts/run_real_text_demo.py
    python frozenllm/scripts/run_real_text_demo.py --model EleutherAI/pythia-70m
    python frozenllm/scripts/run_real_text_demo.py --text hamlet.txt --layers 0,5,11
    python frozenllm/scripts/run_real_text_demo.py --steer 2.0
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Any

# Ensure repo root and frozenllm are on sys.path even when run directly as a script
FROZENLLM_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = FROZENLLM_DIR.parent
for _p in (REPO_ROOT, FROZENLLM_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# Compatibility patch for torchax versions expecting FP4 dtype on PyTorch < 2.5
if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = getattr(torch, "float8_e4m3fn", torch.uint8)


from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from frozenllm.substrate import (  # noqa: E402
    FrozenSubstrate,
    check_memory_headroom,
    compute_kl_drift,
    enable_torchax,
    get_memory_status,
)

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

# Public-domain fallback so the demo also works fully offline.
FALLBACK_TEXT = """\
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die, to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream. Ay, there's the rub,
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes calamity of so long life.
"""


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def load_text(path: Path | None) -> str:
    """User-supplied file, else local data folder, else download, else fallback."""
    candidates: list[Path] = []
    if path is not None:
        candidates.append(path)
    else:
        candidates += [
            REPO_ROOT / "data" / "shakespeare.txt",
            REPO_ROOT / "data" / "wiki.txt",
            FROZENLLM_DIR / "data" / "shakespeare.txt",
            FROZENLLM_DIR / "data" / "wiki.txt",
        ]
    for candidate in candidates:
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8")
            print(f"Text source : file {candidate}")
            return text
    try:
        with urllib.request.urlopen(SHAKESPEARE_URL, timeout=20) as resp:
            raw = resp.read()
            text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        print(f"Text source : downloaded tinyshakespeare ({len(text)} chars)")
        return text
    except Exception as exc:
        print(f"Text source : download failed ({exc}); using embedded excerpt")
        return FALLBACK_TEXT


def parse_layers(spec: str, num_layers: int) -> list[int]:
    """'all' or a comma-separated list of zero-based layer indices."""
    if spec.strip().lower() == "all":
        return list(range(num_layers))
    layers = [int(x) for x in spec.split(",") if x.strip()]
    if not layers:
        raise ValueError("No valid layer indices given")
    for i in layers:
        if not 0 <= i < num_layers:
            raise ValueError(
                f"Layer {i} out of range: model has {num_layers} layers "
                f"(0..{num_layers - 1})"
            )
    return sorted(set(layers))


def hidden_stats(h: jax.Array) -> str:
    a = np.asarray(h)
    return (
        f"mean={a.mean():+.4f} std={a.std():.4f} "
        f"min={a.min():+.4f} max={a.max():+.4f} l2={np.linalg.norm(a):.2f}"
    )


def make_steer(strength: float):
    """Dimension-varying perturbation (survives LayerNorm, so KL > 0)."""

    def steer(h: jax.Array, layer_idx: int) -> jax.Array:
        pattern = jnp.arange(h.shape[-1], dtype=h.dtype) / h.shape[-1]
        return h + strength * pattern

    return steer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="gpt2",
        help="HuggingFace model id (gpt2 family or Pythia/GPT-NeoX)",
    )
    parser.add_argument(
        "--text",
        type=Path,
        default=None,
        help="Path to a .txt file; defaults to tinyshakespeare",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Layers to intercept: 'all' or comma-separated zero-based ids",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Truncate the sequence to this many tokens",
    )
    parser.add_argument(
        "--steer",
        type=float,
        default=0.0,
        help="Steering strength; 0 keeps the pure identity hook",
    )
    parser.add_argument("--topk", type=int, default=5, help="Next-token candidates")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Model compute dtype (default: float32 to prevent FP16 attention overflow)",
    )
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        choices=["eager", "sdpa"],
        help="Attention implementation (default: eager to avoid cuDNN version mismatches)",
    )
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    selected_dtype = dtype_map.get(args.dtype, torch.float32)

    section("1. ENVIRONMENT")
    print(f"JAX backend : {jax.default_backend()}")
    print(f"Devices     : {[str(d) for d in jax.devices()]}")

    # ── Real text ───────────────────────────────────────────────────────────
    section("2. REAL TEXT")
    text = load_text(args.text)
    print(f"Preview     : {text[:70]!r}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoded = tokenizer(text, return_tensors="np")
    input_ids = encoded["input_ids"][:, : args.max_tokens]
    print(f"Tokenizer   : {type(tokenizer).__name__}")
    print(f"Tokens      : {input_ids.shape[1]} (truncated to --max-tokens)")
    round_trip = tokenizer.decode(input_ids[0].tolist())
    assert round_trip == tokenizer.decode(
        encoded["input_ids"][0][: args.max_tokens].tolist()
    )

    # Load native PyTorch reference model BEFORE enabling TorchAX dispatch,
    # so it runs on vanilla CPU PyTorch (prevents nan from TorchAX interception).
    print(f"Pre-loading native PyTorch reference model on CPU ({args.dtype})...")
    ref_kwargs: dict[str, Any] = {"dtype": selected_dtype}
    if args.attn_implementation:
        ref_kwargs["attn_implementation"] = args.attn_implementation
    try:
        torch_ref_model = AutoModelForCausalLM.from_pretrained(
            args.model, **ref_kwargs
        ).eval()
    except TypeError:
        ref_kwargs.pop("dtype", None)
        ref_kwargs["torch_dtype"] = selected_dtype
        try:
            torch_ref_model = AutoModelForCausalLM.from_pretrained(
                args.model, **ref_kwargs
            ).eval()
        except TypeError:
            ref_kwargs.pop("attn_implementation", None)
            torch_ref_model = AutoModelForCausalLM.from_pretrained(
                args.model, **ref_kwargs
            ).eval()

    with torch.no_grad():
        ref_logits_np = torch_ref_model(
            input_ids=torch.from_numpy(input_ids)
        ).logits.numpy()
    del torch_ref_model  # free CPU memory before TorchAX model load

    # Now enable TorchAX global dispatch
    enable_torchax()
    print("TorchAX     : enabled (monolithic JAX-backed PyTorch execution)")

    # ── Frozen substrate from the real checkpoint ───────────────────────────
    section("3. LOADING FROZEN SUBSTRATE")
    print(
        f"Loading     : {args.model} via TorchAX ({args.dtype}, attn={args.attn_implementation})"
    )
    sub = FrozenSubstrate(
        model_id_or_model=args.model,
        tokenizer=tokenizer,
        dtype=selected_dtype,
        attn_implementation=args.attn_implementation,
    )
    arch = sub.architecture
    layers = parse_layers(args.layers, arch.num_layers)
    print(
        f"Detected    : family={arch.model_family} layers={arch.num_layers} "
        f"hidden={arch.hidden_size} vocab={arch.vocab_size}"
    )
    print(f"Intercepting: {layers}")

    recorded: dict[int, tuple[jax.Array, jax.Array]] = {}

    def base_hook(h: jax.Array, layer_idx: int) -> jax.Array:
        return h + 0.0 if not args.steer else make_steer(args.steer)(h, layer_idx)

    def recording_hook(h: jax.Array, layer_idx: int) -> jax.Array:
        """Wraps the active hook and records what goes in / comes out."""
        out = base_hook(h, layer_idx)
        recorded[layer_idx] = (h, out)
        return out

    # ── Forward pass with interception ──────────────────────────────────────
    section("4. FORWARD PASS (EVERY INTERCEPTED LAYER)")
    ids = jnp.asarray(input_ids, dtype=jnp.int32)

    # 1. Plain reference run (no modification)
    plain_result = sub.run_with_interception(
        input_ids=ids,
        modify_fn=None,
        intercept_layers=layers,
    )
    # Materialise logits: to_jax_array uses a zero-copy view into the XLA
    # output buffer.  A second forward pass may reuse that buffer, silently
    # overwriting plain_result.logits.  Roundtrip through numpy to guarantee
    # an owned copy (jnp.array(copy=True) may be unsupported in older JAX).
    plain_logits = jnp.asarray(np.array(plain_result.logits))

    # 2. Recorded run with active hook
    result = sub.run_with_interception(
        input_ids=ids,
        modify_fn=recording_hook,
        intercept_layers=layers,
    )
    steered_logits = jnp.asarray(np.array(result.logits))

    print(
        f"logits      : shape={tuple(result.logits.shape)} "
        f"finite={bool(jnp.isfinite(result.logits).all())}"
    )

    identity_ok = True
    for idx in layers:
        h_in, h_out = recorded[idx]
        same = bool(np.allclose(np.asarray(h_in), np.asarray(h_out), atol=1e-5))
        identity_ok &= same
        cached = result.hidden_state(idx)
        cache_matches = bool(
            np.allclose(np.asarray(cached), np.asarray(h_in), atol=1e-5)
        )
        print(
            f"layer {idx:>2}: in==out: {same!s:<5} "
            f"| cache matches: {cache_matches!s:<5} "
            f"| {hidden_stats(h_out)}"
        )
    if args.steer:
        print(
            "HOOK PROOF  : every layer received its hidden state, the steering "
            f"hook modified it (in==out: {identity_ok}), and the cache kept "
            "the pre-modification state."
        )
    else:
        print(
            "HOOK PROOF  : every intercepted layer received its hidden state, "
            f"applied +0.0 and returned it unchanged: {identity_ok}"
        )

    # ── Original-vs-wrapper equivalence on real data ────────────────────────
    section("5. ORIGINAL TORCH MODEL VS TORCHAX SUBSTRATE")
    print("Comparing against native PyTorch reference (pre-loaded on CPU)...")
    max_abs = float(np.max(np.abs(ref_logits_np - np.asarray(plain_logits))))
    kl_ref = compute_kl_drift(jnp.asarray(ref_logits_np), plain_logits)["kl_divergence"]
    print(f"max |torch - torchax| logit diff : {max_abs:.3e}")
    print(f"KL(torch || torchax substrate)   : {kl_ref:.3e}")

    # ── Next-token predictions you can read ────────────────────────────────
    section("6. TOP NEXT-TOKEN PREDICTIONS (LAST POSITION)")
    last_logits = np.asarray(plain_logits)[0, -1]
    top_ids = np.argsort(last_logits)[::-1][: args.topk]
    print("baseline: " + " | ".join(repr(tokenizer.decode([int(t)])) for t in top_ids))

    if args.steer:
        section("7. STEERED RUN AND KL DRIFT")
        kl = compute_kl_drift(plain_logits, steered_logits)["kl_divergence"]
        print(f"steer={args.steer} at layers {layers}")
        print(f"KL(baseline || steered)      : {kl:.4f}  (> 0 means steering worked)")
        s_last = np.asarray(steered_logits)[0, -1]
        s_top = np.argsort(s_last)[::-1][: args.topk]
        print(
            "steered : " + " | ".join(repr(tokenizer.decode([int(t)])) for t in s_top)
        )

    # ── Memory monitoring and headroom rule ─────────────────────────────────
    section("8. MEMORY STATUS AND HEADROOM RULE")
    status = get_memory_status()
    print(f"platform    : {status.platform}")
    print(f"available   : {status.available}")
    if (
        status.available
        and status.total_bytes is not None
        and status.allocated_bytes is not None
        and status.available_bytes is not None
    ):
        print(f"total       : {status.total_bytes / 1e9:.2f} GB")
        print(f"allocated   : {status.allocated_bytes / 1e9:.2f} GB")
        print(f"free        : {status.available_bytes / 1e9:.2f} GB")
    else:
        print(f"diagnostic  : {status.diagnostic}")
    warnings = check_memory_headroom(status, min_headroom=0.5)
    for w in warnings:
        print(w)
    if not warnings:
        print("headroom    : OK (>= 50%)")

    _, report = sub.run_with_memory_guard(ids, min_headroom=0.5)
    print(
        f"guard report: reduced={report['batch_size_reduced']} "
        f"effective_batch={report['effective_batch_size']}"
    )

    # ── Freeze guarantee ────────────────────────────────────────────────────
    section("9. FREEZE VERIFICATION")
    frozen = sub.verify_frozen()
    print(f"params_unchanged : {frozen['params_unchanged']}")
    print(f"param leaves     : {frozen['param_leaves']}")
    print(f"architecture     : {frozen['architecture']}")

    section("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
