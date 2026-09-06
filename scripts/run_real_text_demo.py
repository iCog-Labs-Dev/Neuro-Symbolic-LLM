#!/usr/bin/env python
"""End-to-end real-text demo for the torchax substrate.

Feeds real text (Shakespeare by default, or any ``.txt`` file you supply)
through a pretrained causal LM wrapped in the torchax functional API:

1. tokenizes it with the model's real HuggingFace tokenizer,
2. runs the functional torchax forward pass with every requested layer intercepted,
3. proves the default hook is a pure identity (``+0.0``) passthrough per
   layer by recording what goes in and what comes out,
4. compares wrapper logits against the untouched HuggingFace torch model,
5. optionally applies a steering hook and measures the KL drift it causes.

Usage:
    python scripts/run_real_text_demo.py
    python scripts/run_real_text_demo.py --model EleutherAI/pythia-70m
    python scripts/run_real_text_demo.py --text hamlet.txt --layers 0,5,11
    python scripts/run_real_text_demo.py --steer 2.0
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import jax
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from substrate import compute_kl_drift
from substrate.loader import load_torchax_model
from substrate.torchax_models import functional_model

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
    text: str
    candidates: list[Path] = []
    if path is not None:
        candidates.append(path)
    else:
        data_dir = Path(__file__).resolve().parent.parent / "data"
        candidates += [data_dir / "shakespeare.txt", data_dir / "wiki.txt"]
    for candidate in candidates:
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8")
            print(f"Text source : file {candidate}")
            return text
    try:
        with urllib.request.urlopen(SHAKESPEARE_URL, timeout=20) as resp:
            text = resp.read().decode("utf-8")
        print(f"Text source : downloaded tinyshakespeare ({len(text)} chars)")
        return text
    except Exception as exc:
        print(f"Text source : download failed ({exc}); using embedded excerpt")
        return FALLBACK_TEXT


def get_num_layers(config) -> int:
    """Safely extract the number of layers across different model architectures."""
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    if hasattr(config, "n_layer"):
        return config.n_layer
    raise ValueError("Could not determine number of layers from model config.")


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


def hidden_stats(h: torch.Tensor) -> str:
    a = h.detach().cpu().numpy()
    return (
        f"mean={a.mean():+.4f} std={a.std():.4f} "
        f"min={a.min():+.4f} max={a.max():+.4f} l2={np.linalg.norm(a):.2f}"
    )


def make_steer(strength: float):
    """Dimension-varying perturbation (survives LayerNorm, so KL > 0)."""

    def steer(h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        pattern = torch.arange(h.shape[-1], dtype=h.dtype, device=h.device) / h.shape[-1]
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
    args = parser.parse_args()

    section("1. ENVIRONMENT")
    print(f"JAX backend (via torchax) : {jax.default_backend()}")
    print(f"Devices                   : {[str(d) for d in jax.devices()]}")

    # ── Real text ───────────────────────────────────────────────────────────
    section("2. REAL TEXT")
    text = load_text(args.text)
    print(f"Preview     : {text[:70]!r}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"][:, : args.max_tokens]
    print(f"Tokenizer   : {type(tokenizer).__name__}")
    print(f"Tokens      : {input_ids.shape[1]} (truncated to --max-tokens)")
    round_trip = tokenizer.decode(input_ids[0].tolist())
    assert round_trip == tokenizer.decode(
        encoded["input_ids"][0][: args.max_tokens].tolist()
    )

    # ── Torchax substrate from the real checkpoint ──────────────────────────
    section("3. LOADING TORCHAX SUBSTRATE")
    print(f"Loading     : {args.model} (torch weights -> torchax layout)")
    torch_model = AutoModelForCausalLM.from_pretrained(args.model)
    torch_model.eval()
    
    # Ingest PyTorch model to extract parameters
    params = load_torchax_model(torch_model)
    
    num_layers = get_num_layers(torch_model.config)
    layers = parse_layers(args.layers, num_layers)
    
    print(
        f"Detected    : layers={num_layers} "
        f"hidden={torch_model.config.hidden_size} vocab={torch_model.config.vocab_size}"
    )
    print(f"Intercepting: {layers}")

    recorded: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def base_hook(h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return h + 0.0 if not args.steer else make_steer(args.steer)(h, layer_idx)

    def recording_hook(h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Wraps the active hook and records what goes in / comes out."""
        out = base_hook(h, layer_idx)
        recorded[layer_idx] = (h, out)
        return out

    # ── Forward pass with interception ──────────────────────────────────────
    section("4. FORWARD PASS (EVERY INTERCEPTED LAYER)")
    
    # Run the model functionally
    result = functional_model(
        torch_model, params, input_ids, intercept_layers=layers, hook=recording_hook
    )
    # Untouched execution (default identity hook) for reference comparisons
    plain_result = functional_model(
        torch_model, params, input_ids, intercept_layers=layers
    )
    
    print(
        f"logits      : shape={tuple(result.logits.shape)} "
        f"finite={bool(torch.isfinite(result.logits).all())}"
    )

    identity_ok = True
    for idx in layers:
        h_in, h_out = recorded[idx]
        h_in_np = h_in.detach().cpu().numpy()
        h_out_np = h_out.detach().cpu().numpy()
        
        same = bool(np.array_equal(h_in_np, h_out_np))
        identity_ok &= same
        
        cached = result.hidden_states[idx].detach().cpu().numpy()
        print(
            f"layer {idx:>2}: in==out: {same!s:<5} "
            f"| cache matches: {bool(np.array_equal(cached, h_in_np))!s:<5} "
            f"| {hidden_stats(h_out)}"
        )
        
    if args.steer:
        print(
            f"HOOK PROOF  : every layer received its hidden state, the steering "
            f"hook modified it (in==out: {identity_ok}), and the cache kept "
            f"the pre-modification state."
        )
    else:
        print(
            f"HOOK PROOF  : every intercepted layer received its hidden state, "
            f"applied +0.0 and returned it unchanged: {identity_ok}"
        )

    # ── Original-vs-wrapper equivalence on real data ────────────────────────
    section("5. ORIGINAL TORCH MODEL VS TORCHAX WRAPPER")

    with torch.no_grad():
        ref = torch_model(input_ids=input_ids).logits
        
    plain_logits_np = plain_result.logits.detach().cpu().numpy()
    ref_np = ref.cpu().numpy()
    
    max_abs = float(np.max(np.abs(ref_np - plain_logits_np)))
    
    # Compute KL drift
    kl_ref = compute_kl_drift(ref, plain_result.logits)["kl_divergence"]
    print(f"max |torch - torchax| logit diff : {max_abs:.3e}")
    print(f"KL(torch || torchax wrapper)     : {kl_ref:.3e}")

    # ── Next-token predictions you can read ────────────────────────────────
    section("6. TOP NEXT-TOKEN PREDICTIONS (LAST POSITION)")
    last_logits = plain_logits_np[0, -1]
    top_ids = np.argsort(last_logits)[::-1][: args.topk]
    print("baseline: " + " | ".join(repr(tokenizer.decode([int(t)])) for t in top_ids))

    if args.steer:
        section("7. STEERED RUN AND KL DRIFT")
        kl = compute_kl_drift(plain_result.logits, result.logits)["kl_divergence"]
        print(f"steer={args.steer} at layers {layers}")
        print(f"KL(baseline || steered)      : {kl:.4f}  (> 0 means steering worked)")
        
        s_last = result.logits.detach().cpu().numpy()[0, -1]
        s_top = np.argsort(s_last)[::-1][: args.topk]
        print(
            "steered : " + " | ".join(repr(tokenizer.decode([int(t)])) for t in s_top)
        )

    section("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())