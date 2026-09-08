# Neuro-Symbolic-LLM — Frozen LLM Substrate (TorchAX Backend)

A high-performance, reusable **frozen LLM substrate** built on **Google TorchAX**: wraps pretrained causal language models (**GPT-2** and **Pythia/GPT-NeoX**) while keeping the base model parameters strictly frozen ($\nabla \theta_0 = 0$).

Supports arbitrary per-layer hidden-state interception, activation caching, bi-directional TorchAX $\leftrightarrow$ JAX tensor bridging, dimension-varying steering interventions, causal cross-entropy loss computation, KL drift monitoring, and device memory management with a 50% headroom safety guard.

---

## Table of Contents

1. [Architectural Overview](#architectural-overview)
2. [What Was Built & Current Status](#what-was-built--current-status)
3. [Implementation Design & Data Flow](#implementation-design--data-flow)
4. [File Breakdown](#file-breakdown)
5. [Test Suite Documentation](#test-suite-documentation)
   - [Consolidated Unit Test Modules](#consolidated-unit-test-modules)
6. [Installation & Requirements](#installation--requirements)
7. [How to Run](#how-to-run)
   - [Prepare offline text and tokenizers](#1-prepare-offline-data)
   - [Run the real-text demo](#2-run-the-real-text-demo)
   - [Run the test suites](#3-run-tests)
   - [Run pre-commit / lint checks](#4-code-quality-checks)
   - [Quickstart: Use FrozenSubstrate in Python](#5-quickstart-python-example)
8. [Core Architectural Guarantees](#core-architectural-guarantees)

---

## Architectural Overview

The repository uses **monolithic TorchAX dispatch architecture**:

1. **TorchAX Lowering & Monolithic Execution:** Rather than slicing the Hugging Face model into multiple fragile submodules, the entire model (`AutoModelForCausalLM`) is placed onto TorchAX's `"jax"` device. Operations are lowered via `__torch_dispatch__` directly into JAX/XLA primitives without altering Hugging Face's internal state (e.g. RoPE rotary caching, attention masks, KV tuples).
2. **Numerical Parity:** Substrate forward execution matches native PyTorch CPU/GPU to within $10^{-6}$ maximum absolute difference on identical weights.
3. **Strict Freezing Invariant:** Base model parameters $\theta_0$ have `requires_grad=False`. Parameter immutability is verified against pristine snapshots via `sub.params_unchanged()`.
4. **Non-Destructive Interception:** PyTorch forward hooks are managed inside a clean context (`InterceptionContext`) attached to block outputs (`transformer.h[i]` for GPT-2, `gpt_neox.layers[i]` for Pythia). Interception is guaranteed zero-leakage upon normal exit or runtime exceptions.
5. **LayerNorm Invariance Resolution:** Standardized all steering interventions to use dimension-varying vectors (e.g. `jnp.linspace`), resolving mathematical cancellation where uniform perturbations are eliminated by LayerNorm.
6. **Bi-directional JAX $\leftrightarrow$ TorchAX Bridge:** Passes a pure `jax.Array` to `modify_fn` when `to_jax=True` via zero-copy `torchax.interop.jax_view`, and automatically converts returned JAX arrays back via `torchax.interop.torch_view` before handing control to downstream blocks.
7. **Pristine State Guarantee:** Cached intermediates strictly store pre-modification activations $h_l^0$, ensuring downstream steering cannot corrupt recorded activations.

---

## What Was Built & Current Status

| Requirement / Milestone | Status | Details |
|---|---|---|
| **Monolithic TorchAX Substrate** | **Completed** | `FrozenSubstrate` in `frozenllm/substrate/substrate.py` executes full models on device `"jax"`. |
| **Multi-Architecture Support** | **Completed** | Auto-detects GPT-2 and Pythia/GPT-NeoX configurations and weight hierarchies. |
| **Strict Base Freezing ($\nabla \theta_0 = 0$)** | **Completed** | `requires_grad=False` on all parameter leaves; verified with `sub.params_unchanged()`. |
| **In-Flight Layer Interception** | **Completed** | Dynamic forward hooks with lifecycle safety in `InterceptionContext`. |
| **Pristine Activation Caching** | **Completed** | Pre-modification hidden states $h_l^0$ cached in `ForwardResult.intermediates`. |
| **Bi-Directional JAX Bridge** | **Completed** | Zero-copy `to_jax_array` and `from_jax_array` via TorchAX interop views. |
| **Autograd Residual Bridge** | **Completed** | `call_jax_differentiable` via `torchax.interop.j2t_autograd` enables gradient flow to trainable parameters $\phi$. |
| **Causal Loss Computation** | **Completed** | `FrozenSubstrate.compute_loss` evaluates shifted cross-entropy with padding masking. |
| **KL Drift Monitoring** | **Completed** | `compute_kl_drift` computes numerically stable $D_{\text{KL}}(P_{\text{base}} \parallel P_{\text{steered}})$. |
| **Device Memory Guard** | **Completed** | Headroom monitoring + 50% safety rule with automatic batch-size reduction option. |
| **Verification Suite** | **Completed** | Comprehensive test suite covering substrate execution, interception invariants, and autograd bridging. |

---

## Implementation Design & Data Flow

```text
Input Tokens (jax.Array or torch.Tensor)
     │
     ▼
FrozenSubstrate.run_with_interception()
     │ [to_torchax_device / from_jax_array]
     ▼
InterceptionContext.__enter__()
     │ [attach forward hooks to block modules via get_block_accessor]
     ▼
torch.func.functional_call(model, params, (input_ids,))
     │
     ├── Token & Position Embeddings (wte, wpe / embed_in)
     ├── Transformer Blocks 0 to l-1
     ▼
Transformer Block l Output (h_l)
     │
     ▼ [Forward Hook Fires]
     ├── 1. _extract_hidden(): unwrap block output tuple
     ├── 2. Pristine caching: clone & convert h_l^0 to jax.Array
     │      Stored in ctx.intermediates[l]
     ├── 3. modify_fn(h_l^0, l): steering / residual modification
     │      (e.g. h_l' = h_l^0 + R_phi(h_l^0))
     ├── 4. from_jax_array(): re-wrap modified tensor onto 'jax' device
     └── 5. _wrap_hidden(): return h_l' as effective block output
     │
     ▼
Downstream Blocks l+1 to L-1 (consume modified activation h_l')
     │
     ▼
Final LayerNorm & LM Head (ln_f, lm_head / embed_out)
     │
     ▼
Raw Logits [B, S, V] on TorchAX 'jax' device
     │
     ▼
InterceptionContext.__exit__() -> remove_hooks()
     │
     ▼
to_jax_array(logits) -> ForwardResult(logits, intermediates)
     │
     ▼
FrozenSubstrate.compute_loss(logits, labels) [Optional Loss Step]
```

---

## File Breakdown

### Substrate Core (`frozenllm/substrate/`)

- **`__init__.py`**: Central package entrypoint re-exporting the primary public API for substrate orchestration, layer interception, architecture discovery, and TorchAX bridging.
- **`substrate.py`**: Houses the main `FrozenSubstrate` execution manager (enforcing $\nabla \theta_0 = 0$ parameter freezing, dynamic hook attachment, and causal cross-entropy loss) and `ForwardResult` (JAX dataclass packaging output logits and intercepted activations).
- **`interception.py`**: Implements the hidden-state interception engine via `InterceptionContext` and `run_with_hooks`; manages dynamic forward hooks on transformer blocks, pristine activation caching ($h_l^0$), and in-flight `modify_fn` state modification.
- **`architecture.py`**: Provides automatic model family detection (GPT-2 vs. Pythia/GPT-NeoX), dynamic block submodule resolution (`get_block_accessor`), and interception layer validation.
- **`torchax_backend.py`**: Manages TorchAX environment setup (`enable_torchax`), device transfers (`to("jax")`), zero-copy tensor/array conversions (`to_jax_array`, `from_jax_array`), and autograd gradient bridging (`call_jax_differentiable`).
- **`torchax_models.py`**: Thin wrapper around `torch.func.functional_call` and `check_numerical_fidelity` for verifying parity between native PyTorch and TorchAX execution.
- **`loader.py`**: Checkpoint loading utilities that initialize pretrained Hugging Face models directly onto TorchAX's `"jax"` device with frozen parameters, alongside their tokenizers.
- **`memory.py`**: Device memory monitoring (`get_memory_status`) and the 50% headroom safety rule (`check_memory_headroom`, `maybe_reduce_batch_size`) with optional automatic batch-size halving.
- **`drift.py`**: Computes numerically stable Kullback-Leibler divergence (`compute_kl_drift`) between baseline and steered model logits to quantify intervention impact.

### Scripts (`frozenllm/scripts/`)

- **`prepare_data.py`**: One-time offline downloader for the Shakespeare corpus, sample Wikipedia text, and Hugging Face tokenizers.
- **`run_real_text_demo.py`**: Interactive end-to-end demonstration featuring text tokenization, monolithic forward execution, layer-by-layer identity verification, steering interventions, and memory telemetry.

---

## Test Suite Documentation

### Consolidated Unit Test Modules

The test suite in `tests/frozenllm/unit/` covers all critical architectural invariants and operational guarantees:

| Test File | Focus Areas |
|---|---|
| `test_interception.py` | Hook registration lifecycle, context manager cleanup on normal exit and exceptions, pristine caching invariance, and multi-layer cascade steering. |
| `test_substrate.py` | `FrozenSubstrate` initialization, GPT-2 & Pythia architecture detection, parameter freezing invariance, forward execution, determinism, input validation, and causal loss computation. |
| `test_torchax_backend.py` | Idempotent TorchAX enablement, `to_jax_array` / `from_jax_array` round-trips, device checks, and `call_jax_differentiable` autograd gradient verification. |
| `test_torchax_gpt2.py` | GPT-2 functional call correctness, numerical parity against CPU PyTorch, freezing invariants, and hook replacement. |
| `test_torchax_pythia.py` | Pythia/NeoX functional call correctness, RoPE attention handling, numerical parity, and hook replacement. |
| `test_substrate_memory.py` | Memory headroom calculation, CPU diagnostic handling, automatic batch-size reduction rules, and KL drift stability. |

---

## Installation & Requirements

### Dependencies
Defined in `pyproject.toml`:
- **Python:** $\ge 3.10$
- **PyTorch:** `torch==2.6.0`
- **TorchAX:** `torchax==0.0.14.dev20260617`
- **JAX Ecosystem:** `jax>=0.4.30`, `jaxlib>=0.4.30`, `optax>=0.2.0`, `orbax-checkpoint>=0.5.0`
- **Transformers:** `transformers>=4.40.0`, `datasets>=2.19.0`
- **Dev Tools:** `pytest>=8.0`, `ruff>=0.4.1`, `black>=24.3.0`, `mypy>=1.9.0`, `pre-commit>=3.7.0`

### Setup Commands
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Or .\.venv\Scripts\Activate.ps1 on Windows

# Install package in editable mode with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

---

## How to Run

### 1. Prepare Offline Data
Downloads datasets and tokenizers to `data/`:
```bash
python frozenllm/scripts/prepare_data.py
```

### 2. Run the Real-Text Demo
```bash
# Run GPT-2 on Shakespeare text with all layers intercepted
python frozenllm/scripts/run_real_text_demo.py --max-tokens 96

# Run Pythia-70m with dimension-varying steering at layers 0, 3, 5
python frozenllm/scripts/run_real_text_demo.py --model EleutherAI/pythia-70m --layers 0,3,5 --steer 2.0
```

### 3. Run Tests
```bash
# Run all unit tests
pytest tests/frozenllm/unit/ -v
```

### 4. Code Quality Checks
All checks configured in `.pre-commit-config.yaml`:
```bash
# Run all hooks across the codebase
pre-commit run --all-files
```

### 5. Quickstart: Python Example

```python
import jax.numpy as jnp
from frozenllm.substrate import FrozenSubstrate, compute_kl_drift

# 1. Load frozen substrate (GPT-2) with layer 5 intercepted
sub = FrozenSubstrate("gpt2", intercept_layers=[5])

# 2. Tokenize input prompt
ids = sub.tokenize("Hello, neuro-symbolic world!", return_tensors="jax")

# 3. Define dimension-varying steering hook
def steering_hook(h: jax.Array, layer_idx: int) -> jax.Array:
    perturbation = jnp.linspace(1.0, 5.0, h.shape[-1])
    return h + 0.5 * perturbation

# 4. Execute baseline and steered forward passes
baseline_res = sub(ids)
steered_res = sub.run_with_interception(ids, modify_fn=steering_hook)

# 5. Inspect outputs and cache
print(f"Logits shape: {steered_res.logits.shape}")
h5_pristine = steered_res.hidden_state(5)  # Pristine pre-modification state
print(f"Layer 5 hidden shape: {h5_pristine.shape}")

# 6. Measure KL drift caused by steering
drift = compute_kl_drift(baseline_res.logits, steered_res.logits)
print(f"KL Divergence: {drift['kl_divergence']:.4f}")

# 7. Verify base parameters remained completely frozen
assert sub.params_unchanged()
```

---

## Core Architectural Guarantees

1. **Strict Immutability ($\nabla \theta_0 = 0$):** Every base model parameter tensor has `requires_grad=False`. Snapshot comparison via `sub.params_unchanged()` guarantees no parameter corruption across runs.
2. **Zero-Copy Memory Boundary:** `to_jax_array` and `from_jax_array` utilize TorchAX interop views (`jax_view` and `torch_view`) to share XLA device buffers without host-device or device-device copying.
3. **Pristine State Isolation:** Intermediates cache $h_l^0$ *before* `modify_fn` is applied; modifications propagate strictly downstream to blocks $l+1 \dots L-1$.
4. **Tuple Safety:** Unpacks and restores Hugging Face block return tuples (`(hidden_states, past_key_values, ...)`), ensuring KV-cache operations continue without breakage.
5. **Memory Telemetry & Safety:** Active monitoring of XLA device memory with an automated $50\%$ headroom rule to prevent Out-Of-Memory (OOM) failures.
