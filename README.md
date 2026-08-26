# Neuro-Symbolic-LLM — Frozen JAX Substrate

A reusable **frozen LLM substrate** in JAX/Flax: a wrapper around pretrained
causal language models (**GPT-2** and **Pythia/GPT-NeoX**) that keeps the base
model completely frozen while supporting arbitrary per-layer hidden-state
interception, activation caching, JIT compilation, KL drift monitoring and
device-memory monitoring with a 50% headroom safety rule.

---

## Table of Contents

1. [What Was Built](#what-was-built)
2. [Installation](#installation)
3. [File-by-File Documentation](#file-by-file-documentation)
   - [`substrate/__init__.py`](#substrate__init__py)
   - [`substrate/architecture.py`](#substratearchitecturepy)
   - [`substrate/models.py`](#substratemodelspy)
   - [`substrate/substrate.py`](#substratesubstratepy)
   - [`substrate/drift.py`](#substratedriftpy)
   - [`substrate/memory.py`](#substratememorypy)
   - [`substrate/loader.py`](#substrateloaderpy)
   - [`tests/conftest.py`](#testsconftestpy)
   - [`tests/unit/test_substrate_gpt2.py`](#testsunittest_substrate_gpt2py)
   - [`tests/unit/test_substrate_pythia.py`](#testsunittest_substrate_pythiapy)
   - [`tests/unit/test_substrate_invalid.py`](#testsunittest_substrate_invalidpy)
   - [`tests/unit/test_substrate_memory.py`](#testsunittest_substrate_memorypy)
   - [`pyproject.toml`](#pyprojecttoml)
   - [`.gitignore`](#gitignore)
   - [`scripts/prepare_data.py`](#scriptsprepare_datapy)
   - [`scripts/run_real_text_demo.py`](#scriptsrun_real_text_demopy)
4. [How to Run](#how-to-run)
   - [Prepare the real-text data folder](#prepare-the-real-text-data-folder)
   - [Run the real-text end-to-end demo](#run-the-real-text-end-to-end-demo)
   - [Run the tests](#run-the-tests)
   - [Run lint / format / type checks](#run-lint--format--type-checks)
   - [Run pre-commit (same as CI)](#run-pre-commit-same-as-ci)
   - [Use the substrate in your own code](#use-the-substrate-in-your-own-code)
5. [Design Guarantees](#design-guarantees)

---

## What Was Built

| Requirement | Status |
|---|---|
| Reusable `FrozenJAXSubstrate` wrapper class | Done — `substrate/substrate.py` |
| Works with GPT-2 **and** Pythia/GPT-NeoX | Done — auto-detected from weights |
| Base model completely frozen | Done — immutable `FrozenDict` + `stop_gradient` + identity check |
| Arbitrary per-layer hidden-state interception | Done — zero-based indices, validated |
| Activation caching at intercepted layers | Done — `ForwardResult.intermediates` |
| JIT compatibility (`jax.jit`) | Done — default hook is pure array math |
| KL drift monitoring vs original model | Done — `compute_kl_drift` |
| Memory monitoring + 50% headroom rule | Done — never crashes on CPU, batch reduction always reported |
| Tests comparing against untouched HF torch models | Done — 66 tests passing |

Verified environment: Python 3.14, JAX 0.11 (CPU), Flax 0.12.8,
transformers 5.14.1, torch 2.13 (CPU), numpy 2.5.1.
Wrapper logits match the untouched HuggingFace torch forward pass to
~1e-7 (GPT-2) / ~3e-6 (NeoX) on small test configs.

---

## Installation

### 1. Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install the package with dev tools

```bash
pip install -e ".[dev]"
```

This installs the `substrate` package in editable mode plus pytest, ruff,
black, mypy and pre-commit.

> **Note:** the base dependencies include `fabricpc @ git+...@v0.3.0`,
> torch, transformers, JAX, FastAPI, redis, etc. If you only want to run the
> substrate tests you can instead install the minimal set directly:
>
> ```bash
> pip install jax flax transformers torch numpy pytest
> ```

### 3. GPU support (optional)

```bash
pip install -e ".[gpu]"      # adds jax[cuda12]
```

Without a GPU everything still works: JAX falls back to CPU and the memory
monitor returns a diagnostic status instead of crashing.

### 4. Set up pre-commit hooks (optional, recommended)

```bash
pre-commit install
```

Now black/ruff/mypy/end-of-file-fixer run automatically on every `git commit`,
identically to CI.

---

## File-by-File Documentation

Everything below was created in this work. The repo previously contained only
configuration files; the entire `substrate/` package and test suite are new.

### `substrate/__init__.py`

Public API of the package. Re-exports every public symbol so users can write
`from substrate import FrozenJAXSubstrate, load_substrate_from_hf, ...`
without knowing the internal module layout. Also defines
`__version__ = "0.1.0"`.

Exported names:

- `Architecture`, `detect_architecture`, `discover_layers`, `validate_interception_layers`
- `ForwardResult`, `FrozenJAXSubstrate`
- `compute_kl_drift`
- `MemoryStatus`, `get_memory_status`, `compute_memory_headroom`, `check_memory_headroom`, `maybe_reduce_batch_size`
- `state_dict_to_jax_pytree`, `load_substrate_from_hf`, `build_substrate_from_state_dict`

### `substrate/architecture.py`

Automatic architecture detection — no hardcoded layer counts anywhere.

- **`Architecture`** (frozen dataclass): normalized description of the model —
  `model_family` (`"gpt2"` or `"neox"`), `num_layers`, `hidden_size`,
  `num_heads`, `head_dim`, `vocab_size`, plus NeoX-specific fields
  (`rope_theta`, `rotary_pct`, `use_parallel_residual`, `layer_norm_eps`).
- **`detect_architecture(params, config=None)`**: inspects the top-level keys
  of the parameter PyTree (`transformer` → GPT-2, `gpt_neox` → NeoX) and reads
  shapes straight out of the embedding weights to get vocab size, hidden size
  and layer count. Raises `ValueError` for unsupported layouts.
- **`discover_layers(params)`**: convenience wrapper returning just the block
  count.
- **`validate_interception_layers(intercept_layers, num_layers)`**: rejects
  negative indices, out-of-range indices and duplicates with explicit error
  messages; `None`/empty means "no interception". Returns a sorted tuple.

### `substrate/models.py`

Pure-JAX forward implementations for both model families, written as pure
functions over a Flax-convention parameter PyTree whose leaf names match the
HuggingFace checkpoint names exactly (so weights convert without renaming).

Key facts encoded here (discovered while matching HF torch outputs):

- GPT-2 uses `Conv1D` layers whose weight is `[input, output]` → forward is
  `x @ w + b` (no transpose). NeoX uses standard `nn.Linear` layout
  `[output, input]` → `x @ w.T + b`.
- For NeoX checkpoints the LM head is the top-level `lm_head.weight`
  (transformers ≥ 5 does not emit `embed_out`); it is **not** tied to
  `embed_in`. Fallbacks to `embed_out` / `embed_in` are kept for older
  checkpoints.
- The final LayerNorm (`ln_f` / `final_layer_norm`) is applied **before** the
  LM head for both families.

Functions:

- Shared: `layer_norm`, `_mlp` (both weight layouts), `_causal_mask`.
- GPT-2: `gpt2_embed` (token + learned position embeddings), `gpt2_attention`
  (multi-head causal self-attention via einsum), `gpt2_block` (pre-LN residual
  block), `gpt2_lm_head`.
- NeoX: `neox_embed`, `_neox_rope` (rotary position embeddings),
  `neox_attention` (fused QKV + RoPE + causal mask), `neox_block` (sequential
  or parallel residual per config), `neox_lm_head`.
- Generic driver used by the wrapper:
  - `run_embeddings(params, arch, input_ids)`
  - `run_transformer_blocks(params, arch, hidden, intercept_layers, hook, position_ids)`
    — runs blocks sequentially; at each requested layer it caches the
    **pre-modification** hidden state into `intermediates[idx]`, then applies
    `hook(cached, idx)` to produce the hidden state fed to the next block.
  - `run_lm_head(params, arch, hidden)`

No parameter is ever mutated; everything is JIT-traceable.

### `substrate/substrate.py`

The main wrapper.

- **`ForwardResult`** (frozen dataclass, registered as a JAX dataclass so it
  works under `jax.jit`):
  - `logits`: `[batch, seq_len, vocab_size]`
  - `intermediates`: `{layer_idx: hidden_state}` cache
  - helpers: `layer_indices()`, `hidden_state(idx)` (raises `KeyError` with a
    clear message on a miss), `hidden_shapes()`
- **`FrozenJAXSubstrate(params, config=None, intercept_layers=None,
  modify_hook=None, min_memory_headroom=0.5)`**:
  - Auto-detects architecture and validates interception layers at
    construction time (bad input fails fast).
  - Stores params as an immutable Flax `FrozenDict`; `_pristine` is a second
    reference to the *same* arrays (no 2× memory copy).
  - `__call__(input_ids)`: validates shape `[batch, seq_len]`, applies
    `jax.lax.stop_gradient` to every param leaf (gradient flow into the base
    model is impossible), runs embeddings → blocks (with interception) → LM
    head, and returns a `ForwardResult`. Fully `jax.jit` compatible.
  - `intercept_and_modify(hidden_state, layer_idx)`: the default hook — an
    identity written as `hidden_state + 0.0` so it is JIT-safe, preserves
    numerics exactly, and never converts tensors to NumPy. Override it or pass
    `modify_hook` for custom steering.
  - Freezing guarantees:
    - `params_unchanged()` — leaf-identity check against the pristine snapshot.
    - `verify_frozen()` — returns a report dict
      (`params_unchanged`, `param_leaves`, architecture summary).
  - Memory integration:
    - `memory_status()`, `memory_warnings()`
    - `run_with_memory_guard(input_ids, min_headroom=None,
      auto_reduce_batch_size=False)` — forward pass plus the headroom rule;
      when auto-reduce is off the user's configuration is only warned about,
      never changed silently; when on, the batch is halved until safe and the
      reduction is reported in the returned report dict.
  - Internal `_run_forward` is a `@staticmethod` pure function so the whole
    call graph traces cleanly under `jax.jit`.

### `substrate/drift.py`

Drift monitoring between the frozen substrate and the untouched model.

- **`compute_kl_drift(original_logits, modified_logits)`** →
  `{"kl_divergence": float}`: mean KL divergence `KL(original || modified)`
  over all batch/sequence positions, computed with a numerically stable
  log-softmax (max-shift) and clamped at 0 to remove float noise.
  With the identity hook this is ~0 (< 1e-6 in tests); any real modification
  pushes it up.

### `substrate/memory.py`

Device memory monitoring with the configurable headroom safety rule.

- **`MemoryStatus`** (frozen dataclass): `available`, `total_bytes`,
  `allocated_bytes`, `available_bytes`, `headroom_ratio`, `device`,
  `platform`, `diagnostic`, `raw`.
- **`get_memory_status(device=None)`**: queries `device.memory_stats()`.
  On platforms without per-device stats (e.g. CPU-only JAX, where
  `memory_stats()` returns `None`) it returns a diagnostic status instead of
  raising — this is the "never crash" requirement.
- **`compute_memory_headroom(status)`**: available/total ratio in [0, 1], or
  `None` when unavailable. Uses explicit `is None` narrowing so it passes both
  old (1.9.0) and new mypy versions.
- **`check_memory_headroom(status, min_headroom=0.5)`**: returns warning
  strings when headroom < threshold (or cannot be verified).
- **`maybe_reduce_batch_size(status, batch_size, min_headroom=0.5,
  auto_reduce=False)`**: implements the safety rule. With `auto_reduce=True`
  the batch is halved until the estimated headroom satisfies the rule (never
  below 1) and a `NOTICE:` line is appended to the warnings — the reduction is
  always reported, never silent. With `False` the configuration is untouched.

### `substrate/loader.py`

Loading HuggingFace checkpoints into JAX parameter PyTrees.

- **`state_dict_to_jax_pytree(state_dict)`**: converts a flat torch/HF state
  dict (`{"transformer.h.0.attn.c_attn.weight": tensor, ...}`) into a nested
  JAX PyTree. Numeric segments become list entries so the layer count is
  discoverable; tensors are moved to CPU NumPy then `jnp.asarray`-ed.
- **`load_substrate_from_hf(model_id, intercept_layers=None)`**: one-call
  helper — downloads the checkpoint via `AutoModelForCausalLM`, checks
  `config.model_type` against the supported families (`gpt2`, `gpt_neox`)
  with a clear error otherwise, converts the state dict, and wraps everything
  in a `FrozenJAXSubstrate`. Heavy imports are local so importing `substrate`
  stays fast.
- **`build_substrate_from_state_dict(state_dict, config=None,
  intercept_layers=None)`**: same, but from an already-loaded state dict (used
  by the tests to guarantee the wrapper sees exactly the reference weights).

### `tests/conftest.py`

Shared fixtures. Reference "original models" are real HuggingFace **torch**
models built from small random-weight configs:

- `GPT2_CFG`: 12 layers, 4 heads, hidden 32, vocab 64, all dropout 0.
- `NEOX_CFG`: 12 layers, 4 heads, hidden 32, vocab 64, RoPE 100%, no parallel
  residual.
- `make_substrate(family, intercept_layers=None, modify_hook=None)` — builds
  the torch model, converts its state dict through the real loader code path,
  and returns `(model, substrate)`.
- `torch_logits(model, ids)` — reference logits from the untouched torch
  forward pass.
- Session-scoped `gpt2_reference` / `pythia_reference` fixtures.

### `tests/unit/test_substrate_gpt2.py`

GPT-2 suite (grouped by requirement):

1. Model detection (family, hidden size, no hardcoded layer count)
2. Correct transformer-block count (12)
3–4. Hidden-state interception at multiple points (`[3, 7, 10]`)
5. Activation caching — shapes `[batch, seq, hidden]`, `KeyError` on miss,
   and proof that a modifying hook still caches the **unmodified** state
6. JIT execution — `jax.jit(sub)` matches eager output and the torch reference
7. Valid logits — correct shape, all finite
8. Original-vs-wrapper equivalence — identity interception preserves logits
   across several intercept sets (`[]`, `[1]`, `[3,7,10]`, `[0,5,11]`)
9. KL divergence — ~0 for identity, > 1e-3 for a dimension-varying perturbation
10. Memory diagnostics — CPU backend reports `available=False`, guard runs
Freezing guarantee — `params_unchanged()` / `verify_frozen()` after eager and
JIT runs

### `tests/unit/test_substrate_pythia.py`

Same coverage as the GPT-2 suite but for Pythia/GPT-NeoX (intercept
`[2, 5, 8, 11]`), additionally exercising RoPE attention, fused QKV layout and
the NeoX LM-head path. Layer count is discovered from the parameter tree.

### `tests/unit/test_substrate_invalid.py`

Robustness: negative layer index, out-of-range index, duplicate indices,
explicit error messages, unsupported parameter layouts, non-mapping params,
and malformed `input_ids` (wrong rank / empty sequence) all raise clear
`ValueError`/`TypeError`s.

### `tests/unit/test_substrate_memory.py`

Memory module in isolation: CPU diagnostic behavior (no crash),
headroom computation, warning thresholds around 50%, batch-size halving with
`auto_reduce=True` (including the floor at 1 and the mandatory notice), and
"config untouched, warn only" with `auto_reduce=False`.

### `pyproject.toml`

(Pre-existing file; modified.)

- Added `"substrate*"` to `[tool.setuptools.packages.find]` include list so
  `pip install -e .` picks up the new package.
- Tool config used by CI lives here: ruff (E/W/F/I/N/UP/B/C4/SIM, line 88),
  black (line 88), mypy (`python_version = "3.12"` with
  `follow_imports = "skip"` overrides for jax/flax/numpy so their PEP 695
  stubs parse on any environment, `ignore_missing_imports = true`), pytest
  (`testpaths = ["tests"]`).

### `.gitignore`

(Pre-existing file; modified.) Added `__pycache__/`, `*.pyc`,
`*.egg-info/` and `data/` so build artifacts and the (regenerable, >500 KB)
real-text data folder never enter git.

### `scripts/prepare_data.py`

Builds the local `data/` folder used by the real-text demo. Run once while
online; afterwards everything works offline:

```
data/
├── shakespeare.txt      full tinyshakespeare corpus (~1.1 MB, 40 000 lines)
├── wiki.txt             real Wikipedia text (wikitext-2 train sample, ~400 KB)
├── tokenizer_gpt2/      GPT-2 tokenizer saved with save_pretrained
└── tokenizer_pythia/    Pythia/GPT-NeoX tokenizer saved with save_pretrained
```

Idempotent: existing files are skipped, so re-running only fills gaps.

### `scripts/run_real_text_demo.py`

End-to-end manual test with REAL data and REAL pretrained checkpoints:

1. loads real text (`--text` file, else `data/shakespeare.txt`, else
   `data/wiki.txt`, else downloads tinyshakespeare),
2. tokenizes it with the model's real HuggingFace tokenizer,
3. runs the frozen JAX forward pass intercepting every requested layer
   (default: ALL layers),
4. proves per layer that the hook received the hidden state and returned it
   unchanged (`+0.0` identity), and that the cache kept the pre-modification
   state,
5. compares wrapper logits against the untouched HuggingFace torch model,
6. prints readable top next-token predictions,
7. optionally applies a steering hook (`--steer`) and reports KL drift,
8. reports device memory status + the 50% headroom rule,
9. verifies parameters were never modified (`verify_frozen()`).

> **Known limitation:** on real Pythia/GPT-NeoX checkpoints the attention
> block still diverges slightly from torch (LayerNorm bias, exact-erf GELU
> and fp32 casting are already fixed and verified; attention is being
> investigated). GPT-2 matches to KL ~2e-8. Unit tests pass for both
> families on reference configs.

---

## How to Run

All commands assume you are in the repository root with the virtual
environment activated.

### Prepare the real-text data folder

```powershell
python scripts/prepare_data.py
```

Downloads Shakespeare + Wikipedia text and saves both tokenizers into
`data/`. One-time; afterwards fully offline.

### Run the real-text end-to-end demo

```powershell
# GPT-2 on Shakespeare, all 12 layers intercepted (default)
python scripts/run_real_text_demo.py --max-tokens 96

# Pythia-70m with steering at every layer
python scripts/run_real_text_demo.py --model EleutherAI/pythia-70m --steer 2.0

# Your own text file, specific layers only
python scripts/run_real_text_demo.py --text my_file.txt --layers 0,5,11
```

Sample output (GPT-2, trimmed):

```
Text source : file data/shakespeare.txt
Tokens      : 96
Detected    : family=gpt2 layers=12 hidden=768 vocab=50257
Intercepting: [0, 1, ..., 11]

layer  0: in==out: True | cache matches: True | mean=-0.0102 std=1.8916 ...
...
layer 11: in==out: True | cache matches: True | mean=-0.0599 std=19.2439 ...
HOOK PROOF  : every intercepted layer received its hidden state, applied
              +0.0 and returned it unchanged: True

max |torch - jax| logit diff : 1.411e-03
KL(torch || jax wrapper)     : 2.139e-08

baseline: ' him' | ' C' | ' you' | ' the' | ' this'

params_unchanged : True
```

### Run the tests

```powershell
# Everything (66 tests, ~100 s on CPU)
python -m pytest -q

# Only unit tests
python -m pytest tests/unit/ -q

# One suite
python -m pytest tests/unit/test_substrate_gpt2.py -v

# Single test
python -m pytest tests/unit/test_substrate_gpt2.py::TestEquivalence::test_identity_interception_preserves_logits -v

# With print output visible
python -m pytest -v -s
```

Expected result: `66 passed`.

### Run lint / format / type checks

```powershell
# Lint (ruff)
python -m ruff check substrate/ tests/

# Format check (black)
python -m black --check substrate/ tests/

# Auto-format
python -m black substrate/ tests/

# Type check
python -m mypy substrate/
python -m mypy tests/
```

The mypy configuration targets Python 3.12 parsing (`python_version = "3.12"`
in `pyproject.toml`) because modern JAX/NumPy/Flax ship PEP 695 stubs that
older parse targets reject, and treats those libraries as opaque via
`follow_imports = "skip"` overrides. The package itself remains compatible
with Python 3.10+ at runtime.

### Run pre-commit (same as CI)

```powershell
pre-commit install          # one-time setup
pre-commit run --all-files  # manually on everything
```

Hooks: black, ruff, ruff-format, mypy, check-yaml, check-toml,
check-ast, debug-statements, end-of-file-fixer, trailing-whitespace.

### Use the substrate in your own code

Load a real pretrained checkpoint:

```python
import jax.numpy as jnp
from substrate import load_substrate_from_hf, compute_kl_drift

# Downloads from HuggingFace Hub and freezes the weights
sub = load_substrate_from_hf("gpt2", intercept_layers=[3, 7, 10])
print(sub)  # FrozenJAXSubstrate(model_family='gpt2', num_layers=12, ...)

ids = jnp.asarray([[50256, 464, 1938]])          # [batch, seq_len]
result = sub(ids)

print(result.logits.shape)                        # (1, 3, 50257)
h7 = result.hidden_state(7)                       # cached activation at layer 7
print(result.hidden_shapes())                     # {3: (1, 3, 768), 7: ..., 10: ...}

assert sub.params_unchanged()                     # freeze guarantee
print(sub.verify_frozen())
```

Steer hidden states with a custom hook (must be pure array math for JIT):

```python
import jax
import jax.numpy as jnp
from substrate import load_substrate_from_hf, compute_kl_drift

def steer(h, layer_idx):
    # example: amplify one dimension at layer 5
    return h.at[:, :, 0].multiply(1.5)

sub = load_substrate_from_hf("gpt2", intercept_layers=[5], modify_hook=steer)
ids = jnp.asarray([[50256, 464, 1938]])

plain = load_substrate_from_hf("gpt2")(ids)
steered = sub(ids)
drift = compute_kl_drift(plain.logits, steered.logits)
print(drift["kl_divergence"])   # > 0 when steering actually changes the output

# JIT works out of the box
fast = jax.jit(sub)(ids)
```

Build from an existing torch model without downloading:

```python
from transformers import GPT2LMHeadModel
from substrate import build_substrate_from_state_dict

torch_model = GPT2LMHeadModel.from_pretrained("gpt2")
torch_model.eval()
sub = build_substrate_from_state_dict(
    torch_model.state_dict(),
    config=torch_model.config,
    intercept_layers=[0, 11],
)
```

Memory-guarded execution:

```python
result, report = sub.run_with_memory_guard(
    ids,
    min_headroom=0.5,
    auto_reduce_batch_size=False,   # True -> halve batch until safe (reported)
)
print(report["warnings"])           # [] on healthy GPUs, diagnostic on CPU
```

---

## Design Guarantees

1. **Frozen forever.** Params live in an immutable Flax `FrozenDict`;
   `stop_gradient` is applied to every leaf before each forward; JAX arrays
   cannot be mutated in place; `params_unchanged()` verifies leaf identity
   against the construction-time snapshot.
2. **Numerically faithful.** The JAX forward math reproduces the untouched
   HuggingFace torch models to float precision (~1e-7 GPT-2, ~3e-6 NeoX on
   test configs), including Conv1D-vs-Linear weight layouts, RoPE, fused QKV
   and untied LM heads.
3. **Interception is transparent.** Cached activations are always the
   *pre-modification* states; the hook output only affects downstream
   computation.
4. **JIT-safe by construction.** The default hook is `hidden + 0.0`; hooks
   must be pure array ops (no `.tolist()`, no host I/O) to stay traceable.
5. **Never crashes on limited platforms.** Missing device memory statistics
   produce diagnostics, not exceptions; batch-size changes are always
   reported, never applied silently.
6. **Fail fast with clear messages.** Bad layer indices, unsupported
   architectures and malformed inputs raise descriptive errors at
   construction/call time, not deep inside JAX tracing.
