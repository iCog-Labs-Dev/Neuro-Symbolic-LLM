# Stage A1 — Hidden-State Residual Adapter

Stage A1 is the baseline for adapting a frozen language model. A small, trainable, low-rank residual is added to the hidden states of the model's late blocks, and only that residual is trained. The base model runs inside a [`FrozenSubstrate`](../../frozenllm/README.md) and never changes.

This is **not** weight-level LoRA: the adapter changes block *outputs*, and the base weights receive no gradient ($\nabla \theta_0 = 0$).

---

## Contents

- [Overview](#overview)
- [The Adapter](#the-adapter)
- [The A1 Objective](#the-a1-objective)
- [Training Loop](#training-loop)
- [Code Layout](#code-layout)
- [Adding a Stage](#adding-a-stage)
- [API](#api)
- [Configuration](#configuration)
- [Installation](#installation)
- [Usage](#usage)
- [Testing and Guarantees](#testing-and-guarantees)

---

## Overview

| | |
|---|---|
| **Base model** | Frozen (GPT-2 by default), run through `FrozenSubstrate` |
| **Trainable part** | Low-rank residual on late blocks, $2 \cdot d \cdot r$ parameters per block |
| **Objective** | Next-token cross-entropy, optional KL to the base, optional weight decay |
| **Data** | Three related legal domains from LexGLUE, in sequence: EU legislation (EUR-Lex) → contracts (LEDGAR) → US case law (SCOTUS) |
| **Optimizer** | `optax.adam`, over the adapter parameters only |
| **Starting point** | Identical to the base model (zero residual at step 0) |

---

## The Adapter

### Residual

At each adapted block $l$, the frozen output $h_l \in \mathbb{R}^{d}$ is replaced by:

$$\tilde h_l = h_l + s \cdot \sigma(h_l A_l)\, B_l, \qquad A_l \in \mathbb{R}^{d \times r},\ B_l \in \mathbb{R}^{r \times d}$$

| Symbol | Meaning |
|---|---|
| $r$ | Rank (`rank`) |
| $s$ | Scale: $\alpha / r$, or $1.0$ when `alpha` is `null` |
| $\sigma$ | Activation: `identity` (default), `gelu`, `relu` or `tanh` |

The residual is split into two halves that later stages can use separately:

| Method | Computes | Shape |
|---|---|---|
| `encode(layer_params, h)` | $z = \sigma(h A)$ | $[\dots, d] \to [\dots, r]$ |
| `decode(layer_params, z)` | $s \cdot z B$ | $[\dots, r] \to [\dots, d]$ |
| `residual(layer_params, h)` | `decode(encode(h))` | $[\dots, d] \to [\dots, d]$ |

The residual is computed in float32 and cast back to the dtype of $h_l$, so the bottleneck keeps its precision on bf16/fp16 models.

### Layer Selection

For a model with $L$ blocks, the adapter targets the late blocks $(L/\!/2,\ L-2]$; the final block is never adapted.

| $L$ | Adapted blocks |
|---|---|
| 12 (GPT-2 small) | 7 – 10 |
| 24 | 13 – 22 |
| 32 | 17 – 30 |

- `late_start` / `late_end` (inclusive) override the range. Invalid bounds or $L < 5$ raise `ValueError`; bounds are never clamped.
- `layers` (e.g. `[6, 10]`) selects any blocks explicitly and replaces the late range. It cannot be combined with `late_start` / `late_end`.

### Initialisation

- $A_l \sim \mathcal{N}(0, 1/d)$ (override with `init_std`) and $B_l = 0$, so the adapted model equals the base at step 0 while $\partial \mathcal{L} / \partial B_l \neq 0$.
- Deterministic in `seed`; keys are derived from the block index, so $A_l$ is stable when the layer range moves.

### Parameters

A PyTree separate from the substrate's weights:

```python
{"layer_7": {"A": f32[d, r], "B": f32[r, d]}, ..., "layer_10": {...}}
```

GPT-2 small ($d = 768$, $r = 16$, blocks 7–10): **98,304** trainable parameters.

---

## The A1 Objective

Implemented in [`a1.py`](a1.py):

$$\mathcal{L}_{A1} = \mathcal{L}_{\text{task}} + \lambda_{\text{kl}} \cdot \mathrm{KL}\big(p_{F_0} \,\|\, p_{F_0 + R}\big) + \lambda_{\text{wd}} \cdot \lVert\phi\rVert^2$$

| Term | Description |
|---|---|
| $\mathcal{L}_{\text{task}}$ | Next-token cross-entropy of the adapted model (labels default to the input ids). |
| $\mathrm{KL}$ | Forward KL from base $F_0$ to adapted $F_0 + R$; base logits are under `stop_gradient`. |
| $\lVert\phi\rVert^2$ | Squared L2 norm of the adapter parameters. |

`a1_objective` returns `(total, terms)` with `total`, `task`, `wd` and, when computed, `kl`. With $\lambda_{\text{kl}} = 0$ and no precomputed `base`, the base forward is skipped and `kl` is omitted, so a step costs one forward pass instead of two. Evaluation always computes KL.

---

## Training Loop

The loop is shared by every stage and lives in [`stages/stage_base.py`](../stage_base.py). [`experiments/run_stage_a1.py`](../../experiments/run_stage_a1.py) only wires it up:

1. Load the YAML into a `RunConfig` and the model into `FrozenSubstrate`; check the weights are frozen.
2. Build and initialise the adapter.
3. Tokenise each domain into `[n, seq_len]` blocks (seeded shuffled train batches, fixed eval batches).
4. Create the run directory; write `config.yaml` and `meta.json`.
5. Evaluate every domain at init.
6. Train on each domain in order with `optax.adam` (moments reset per domain unless `reset_optimizer: false`); after each task, save the adapter and evaluate every domain to fill `L[t][j]`.
7. Write `results.json` and exit non-zero if the base weights changed.

```text
input ids ──► FrozenSubstrate ──────────────────────────────► base logits (stop_gradient)
          │                                                         │
          └─► FrozenSubstrate + adapter hook on late blocks ──► adapted logits
                                                                    │
                                   task CE + λ_kl·KL + λ_wd·‖φ‖² ◄──┘
                                                │
                                   jax.grad w.r.t. adapter only ──► optax.adam
```

The step is not JIT-compiled, so CPU training is slow.

---

## Code Layout

| File | Purpose |
|---|---|
| [`residual/adapter.py`](../../residual/adapter.py) | `ResidualConfig`, `ResidualAdapter`, layer selection (shared by every stage) |
| [`stages/stage_base.py`](../stage_base.py) | Shared configs, data loading, `evaluate`, `run_sequential`, `summarize`, `Objective` |
| [`stages/stage_A/a1.py`](a1.py) | A1 only: `A1Config`, `a1_objective` |
| [`stages/results.py`](../results.py) | Run artifacts: `RunWriter`, `run_metadata`, `load_params` |
| [`metrics/performance.py`](../../metrics/performance.py) | Loss, KL, perplexity, forgetting, step timing |
| [`metrics/representation.py`](../../metrics/representation.py) | Loss improvement and residual compressibility |
| [`experiments/run_stage_a1.py`](../../experiments/run_stage_a1.py) | CLI runner |
| [`configs/stage_A/a1_lora_baseline.yaml`](../../configs/stage_A/a1_lora_baseline.yaml) | Run configuration |
| [`tests/residual/`](../../tests/residual/) | Unit and end-to-end tests |

---

## Adding a Stage

Stages A2–A7 reuse everything except the objective. A new stage needs:

1. **An objective config**: a frozen dataclass whose `from_dict` uses `stages.stage_base.checked_kwargs`.
2. **An objective function** matching `stages.stage_base.Objective`:

   ```python
   def my_objective(params, *, substrate, adapter, config, input_ids,
                    labels=None, base=None) -> tuple[jax.Array, dict[str, jax.Array]]:
       ...
       return total, {"total": total, "task": task, ...}
   ```

   `terms` must contain `"total"` and `"task"`; any other term is logged as `loss_<term>`.
3. **A runner**: copy `run_stage_a1.py` and swap in `load_configs(path, MyConfig)` and `my_objective`.

---

## API

### `residual`

| Name | Description |
|---|---|
| `ResidualConfig` | Frozen, validated dataclass: `rank=16`, `activation="identity"`, `alpha=None`, `init_std=None`, `late_start=None`, `late_end=None`, `layers=None`, `seed=0`. `.scale` returns $s$. |
| `ResidualAdapter(config, architecture)` | Attributes: `layers`, `hidden_size`, `scale`, `init_std`. |
| `.init_params()` | Initial `AdapterParams`. |
| `.apply(params, h, layer_idx)` | Adds the residual; other blocks pass through. |
| `.encode` / `.decode` / `.residual` | $\sigma(hA)$, $s \cdot zB$, and their composition (float32). |
| `.modify_fn(params)` | Hook for `FrozenSubstrate.run_with_interception`. |
| `ResidualAdapter.num_params(params)` / `.l2_norm_sq(params)` | Parameter count / $\lVert\phi\rVert^2$. |
| `resolve_late_layers(num_layers, late_start=None, late_end=None)` | Adapted block indices. |
| `layer_key(layer_idx)` | `f"layer_{layer_idx}"`. |

### `stages.stage_A.a1`

| Name | Description |
|---|---|
| `A1Config` | `lambda_kl=0.0`, `lambda_wd=0.0` (both `>= 0`). |
| `a1_objective(params, *, substrate, adapter, config, input_ids, labels=None, base=None)` | Returns `(total, terms)`. |

### `stages.stage_base`

| Name | Description |
|---|---|
| `checked_kwargs(cls, data)` | Dataclass kwargs, rejecting unknown keys. |
| `TrainingConfig`, `DomainConfig`, `RunConfig` | The `training:`, `domains:` and full-file configs. |
| `load_configs(path, objective_cls)` | Parses the YAML into a `RunConfig`. |
| `load_token_blocks`, `make_eval_set`, `batches` | Tokenised blocks, fixed eval batches, seeded shuffled stream. |
| `base_logits(substrate, input_ids)` | Base logits with gradients stopped. |
| `evaluate(substrate, adapter, params, eval_batches)` | Mean base/adapted loss, perplexity and KL. |
| `run_sequential(...)` | Multi-domain training loop; returns `(params, summary)`. |
| `summarize(names, init, rows)` | Eval matrices, mean seen loss, mean forgetting. |

---

## Configuration

[`configs/stage_A/a1_lora_baseline.yaml`](../../configs/stage_A/a1_lora_baseline.yaml) has four sections, each mapping to a dataclass. Unknown sections or keys are rejected.

```yaml
adapter:                # → ResidualConfig
  rank: 16
  activation: identity  # identity | gelu | relu | tanh
  alpha: null           # null → scale 1.0, else alpha / rank
  init_std: null        # null → 1/sqrt(d)
  late_start: null      # null → L//2 + 1
  late_end: null        # null → L - 2
  layers: null          # e.g. [6, 10]: explicit blocks; excludes late_start/late_end
  seed: 0

objective:              # → A1Config
  lambda_kl: 0.0
  lambda_wd: 0.0

training:               # → TrainingConfig
  model: gpt2
  seq_len: 128
  batch_size: 8
  steps: 200            # per domain, unless the domain sets its own
  learning_rate: 1.0e-3
  eval_batches: 8
  log_every: 10
  seed: 0
  reset_optimizer: true # fresh Adam moments at each domain boundary
  output_dir: runs

domains:                # → DomainConfig each; trained in order, names unique
  - name: eurlex        # [A-Za-z0-9_.-]+, used in checkpoint file names
    dataset: coastalcph/lex_glue
    dataset_config: eurlex
    text_field: text
    train_split: "train[:5000]"   # HF slicing
    eval_split: "validation[:500]"
    separator: "\n\n"   # joins rows into one token stream
    steps: null         # null → training.steps
  - name: ledgar        # contracts; same fields
    ...
  - name: scotus        # case law; same fields
    ...
```

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

On macOS (Intel Macs have no `torch>=2.6` wheels), use the Docker image with CPU-only torch:

```bash
docker build -t neuro-symbolic-llm .
docker run --rm -it -v "$PWD":/app -v nsllm-hf-cache:/root/.cache/huggingface neuro-symbolic-llm
```

The repo is mounted at `/app` (edits apply without a rebuild); `nsllm-hf-cache` persists downloaded models and datasets.

---

## Usage

### Train

```bash
python experiments/run_stage_a1.py --config configs/stage_A/a1_lora_baseline.yaml

# Overrides: rank, model, steps (every domain), output location
python experiments/run_stage_a1.py --config configs/stage_A/a1_lora_baseline.yaml \
    --rank 8 --steps 20 --output-dir runs --run-name smoke
```

> **Docker memory:** the default config exceeds Docker Desktop's ~4 GB VM (exit code 137). For a smoke run, use a smaller copy and run one container at a time:
>
> ```bash
> mkdir -p runs
> sed -e 's/^  seq_len: 128$/  seq_len: 64/' -e 's/^  batch_size: 8$/  batch_size: 2/' \
>     -e 's/^  eval_batches: 8$/  eval_batches: 2/' \
>     configs/stage_A/a1_lora_baseline.yaml > runs/smoke.yaml
> docker run --rm -v "$PWD":/app -v nsllm-hf-cache:/root/.cache/huggingface neuro-symbolic-llm \
>     python experiments/run_stage_a1.py --config runs/smoke.yaml --rank 8 --steps 5 --run-name smoke
> ```

### Run Artifacts

Each run writes a new `<output_dir>/<run_name>/` (default `a1-<UTC timestamp>`); existing directories are never overwritten.

| File | Contents |
|---|---|
| `config.yaml` | Resolved config; can be passed back with `--config` |
| `meta.json` | Git commit, library versions, argv, model, adapted layers, parameter counts |
| `metrics.jsonl` | `train` events (`loss_<term>`, `ppl`, `step_time`, …) and `eval` events (`after_task` `-1` = init) |
| `results.json` | `loss`, `ppl`, `kl`, `improvement` matrices `[t][j]`, `mean_seen_loss`, `mean_forgetting`, `params_unchanged` |
| `params/task_<t>_<name>.npz` | Adapter after task `t`; load with `stages.results.load_params` |

Forgetting is computed on loss: a positive value means an earlier domain's loss rose above its best since it was trained.

### Python

```python
import jax
import jax.numpy as jnp
from transformers import GPT2Config, GPT2LMHeadModel

from frozenllm.substrate import FrozenSubstrate
from residual import ResidualAdapter, ResidualConfig
from stages.stage_A.a1 import A1Config, a1_objective

# Tiny 12-block GPT-2; use FrozenSubstrate("gpt2") for the real checkpoint.
cfg = GPT2Config(n_layer=12, n_head=4, n_embd=32, n_positions=32, vocab_size=64,
                 bos_token_id=1, eos_token_id=2)
substrate = FrozenSubstrate(GPT2LMHeadModel(cfg).eval(), config=cfg)

adapter = ResidualAdapter(ResidualConfig(rank=4), substrate.architecture)
params = adapter.init_params()  # adapter.layers == (7, 8, 9, 10)

ids = jnp.arange(9, dtype=jnp.int32)[None, :]
(total, terms), grads = jax.value_and_grad(a1_objective, has_aux=True)(
    params, substrate=substrate, adapter=adapter,
    config=A1Config(lambda_kl=0.1), input_ids=ids,
)
assert substrate.params_unchanged()
```

---

## Testing and Guarantees

```bash
pytest tests/residual/ -v
pre-commit run --all-files
```

The suite in [`tests/residual/`](../../tests/residual/) covers the adapter, the A1 objective end to end on GPT-2 and Pythia fixtures, the sequential loop, metrics and run artifacts. It enforces:

- **Frozen base**: base weights get no gradient, are absent from the optimizer state and are unchanged after training.
- **Zero-residual start**: adapted logits equal base logits at init; non-adapted blocks and intermediates are untouched.
- **Strict configuration**: invalid layers or bounds and unknown config keys raise `ValueError`.
- **Reproducibility**: a written `config.yaml` loads back to an equal `RunConfig`.
