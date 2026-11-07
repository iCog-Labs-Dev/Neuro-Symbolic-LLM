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

Training steps and evaluation are JIT-compiled with XLA by default (`training.jit: true`), significantly speeding up CPU and accelerator execution. Pass `--no-jit` (or set `jit: false` in YAML) for eager execution when debugging Python-side hooks.

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
  alpha: 16             # scale = alpha / rank (1.0)
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
  batch_size: 16
  epochs: 3         # per domain, unless the domain sets its own
  learning_rate: 1.0e-3
  eval_batches: 8
  log_every: 10
  seed: 0
  reset_optimizer: true # fresh Adam moments at each domain boundary
  output_dir: runs
  jit: true             # JAX XLA JIT compilation enabled

domains:                # → DomainConfig each; trained in order, names unique
  - name: eurlex        # [A-Za-z0-9_.-]+, used in checkpoint file names
    dataset: coastalcph/lex_glue
    dataset_config: eurlex
    text_field: text
    train_split: "train[:5000]"   # HF slicing
    eval_split: "validation[:500]"
    separator: "\n\n"   # joins rows into one token stream
    epochs: null        # null → training.epochs
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

# Overrides: rank, model, epochs (every domain), output location
python experiments/run_stage_a1.py --config configs/stage_A/a1_lora_baseline.yaml \
    --rank 8 --epochs 1 --output-dir runs --run-name smoke
```

> **Docker memory:** the default config exceeds Docker Desktop's ~4 GB VM (exit code 137). For a smoke run, use a smaller copy and run one container at a time:
>
> ```bash
> mkdir -p runs
> sed -e 's/^  seq_len: 128$/  seq_len: 64/' -e 's/^  batch_size: 8$/  batch_size: 2/' \
>     -e 's/^  eval_batches: 8$/  eval_batches: 2/' \
>     configs/stage_A/a1_lora_baseline.yaml > runs/smoke.yaml
> docker run --rm -v "$PWD":/app -v nsllm-hf-cache:/root/.cache/huggingface neuro-symbolic-llm \
>     python experiments/run_stage_a1.py --config runs/smoke.yaml --rank 8 --epochs 1 --run-name smoke
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

## Stage A4 training

Stage A4 trains a symbolic head on a frozen language model and frozen A3
residual. The optimizer contains `W_sym`, `b_sym`, `U_out` and `V_pred` only.
Its objective is CE plus weighted key-space and value-space alignment.

This is a head-only isolation condition. Joint A3/A4 learning is a separate
experiment. Continuous-state settlement still runs with fixed A3 weights; its
inference energy is not added to the symbolic-head optimizer objective.
`V_pred` maps hidden states into value space, while `U_out` maps attended values
back into hidden space. They are independent: value alignment directly trains
`V_pred`; CE trains the writeback through the model's predictions.

```bash
python -m experiments.run_stage_a4 --help
python -m experiments.run_stage_a4 \
  --config configs/stage_A/a4_symbolic_head.yaml --model LOCAL_MODEL \
  --warm-start TRAINED_A3_CHECKPOINT.npz \
  --train-sources train.jsonl --validation-sources validation.jsonl --skip-test \
  --patterns patterns.jsonl --embeddings embeddings.jsonl \
  --alignment-labels firing-labels.json \
  --native-library native/retrieval.so --mork-url http://127.0.0.1:18000 \
  --output-dir runs --run-name a4-seed-0 --epochs 3
```

The configuration's `residual_config` resolves relative to the A4 YAML file and
specifies the inherited A3 architecture and PC-settlement settings. It must
match the checkpoint's model, rank, layers, activation and scale. Training settles
PC latents without updating A3 weights; evaluation uses predicted latents without
target-conditioned settlement.

In an existing Python session, pass the initialized `FrozenSubstrate` to the
runner to reuse its model and tokenizer without loading another copy:

```python
from experiments.run_stage_a4 import main

main(training_arguments, substrate=substrate)
```

`training_arguments` is the list of command-line arguments shown above. The
supplied substrate must match the configured model and inherited checkpoint.
A standalone command starts a new process and loads through the existing shared
loader; it cannot reuse an instance from another process. A4 does not change
the shared loader or TorchAX's loading modes.

Prepare source-linked examples with `python -m scripts.prepare_alignment_labels`.
The command accepts `--config preparation.yaml`, or explicit input flags. A
preparation configuration supplies `training_config`, `documents`, `exact`,
`bundle`, `patterns`, `embeddings` and `output`; `sources` defaults to `documents`.
`training_config` references the A4 YAML for sequence length and seed; `model`
can override its model identifier with a pinned local cache. Input/output paths
resolve relative to the preparation YAML. `exact` accepts files or a directory
of completed mining shards; failed shards are excluded without a merge step.

Occurrence preparation retains short documents with right padding and selects
deduplicated template/source/window/endpoint pairs. It targets 10,000 pairs by
default and fails below 5,000. Multiple targets sharing a context remain separate
supervised pairs under the existing single-positive alignment objective. Pair,
template, context and document counts are reported separately; pair counts do
not establish independent text examples or semantic retrieval precision.

The `alignment-occurrences-v2` JSON manifest references adjacent compressed token,
mask and target arrays. Admission checks bank/source/array identity, re-tokenizes
the source windows, verifies evidence endpoints, and rejects duplicate pairs.
The runner's `--minimum-examples` defaults to 5,000. Padding is excluded from CE
and from the inherited A3 settlement energy. Right-padded causal inputs cannot
expose future padding to real-token queries. A4 parameter training remains CE
plus key/value alignment; no KL or spring term is added to that objective.

Validation/test documents are independently chunked with masked right-padded
tails and must not overlap supplied training or bank-source text. Preserve
dataset parent/group separation as well; text-hash admission alone is not a
complete dataset-level leakage audit. Legacy `alignment-labels-v1` manifests
remain readable, subject to the runner's example-count floor.
`--skip-test` leaves test sources unread; supply `--test-sources` for final testing.

| Module | Responsibility |
| --- | --- |
| `experiments/run_stage_a4.py` | Configuration, model/data/backend setup and run artifacts |
| `a4/config.py` | Validated A4 objective, retrieval and shared training settings |
| `a4/data.py` | Source-split and firing-label admission |
| `a4/checkpoint.py` | Restore inherited A3 residual weights |
| `a4/model.py` | Continuous-residual and symbolic-head forward composition |
| `a4/objective.py` | Cross-entropy and key/value alignment |
| `a4/training.py` | Head-only optimizer steps, evaluation and checkpointing |
| `a4/settlement.py` | Frozen continuous-state inference without changing A3 |

Defaults require 10,000 bank records and retrieve top-8. Native retrieval requires
a separately built Linux CUDA/JAX library and its matching manifest. Use
`--retrieval-backend tcp` for the service backend.

Each run writes raw metrics, configuration and data fingerprints, evaluation
results, `params/a3-frozen.npz`, and `checkpoints/a4-final.npz`. The A4 checkpoint
contains only symbolic-head weights and their optimizer state; the frozen A3
weights are saved separately. Mining inspections, benchmarks and historical
comparisons are local tools under ignored `data/check/`.

`results.json` includes initial and final validation loss/perplexity, validation
CE change, training-example and alignment-coverage counts. Initial validation
uses the zero-writeback head, providing the frozen-A3 starting reference. A
negative CE change indicates lower loss on the chosen validation workload; it
does not by itself establish semantic retrieval quality or the latency target.

### A4 metrics CLI

After installing the project, use `a4-metrics`; without installation, use
`python -m metrics.a4` with the same arguments:

```bash
a4-metrics show runs/a4-seed-0
a4-metrics retrieval probes.json
a4-metrics latency data/a4-timing/result.json
```

These three read-only commands inspect recorded A4 observations, not new model
inference or benchmarks. Add `--json` for structured output. `show` displays
training and evaluation losses and preserves incomplete-run status. It does not
infer semantic retrieval quality from alignment losses.

`retrieval` accepts explicitly labelled ranked IDs:

```json
{
  "label_kind": "semantic_firing",
  "split": "validation",
  "probes": [
    {"retrieved_ids": ["template-a", "template-b"], "relevant_ids": ["template-a"]}
  ]
}
```

Use `exact_neighbor` instead of `semantic_firing` for HNSW-versus-exact-index
agreement. Scores use the supplied relevance sets and actual returned counts;
complete semantic labels and source-disjoint probes require independent review.

`latency` recomputes statistics from the existing checkpoint benchmark's raw
paired `samples`, not its saved pass flags. When adjacent
`cuda-transfers-training.json` or `cuda-transfers-deployment.json` exists, it
checks sample/batch/denominator alignment before computing transfer-only shares.
Transfer-only, bridge-minus-CPU-search and full-bridge shares remain separate;
none is presented as an unqualified overall latency pass. The command neither
loads checkpoints nor modifies run artifacts.

### Measure A4 latency

The fresh measurement path is included in the repository; it does not import
scripts from ignored `data/check/`. In the Linux CUDA/JAX environment used for
training, with the recorded inputs and MORK endpoint still available:

```bash
a4-metrics list
a4-metrics measure-latency --latest
```

`--latest` explicitly selects the completed A4 checkpoint with the newest
`status.json` modification time under `runs/` and prints its resolved path.
Alternatively, pass a specific run directory. `--runs-root` changes the search
directory. A fresh clone has no runs until training or artifact restoration.

The defaults measure both training and deployment paths with 100 paired samples,
10 distinct batches and 10 warmups, followed by a separate CUDA-event transfer
pass. Compilation and warmups are excluded. Both paths use the same recorded
training batches; deployment here means a forward/CE kernel, not generation or
held-out task-quality evaluation. Other GPU processes must be stopped.

Outputs are created automatically under the selected run:

```text
evaluation/latency-<UTC timestamp>/
  config.yaml
  status.json
  result.json
  metrics.jsonl
  cuda-transfers-training.json
  cuda-transfers-deployment.json
```

The checkpoint parameters and optimizer state are restored and reused without
advancing them. Candidate optimizer outputs are discarded, frozen-weight and
checkpoint-file invariance are checked, and no checkpoint is written. MORK bank
readback, source/data identities and the saved optimizer schema are checked.
Optimizer restoration does not unpickle checkpoint metadata.

Changed training source is rejected by default. `--allow-source-change` records
an intentional runtime comparison instead of claiming exact historical replay;
it does not bypass data, bank or checkpoint checks. `--native-library` explicitly
selects a replacement compiled library. The runtime can also accept an existing
substrate through the Python `measure_latency(..., substrate=...)` API without
changing the shared loader.

Measurement orchestration lives in `a4/benchmark.py`, checkpoint restoration in
`a4/replay.py`, and pure timing calculations in `metrics/a4/latency.py`. GPU
execution of the promoted driver must be qualified separately from host tests.

### Evaluate A4 retrieval

`a4-metrics measure-retrieval --latest --probes probes.json` restores the
completed checkpoint and evaluates its deployment-time queries against prepared
reference labels. It compares the trained head with its seeded initial head,
using identical frozen A3/base weights. Results are written under the run's
`evaluation/retrieval-<UTC timestamp>/` directory; weights are not updated.
An additional query-independent prior ranks templates by their original
bank-source qualifying-binding counts, without fitting to held-out labels.

Prepare labels with
`python -m scripts.prepare_retrieval_probes --config preparation.yaml`.
The YAML supplies explicit `documents` and `exclude_sources` file lists,
`bundle`, `patterns`, `embeddings`, `model`, `seq_len`, `petta_root`, `output`
and a descriptive `split_policy`. Optional `swipl` selects an existing
SWI-Prolog executable. The installed miner must support the bank's versioned
predicates; mismatches abort rather than dropping templates.

Preparation excludes complete training/bank documents by ID and source text,
admits each remaining annotated graph through PeTTa, and checks every fixed-bank
pattern. Constant/arity impossibility is an exact rejection, not ANN pruning.
Relevance sets contain every distinct-ground-fact binding whose source evidence
is visible in the causal token window. They are not sampled training positives.
Bindings, admission artifacts, source hashes and coverage remain with the probe
manifest. Dataset-reference graph firing does not certify natural-language parser
accuracy or executable rule entailment; semantic review remains explicit.

The evaluator retokenizes probe windows, rechecks source separation and artifact
identities, and reports precision/recall at measured depths among 1, 4, 8 and 16.
It does not report depth 16 when the run retrieves only 8. Positive-probe recall
and all-probe precision are distinguished; known-negative probes are retained.
The CE comparison uses the same causal windows and counts non-padding targets,
not an official task-test split or a whole-document perplexity estimate.
