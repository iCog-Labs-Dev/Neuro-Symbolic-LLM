# Stage A4

A4 trains a symbolic head on a frozen language model and frozen A3 residual:

```text
Hidden state + A3 residual -> q_sym = h @ W_sym + b_sym
  -> CPU FAISS top-m retrieval -> GPU attention over returned K/V
  -> U_out writeback -> remaining transformer layers
```

## Objective and configuration

The optimizer updates only `W_sym`, `b_sym`, `U_out` and `V_pred`:

```text
loss = CE + key_weight * key_alignment + value_weight * value_alignment
```

Key alignment supervises the query against bank keys. Value alignment trains
`V_pred[d,k]`, a hidden-to-value predictor distinct from the value-to-hidden
writeback matrix `U_out[k,d]` (row-vector convention). With one attached head,
`U_out` learns writeback through CE. With multiple attached heads, later-layer
alignment losses also backpropagate through earlier heads: value loss can update
an earlier `U_out`, and its query projection once writeback is nonzero. It does
not directly supervise its own layer's `U_out`. This cross-layer coupling is
retained; setting either alignment weight to zero permits controlled ablations.
Labels are still required for input validation. KL and spring terms are not included.

Training and evaluation both reuse A3's frozen `clean_pass` and `predict_latents`
methods, without target-conditioned settling or A3 parameter updates.
Run metadata identifies this protocol as `frozen-a3-prediction-v1`.
Unlike the post-settlement reference in paper section 5.2, the head consumes
frozen A3 predictions rather than iteratively settled, target-conditioned states.

Stage A uses one shared, fixed K/V bank for all attached head layers, with
separate trainable projections per layer. The experiment's bank was built from
the encoder's last-layer representations; it is not a collection of distinct
layer-specific banks as denoted by the paper's per-layer keys and values.

[A4 YAML](../../../configs/stage_A/a4_symbolic_head.yaml) sets batch size,
sequence length, epochs, learning rate, alignment weights and top-m. Its
`residual_config` resolves relative to the YAML and must match the A3 checkpoint's
model, rank, layers, activation and scale. Defaults require a 10,000-record bank
and retrieve top-8.

## Inputs and data preparation

Use the installed JAX/CUDA and FAISS environment. Training requires an A3
checkpoint, source text splits, a pattern/embedding bank, alignment labels,
a MORK endpoint and a compiled native retrieval library.
See [symbolic memory](../../../symbolic_memory/README.md) for bank installation,
indexing and native builds.

```text
Dataset snapshot -> source JSONL -> parsing/mining -> exact source matches
                       +------------------------------------+
                                                            |
                                                  alignment JSON + NPZ
```

Source JSONL rows contain `document_id`, `corpus_id` and intact `source_text`.
The optional `scripts.prepare_symbolic_corpus` exporter preserves text and
provenance from local dataset snapshots; it does not run the parser. Existing
equivalent source files can be used directly.

After parsing/mining, create `data/a4/alignment.yaml`:

```yaml
training_config: ../../configs/stage_A/a4_symbolic_head.yaml
sources: train.sources.jsonl
documents: documents.jsonl
exact: [exact.jsonl]
bundle: bundle.json
patterns: patterns.jsonl
embeddings: embeddings.jsonl
output: alignment.json
```

```bash
python -m scripts.prepare_alignment_labels --config data/a4/alignment.yaml
```

Paths resolve relative to this YAML. `model` can select a local model cache;
`exact` also accepts completed-shard directories.

Preparation selects source-window/position/template pairs: 10,000 targeted,
5,000 minimum by default. Multiple labels may share a window, so pair count
does not mean distinct text count. Coverage reports both. The
`alignment-occurrences-v2` manifest references NPZ token, mask and target arrays;
padding contributes neither CE nor alignment targets. Legacy
`alignment-labels-v1` is also accepted.

Preparation excludes hypothetical-only assertion bindings and reports their
count. Explicit rule-structure bindings remain eligible; unclassified factuality
is retained and counted. The same policy applies to retrieval probes. This does
not repair upstream polarity/modality encoding: after an upstream semantic fix,
rebuild patterns, exact matches, embeddings and labels together before retraining.

Admission checks fingerprints, retokenized windows and evidence positions.
Validation/test text must be separate from training and bank-source text.
Preserve dataset parent/group separation too; text-hash checks alone cannot
enforce it.

## Training

From the repository root, with prepared artifacts in `data/a4/`:

```bash
python -m experiments.run_stage_a4 \
  --config configs/stage_A/a4_symbolic_head.yaml \
  --warm-start data/a4/a3-checkpoint.npz \
  --train-sources data/a4/train.sources.jsonl \
  --validation-sources data/a4/validation.sources.jsonl --skip-test \
  --patterns data/a4/patterns.jsonl --embeddings data/a4/embeddings.jsonl \
  --alignment-labels data/a4/alignment.json \
  --native-library data/native/retrieval.so --mork-url http://127.0.0.1:8000 \
  --output-dir runs --run-name a4-seed-0
```

Use a fresh run name. `--skip-test` leaves test sources unread; replace it with
`--test-sources` for final testing. `--model` selects a local cache and
`--retrieval-backend tcp` selects the service backend instead of native.
Run `python -m experiments.run_stage_a4 --help` for overrides.

Training prints task/epoch progress and a batch table of total, task and weighted
symbolic loss plus task perplexity. `training.log_every: 1` prints every update.
The training summary includes mean step time excluding the first update.
Compiled native runs measure CUDA copy intervals inside every 100th real optimizer
update, excluding the first 10 updates and first use of each input shape.
`--latency-every N` changes this interval; `--skip-latency` disables it.
Each epoch ends with repeated column headings and a communication summary in the
console and `training.log`. Sampled updates include their timing in `metrics.jsonl`;
`results.json` contains the aggregate. The denominator is the same instrumented
update's elapsed time. These copy intervals exclude CPU search and host overhead;
they are not a full bridge-overhead qualification. TCP and non-JIT runs report
timing as unavailable. Separate deployment/latency benchmarks remain available
through the metrics CLI; training does not replay the checkpoint afterward.

An existing Python session can reuse its initialized substrate through
`experiments.run_stage_a4.main(training_arguments, substrate=substrate)`.
A CLI process loads its own model; it cannot reuse another process's object.

## Run artifacts

Each run creates an ignored `runs/<run-name>/` directory:

| Artifact | Content |
| --- | --- |
| `config.yaml`, `meta.json`, `data.json` | Effective settings, runtime and input fingerprints |
| `status.json` | Running, completed or failed |
| `training.log` | Readable A4 training progress and final summary |
| `metrics.jsonl` | Per-step losses/gradients and epoch evaluations |
| `results.json` | Initial/final held-out CE and perplexity |
| `params/a3-frozen.npz` | Frozen A3 residual |
| `checkpoints/a4-final.npz` | Head/value-predictor weights and optimizer state |

Initial validation uses zero writeback as the frozen-A3 reference.
Lower CE alone does not establish retrieval quality or a latency target.

## Evaluation and metrics

Use `a4-metrics` after installation, or `python -m metrics.a4`:

```bash
a4-metrics list
a4-metrics show runs/a4-seed-0
a4-metrics measure-retrieval runs/a4-seed-0 --probes data/a4/probes.json
a4-metrics measure-latency runs/a4-seed-0
a4-metrics measure-representation runs/a4-seed-0 --probes data/a4/probes.json
```

`list` and `show` read saved artifacts. `retrieval FILE` scores existing ranked
IDs; `latency FILE` recalculates saved timing samples. These read-only commands
accept `--json`. The `measure-*` commands load a completed checkpoint without
retraining. `measure-retrieval` and `measure-latency` accept `--latest` to select
the completed run with the newest status-file modification time; `--runs-root`
changes the default `runs/` search.
Use `a4-metrics COMMAND --help` for options.

Measurement commands print readable results and save structured artifacts.
`report` displays a completed retrieval, representation or latency evaluation
without running the model. Representation output includes alignment, retrieval,
channel organization, query geometry, writeback and capacity controls when present.
Capture console progress and errors with `2>&1 | tee evaluation.log` in Bash;
use `set -o pipefail` when checking the measurement's exit status through a pipe.

### Retrieval quality

Prepare held-out labels with
`python -m scripts.prepare_retrieval_probes --config data/a4/probes.yaml`.
Its YAML supplies `documents` and `exclude_sources` lists, `bundle`, `patterns`,
`embeddings`, `model`, `seq_len`, `petta_root`, `output` and `split_policy`;
optional `swipl` selects SWI-Prolog. Exclude training/bank-source documents.

Expected IDs come from exact pattern bindings on admitted reference graphs,
with evidence visible in the causal window. Evaluation validates source
separation and compares the trained head, initial head and bank-support prior.
It currently accepts validation probes and reports available depths among
1, 4, 8 and 16: precision across all probes and recall on positive probes.
These labels do not measure parser accuracy or rule entailment.

For saved ranked IDs, `retrieval` accepts JSON with `label_kind`, `split` and
`probes`; each probe contains `retrieved_ids` and `relevant_ids`.
Use `semantic_firing` for template relevance or `exact_neighbor` for
HNSW-versus-Flat agreement. Neighbor agreement is not semantic quality.

### Representation diagnostics

`measure-representation` compares the initial and trained heads without updating
weights. It reports CE/PPL, next-token accuracy, base-to-adapted KL, reference-pair
alignment errors, retrieval at available depths through 16, and endpoint writeback
size. Bank-key cosine statistics describe the fixed bank, not head learning.

Channel diagnostics zero one frozen-A3 residual coordinate across the input
window. A factor is a reference template present in at least two documents;
selection uses label support only (up to `--maximum-factors`, default 32).
Positive increases in full-bank target-key NLL estimate component contributions,
averaged equally across documents. Fracture and overlap follow paper sections
14.3/14.4; zero-effect factors remain undefined, and beneficial removals are saved
separately. These are basis-dependent diagnostics, not proof of disentanglement.
Query silhouette uses Euclidean distance, expanded multilabel memberships and at
most 256 eligible probe windows. Coverage and selection policies accompany scores.

Results go under `run/evaluation/representation-<timestamp>/`: `result.json`,
`contributions.npz`, `summary.csv`, and `review.json`. Capacity results are
included in `result.json`.
Capacity controls use captured trained queries with smaller query/key dimensions
and nested ID-selected bank subsets. All controls use exact cosine search and
report reference coverage; they do not retrain smaller heads or resize values.
The full-space control also reports HNSW-versus-exact top-8 ID agreement, which is
tie-sensitive and distinct from semantic precision.
Review records contain text,
expressions and attention weights; exporting them does not constitute human review.
These measurements do not replace controlled capacity/retraining studies or the
shared Stage A residual/continual-learning evaluations.

### Latency

Run in Linux CUDA/JAX with recorded inputs, the native library and MORK available,
without competing GPU workloads. Defaults are 100 paired samples, 10 batches
and 10 warmups in training and deployment modes; compilation/warmups are excluded.
Both modes use the same recorded batches. Training discards candidate optimizer
updates; deployment measures forward/CE, not autoregressive generation.
Parameters, optimizer state and checkpoint files remain unchanged.

| Timing boundary | Includes |
| --- | --- |
| Full bridge | Synchronized GPU -> CPU search/gather -> GPU |
| Bridge minus CPU search | Transfers, dispatch and synchronization outside CPU index work |
| CUDA copy intervals | D2H/H2D event intervals from a separate instrumented pass |

Shares use the complete uninstrumented A4 step, not an A3 baseline or incremental
head slowdown. CUDA events are not PCIe bus measurements; a copy-only result
below 5% does not qualify complete-bridge overhead. `--skip-transfers` omits
the event pass.

Replay checks source/bank/checkpoint identities. `--native-library` selects a
replacement library; `--allow-source-change` explicitly records a runtime/protocol
comparison rather than exact historical replay, without bypassing data checks.
Results and raw samples go under `run/evaluation/latency-<UTC timestamp>/` or
`retrieval-<UTC timestamp>/`; checkpoints are not overwritten.

## Implementation and tests

| Location | Responsibility |
| --- | --- |
| `experiments/run_stage_a4.py` | CLI and model/data/backend setup |
| `a4/config.py`, `data.py`, `checkpoint.py` | Settings, labels/splits and A3 restoration |
| `a4/model.py`, `objective.py`, `training.py` | Forward composition, losses and head optimization |
| `a4/evaluation.py`, `benchmark.py`, `diagnostics.py`, `replay.py` | Retrieval, latency, representation evaluation and replay |
| `residual/symbolic_head/jax_head.py`, `alignment.py` | Query projection, attention, writeback and alignment |
| `residual/symbolic_head/contracts/`, `retrieval/` | Record validation and CPU index |
| `symbolic_memory/` | MORK storage, native bridge and TCP service |
| `metrics/a4/` | CLI and metric calculations |

```bash
python -m pytest tests/stages/stage_A/test_a4_model.py \
  tests/unit/symbolic_head/test_alignment.py tests/unit/symbolic_memory/test_retrieval_index.py -q
```

Run host-side A4 and metrics tests with
`python -m pytest tests/unit/a4/ tests/unit/metrics/a4/ -q`.
Numerical tests require JAX/FAISS. Native tests additionally require the compiled
library; live MORK tests require a running server.
