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

Key alignment supervises queries against bank keys. Value alignment trains
`V_pred[d,k]`, separate from writeback `U_out[k,d]`. Writeback learns through CE;
later-head alignment losses can also update earlier heads through their modified
hidden states. Value loss does not directly supervise its own head's writeback.
Either alignment weight can be zero; labels remain required. The objective
excludes KL and spring terms.

Training and evaluation use frozen A3 `clean_pass` and `predict_latents`
(`frozen-a3-prediction-v1`), without target-conditioned settling. This differs
from the post-settlement head in paper section 5.2. All head layers share one
fixed K/V bank built from last-layer encoder representations, with separate
trainable head projections rather than the paper's layer-specific banks.

[A4 YAML](../../../configs/stage_A/a4_symbolic_head.yaml) sets batch size,
sequence length, epochs, learning rate, alignment weights and top-m. Its
`residual_config` resolves relative to the YAML and must match the A3 checkpoint's
model, rank, layers, activation and scale. Defaults require a 10,000-record bank
and retrieve top-8.

## Inputs and data preparation

Training requires JAX/CUDA, FAISS, an A3 checkpoint, source text splits,
a pattern/embedding bank, alignment labels, MORK and a native retrieval library.
See [symbolic memory](../../../symbolic_memory/README.md) for bank installation,
indexing and native builds.

Source JSONL rows contain `document_id`, `corpus_id` and intact `source_text`.
`scripts.prepare_symbolic_corpus` optionally exports this format from dataset
snapshots, preserving provenance without parsing the text.

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

Preparation defaults to single-template pairs (`alignment-occurrences-v2`),
targeting 10,000 examples with a 5,000 minimum. Set `group_positives: true` for
query-balanced sets (`alignment-query-sets-v1`): selection balances documents
and windows, and each query retains supplied exact-match positives with complete,
visible evidence. Target/minimum counts then refer to queries, not pairs.

Grouped key loss compares each positive against unlabelled bank entries,
excluding other known positives; unlabelled entries are not certified negatives.
Value errors are averaged over positives per query. Coverage reports distinct
windows, visible prefixes and positive memberships. JSON manifests reference
NPZ token, mask and target arrays; padding is excluded from losses. Legacy
`alignment-labels-v1` inputs retain their single-positive objective.

Training labels and retrieval probes exclude hypothetical-only assertion
bindings, retain explicit rule-structure bindings, and count retained bindings
with unclassified factuality. After upstream semantic changes, rebuild patterns,
exact matches, embeddings and labels together.

Loading checks fingerprints, tokenized windows and evidence positions.
Validation/test text must exclude training and bank-source text; dataset
parent/group separation must also be preserved by the source split.

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

Console and `training.log` show epoch progress, total/task/weighted symbolic
losses, task PPL and repeated column headings. `training.log_every` controls
update frequency. Mean step time excludes the first update.

Compiled native runs sample CUDA copy intervals every 100th optimizer update,
excluding the first 10 updates and first use of each input shape. Configure with
`--latency-every N` or disable with `--skip-latency`. Samples are saved in
`metrics.jsonl`, with epoch summaries and an aggregate in `results.json`.
Copy shares use the same instrumented update's elapsed time and exclude CPU
search and host overhead. TCP and non-JIT runs do not provide these timings.

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

`list`, `show` and `report` display saved artifacts. `retrieval FILE` scores saved
ranked IDs; `latency FILE` summarizes saved timings. `measure-*` loads a completed
checkpoint without retraining, prints results and saves structured artifacts.
Retrieval and latency measurements support `--latest` and `--runs-root`.
Use `a4-metrics COMMAND --help` for options.

### Retrieval quality

Prepare held-out labels with
`python -m scripts.prepare_retrieval_probes --config data/a4/probes.yaml`.
Its YAML supplies `documents` and `exclude_sources` lists, `bundle`, `patterns`,
`embeddings`, `model`, `seq_len`, `petta_root`, `output` and `split_policy`;
optional `swipl` selects SWI-Prolog. Exclude training/bank-source documents.

Expected IDs come from exact pattern bindings on admitted reference graphs,
with evidence visible in the causal window. Validation-probe evaluation checks
source separation and compares trained/initial heads and a bank-support prior.
It reports precision across all probes and recall on positive probes at available
depths among 1, 4, 8 and 16, not parser accuracy or rule entailment.

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
`contributions.npz`, `summary.csv` and `review.json`.
Capacity controls use captured trained queries with smaller query/key dimensions
and nested ID-selected bank subsets. All controls use exact cosine search and
report reference coverage; they do not retrain smaller heads or resize values.
The full-space control also reports HNSW-versus-exact top-8 ID agreement, which is
tie-sensitive and distinct from semantic precision.
Review records contain text, expressions and attention weights, not human-review
certification. These diagnostics do not replace capacity/retraining studies or
shared Stage A residual/continual-learning evaluations.

### Latency

Standalone search-effort and transport comparisons are documented in
[benchmarks](benchmarks/README.md). They do not run training or replace the
complete-model latency measurement below.

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
Retrieval and latency results go under timestamped `run/evaluation/` directories;
checkpoints are not overwritten.

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
python -m pytest tests/residual/stage_A/test_a4_model.py \
  tests/unit/symbolic_head/test_alignment.py tests/unit/symbolic_memory/test_retrieval_index.py -q
```

Run host-side A4 and metrics tests with
`python -m pytest tests/unit/a4/ tests/unit/metrics/a4/ -q`.
Numerical tests require JAX/FAISS. Native tests additionally require the compiled
library; live MORK tests require a running server.
