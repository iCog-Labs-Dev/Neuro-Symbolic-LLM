# A4 infrastructure benchmarks

Opt-in comparisons, separate from A4 training and checkpoint evaluation:

| Command | Comparison | Timing boundary |
| --- | --- | --- |
| `search` | HNSW `efSearch` sweep versus exact cosine Flat | CPU query validation, normalization, search and gather |
| `tcp-local` | Loopback TCP versus same-process Python callback | Synchronized JAX GPU -> CPU retrieval -> GPU |
| `native-prototype` | Pinned-memory CUDA-FFI prototype versus Python callback | Same synchronized boundary, with CPU search-excluded timings |

The prototype deliberately calls **Python CPU FAISS in both conditions**. It
isolates the native boundary; it is not the production all-C++ bridge and must
not be presented as a direct production-native-versus-TCP comparison.

## Prerequisites

- Install the project and FAISS in a compatible environment. Run from the
  repository root or an editable installation.
- Supply matching `patterns.jsonl` and `embeddings.jsonl` and an already installed
  MORK bank. These commands verify/read it; they do not seed or modify it. See
  [symbolic memory](../../../../symbolic_memory/README.md) for installation.
- Transport comparisons need Linux with CUDA-enabled JAX and an available GPU.
  Avoid competing GPU/CPU workloads. `search` does not need a GPU.
- The prototype additionally needs a C++17 compiler, CUDA runtime headers and
  `libcudart`. Discovery uses NVIDIA Python packages, or `--cuda-root /path/to/cuda`.
  Compilation occurs once before sampling, with no package installation.

No language-model weights, A3 checkpoint, alignment labels or training are
required: these are bank/query infrastructure controls, not model-quality tests.

## Commands

Use a fresh output directory for each invocation:

```bash
python -m stages.stage_A.a4.benchmarks search --patterns data/a4/patterns.jsonl --embeddings data/a4/embeddings.jsonl --output runs/a4-search
python -m stages.stage_A.a4.benchmarks tcp-local --patterns data/a4/patterns.jsonl --embeddings data/a4/embeddings.jsonl --output runs/a4-tcp-local
python -m stages.stage_A.a4.benchmarks native-prototype --patterns data/a4/patterns.jsonl --embeddings data/a4/embeddings.jsonl --output runs/a4-native-prototype
```

Defaults: MORK `http://127.0.0.1:18000`, 256 queries, top-8, two CPU FAISS
threads, ten warmups and 100 samples per variant. Override with `--mork-url`,
`--query-count`, `--top-m`, `--threads`, `--warmups`, `--samples`; `search` also
accepts `--ef-search 16 50 100`.

Without `--queries`, evenly spaced bank keys form the query control. Optional
`--queries queries.npy` supplies finite, nonzero float32 vectors of shape
`[query-count, bank_dimension]`; retain their external provenance. Neither input
choice provides semantic relevance labels.

## Measurement and artifacts

Setup, MORK readback, index construction, compilation and warmups are outside
sampling. Variant order is rotated and reversed. All device calls synchronize
before their timer stops; CPU thread settings are applied in the executing
callback/server thread. Transport checks compare complete K/V, scores and
registry indices against the CPU index on the original and negated queries.
The prototype also checks that stopped retrieval gradients preserve the direct
query gradient.

Console summaries show mean, p50 and p95. Each output directory contains:

- `result.json`: settings, versions, bank/query/source fingerprints, checks and
  summary timings; failed execution is recorded rather than marked completed.
- `samples.csv`, `summary.csv`: raw timed samples and aggregate milliseconds.
- Prototype only: `build.log` and `native_boundary.so`.

Keep these generated artifacts under ignored `runs/`. The `.cc` source is
packaged; compiled libraries are not. Run hardware benchmarks manually, not as
CPU-only CI performance gates.

Search agreement is measured against exact vector neighbors, not alignment
labels. Ties can give different IDs with equivalent scores; the tie-aware
agreement uses the exact top-m score cutoff with a recorded tolerance.

Prototype search-excluded time subtracts the timed shared `index.query_top_k`
interval from its synchronized round trip. It retains boundary/packing work.
Its D2H/H2D subintervals are CPU-clock enqueue-and-wait timings, **not CUDA-event
copy-only timings or physical PCIe occupancy**.

These maintained commands derive from the recorded standalone experiments, but
use the current implementation and explicit settings. New results are new
measurements, not guaranteed reproductions of historical report numbers. In
particular, the prototype uses the shared public CPU query operation rather than
the historical native driver's private optimized search/gather routine.

No model denominator or `<5%` claim is computed here. Use the existing
`python -m metrics.a4 measure-latency RUN` for complete-A4 training/forward
timings. Do not multiply speedups from different comparisons.
