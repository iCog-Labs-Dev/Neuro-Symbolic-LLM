# Symbolic Head and Tier 2 Retrieval

This component implements the Stage A4 symbolic-head boundary described in the
PC residual paper. It owns the GPU-side symbolic projection and integration,
the Tier 1/Tier 2 packet contract, and the CPU-side MORK client with its local
vector retrieval cache.

## Responsibility

The component sits between the PC residual output and the symbolic template
store:

```text
residual hidden state
    -> symbolic query projection
    -> compact query packet
    -> Tier 2 template retrieval
    -> compact template packet
    -> symbolic attention
    -> residual update
```

The semantic parser remains owned by `parser/`. The parser-to-template-mining
pipeline is an upstream ingestion boundary and is intentionally not coupled to
the parser model or provider implementation.

## Modules

- `head.py`: projects hidden states into symbolic key space, retrieves templates,
  computes symbolic attention, and applies the output projection.
- `mork_client.py`: MORK upload client, template records, and the selectable
  exact FAISS, HNSW, or NumPy local retrieval cache.
- `communication.py`: versioned binary query/template packet contract.
- `retrieval_bridge.py`: Tier 1/Tier 2 retrieval adapter.
- `miner_adapter.py`: provider-neutral adapter from hybrid-miner payloads to
  validated Tier 2 records.
- `losses.py`: Stage A key-space and value-space alignment losses.

## Stage A4 contract

For a hidden vector `h` with model dimension `d`:

1. `q_sym = W_sym @ h + b_sym`, where `q_sym` has dimension `k`.
2. Tier 2 returns top-`m` template key/value pairs `(p_j, v_j)`.
3. The head computes a softmax-weighted value summary `SAtt`.
4. The output update is `h_out = h + U_out @ SAtt`.

The implementation currently uses this contract as a shape and integration
baseline. It does not claim trained retrieval quality by itself.

## Current implementation status

Implemented:

- MORK upload and export smoke path through the Docker HTTP API.
- Selectable exact cosine retrieval through FAISS `IndexFlatIP` or approximate
  cosine retrieval through FAISS `IndexHNSWFlat`, with a NumPy fallback.
- Rebuilding the derived local index from exported MORK record expressions,
  including chunked vectors.
- Query/template binary serialization and round-trip tests.
- Symbolic-head projection, attention, residual update, and alignment losses.
- Thread-safe local index insertion and query operations.

Not yet implemented:

- Offline parser-output ingestion and pattern-miner orchestration.
- Support-counted template selection over a corpus.
- Frozen-LLM sentence-average key/value embedding generation.
- Held-out template precision/recall and end-to-end JAX integration.

## Team integration boundary

The parser team should provide validated MeTTa expressions or a stable
structured result that this component can consume. The Tier 2 ingestion path
should then validate, canonicalize, mine/generalize, embed, and store records.
It must not depend on a particular parser provider, prompt, or model backend.

MORK is the authoritative symbolic store. FAISS/HNSW is a derived local
retrieval index and must be rebuildable after process restart or cache loss.
`DockerMorkClient.rebuild_vector_index_from_mork()` performs that export and
rebuild operation. The `flat` backend remains the reproducible exact baseline;
the `hnsw` backend is selected explicitly with `index_backend="hnsw"` or
`MORK_INDEX_BACKEND=hnsw`.

The hybrid miner integrates through `MinerTemplatePayload`, a structural
protocol with `template_id`, `sexpr`, `key`, and `value` fields. The adapter
validates dimensions and finite values, then publishes through `MorkClient`.
The main repository therefore does not import the experimental miner package.

## Benchmark method

Benchmark both backends on the same seeded template vectors and query vectors:

```bash
python -m experiments.benchmark_tier2_mork --index-backend flat \
  --num-templates 10000 --num-queries 1000 --output-json flat.json
python -m experiments.benchmark_tier2_mork --index-backend hnsw \
  --num-templates 10000 --num-queries 1000 --output-json hnsw.json
```

Compare p50/p95/p99 latency, throughput, indexing time, and retrieval recall
against the exact Flat result. HNSW is a candidate only if it reduces latency
at the target library size without an unacceptable recall loss.

## Verification

From the repository root:

```bash
python -m pytest tests/unit/test_symbolic_losses.py -v
python -m pytest tests/unit/test_tier2_mork.py -v
python -m experiments.run_stage_a4
python -m experiments.benchmark_tier2_mork --num-templates 1000 --num-queries 100
```

The MORK integration tests and Stage A4 script require the Docker service from
`docker compose up -d --build`. The exact benchmark is a baseline; the quarter
plan's HNSW target is not satisfied by `IndexFlatIP`.