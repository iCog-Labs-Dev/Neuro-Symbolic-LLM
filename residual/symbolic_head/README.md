# A4 symbolic head

The A4 runtime projects a hidden state into a symbolic query, retrieves template
keys and values from CPU memory, attends over them on the GPU, and writes the
result back into the hidden state.

```text
Frozen LLM + frozen A3 residual
  -> q_sym = W_sym h + b_sym
  -> CPU FAISS top-m lookup over a MORK-verified bank
  -> GPU attention over returned keys and values
  -> U_out writeback
  -> remaining transformer layers
```

## Modules

| Module | Responsibility |
| --- | --- |
| `jax_head.py` | Query projection, symbolic attention and writeback |
| `alignment.py` | Key-space alignment and independent value prediction |
| `contracts/pattern_record.py` | Versioned pattern-record validation |
| `contracts/template_record.py` | Runtime template keys, values and identifiers |
| `retrieval/config.py` | CPU index settings |
| `retrieval/faiss_index.py` | HNSW retrieval and exact Flat controls |
| `mork_client.py` | MORK HTTP client |

`symbolic_memory/` loads artifact banks, verifies their MORK representation and
provides TCP and native CUDA/JAX retrieval boundaries. Its native backend uses
CPU FAISS, not GPU search. Full hidden states are not transferred to the CPU.

The default CPU index is HNSW. Flat is an explicit exact-search control, not
semantic ground truth. Index-neighbor agreement and source-labelled template
retrieval precision measure different things.

## Training and ownership

The head-only A4 condition freezes both the language model and inherited A3
parameters. It trains `W_sym`, `b_sym`, `U_out` and independent `V_pred` with CE
plus weighted key/value alignment. CE trains writeback; the independent value
alignment term directly trains `V_pred`.

Parser implementation, ontology definitions and miner adapters belong to their
upstream components. This runtime does not maintain a second ontology loader or
modify those components. Parsing, mining and embedding generation are offline
artifact-production steps, not operations inside the retrieval forward pass.

See [A4 training and evaluation](../../stages/stage_A/README.md#stage-a4-training)
and [bank installation and native builds](../../symbolic_memory/README.md).

## Tests

```bash
python -m pytest tests/unit/symbolic_head/test_pattern_record.py \
  tests/unit/symbolic_head/test_alignment.py tests/unit/test_retrieval_index.py \
  tests/unit/test_retrieval_protocol.py tests/unit/test_template_bank.py \
  tests/unit/test_template_storage.py
```

Numerical tests require the project JAX/FAISS environment. Native CUDA tests
additionally require the compiled library selected by `NSLLM_NATIVE_LIBRARY`.
Live MORK integration tests require a separately running compatible server.
