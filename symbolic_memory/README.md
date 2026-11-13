# Symbolic memory

Stores mined templates and embeddings in MORK, verifies readback, and builds a
derived CPU FAISS index. Parsing, mining and embedding generation happen offline.

## Inputs and indexing

Install Python dependencies with `pip install -e '.[miner]'` and provide:

| Artifact | Required content |
| --- | --- |
| Pattern JSONL | Released `hybrid_miner.contracts.PatternRecord` records |
| Embedding JSONL | `pattern_id`, `key` and `value`; optional producer metadata |

Files must cover the same unique IDs. Vectors need consistent dimensions,
finite float32 values and nonzero keys. Loading joins records by ID, preserves
original vectors and fingerprints the input files.

HNSW is the default CPU index; Flat is an explicit exact-search control.
Retrieval returns cosine scores, full IDs and original K/V. If the bank has
fewer than top-m records, only existing records are returned.

## MORK deployment

```bash
docker compose up -d --build mork-server
docker compose ps mork-server
```

MORK builds from `docker/mork/`; the root `Dockerfile` provides Python CPU
tooling. The host endpoint is `http://127.0.0.1:8000`; Compose containers use
`http://mork-server:8000`. Pass it as `--server-url` to the bank CLI or
`--mork-url` to A4. A4's default port 18000 is a separate deployment setting.
HTTP 200 or 404 at the root passes the health check; bank readback below checks
functionality. Fixed-bank retrieval does not require Redis.

The build pins the Rust image/digest, MORK revision and compatible Git revisions
through `dependencies.patch`. Cargo generates a lockfile inside the image and
builds with `--locked`; no local `Cargo.lock` is used. Registry/system packages
can vary between uncached builds. Retain the generated lockfile and image digest
with build artifacts.

**Storage warning:** the server clears `MORK_SERVER_DIR` at startup. Use dedicated
scratch space, never source, bank, checkpoint or results directories.
The `./data/mork_data:/app/data` mount alone does not persist the in-memory bank;
reinstall and verify after restart.

## Install and verify a bank

With prepared artifacts in `data/a4/`, run from the repository root:

```bash
python -m symbolic_memory inspect-bank \
  --patterns data/a4/patterns.jsonl --embeddings data/a4/embeddings.jsonl

python -m symbolic_memory install-bank \
  --patterns data/a4/patterns.jsonl --embeddings data/a4/embeddings.jsonl \
  --server-url http://127.0.0.1:8000

python -m symbolic_memory check-bank \
  --patterns data/a4/patterns.jsonl --embeddings data/a4/embeddings.jsonl \
  --server-url http://127.0.0.1:8000
```

Inspection is local; `check-bank` is read-only. Installation uploads missing
records and verifies native structure and linked payloads. Repeating it skips
identical records. Success reports `matches: true` with empty `missing_ids`,
`conflicting_ids` and `extra_handles`; mismatches/errors return exit code 2.

Counts represent logical templates, not fragments. Installation reports
`uploaded_count`, `already_present_count`, `verified_count` and `native_count`.
`index_updated` remains false: index loading is separate.
Defaults are 64 records per batch and a 30-second HTTP timeout; use
`--batch-size` and `--timeout` to override them.

## Structural matching

MORK matching unifies native terms and enforces shared-variable constraints:

```bash
python -m symbolic_memory match-bank \
  --patterns data/a4/patterns.jsonl --embeddings data/a4/embeddings.jsonl \
  --server-url http://127.0.0.1:8000 \
  --pattern '(, (Agent $event $agent) (Event $event))'
```

Results contain full template IDs in `matching_ids`; an empty list is valid.
This is structural matching, not FAISS similarity search.

For Python CPU retrieval, use `bank.load_bank`, then
`MorkTemplateStore.load_index(bank)` and `index.query_top_k(query, top_m=8)`.
Index loading verifies MORK records once; retain the index for subsequent queries.

## Native GPU/CPU bridge

The default A4 backend uses `NativeJaxRetriever` and JAX CUDA FFI:

```text
GPU query -> D2H copy -> CPU normalization/search/KV gather -> H2D results
```

`from_mork` verifies the bank, builds its CPU index and serializes it once into
a C++ FAISS snapshot, preserving graph/search settings and ordering. No Python
callback or MORK request runs per query. Only queries and selected float32 K/V,
scores and registry indices cross the device boundary; the bank stays on CPU.

### Build and use

Build in Linux with CUDA/JAX, a C++17/OpenMP compiler and a matching FAISS SDK:

```bash
python -m symbolic_memory.native_bridge build \
  --faiss-prefix /path/to/faiss-sdk --output data/native/retrieval.so
```

The SDK needs `include/faiss/Index.h` and `lib/libfaiss.so` (or `lib64`),
matching Python's FAISS version; a Python wheel may not supply them. CUDA runtime
and JAX FFI headers are also required. `--faiss-library` selects a compatible
shared variant; use only instruction sets supported by the CPU.
Without a prefix, the builder checks the active environment, `/usr/local` and
`/usr`. Dependencies are not downloaded or compiled at import.

The adjacent `.so.json` records ABI 2 and source/JAX/FAISS fingerprints. Loading
rejects mismatches; rebuild after relevant source or dependency changes.
Different FAISS compile variants can affect parity and performance.

```python
from pathlib import Path

import jax

from symbolic_memory.bank import load_bank
from symbolic_memory.native_bridge import NativeJaxRetriever

bank = load_bank(Path("data/a4/patterns.jsonl"), Path("data/a4/embeddings.jsonl"))
with NativeJaxRetriever.from_mork(
    bank, server_url="http://127.0.0.1:8000",
    library_path=Path("data/native/retrieval.so"), top_m=8,
) as retrieve:
    templates = jax.jit(retrieve)(gpu_queries)
    jax.block_until_ready(templates)
```

Queries are rank-1/rank-2 float32 arrays with the bank's vector dimension.
Pass `retrieve` to `SymbolicHead` for model integration. Result K/V shapes are
`[batch, min(top_m, bank_count), dimension]`; scores/indices omit the last axis.
Indices map through `template_ids`. Selection is discrete; retain the original
query for differentiable attention over constant bank vectors.

Finish submitted JAX work before closing. Handles are not reused; closed or
invalidated handles fail. Each retriever supports eight query shapes and one
selected CUDA device. Index mutations require a new retriever. Failed CUDA
cleanup retains buffers rather than freeing in-flight memory.

### Modes and diagnostics

Defaults are `return_mode="sync"` and `output_layout="packed"`:

| Option | Behavior |
| --- | --- |
| `sync` | Host waits for D2H readiness and H2D completion |
| `event` | Wait for D2H, enqueue H2D/completion event without the final host wait |
| `packed` | One H2D result submission, followed by GPU unpacking |
| `direct` | Four H2D submissions into separate result buffers |

Event-mode consumers stay stream-ordered; buffers wait for completion before
reuse/freeing. All layouts transfer the same bytes. Event-mode `h2d_seconds`
measures enqueue time, not completed transfer latency.

`capture_queries()` captures C++ query batches for inspection (default limit:
64 calls). `transfer_timing()` with `transfer_stats()` measures CUDA copy-event
intervals separately from CPU search. These scopes cannot nest, overlap, close
the retriever or receive concurrent submissions.
Event intervals can include scheduling/submission gaps; they are not PCIe bus
measurements. Use separate instrumented/uninstrumented passes and the
[A4 metrics CLI](../stages/stage_A/a4/README.md#evaluation-and-metrics) for timings.

## TCP service and adapters

For a separate retrieval process:

```bash
python -m symbolic_memory serve-bank \
  --patterns data/a4/patterns.jsonl --embeddings data/a4/embeddings.jsonl \
  --server-url http://127.0.0.1:8000 --host 127.0.0.1 --port 9090
```

Startup verifies the bank and builds one index. `RetrievalClient(host, port,
bank.input_snapshot_sha256)` binds a persistent connection to that snapshot;
`query_top_k` returns full IDs, scores and K/V. `client.reconnect()` rechecks the
descriptor after idle periods; it does not retry failed requests or reload data.
New banks require restarting the service and reconnecting clients.
MORK restarts do not automatically invalidate a loaded index.

Defaults are 1,024 queries/request, top-m up to 64, eight connections and a 64 MiB
message ceiling. FAISS access is locked. The protocol has no TLS/authentication:
use loopback or an access-controlled network/tunnel. See `serve-bank --help`.

`jax_bridge.JaxRetriever` uses an ordered JAX callback around the TCP client;
enable `JAX_PLATFORMS=cuda,cpu`, construct outside JIT and pass batches directly
rather than vmapping. `device.retrieve_cuda` provides an eager PyTorch CUDA
boundary. Both return results to the GPU without projection, attention or
writeback; failures propagate without synthetic fallback.

## Storage and recovery

A namespace derives from the bank fingerprint and storage version:

```text
(BankTemplateV1 <bank> <handle> <native-template-expression>)
(BankRecordV1   <bank> <handle> <part-count> <checksum>)
(BankPartV1     <bank> <handle> <part-index> <hex-payload>)
```

Native bodies preserve structural access. Linked payloads retain full IDs,
original templates, metadata and float32 K/V with checksums. Long constants and
reserved `BankSym_` names use collision-checked bounded symbols without changing
variable sharing. Unsupported arity/export sizes are rejected, not silently
replaced by opaque payloads. The low-level
`residual/symbolic_head/mork_client.py` API remains independent.

Installation requires a single writer and is not transactional. Interrupted
uploads can leave fragments; inspect before retrying. Compatible fragments are
reused, conflicting data is rejected, and unrelated namespaces remain untouched.
After server restart, reinstall retained artifacts and explicitly reload indexes.

## Tests

```bash
export NSLLM_NATIVE_LIBRARY="$PWD/data/native/retrieval.so"
python -m pytest tests/unit/symbolic_memory/ -q -rs
```

Compiled tests skip without the library; CUDA tests need a real JAX GPU and live
MORK tests need a server. Only locally serialized verified indexes are supported.
Bank consistency and self-key tests do not establish held-out retrieval quality;
see [A4 evaluation](../stages/stage_A/a4/README.md#evaluation-and-metrics).
