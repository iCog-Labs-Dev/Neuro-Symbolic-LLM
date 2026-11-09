# Symbolic Template Banks

Associates released miner patterns with offline embeddings, stores native MeTTa
templates in MORK and builds a FAISS retrieval index.

## Requirements

- Python dependencies installed with `pip install -e '.[miner]'`.
- A compatible MORK HTTP server.
- Pattern and embedding JSONL artifacts covering the same set of unique IDs.

The package does not start MORK or run parsing, mining, embedding generation,
training or candidate promotion.

## Docker deployment

`docker/mork/Dockerfile` builds the MORK Rust server using only the
`docker/mork/` directory as its build context. Its entrypoint is in the same
directory. The root `Dockerfile` and `make build` retain Base's Python CPU
application/testing image.

```bash
docker compose up -d --build mork-server
docker compose ps mork-server
```

The MORK endpoint on the host is `http://127.0.0.1:8000`; containers on the
Compose network use `http://mork-server:8000`. Pass this endpoint explicitly to
the bank CLI's `--server-url` or the A4 runner's `--mork-url`. The A4 runner's
JupyterHub-oriented default port 18000 is a different deployment configuration.
`./data/mork_data` is mounted at `/app/data`. The HTTP health check accepts 200
or 404 at the root; bank installation/readback is a separate functional check.

The original deployment also defines `redis-queue`; fixed-bank A4 does not use
it, and starting the explicit `mork-server` service does not start Redis.

```bash
docker compose run --rm app bash
docker compose run --rm test
```

The Python image is CPU tooling, not a qualified CUDA training environment.
The MORK recipe pins the Rust image by version and digest and the upstream MORK
revision. `docker/mork/dependencies.patch` pins compatible PathMap, Freeze and
gxhash Git revisions in the downloaded workspace manifest. Cargo generates its
lockfile inside the image, then builds the server with `--locked`; no local
`Cargo.lock` is required or copied into the image. Cargo builds PathMap as a
dependency, without a separate standalone build.

Registry dependencies are resolved during lockfile generation and can vary
between uncached builds. Registry and system package versions are not fully
pinned, so this recipe does not guarantee an identical dependency graph or
binary across builds. Retain the generated lockfile and image digest with build
artifacts when recording results. A successful build still requires server
startup and native bank installation/readback checks before qualification.

`MORK_SERVER_DIR` is temporary request storage: the pinned server clears that
directory at startup. Use a dedicated scratch directory, never a source, bank,
checkpoint or results directory. The `/app/data` mount alone does not configure
persistence of the in-memory bank; reinstall and verify the bank after restart.

## Input contract

| Artifact | Required content |
| --- | --- |
| Pattern JSONL | Released `hybrid_miner.contracts.PatternRecord` records |
| Embedding JSONL | `pattern_id`, `key` and `value`; optional producer metadata |

Records are associated by ID, independently of row order. Vectors must have
consistent dimensions, finite float32 values and nonzero keys. Loading preserves
producer values without normalization or projection.

`load_bank(patterns_path, embeddings_path)` returns `TemplateRecord` instances
and retains the source pattern records. SHA-256 fingerprints identify the exact
input files and their combined identity. These identify artifacts, not model
executions or runtime environments. Reproduction also requires producer code,
model revisions and preprocessing settings.

## Command-line interface

Run commands from an installed environment or the repository root. The following
PowerShell variables specify the input artifacts and endpoint:

```powershell
$patterns = "path/to/patterns.jsonl"
$embeddings = "path/to/embeddings.jsonl"
$server = "http://127.0.0.1:8000"
```

### Inspect inputs

```powershell
python -m symbolic_memory inspect-patterns --input $patterns
python -m symbolic_memory inspect-bank --patterns $patterns --embeddings $embeddings
```

Inspection reports record counts, dimensions, input fingerprints and example
ID/expression associations. It does not contact MORK.

### Install and check a bank

```powershell
python -m symbolic_memory install-bank --patterns $patterns --embeddings $embeddings --server-url $server
python -m symbolic_memory check-bank --patterns $patterns --embeddings $embeddings --server-url $server
```

Installation uploads missing records and verifies payload integrity and native
template structure. Repeating installation skips identical records.
`--batch-size` controls the number of logical records per upload batch
(default: 64); `--timeout` controls HTTP timeouts in seconds (default: 30).

| Output | Meaning |
| --- | --- |
| `uploaded_count` | Logical records newly completed by the installation |
| `already_present_count` | Records already present before installation |
| `verified_count` | Records verified after installation |
| `native_count` | Native template structures |
| `index_updated` | Always false for installation; index loading is separate |
| `matches` | Whether the requested bank matches its stored representation |
| `missing_ids`, `conflicting_ids` | Missing or mismatched logical records |
| `extra_handles` | Unexpected entries within the bank namespace |

Counts describe logical templates, not physical storage fragments. Successful
checks report `matches: true` and empty discrepancy lists. A bank mismatch or
an execution error returns exit code 2. Inspection and checking do not mutate
server data.

### Structural queries

```powershell
python -m symbolic_memory match-bank --patterns $patterns --embeddings $embeddings --server-url $server --pattern '(, (Agent $event $agent) (Event $event))'
```

`match-bank` uses MORK structural unification and returns full template IDs in
`matching_ids`. Repeated variables constrain shared arguments. An empty list is
a valid result. Structural queries are independent of FAISS similarity search.

## Retrieval API

The example uses a stored key as the query. Runtime callers supply projected
query vectors with the bank's dimension.

```python
from pathlib import Path

from symbolic_memory.bank import load_bank
from symbolic_memory.storage import MorkTemplateStore

bank = load_bank(Path("patterns.jsonl"), Path("embeddings.jsonl"))
with MorkTemplateStore("http://127.0.0.1:8000") as store:
    index = store.load_index(bank)

query = bank.records[0].key_vector
result = index.query_top_k(query, top_m=8)
print(result.template_ids)
```

`load_index` downloads server records, verifies their payloads and native
structures, then builds the configured FAISS index. The returned object provides
template IDs, original keys/values and similarity scores. Retain it for
subsequent queries rather than rebuilding it per request.

## Retrieval service

```powershell
python -m symbolic_memory serve-bank --patterns $patterns --embeddings $embeddings --server-url $server --host 127.0.0.1 --port 9090
```

Startup verifies the installed MORK bank and builds one FAISS index before
opening the listening socket. Missing or conflicting records prevent startup.
The startup descriptor reports the bank fingerprint, dimension, record count,
index backend and request limits.

Clients maintain a TCP connection and require the expected input fingerprint:

```python
from symbolic_memory.service import RetrievalClient

with RetrievalClient("127.0.0.1", 9090, bank.input_snapshot_sha256) as client:
    result = client.query_top_k(bank.records[0].key_vector, top_m=8)
    print(client.info)
    print(result.template_ids, result.scores)
```

Queries use the loaded snapshot without MORK downloads or index rebuilds per
request. Results contain full IDs, original float32 keys/values and cosine
scores. If the bank contains fewer than `top_m` records, only existing records
are returned; no synthetic templates are padded into the result. Loading a
different bank requires restarting the service and reconnecting clients with
the new fingerprint. A MORK restart does not invalidate the already loaded
snapshot or trigger an automatic reload.

### Transport

The application protocol uses persistent TCP connections with `TCP_NODELAY`.
Requests contain query vectors; responses contain retrieved keys, values,
similarity scores and template IDs.

| Message | Layout |
| --- | --- |
| Bank descriptor, once per connection | `BANK`, uint32 JSON byte length, UTF-8 JSON |
| Query | `QSYM`, uint32 version=2, query count, dimension, `top_m`; float32 queries |
| Result | `TMPL`, uint32 version=2, query count, returned count, dimension; float32 keys, values, scores; uint32 ID byte length; UTF-8 IDs |

Integers and floats are little-endian; arrays are row-major. IDs are separated
by tabs within each result row and newlines between query rows. IDs containing
tabs or newlines cannot be served by this protocol. Version 1 is rejected.
The client verifies bank identity on connection and response shapes on each
query. Malformed, oversized or nonfinite requests close that connection without
changing the bank.

Defaults allow 1,024 queries per request, `top_m` up to 64 and eight concurrent
connections. `--max-queries`, `--max-top-m` and `--max-connections` adjust these
limits; requests and responses also have a 64 MiB transport ceiling. `--timeout`
sets the MORK startup timeout and connection inactivity timeout. Clients should
reconnect after an idle connection expires. FAISS access uses the existing index
lock; accepting multiple connections does not imply parallel FAISS execution.

`client.reconnect()` explicitly opens a fresh connection and checks that the
bank and complete service descriptor remain unchanged. Call it after a long
compilation or planned idle period before resuming queries. It does not retry
failed requests or reload the bank. Reconnection time is setup cost and should
be recorded separately from steady-state query latency.

The default listener is loopback-only. The protocol does not provide TLS or
authentication; bind a remote interface only within an access-controlled network
or tunnel. TCP round-trip measurements are not PCIe or GPU transfer measurements.

### CUDA boundary

`symbolic_memory.device.retrieve_cuda(client, queries, top_m)` accepts a rank-1
or rank-2 float32 PyTorch CUDA tensor. It copies a detached query to host memory,
uses the existing TCP client, then returns keys, values and scores on the original
CUDA device. CPU inputs are rejected; there is no device fallback or projection.

```python
from symbolic_memory.device import retrieve_cuda

result = retrieve_cuda(client, queries, top_m=8)
print(result.keys.device, result.template_ids)
print(result.device_to_host_ms, result.retrieval_round_trip_ms)
print(result.host_to_device_ms, result.total_ms)
```

The caller supplies the CUDA query and an already-connected `RetrievalClient`.
This is an eager PyTorch boundary, not a JAX/TorchAX model integration. Discrete
retrieval is outside autograd; retain the original query for differentiable GPU
attention. The adapter does not compute attention or residual writeback.

Timing uses synchronized wall-clock stages. Device/host measurements include
allocation and conversion; the retrieval stage includes serialization, network
transport, CPU search and decoding. Pending producer work is synchronized before
the measured interval. These measurements are not isolated PCIe bandwidth or
complete training-step overhead.

### Compiled JAX boundary

`symbolic_memory.jax_bridge.JaxRetriever` runs the existing TCP client through an
ordered JAX I/O callback. Construct the client and retriever outside JIT:

```python
import jax

from symbolic_memory.jax_bridge import JaxRetriever

retrieve = JaxRetriever(
    client, tuple(row.template_id for row in bank.records), top_m=8
)
compiled_retrieve = jax.jit(retrieve)
templates = compiled_retrieve(queries)
jax.block_until_ready(templates)
```

Queries are rank-1 or rank-2 float32 JAX arrays. Keys and values have shape
`[batch, min(top_m, bank_count), dimension]`; scores and template indices have
shape `[batch, min(top_m, bank_count)]`. Indices refer to `retrieve.template_ids`,
which retains full IDs outside device arrays. The client binds the connection
to the expected bank fingerprint.

Selection receives a stopped-gradient query. Retain the original query for
differentiable attention and alignment; the returned bank vectors are constants.
The adapter does not implement a learned projection, attention, writeback or
training objective. Network failures propagate rather than returning cached or
synthetic templates.

GPU execution with host callbacks requires both CUDA and CPU backends enabled
(for example, `JAX_PLATFORMS=cuda,cpu` before importing JAX). The CPU backend
executes the host callback; device placement of model computation remains the
caller's responsibility. Supply query batches directly instead of vmapping an
ordered callback. Shapes are static for compilation; a new batch shape can
require a new executable. Lowering/compilation and synchronized steady-state
execution must be measured separately.

## Colocated native CUDA retrieval for A4

`NativeJaxRetriever` provides the same `RetrievedTemplates` interface without
the per-query TCP hop. The bank is still MORK-authoritative: `from_mork` verifies
native structures and linked payloads before building a private CPU FAISS index.
The verified index is serialized once and deserialized into a CPU-owned C++
FAISS snapshot, preserving its graph, search parameters and insertion ordering.
At runtime JAX invokes the CUDA FFI handler, which copies the query, normalizes
and searches in C++, gathers the original float32 K/V and returns the results.
No Python callback executes in this search path. Only queries and complete
float32 K/V, scores and indices cross the GPU/CPU boundary. There is no GPU
retrieval-bank cache or synthetic fallback. Small floating-point differences in
normalization/FAISS builds require parity checks, particularly for tied scores.

Build explicitly in the installed **Linux CUDA/JAX** environment, using a fresh
output path (no package installation or automatic compilation at import):

```bash
python -m symbolic_memory.native_bridge build --faiss-prefix /path/to/faiss-sdk --output native/a4-retrieval-NEW.so
```

The SDK must contain `include/faiss/Index.h` and `lib/libfaiss.so` (or `lib64`),
with the same FAISS version as Python. A pip wheel alone may not supply that SDK.
`--faiss-library PATH` selects an explicitly built compatible shared variant
(for example `libfaiss_avx2.so` on a supporting CPU). Native and Python compile
options are recorded; comparing a generic C++ library with an optimized wheel
does not isolate Python overhead. Never select an instruction set the CPU lacks.
Without `--faiss-prefix`, the build checks the active environment, `/usr/local`
and `/usr`; it never downloads/builds FAISS automatically. A C++17 compiler with
OpenMP, CUDA runtime headers/library and JAX FFI headers must already be installed.
The adjacent `.so.json` records ABI 2, source/library/JAX fingerprints and the
FAISS version and shared-library fingerprint. The loader rejects mismatches;
rebuild after relevant changes. ABI-1 callback libraries are not interchangeable.
The implementation uses [JAX's CUDA FFI interface](https://docs.jax.dev/en/latest/ffi.html).

```python
from pathlib import Path
from symbolic_memory.native_bridge import NativeJaxRetriever

with NativeJaxRetriever.from_mork(
    bank, server_url="http://127.0.0.1:18000",
    library_path=Path("native/a4-retrieval-NEW.so"), top_m=8,
) as retrieve:
    # Pass retrieve directly to SymbolicHead; retain it until compiled calls finish.
    compiled_retrieve = jax.jit(retrieve)
    templates = compiled_retrieve(gpu_queries)
    jax.block_until_ready(templates)
```

The A4 training runner (`python -m experiments.run_stage_a4`) defaults to native:
supply `--native-library PATH.so`.
`--retrieval-backend tcp` explicitly selects the existing service path.
Training and evaluation use the same selection. Native mode does not
need `serve-bank` or a retrieval TCP port, but does need the existing MORK endpoint.
It does not replace or restart an existing service.

Each bank/query shape has an opaque, never-reused native handle and reusable
pinned buffers. At most eight shapes are retained per retriever. Supported index
mutations invalidate its generation; create a new retriever rather than rebinding
compiled IDs. ID indices always refer to `template_ids`, not FAISS insertion
order. Closing waits for native calls already executing, releases buffers, and makes
old compiled handles fail instead of reading another bank. Synchronize submitted
JAX calls before closing; a call still queued after close fails explicitly.
One selected CUDA device is supported; sharded/multi-GPU
execution is not qualified. Use a context manager and synchronize before exit.

For explicit diagnostics, `with retrieve.capture_queries() as queries:` captures
queries in C++ and fills the Python list when the scope exits, outside timed
execution. Capture is bounded (default 64 calls), cannot nest or close its
retriever, and is disabled in ordinary calls. The former `set_observer` callback
API is removed. Concurrent submission during capture is not supported.

Native counters report actual CPU index time separately from transfers and
`host_search_seconds`. Both transfer synchronization points remain in place in
the default `return_mode="sync"`.
Latency benchmarks must synchronize execution and record the workload and timing
boundaries. Benchmark drivers are separate from the A4 training entry point.

The opt-in `return_mode="event"` queues CPU-to-GPU copies and a completion event
on XLA's CUDA stream, without waiting on the host before returning. GPU consumers
remain stream-ordered. The context waits for its event before reusing pinned
buffers or freeing them on close. GPU-to-CPU readiness is still synchronized
before CPU search. Partially submitted transfers are drained on error; a failed
drain prevents further reuse. If CUDA cleanup fails, release reports an error
and retains the native context rather than freeing potentially in-flight memory.
Callers must still finish submitted JAX execution before closing the retriever.

In event mode `h2d_seconds` measures enqueue/event-record host time only, not the
completed transfer. A synchronized full-call benchmark still includes the
remaining copy/completion time; subtracting CPU index work must not discard it.
The build manifest records supported return modes. Rebuild explicitly after
this source update; previous ABI-2 libraries lack the new context-creation API.

CUDA copy-event profiling is a separate opt-in scope:

```python
with retrieve.transfer_timing():
    jax.block_until_ready(compiled_model(...))
    intervals = retrieve.transfer_stats()
```

`transfer_stats()` reports profiled call count and D2H/H2D event seconds. Each
copy direction has its own timing-enabled event pair on XLA's stream. The CPU
search between pairs and GPU query production preceding the D2H start event are
excluded. Event intervals can include scheduling/submission gaps; they are not
hardware PCIe bus-utilization measurements. Stats are collected before event
reuse, so repeated reads do not double-count completed calls. Timing is disabled
outside the scope, and new query shapes created inside it inherit profiling.
Capture, nested profiling and closing inside the scope are rejected. Concurrent
submission during profiling is unsupported.

Run instrumented model calls separately from uninstrumented latency samples.
Report the uninstrumented denominator, instrumented wall time and transfer-only
fraction explicitly; a transfer-only result does not qualify complete bridge
overhead. Native source updates require an explicit rebuild, including the new
profiling-control API. This instrumentation does not change the default return
mode or A4 objective.

The default output layout is `packed`. The opt-in constructor argument
`output_layout="direct"` copies K, V, scores and int32 registry indices into
separate JAX result buffers without GPU unpacking. Both layouts use the same
CPU search, transfer the same number of bytes and retain both synchronization
points. Direct output uses four H2D submissions rather than one; performance
must be measured, not inferred from removing unpacking. Rebuild the library
explicitly after updating native source; earlier ABI-2 builds do not contain
the additional direct-output handler.

After building, qualify the compiled CPU and CUDA paths explicitly:

```bash
export NSLLM_NATIVE_LIBRARY=/absolute/path/to/native/a4-retrieval-NEW.so
python -m pytest tests/unit/test_native_bridge.py tests/unit/test_retrieval_index.py -q -rs
```

Without that environment variable, compiled parity checks are skipped; handle
controls alone do not establish native execution or latency. CUDA checks also
require a real JAX GPU. Snapshot deserialization receives only the locally
serialized, verified in-memory index; arbitrary serialized indexes are not a
supported input contract.

## Storage representation

Each immutable bank occupies a namespace derived from its input fingerprint
and storage-format version.

```text
(BankTemplateV1 <bank> <handle> <native-template-expression>)
(BankRecordV1   <bank> <handle> <part-count> <checksum>)
(BankPartV1     <bank> <handle> <part-index> <hex-payload>)
```

`BankTemplateV1` contains the native MeTTa body, preserving structural access for
symbolic matching and downstream reasoning. The wrapper associates the body with
its bank and storage handle; it is not part of the canonical term or embedding.

Linked payloads preserve the full ID, original template text, source
`PatternRecord` metadata and vectors. Payload bytes consist of a UTF-8 JSON
metadata line followed by little-endian float32 key and value arrays. Hex
fragments occupy numbered `BankPartV1` rows; `BankRecordV1` records the fragment
count and checksum.

Storage handles are 53-character SHA-256/base32 symbols. Payload fragments are
at most 60 characters. Neither representation truncates public IDs or changes
vector values. MORK's variable notation (`$` and `_N`) is compared by structure
and variable sharing; original variable names remain in the linked metadata.

Constants exceeding 63 UTF-8 bytes, or beginning with the reserved `BankSym_`
prefix, are mapped to deterministic bounded symbols. Collision checks apply
within each bank; linked payloads retain the original constants and template.
The native expression tree and variable sharing remain intact.

Native terms must fit the deployed server's capabilities. The adapter rejects
arity above 63 and native expressions exceeding its conservative export-size
budget. Unsupported terms are not replaced by opaque payloads. Larger native
structures require a compatible server build.

The existing `residual/symbolic_head/mork_client.py` interface is independent
of this application-layer representation.

## Installation and recovery

Installation requires a single writer. It rejects conflicting stored data before
uploading, writes missing payload fragments, publishes headers, then publishes
native structures. Acceptance requires exact payload integrity and matching
native structure. Other namespaces and unrelated MORK expressions are untouched.

HTTP batches are not transactions. Interrupted installations may leave partial
writes; automatic rollback is not provided. Inspect the bank before retrying.
Compatible existing fragments are reused, while conflicting or malformed data
produces an error.

The deployed server graph is in memory. After restart, reinstall the retained
input artifacts and reload the index. This is explicit reconstruction, not
automatic persistence. Server-generation changes are not monitored automatically.

## Evaluation scope

Consistency checks cover structural matching, payload integrity, repeated
installation and restart/reload. Self-key retrieval measures consistency, not
held-out semantic quality. Large-bank throughput, semantic precision/recall,
remote deployment and GPU/CPU latency require separate evaluation.
