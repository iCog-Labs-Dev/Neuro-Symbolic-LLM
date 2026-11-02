# Symbolic Template Banks

Associates released miner patterns with offline embeddings, stores native MeTTa
templates in MORK and builds a FAISS retrieval index.

## Requirements

- Python dependencies installed with `pip install -e '.[tier2]'`.
- A compatible MORK HTTP server.
- Pattern and embedding JSONL artifacts covering the same set of unique IDs.

The package does not start MORK or run parsing, mining, embedding generation,
training or candidate promotion.

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

Native terms must fit the deployed server's capabilities. The current adapter
rejects symbols exceeding 63 UTF-8 bytes, arity above 63 and native expressions
exceeding its conservative export-size budget. Unsupported terms are not replaced
by opaque payloads. Larger native structures require a compatible server build.

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
