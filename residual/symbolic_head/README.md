# Symbolic Head and Tier 2

This component contains neural-symbolic promotion, storage, retrieval, and the
runtime boundary connecting CPU symbolic templates to GPU symbolic-head
execution. The independent `hybrid-miner` repository owns semantic formation
and the paper-defined mining program.

The target architecture is defined by the PC-residual paper, the hybrid-miner
paper, and the Q1 plan. Status sections distinguish implemented behavior from
unimplemented requirements.

## Ownership boundary

### This component owns

- validation of versioned promotion requests from `hybrid-miner`;
- construction of embedded template records using a frozen-LLM embedding API;
- MORK persistence and rebuilding the derived FAISS index;
- the CPU retrieval service and QSYM/TMPL wire protocol; and
- promotion, packet, retrieval, provenance, and latency metrics.

### This component does not own

- frozen-transformer execution or hidden-state interception;
- the A1--A3 predictive-coding residual;
- GPU-side `q_sym = W_sym h + b_sym`;
- GPU-side symbolic attention or the `U_out` residual update;
- parser-model training and inference backends; or
- PeTTa execution, Hyperon Miner orchestration, candidate generation, SVM/STLM
  control, relation invention, or causal-effect estimation.

Those operations belong to the frozen-LLM, PC-residual, parser, or independent
`hybrid-miner` components. This component implements their neural-runtime
boundaries but excludes their internal computation.

## Runtime boundary

The intended Stage A4 runtime path is:

```text
Tier 1 / GPU                            Tier 2 / CPU
------------------------------          -------------------------------
frozen LLM hidden state
  -> PC residual
  -> q_sym projection
  -> QSYM v2 packet -------------------> retrieval service
                                           -> FAISS top-m lookup
                                           -> MORK-backed template records
  <- TMPL packet <---------------------
  -> symbolic attention
  -> U_out residual update
```

Full hidden states do not cross into Tier 2. The GPU sends compact `q_sym`
vectors and receives the selected template keys, values, scores, identifiers,
and validity information.

MORK HTTP storage and the retrieval socket are different services:

- MORK HTTP on port 8000 is the authoritative template store.
- The planned QSYM/TMPL retrieval service on port 9090 serves the derived
  FAISS index.

## Offline promotion boundary

Mining, promotion, and retrieval are separate processes:

```text
hybrid-miner repository
  validated semantic document
  -> canonical PeTTa atoms
  -> pinned Hyperon Miner greedy baseline
  -> PatternRecord

Neuro-Symbolic-LLM repository
  -> validated promotion request
  -> frozen-LLM F0 embedding
  -> TemplateRecord
  -> MORK
  -> rebuildable FAISS index
```

The runtime retrieval service never runs the parser or miner. The repositories
exchange versioned artifacts; they do not import each other through filesystem
paths. A miner result cannot be published directly to MORK without becoming a
validated `PatternRecord` and passing the embedding/promotion stage.

## Repository layout

Current paper-aligned structure:

```text
Residual/symbolic_head/
  contracts/
    ontology_catalog.py
    pattern_record.py
  README.md

parser/ontology/
  upper.yaml
  discourse.yaml
  proof.yaml
  narrative.yaml
  lexical.yaml
  program.yaml

tests/unit/symbolic_head/
  test_ontology_catalog.py
  test_pattern_record.py
```

The local ontology and pattern-record implementations are contract prototypes
pending controlled migration to the `hybrid-miner` contract package. They must
not be deleted until the versioned external package is available and canonical
serialization compatibility has been verified.

Additional subpackages will be created only with their corresponding
implementations:

```text
contracts/   runtime wire formats and template-installation receipts
promotion/   validated request-to-template promotion and F0 embedding
storage/     MORK authority and derived FAISS index
service/     CPU retrieval server
client/      Tier 1-facing retrieval client
metrics/     promotion, retrieval, and cross-tier measurements
```

Shared ontology files remain under `parser/ontology/` because the parser and
Tier 2 must consume the same versioned vocabulary.

## Ontology contract

`contracts/ontology_catalog.py` implements the modular catalog required by
Hybrid Miner Appendix A.3. It provides immutable argument, relation, and
catalog records and rejects:

- missing or unknown specification fields;
- duplicate ontology versions or relation names;
- arity and argument-count mismatches;
- duplicate argument names;
- unknown argument types;
- invalid type unions;
- malformed confidence ranges; and
- empty extraction instructions or catalogs.

The catalog currently covers the minimal upper, discourse, proof, narrative,
lexical/event, and program vocabularies selected for the Q1 experiments.

The live parser still reads `configs/parser_config/predicate_schema.yaml`.
Therefore, the catalog is validated but is not yet the parser's runtime source
of truth. Appendix A.4 parser compliance requires completion of the explicit
migration and validation stage.

Example:

```python
from Residual.symbolic_head.contracts import load_ontology_catalog

catalog = load_ontology_catalog()
concession = catalog.relation("Concession")
assert concession.arity == 2
```

## Pattern record contract

`contracts/pattern_record.py` implements the storage-neutral candidate record
defined by Hybrid Miner section 6.1. It retains the complete field set needed
by the staged mining program while applying a separate Q1 policy to prevent
later-stage controller and causal data from entering extraction-first runs.

The contract enforces:

- disjoint interface and existential variable sets;
- type constraints that reference declared variables only;
- half-open source spans linked to provenance document identifiers;
- finite parser confidences in `[0, 1]`;
- required spans, confidences, and ontology grounding for extracted patterns;
- finite, non-negative uncertainty and runtime metrics;
- finite JSON-compatible endpoint, feature, and trace payloads;
- complete parser, ontology, corpus, and source-document provenance; and
- deterministic canonical JSON serialization.

`validate_q1_pattern_record()` accepts only surface or extracted semantic
levels and rejects semantic consequences, definitions, controller outputs, and
causal statistics. It requires miner-version provenance, generator history,
and cheap features. Cheap and exact endpoints and runtime cost remain
available because they are required by the Q1 greedy-mining evaluation.

The contract is not yet connected to Hyperon Miner output or MORK promotion.
Those integrations require the mining and promotion modules.

## Current implementation status

### Implemented and verified

- Strict modular ontology loading and validation.
- Canonical section 6.1 `PatternRecord` validation, deterministic
  serialization, and Q1 scope enforcement.
- MORK HTTP upload/export foundations.
- Rebuilding a local index from exported MORK records.
- Fail-closed FAISS Flat and HNSW index foundations.
- Explicit empty and short retrieval results without fabricated templates.
- Rejection of duplicate identifiers and invalid template/query inputs.

The ontology and pattern-record contracts are covered by paper-aligned unit
tests. MORK/FAISS storage and retrieval remain in one module and require the
structural separation described below.

### Removed non-production paths

The following pre-production paths have been removed from the component:

- the NumPy symbolic-head and loss implementations;
- the in-process QSYM v1 serialization and retrieval bridge;
- the direct embedded-vector miner publication adapter; and
- generated-vector Stage A4 and MORK benchmark scripts.

Their outputs did not establish production GPU integration, cross-tier network
behavior, semantic retrieval quality, or hardware performance. Replacement
implementations must satisfy the ownership boundaries, promotion invariants,
and production protocols defined in this document.

### Not yet implemented

- versioned `hybrid-miner` contract-package consumption;
- promotion-request and template-receipt contracts;
- F0 context-window embedding and L2 key normalization;
- validated PatternRecord-to-TemplateRecord promotion;
- QSYM v2;
- the retrieval TCP server and client;
- separation of MORK storage from the derived FAISS index;
- production frozen-LLM/GPU integration;
- hardware cross-tier latency measurement; and
- held-out semantic retrieval precision/recall.

## Production invariants

The production path must satisfy all of the following:

1. MORK is authoritative; FAISS is derived and rebuildable.
2. Production never silently falls back to NumPy search.
3. Missing MORK, FAISS, PeTTa, or Hyperon dependencies fail clearly.
4. Invalid semantic input is quarantined without losing provenance.
5. Miner candidates cannot bypass `PatternRecord`.
6. Templates cannot enter MORK without validated embeddings and identifiers.
7. Empty retrieval results are explicit and never represented as synthetic zero
   templates.
8. Unsupported protocol versions, dimensions, and payload sizes are rejected.
9. Tier 2 receives symbolic queries, not full hidden states.
10. Benchmarks state exactly which subsystem and metric they measure.

## Testing policy

Pure validation, serialization, and deterministic transformation tests are unit
tests. Tests requiring MORK, PeTTa, Hyperon Miner, a socket service, a model, or
a GPU are integration tests and must be selected explicitly.

Current ontology verification:

```bash
python -m pytest tests/unit/symbolic_head/test_ontology_catalog.py -q
python -m pytest tests/unit/symbolic_head/test_pattern_record.py -q
```

Docker-dependent MORK tests are located under
`tests/integration/symbolic_head/` and must be selected explicitly.

## Q1 implementation order

1. Publish and pin the versioned `hybrid-miner` contract package.
2. Migrate local ontology and pattern-record contract authority.
3. Separate MORK storage from the derived FAISS retrieval index.
4. Implement promotion-request validation, F0 embedding, and template receipts.
5. Implement and freeze QSYM v2 and TMPL.
6. Implement the retrieval service and production Tier 1 client.
7. Integrate real `hybrid-miner` candidate artifacts.
8. Run corpus-derived quality evaluation and hardware measurements.
