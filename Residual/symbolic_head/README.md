# Symbolic Head and Tier 2

This component contains the CPU symbolic pipeline and the contracts connecting
semantic parsing, symbolic mining and storage, and GPU symbolic-head execution.

The target architecture is defined by the PC-residual paper, the hybrid-miner
paper, and the Q1 plan. Status sections distinguish implemented behavior,
transitional code, and unimplemented requirements.

## Ownership boundary

### Tier 2 owns

- the validated semantic-document intake contract;
- ontology loading, canonicalization, and PeTTa emission;
- Hyperon Miner orchestration and mined-candidate parsing;
- the canonical `PatternRecord` and promotion eligibility checks;
- construction of embedded template records using a frozen-LLM embedding API;
- MORK persistence and rebuilding the derived FAISS index;
- the CPU retrieval service and QSYM/TMPL wire protocol; and
- packet, retrieval, mining, provenance, and latency metrics.

### Tier 2 does not own

- frozen-transformer execution or hidden-state interception;
- the A1--A3 predictive-coding residual;
- GPU-side `q_sym = W_sym h + b_sym`;
- GPU-side symbolic attention or the `U_out` residual update;
- parser-model training and inference backends; or
- Stage B/C causal, SVM, or STLM systems.

Those operations belong to the frozen-LLM, PC-residual, or parser components.
Tier 2 implements their boundary contracts but excludes their internal
computation.

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

Mining and retrieval are separate processes:

```text
validated semantic document
  -> canonical PeTTa atoms
  -> pinned Hyperon Miner greedy baseline
  -> PatternRecord
  -> frozen-LLM F0 embedding
  -> TemplateRecord
  -> MORK
  -> rebuildable FAISS index
```

The runtime retrieval service never runs the parser or miner. A miner result
cannot be published directly to MORK without becoming a validated
`PatternRecord` and passing the embedding/promotion stage.

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

Additional subpackages will be created only with their corresponding
implementations:

```text
contracts/   stable records and wire formats
ingestion/   semantic validation, canonicalization, PeTTa emission
mining/      pinned Hyperon execution and output parsing
promotion/   F0 embedding and PatternRecord-to-template promotion
storage/     MORK authority and derived FAISS index
service/     CPU retrieval server
client/      Tier 1-facing retrieval client
metrics/     mining, retrieval, and cross-tier measurements
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
- FAISS Flat and HNSW index foundations.

Only the ontology contract added in the current modular change is covered by
the new paper-aligned tests. Existing MORK/FAISS code still requires the
cleanup and integration work described below.

### Reference-only or transitional code

- `head.py` is a NumPy shape/reference implementation. It is not the GPU A4
  implementation and does not measure GPU/PCIe behavior.
- `losses.py` is a NumPy reference for the A4 loss formulas, not a trainable
  JAX/Flax or TorchAX implementation.
- `communication.py` implements an in-process QSYM/TMPL v1 serialization
  round trip. It is not a network boundary and will be replaced by QSYM v2.
- `retrieval_bridge.py` performs in-process retrieval and is not a TCP client.
- `experiments/run_stage_a4.py` uses generated vectors and verifies shapes
  only.
- `experiments/benchmark_tier2_mork.py` uses generated vectors and may measure
  local infrastructure mechanics only. It cannot establish semantic retrieval
  quality or PCIe overhead.

Reference-only results must never be reported as production integration,
semantic quality, or hardware performance.

### Transitional direct-publication path

`miner_adapter.py` currently accepts already-embedded keys and values and can
publish them directly. This bypasses canonical miner output, `PatternRecord`,
F0 embedding, and promotion checks. It is not the target miner integration and
will be removed when the validated promotion pipeline is introduced.

### Not yet implemented

- Appendix A.4 semantic-document runtime validation;
- canonical PeTTa emission and ingestion guards;
- pinned PeTTa and Hyperon Miner execution;
- construction of `PatternRecord` instances from Hyperon Miner output;
- F0 context-window embedding and L2 key normalization;
- validated PatternRecord-to-TemplateRecord promotion;
- QSYM v2;
- the retrieval TCP server and client;
- fail-closed production FAISS configuration;
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

Some existing Docker-dependent MORK tests are still located under
`tests/unit/`. They require reclassification under
`tests/integration/symbolic_head/`.

## Q1 implementation order

1. Complete and freeze the ontology contract.
2. Implement and test `PatternRecord`.
3. Implement and freeze QSYM v2.
4. Migrate semantic validation and canonical PeTTa emission.
5. Pin and integrate the Hyperon greedy baseline.
6. Implement F0 embedding and promotion.
7. Separate MORK storage from the retrieval service.
8. Integrate the production Tier 1 client.
9. Run corpus-derived quality evaluation and hardware measurements.
