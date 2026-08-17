# Neuro-Symbolic LLM

Natural-language text to validated MeTTa Atomese, with configurable hosted and
local model backends. The current implementation includes the semantic-parser
foundation, Atomese grammar, offline dataset generation, and unit tests.

## Overview

```text
Input:  Ben bought a car.
Output: (Evaluation buy (List ben car))
```


[`ReferenceSemanticParser`](./parser/semantic/semantic_parser.py) uses a larger model to make validated examples for
offline distillation. [`DistilledSemanticParser`](./parser/semantic/semantic_parser.py) uses the same validation path
with a future fine-tuned 1B–3B local model for deployment.

## Architecture

### End-to-end teacher–student flow

```text
┌──────────────────────────────────────────────────────────────────────┐
│                        OFFLINE REFERENCE STAGE                       │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────┐
│ Natural-language sentence    │
│ "Ben bought a car."          │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│ ReferenceSemanticParser      │
│ Large hosted or local model  │
│ Recommended Model (>7B)      │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│ Expected MeTTa output        │
│ (Evaluation buy              │
│   (List ben car))            │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│ Shared Atomese pipeline      │
│ 1. clean_model_output()      │
│ 2. canonical()               │
│ 3. validate_metta_string()   │
│ 4. parse_atom()              │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│     Validated LinkAtom       │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│ Distillation JSONL dataset   │
│ accepted and rejected data   │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│ Fine-tune local student      │
│ future 1B–3B model           │
└───────────────┬──────────────┘
                │
                ▼

┌──────────────────────────────────────────────────────────────────────┐
│                         DEPLOYMENT STAGE                             │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────┐
│ Natural-language sentence    │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│ DistilledSemanticParser      │
│ Local 1B–3B student model    │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│ Shared Atomese pipeline      │
│ clean → canonicalize         │
│ validate → parse             │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│ Validated LinkAtom           │
└──────────────────────────────┘
```

### Parser Component relationship

```text
┌────────────────────┐
│ GeminiBackend      │
├────────────────────┤
│ OpenAIBackend      │
├────────────────────┤
│ AnthropicBackend   │
├────────────────────┤
│ OllamaBackend      │
├────────────────────┤
│ TransformersBackend│
└─────────┬──────────┘
          │ implements
          ▼
┌────────────────────┐
│ ModelBackend       │
│ generate(...)      │
└─────────┬──────────┘
          │ injected into
          ▼
┌──────────────────────────────────────────┐
│ Shared semantic-parser implementation    │
│  • prompt construction                   │
│  • output cleaning                       │
│  • canonicalization                      │
│  • grammar validation                    │
│  • LinkAtom conversion                   │
└───────────────┬──────────────────────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
┌─────────────────┐  ┌────────────────────┐
│ Reference       │  │ Distilled          │
│ Semantic Parser │  │ Semantic Parser    │
└─────────────────┘  └────────────────────┘
```

## Model configuration and running

Copy [`.env.example`](./.env.example) to `.env` and add only the credentials you
use. Model selection belongs in
[`configs/parser_config/model_backend.yaml`](./configs/parser_config/model_backend.yaml).
The active profile picks the backend class.

```yaml
active: ollama

profiles:
  ollama:
    provider: ollama
    model: qwen3:8b
```

For the local Ollama profile, start the server in one terminal:

```bash
ollama serve
```

Then run the parser from another terminal:

```bash
source .venv/bin/activate
python -m pip install -e ".[ollama]"
python -m parser.semantic
```

`python -m parser.semantic` avoids the import-order warning produced by running
the implementation module directly. If the configured local backend is not
running, start it with `ollama serve` first.

In Python, use the factory instead of choosing provider classes manually:

```python
from parser.semantic import build_reference_semantic_parser

parser = build_reference_semantic_parser()
atom = parser.parse("A bird can fly.")
print(atom)
```

## Atomese output contract

The closed vocabulary in [`parser/grammar/atomese.py`](./parser/grammar/atomese.py) is:

```text
Inheritance  Evaluation  CanDo  On  Cause  Has  PartOf  StateOf  List
LocatedIn  MemberOf  UsedFor  Before  After
```

The model must output one or more expressions, preserve semantic roles, use
lowercase concepts and base-form verbs, and return `UNSUPPORTED` when it cannot
represent the sentence reliably. `Evaluation` handles relations without a
dedicated predicate.

## Repository structure

```text
configs/
├── parser_config/
│   ├── model_backend.yaml       # selected provider and model profiles
│   └── parser_prompt.yaml       # semantic-parser prompt
├── tiers.yaml                   # tier configuration
└── stage_A/                     # experiment configurations
experiments/                     # Stage A experiment entry points
parser/
├── grammar/atomese.py           # Atomese model, parser, and validation
└── semantic/
    ├── backends.py              # provider adapters
    ├── model_config.py          # YAML/.env backend factory
    ├── semantic_parser.py       # reference and distilled parsers
    ├── __main__.py              # command-line demonstration
    ├── dataset.py               # accepted/rejected JSONL records
    └── __init__.py              # public semantic-parser API
tests/unit/                      # offline unit tests
```

## Testing

Tests are necessary. They verify deterministic logic without calling a model
provider: Atomese parsing and validation, prompt construction, output cleanup,
error handling, dataset records, and model-profile selection.

Run all offline tests:

```bash
python -m pytest tests/unit -v
```

Do not replace these with live API tests. Test provider integrations separately
with a small, opt-in smoke test because they require credentials, network
access, and may cost money.

## Current implementation status

Implemented: Atomese grammar and validation, configurable provider adapters,
reference/distilled parser roles, dataset-record generation, and offline unit
tests. The student-model fine-tuning and deployment pipeline are planned next.
