"""Generate a deterministic synthetic template bank for A4 smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hybrid_miner.contracts import PatternRecord
from symbolic_memory.bank import load_bank


def generate_bank(
    output_dir: Path,
    count: int = 128,
    dimension: int = 64,
    seed: int = 42,
) -> None:
    if count < 8:
        raise ValueError("At least 8 templates are required")
    if dimension < 1:
        raise ValueError("Embedding dimension must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)

    patterns_path = output_dir / "patterns.jsonl"
    embeddings_path = output_dir / "embeddings.jsonl"

    if patterns_path.exists() or embeddings_path.exists():
        raise FileExistsError(
            "Test bank already exists. Use a new output directory."
        )

    rng = np.random.default_rng(seed)

    with (
        patterns_path.open("w", encoding="utf-8") as pattern_file,
        embeddings_path.open("w", encoding="utf-8") as embedding_file,
    ):
        for index in range(count):
            pattern_id = f"test-pattern-{index:05d}"
            document_id = f"synthetic-document-{index:05d}"

            # Synthetic symbolic relation, not mined from real text.
            canonical_term = (
                f"(TestRelation Entity{index} Value{index})"
            )

            pattern = {
                "contract_version": "pattern-record-v1",
                "id": pattern_id,
                "canonical_term": canonical_term,
                "interface_variables": [],
                "existential_variables": [],
                "type_constraints": {},
                "semantic_level": "invented",
                "ontology_groundings": [],
                "semantic_consequences": [],
                "source_spans": [
                    {
                        "document_id": document_id,
                        "start": 0,
                        "end": len(canonical_term),
                    }
                ],
                "parser_confidences": [1.0],
                "derivation_or_definition": (
                    "Synthetic pattern for A4 integration testing"
                ),
                "provenance": {
                    "corpus_id": "a4-synthetic-smoke-test",
                    "parser_version": "synthetic-test-v1",
                    "ontology_versions": ["synthetic-a4-test-v1"],
                    "source_document_ids": [document_id],
                    "experiment_version": "a4-smoke-test-v1",
                },
                "generator_history": [
                    "synthetic-a4-test-generator"
                ],
                "cheap_features": {},
                "status": "active",
            }

            # Validate using the actual hybrid-miner contract.
            validated = PatternRecord.model_validate(pattern)

            pattern_file.write(
                json.dumps(validated.model_dump(mode="json")) + "\n"
            )

            key = rng.normal(size=dimension).astype(np.float32)
            value = rng.normal(size=dimension).astype(np.float32)

            # Keep vector magnitudes controlled for testing.
            key /= np.linalg.norm(key)
            value /= np.linalg.norm(value)

            embedding = {
                "pattern_id": pattern_id,
                "embedding_dim": dimension,
                "key": key.tolist(),
                "value": value.tolist(),
            }

            embedding_file.write(json.dumps(embedding) + "\n")

    # Validate both files through A4's actual bank loader.
    bank = load_bank(patterns_path, embeddings_path)

    print("\nA4 synthetic bank generated successfully")
    print(f"Patterns:   {patterns_path}")
    print(f"Embeddings: {embeddings_path}")
    print(f"Templates:  {len(bank.records)}")
    print(f"Dimension:  {bank.vector_dim}")
    print(f"Snapshot:   {bank.input_snapshot_sha256}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/a4_test_bank"),
    )
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    generate_bank(
        output_dir=args.output,
        count=args.count,
        dimension=args.dimension,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
