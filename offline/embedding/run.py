"""Command-line entry point for offline template embedding generation."""

from __future__ import annotations

import argparse

from .context_builder import load_pattern_contexts
from .encoder import FrozenLLMEncoder
from .template_embeddings import (
    DEFAULT_PROJECTION_SEED,
    DEFAULT_SYMBOLIC_DIM,
    initialize_template_kv,
    save_template_embeddings,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate source-grounded template K/V embeddings."
    )

    parser.add_argument("--documents", required=True)
    parser.add_argument("--patterns", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--layer-idx", type=int, default=None)
    parser.add_argument(
        "--symbolic-dim",
        type=int,
        default=DEFAULT_SYMBOLIC_DIM,
    )
    parser.add_argument(
        "--projection-seed",
        type=int,
        default=DEFAULT_PROJECTION_SEED,
    )

    args = parser.parse_args()

    templates = load_pattern_contexts(
        pattern_records_path=args.patterns,
        documents_path=args.documents,
    )

    if not templates:
        raise ValueError("No pattern records found to embed")

    encoder = FrozenLLMEncoder(
        model_name=args.model,
        max_length=args.max_length,
        layer_idx=args.layer_idx,
    )

    records = []

    for template in templates:
        record = initialize_template_kv(
            template,
            encoder,
            symbolic_dim=args.symbolic_dim,
            projection_seed=args.projection_seed,
        )

        records.append(record)

        print(
            f"Embedded {template.pattern_id}: "
            f"{record['num_contexts']} contexts, "
            f"{record['embedding_dim']} dimensions"
        )

    save_template_embeddings(records, args.output)

    print(f"Saved {len(records)} template embeddings to {args.output}")


if __name__ == "__main__":
    main()
