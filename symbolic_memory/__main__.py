"""Inspect, install and check externally generated template banks."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import suppress
from pathlib import Path

import httpx

from symbolic_memory.bank import load_bank, read_patterns


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m symbolic_memory")
    commands = parser.add_subparsers(dest="command", required=True)
    patterns_command = commands.add_parser(
        "inspect-patterns", help="Inspect miner records"
    )
    patterns_command.add_argument("--input", type=Path, required=True)
    bank_command = commands.add_parser(
        "inspect-bank", help="Join patterns and embeddings"
    )
    bank_command.add_argument("--patterns", type=Path, required=True)
    bank_command.add_argument("--embeddings", type=Path, required=True)
    for name, help_text in (
        ("install-bank", "Install missing MORK records and verify readback"),
        ("check-bank", "Compare bank records with MORK without writes"),
        ("match-bank", "Structurally match native templates using MORK"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--patterns", type=Path, required=True)
        command.add_argument("--embeddings", type=Path, required=True)
        command.add_argument("--server-url", required=True)
        command.add_argument("--timeout", type=float, default=30.0)
        if name == "install-bank":
            command.add_argument("--batch-size", type=int, default=64)
        elif name == "match-bank":
            command.add_argument("--pattern", required=True)
    serve = commands.add_parser(
        "serve-bank", help="Serve a verified MORK bank over TCP"
    )
    serve.add_argument("--patterns", type=Path, required=True)
    serve.add_argument("--embeddings", type=Path, required=True)
    serve.add_argument("--server-url", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=9090)
    serve.add_argument("--timeout", type=float, default=30.0)
    serve.add_argument("--max-queries", type=int, default=1024)
    serve.add_argument("--max-top-m", type=int, default=64)
    serve.add_argument("--max-connections", type=int, default=8)
    arguments = parser.parse_args(argv)
    result: dict[str, object]
    exit_code = 0
    try:
        if arguments.command == "inspect-patterns":
            patterns = read_patterns(arguments.input)
            result = {
                "pattern_count": len(patterns),
                "statuses": sorted({item.status for item in patterns}),
                "semantic_levels": sorted({item.semantic_level for item in patterns}),
                "contains_vectors": False,
                "preview": [
                    {"id": item.id, "canonical_term": item.canonical_term}
                    for item in patterns[:3]
                ],
            }
        elif arguments.command == "inspect-bank":
            bank = load_bank(arguments.patterns, arguments.embeddings)
            result = {
                "record_count": len(bank.records),
                "vector_dimension": bank.vector_dim,
                "dtype": "float32",
                "normalization_applied": False,
                "patterns_sha256": bank.patterns_sha256,
                "embeddings_sha256": bank.embeddings_sha256,
                "input_snapshot_sha256": bank.input_snapshot_sha256,
                "preview": [
                    {"id": item.template_id, "canonical_term": item.metta_expr}
                    for item in bank.records[:3]
                ],
            }
        else:
            from symbolic_memory.storage import MorkTemplateStore

            bank = load_bank(arguments.patterns, arguments.embeddings)
            with MorkTemplateStore(
                server_url=arguments.server_url,
                timeout_sec=arguments.timeout,
            ) as store:
                if arguments.command == "serve-bank":
                    from symbolic_memory.service import RetrievalServer

                    with RetrievalServer(
                        (arguments.host, arguments.port),
                        bank,
                        store,
                        max_queries=arguments.max_queries,
                        max_top_m=arguments.max_top_m,
                        max_connections=arguments.max_connections,
                        timeout_sec=arguments.timeout,
                    ) as server:
                        print(
                            json.dumps(
                                {
                                    **server.info,
                                    "listening_host": arguments.host,
                                    "listening_port": server.server_address[1],
                                }
                            ),
                            flush=True,
                        )
                        with suppress(KeyboardInterrupt):
                            server.serve_forever()
                    return 0
                elif arguments.command == "install-bank":
                    result = store.install_bank(bank, batch_size=arguments.batch_size)
                elif arguments.command == "match-bank":
                    result = {
                        "matching_ids": store.match_templates(bank, arguments.pattern)
                    }
                else:
                    result = store.check_bank(bank)
                    if not result["matches"]:
                        exit_code = 2
            result.update(
                input_snapshot_sha256=bank.input_snapshot_sha256,
                server_url=arguments.server_url,
            )
    except (OSError, ValueError, RuntimeError, httpx.HTTPError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
