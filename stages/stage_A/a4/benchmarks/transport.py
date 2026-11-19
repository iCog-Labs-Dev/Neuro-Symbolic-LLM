"""Matched loopback TCP and same-process Python JAX retrieval callbacks."""

from __future__ import annotations

import threading
from contextlib import ExitStack
from typing import Any

import numpy as np

from .common import (
    CpuBackend,
    expected_outputs,
    gpu_queries,
    paired_samples,
    verify_outputs,
)


def measure(
    args: Any, bank: Any, store: Any, queries: np.ndarray, result: dict, samples: list
) -> None:
    import jax

    from symbolic_memory.jax_bridge import JaxRetriever
    from symbolic_memory.service import RetrievalClient, RetrievalServer

    gpu = gpu_queries(queries)
    result.update(jax=jax.__version__, gpu=str(gpu.device))
    with ExitStack() as cleanup:
        server: Any = cleanup.enter_context(
            RetrievalServer(
                ("127.0.0.1", 0), bank, store, max_queries=max(1024, len(queries))
            )
        )
        backend = CpuBackend(server.index, server.info, args.threads)
        server.index = backend
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        cleanup.callback(thread.join)
        cleanup.callback(server.shutdown)
        client = cleanup.enter_context(
            RetrievalClient(
                "127.0.0.1", server.server_address[1], bank.input_snapshot_sha256
            )
        )
        backends: dict[str, Any] = {"tcp": client, "local-python": backend}
        compiled = {
            name: jax.jit(JaxRetriever(connection, backend.id_store, top_m=args.top_m))
            .lower(gpu)
            .compile()
            for name, connection in backends.items()
        }
        client.reconnect()
        for host in (queries, -queries):
            expected = expected_outputs(backend, host, args.top_m)
            device = gpu_queries(host)
            for call in compiled.values():
                verify_outputs(expected, jax.device_get(call(device)))

        def invoke(call):
            before = backend.calls
            jax.block_until_ready(call(gpu))
            if backend.calls != before + 1:
                raise RuntimeError("Expected exactly one runtime CPU retrieval")
            return {}

        calls = {
            name: lambda call=call: invoke(call) for name, call in compiled.items()
        }
        paired_samples(calls, args.samples, args.warmups, samples)
    result.update(
        exact_output_equivalence=True,
        distinct_queries_verified=True,
        scope="Synchronized GPU/CPU/GPU boundary; loopback TCP versus same-process Python, not remote networking or model latency",
    )
