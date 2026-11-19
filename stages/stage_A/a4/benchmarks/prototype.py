"""Pinned-memory CUDA-FFI prototype with a shared Python CPU-FAISS callback.

This isolates the boundary, NOT the production all-C++ retrieval implementation.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    CpuBackend,
    digest,
    expected_outputs,
    gpu_queries,
    paired_samples,
    verify_outputs,
)


def build(args: Any) -> tuple[Any, dict]:
    import jax

    if sys.platform != "linux":
        raise RuntimeError("The native prototype requires Linux with CUDA")
    if args.cuda_root:
        roots = [args.cuda_root]
    else:
        import nvidia

        roots = [Path(path) for path in nvidia.__path__]
    headers = sorted({p for root in roots for p in root.glob("**/cuda_runtime_api.h")})
    runtimes = sorted(
        {p for root in roots for p in root.glob("**/libcudart.so*") if p.is_file()}
    )
    crt = sorted({p for root in roots for p in root.glob("**/crt/host_defines.h")})
    if not headers or not runtimes:
        raise RuntimeError(
            "CUDA headers/runtime missing; supply --cuda-root or install matching NVIDIA runtime development packages"
        )
    source = Path(__file__).with_name("native_boundary.cc")
    output = args.output / "native_boundary.so"
    command = [
        args.compiler,
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-pthread",
        "-I" + jax.ffi.include_dir(),
        "-I" + str(headers[0].parent),
        *("-I" + str(path.parent.parent) for path in crt),
        str(source),
        str(runtimes[0]),
        "-Wl,-rpath," + str(runtimes[0].parent),
        "-o",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
    (args.output / "build.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    if completed.returncode:
        raise RuntimeError("Prototype compilation failed; see build.log")
    return ctypes.CDLL(str(output.resolve())), {
        "build_command": command,
        "native_source_sha256": digest(source),
        "native_library_sha256": digest(output),
    }


def measure(
    args: Any, index: Any, queries: np.ndarray, result: dict, samples: list
) -> None:
    import jax
    import jax.numpy as jnp

    from symbolic_memory.jax_bridge import JaxRetriever

    gpu = gpu_queries(queries)
    result.update(
        jax=jax.__version__,
        gpu=str(gpu.device),
        scope="Pinned CUDA-FFI prototype versus Python io_callback; both use Python CPU FAISS; no model denominator or production-native speedup claim",
    )
    library, build_info = build(args)
    result.update(build_info)
    n, d, m = len(queries), index.key_dim, args.top_m
    vector_count, metadata_count = n * m * d, n * m
    shape = (2 * vector_count + 2 * metadata_count,)
    backend: Any = CpuBackend(
        index,
        {
            "record_count": len(index.id_store),
            "vector_dimension": d,
            "max_top_m": 64,
            "max_queries": max(1024, n),
        },
        args.threads,
    )
    errors = []
    lookup = {name: i for i, name in enumerate(index.id_store)}
    callback_type = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_double),
    )

    @callback_type
    def query_callback(query_pointer, result_pointer, elapsed):
        try:
            q = np.ctypeslib.as_array(query_pointer, shape=(n * d,)).reshape(n, d)
            matches = backend.query_top_k(q, m)
            indices = np.asarray(
                [[lookup[name] for name in row] for row in matches.template_ids],
                dtype=np.int32,
            )
            packed = np.ctypeslib.as_array(result_pointer, shape=shape)
            packed[:vector_count] = matches.keys.reshape(-1)
            packed[vector_count : 2 * vector_count] = matches.values.reshape(-1)
            packed[2 * vector_count : 2 * vector_count + metadata_count] = (
                matches.scores.reshape(-1)
            )
            packed.view(np.int32)[2 * vector_count + metadata_count :] = (
                indices.reshape(-1)
            )
            elapsed[0] = backend.last_seconds
            return 0
        except BaseException as error:
            errors.append(f"{type(error).__name__}: {error}")
            return 1

    library.Configure.argtypes = [ctypes.c_size_t, ctypes.c_size_t, callback_type]
    library.Configure.restype = ctypes.c_int
    library.LastStats.argtypes = [ctypes.POINTER(ctypes.c_double)]
    library.LastStats.restype = None
    library.ReleaseBuffers.argtypes = []
    library.ReleaseBuffers.restype = ctypes.c_int
    try:
        status = library.Configure(queries.nbytes, shape[0] * 4, query_callback)
        if status:
            raise RuntimeError(f"Pinned buffer allocation failed: CUDA status {status}")
        target = "nsllm_a4_boundary_" + uuid.uuid4().hex
        jax.ffi.register_ffi_target(
            target, jax.ffi.pycapsule(library.NativeRetrieve), platform="CUDA"
        )

        def retrieve(q):
            packed = jax.ffi.ffi_call(
                target, jax.ShapeDtypeStruct(shape, jnp.float32), has_side_effect=True
            )(jax.lax.stop_gradient(q))
            return (
                packed[:vector_count].reshape(n, m, d),
                packed[vector_count : 2 * vector_count].reshape(n, m, d),
                packed[2 * vector_count : 2 * vector_count + metadata_count].reshape(
                    n, m
                ),
                jax.lax.bitcast_convert_type(
                    packed[2 * vector_count + metadata_count :].reshape(n, m), jnp.int32
                ),
            )

        native = jax.jit(retrieve).lower(gpu).compile()
        python = (
            jax.jit(JaxRetriever(backend, index.id_store, top_m=m)).lower(gpu).compile()
        )
        for host in (queries, -queries):
            expected = expected_outputs(backend, host, m)
            device = gpu_queries(host)
            for call in (native, python):
                verify_outputs(expected, jax.device_get(call(device)))
        gradient = jax.jit(jax.grad(lambda q: jnp.sum(q) + jnp.sum(retrieve(q)[0])))
        np.testing.assert_array_equal(
            jax.device_get(gradient(gpu)), np.ones_like(queries)
        )
        stats = (ctypes.c_double * 4)()

        def invoke(call, native_stats=False):
            before = backend.calls
            jax.block_until_ready(call(gpu))
            if backend.calls != before + 1:
                raise RuntimeError("Expected exactly one runtime CPU retrieval")
            details = {"cpu_index_seconds": backend.last_seconds}
            if native_stats:
                library.LastStats(stats)
                details.update(
                    d2h_enqueue_wait_seconds=stats[0],
                    host_callback_seconds=stats[1],
                    h2d_enqueue_wait_seconds=stats[2],
                )
            return details

        calls = {
            "native-prototype": lambda: invoke(native, True),
            "local-python": lambda: invoke(python),
        }
        paired_samples(calls, args.samples, args.warmups, samples)
        result.update(
            exact_output_equivalence=True,
            distinct_queries_verified=True,
            attention_query_gradient_preserved=True,
            cpu_timing_scope="Shared index.query_top_k: validation, normalization, search and gather; callback packing/metadata outside this interval",
            copy_timing_scope="CPU-clock enqueue-and-wait, not CUDA events or isolated PCIe occupancy",
        )
    finally:
        result["callback_errors"] = errors
        result["release_buffers_status"] = library.ReleaseBuffers()
    if result["release_buffers_status"]:
        raise RuntimeError("Could not release CUDA pinned buffers")
