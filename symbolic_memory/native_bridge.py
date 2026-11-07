"""Bank-bound CUDA/JAX retrieval with native CPU FAISS search.

Build explicitly against installed CUDA/JAX headers and a matching FAISS C++
SDK. Runtime search has no Python callback; Python configures immutable CPU
snapshots, opaque handles and optional out-of-band diagnostic query capture.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import weakref
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from residual.symbolic_head.retrieval.faiss_index import FaissTemplateIndex
from symbolic_memory.bank import TemplateBank
from symbolic_memory.jax_bridge import RetrievedTemplates
from symbolic_memory.storage import MorkTemplateStore

_ABI = 2
_LIBRARIES: dict[Path, tuple[Any, str]] = {}
_LIBRARY_LOCK = threading.Lock()
_FLOAT: Any = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
_INT: Any = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
_BYTE: Any = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")


def _faiss_sdk(
    prefix: Path | None, selected_library: Path | None = None
) -> tuple[Path, Path, str]:
    import faiss

    roots = (
        [prefix.resolve()]
        if prefix
        else [Path(sys.prefix), Path("/usr/local"), Path("/usr")]
    )
    for root in roots:
        include = root / "include"
        header = include / "faiss/Index.h"
        libraries = (
            [selected_library]
            if selected_library
            else [root / name / "libfaiss.so" for name in ("lib", "lib64")]
        )
        if not header.is_file():
            continue
        text = header.read_text(encoding="utf-8")
        parts = [
            re.search(r"#define\s+FAISS_VERSION_" + name + r"\s+(\d+)", text)
            for name in ("MAJOR", "MINOR", "PATCH")
        ]
        if not all(parts):
            raise RuntimeError(f"Cannot determine FAISS C++ header version: {header}")
        version = ".".join(part.group(1) for part in parts if part)
        if version != faiss.__version__:
            raise RuntimeError(
                f"FAISS C++ SDK {version} differs from Python FAISS {faiss.__version__}"
            )
        for library in libraries:
            if library.is_file():
                return include, library.resolve(), version
    raise RuntimeError(
        "Matching FAISS C++ headers and shared libfaiss.so not found. "
        "The Python wheel alone may not include them. Supply --faiss-prefix PREFIX "
        "with include/faiss/Index.h and lib/libfaiss.so (or lib64). "
        "No dependency was downloaded or installed."
    )


def build_native_library(
    output: Path,
    *,
    compiler: str = "g++",
    faiss_prefix: Path | None = None,
    faiss_library: Path | None = None,
) -> dict[str, Any]:
    """Compile against an existing matching FAISS shared SDK; never overwrite."""
    if os.name != "posix":
        raise RuntimeError("The native bridge currently requires Linux with CUDA")
    if not hasattr(jax, "ffi"):
        raise RuntimeError("The installed JAX must expose jax.ffi")
    output = output.resolve()
    if output.exists() or output.with_suffix(output.suffix + ".json").exists():
        raise FileExistsError("Choose a fresh native library output path")
    include, faiss_library, version = _faiss_sdk(faiss_prefix, faiss_library)
    import faiss
    import nvidia

    roots = [
        Path(jax.__file__).resolve().parent.parent / "nvidia",
        *(Path(p) for p in nvidia.__path__),
    ]
    headers = sorted({p for root in roots for p in root.glob("**/cuda_runtime_api.h")})
    runtimes = sorted({p for root in roots for p in root.glob("**/libcudart.so.*")})
    crt = sorted({p for root in roots for p in root.glob("**/crt/host_defines.h")})
    if not headers or not runtimes:
        raise RuntimeError(
            "Installed CUDA runtime headers/library not found; no packages were installed"
        )
    source = Path(__file__).with_name("native") / "retrieval.cc"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="nsllm-native-", dir=output.parent
    ) as temporary:
        built = Path(temporary) / "retrieval.so"
        command = [
            compiler,
            "-O3",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-pthread",
            "-fopenmp",
            "-I" + jax.ffi.include_dir(),
            "-I" + str(include),
            "-I" + str(headers[0].parent),
            *("-I" + str(p.parent.parent) for p in crt),
            str(source),
            str(faiss_library),
            str(runtimes[0]),
            "-Wl,-rpath," + str(faiss_library.parent),
            "-Wl,-rpath," + str(runtimes[0].parent),
            "-o",
            str(built),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
        if completed.returncode:
            raise RuntimeError(
                "Native bridge compilation failed:\n"
                + completed.stdout
                + completed.stderr
            )
        probe = ctypes.CDLL(str(built))
        _configure_library(probe)
        if (
            probe.NativeAbiVersion() != _ABI
            or probe.NativeFaissVersion().decode() != version
        ):
            raise RuntimeError("Built native ABI/FAISS version mismatch")
        manifest = {
            "abi": _ABI,
            "output_layouts": ["packed", "direct"],
            "return_modes": ["sync", "event"],
            "cuda_event_transfer_profiling": True,
            "jax_version": jax.__version__,
            "faiss_version": version,
            "native_faiss_compile_options": probe.NativeFaissCompileOptions().decode(),
            "python_faiss_compile_options": faiss.get_compile_options(),
            "faiss_include": str(include),
            "faiss_library": str(faiss_library),
            "faiss_library_sha256": hashlib.sha256(
                faiss_library.read_bytes()
            ).hexdigest(),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "library_sha256": hashlib.sha256(built.read_bytes()).hexdigest(),
            "command": command,
            "build_output": completed.stdout + completed.stderr,
        }
        os.link(built, output)
        with output.with_suffix(output.suffix + ".json").open(
            "x", encoding="utf-8"
        ) as stream:
            json.dump(manifest, stream, indent=2)
    return manifest


def _configure_library(library: Any) -> None:
    """Declare the complete C ABI, including CPU-only parity controls."""
    signatures = {
        "NativeAbiVersion": ([], ctypes.c_int),
        "NativeFaissVersion": ([], ctypes.c_char_p),
        "NativeFaissCompileOptions": ([], ctypes.c_char_p),
        "NativeLastError": ([], ctypes.c_char_p),
        "NativeCreateBank": (
            [
                _BYTE,
                ctypes.c_size_t,
                _FLOAT,
                _FLOAT,
                _INT,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_int,
            ],
            ctypes.c_int64,
        ),
        "NativeInvalidateBank": ([ctypes.c_int64], ctypes.c_int),
        "NativeCloseBank": ([ctypes.c_int64], ctypes.c_int),
        "NativeCreate": (
            [ctypes.c_int64, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int],
            ctypes.c_int64,
        ),
        "NativeCreateWithReturnMode": (
            [
                ctypes.c_int64,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_int,
            ],
            ctypes.c_int64,
        ),
        "NativeClose": ([ctypes.c_int64], ctypes.c_int),
        "NativeStats": (
            [ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int],
            ctypes.c_int,
        ),
        "NativeSetTransferTiming": ([ctypes.c_int64, ctypes.c_int], ctypes.c_int),
        "NativeTransferStats": (
            [ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int],
            ctypes.c_int,
        ),
        "NativeContextError": (
            [ctypes.c_int64, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t],
            ctypes.c_int,
        ),
        "NativeSearchHost": (
            [
                ctypes.c_int64,
                ctypes.c_size_t,
                ctypes.c_size_t,
                _FLOAT,
                _FLOAT,
                ctypes.POINTER(ctypes.c_double),
            ],
            ctypes.c_int,
        ),
        "NativeCaptureStart": ([ctypes.c_int64, ctypes.c_size_t], ctypes.c_int),
        "NativeCaptureStop": ([ctypes.c_int64], ctypes.c_int64),
        "NativeCaptureSize": ([ctypes.c_int64, ctypes.c_size_t], ctypes.c_int64),
        "NativeCaptureRead": (
            [ctypes.c_int64, ctypes.c_size_t, _FLOAT, ctypes.c_size_t],
            ctypes.c_int,
        ),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, result


def _library(path: Path) -> tuple[Any, str]:
    import faiss

    path = path.resolve(strict=True)
    with _LIBRARY_LOCK:
        if path not in _LIBRARIES:
            manifest = json.loads(path.with_suffix(path.suffix + ".json").read_text())
            source = Path(__file__).with_name("native") / "retrieval.cc"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if (
                manifest["abi"] != _ABI
                or manifest["library_sha256"] != digest
                or manifest["source_sha256"]
                != hashlib.sha256(source.read_bytes()).hexdigest()
                or manifest["jax_version"] != jax.__version__
                or manifest["faiss_version"] != faiss.__version__
                or manifest["faiss_library_sha256"]
                != hashlib.sha256(
                    Path(manifest["faiss_library"]).read_bytes()
                ).hexdigest()
            ):
                raise RuntimeError(
                    "Native source/JAX/FAISS/library fingerprint mismatch; rebuild explicitly"
                )
            library = ctypes.CDLL(str(path))
            _configure_library(library)
            if (
                library.NativeAbiVersion() != _ABI
                or library.NativeFaissVersion().decode() != faiss.__version__
            ):
                raise RuntimeError("Unsupported native ABI or FAISS version")
            target = (
                "nsllm_retrieval_"
                + hashlib.sha256((str(path) + digest).encode()).hexdigest()
            )
            jax.ffi.register_ffi_target(
                target,
                jax.ffi.pycapsule(library.NsllmNativeRetrieveV2),
                platform="CUDA",
            )
            jax.ffi.register_ffi_target(
                target + "_direct",
                jax.ffi.pycapsule(library.NsllmNativeRetrieveDirectV2),
                platform="CUDA",
            )
            _LIBRARIES[path] = library, target
        return _LIBRARIES[path]


def _error(library: Any, message: str) -> RuntimeError:
    detail = library.NativeLastError()
    return RuntimeError(
        f"{message}: {detail.decode() if detail else 'unknown native error'}"
    )


def _release(
    library: Any,
    contexts: dict[int, int],
    bank: int,
    index: FaissTemplateIndex,
    token: int,
) -> None:
    index.unwatch_native_generation(token)
    failures = [
        code
        for handle in list(contexts.values())
        if (code := library.NativeClose(handle))
    ]
    contexts.clear()
    code = library.NativeCloseBank(bank)
    if code:
        failures.append(code)
    if failures:
        raise RuntimeError(f"Native release failed: {failures}")


class NativeJaxRetriever:
    """Native CPU FAISS search with CUDA query/result transport and no GPU bank cache.

    Each shape has reusable pinned and search scratch buffers. Supported source
    index mutations invalidate existing compiled handles. Close after submitted
    JAX calls finish. Diagnostic capture is explicit and drained outside calls.
    """

    def __init__(
        self,
        index: FaissTemplateIndex,
        template_ids: Sequence[str],
        *,
        snapshot_sha256: str,
        library_path: Path,
        top_m: int = 8,
        max_queries: int = 1024,
        device_id: int = 0,
        cpu_threads: int = 2,
        output_layout: str = "packed",
        return_mode: str = "sync",
    ) -> None:
        if output_layout not in ("packed", "direct"):
            raise ValueError("output_layout must be packed or direct")
        if return_mode not in ("sync", "event"):
            raise ValueError("return_mode must be sync or event")
        for name, value in (
            ("top_m", top_m),
            ("max_queries", max_queries),
            ("cpu_threads", cpu_threads),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if len(snapshot_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in snapshot_sha256
        ):
            raise ValueError("Provide the bank's SHA256 snapshot identity")
        if (
            isinstance(device_id, bool)
            or not isinstance(device_id, int)
            or device_id < 0
        ):
            raise ValueError("device_id must be a nonnegative CUDA device ordinal")
        if not any(
            d.platform == "gpu" and d.id == device_id for d in jax.devices("gpu")
        ):
            raise RuntimeError("Requested JAX GPU device is unavailable")
        generation, native_ids, serialized, keys, values = index.native_snapshot()
        identifiers = tuple(template_ids)
        if (
            len(set(identifiers)) != len(identifiers)
            or not all(identifiers)
            or set(identifiers) != set(native_ids)
        ):
            raise ValueError("Native index and bank ID registries differ")
        lookup = {name: position for position, name in enumerate(identifiers)}
        mapping = np.array([lookup[name] for name in native_ids], dtype=np.int32)
        library, target = _library(library_path)
        bank = library.NativeCreateBank(
            serialized,
            serialized.nbytes,
            keys,
            values,
            mapping,
            len(native_ids),
            index.key_dim,
            cpu_threads,
        )
        if bank <= 0:
            raise _error(library, "Native bank creation failed")

        def invalidate():
            if library.NativeInvalidateBank(bank):
                raise RuntimeError("Native bank invalidation failed")

        try:
            token = index.watch_native_generation(generation, invalidate)
        except BaseException:
            library.NativeCloseBank(bank)
            raise
        self._library, self._target, self._bank = library, target, bank
        self._output_layout = output_layout
        self._return_mode = return_mode
        self._index = index
        self.template_ids = identifiers
        self._matches = min(top_m, len(identifiers))
        self._device_id = device_id
        self._contexts: dict[int, int] = {}
        self._max_shapes = 8
        self._lock = threading.RLock()
        self._closed, self._capturing = False, False
        self._timing_transfers = False
        self._finalizer = weakref.finalize(
            self, _release, library, self._contexts, bank, index, token
        )
        self.info = {
            "record_count": len(identifiers),
            "vector_dimension": index.key_dim,
            "max_top_m": top_m,
            "max_queries": max_queries,
            "input_snapshot_sha256": snapshot_sha256,
            "backend": "native",
            "native_abi": _ABI,
            "output_layout": output_layout,
            "return_mode": return_mode,
            "h2d_counter_scope": "enqueue-and-event-record"
            if return_mode == "event"
            else "enqueue-and-wait",
            "cpu_engine": "faiss-cpp",
            "cpu_threads": cpu_threads,
            "faiss_compile_options": library.NativeFaissCompileOptions().decode(),
            "python_search_callback": False,
            "faiss_index_sha256": hashlib.sha256(serialized.tobytes()).hexdigest(),
            "mork_readback_verified": False,
            "no_gpu_bank_cache": True,
            "cuda_event_transfer_profiling": True,
        }

    @classmethod
    def from_mork(
        cls,
        bank: TemplateBank,
        *,
        server_url: str,
        library_path: Path,
        top_m: int = 8,
        max_queries: int = 1024,
        device_id: int = 0,
        cpu_threads: int = 2,
        output_layout: str = "packed",
        return_mode: str = "sync",
    ) -> NativeJaxRetriever:
        with MorkTemplateStore(server_url, timeout_sec=300) as store:
            index = store.load_index(bank)
        retriever = cls(
            index,
            [r.template_id for r in bank.records],
            snapshot_sha256=bank.input_snapshot_sha256,
            library_path=library_path,
            top_m=top_m,
            max_queries=max_queries,
            device_id=device_id,
            cpu_threads=cpu_threads,
            output_layout=output_layout,
            return_mode=return_mode,
        )
        retriever.info["mork_readback_verified"] = True
        return retriever

    @property
    def last_error(self) -> str | None:
        with self._lock:
            for handle in self._contexts.values():
                buffer = ctypes.create_string_buffer(8192)
                if (
                    self._library.NativeContextError(handle, buffer, len(buffer)) == 0
                    and buffer.value
                ):
                    return buffer.value.decode()
        return None

    def _context(self, count: int) -> int:
        with self._lock:
            if self._closed:
                raise RuntimeError("Native retriever is closed")
            if count not in self._contexts:
                if len(self._contexts) >= self._max_shapes:
                    raise RuntimeError(
                        "Native retriever shape limit reached; close it before using more batch shapes"
                    )
                if self._return_mode == "event":
                    handle = self._library.NativeCreateWithReturnMode(
                        self._bank, count, self._matches, self._device_id, 1
                    )
                else:
                    handle = self._library.NativeCreate(
                        self._bank, count, self._matches, self._device_id
                    )
                if handle <= 0:
                    raise _error(
                        self._library, "Native pinned-buffer allocation failed"
                    )
                if self._timing_transfers and self._library.NativeSetTransferTiming(
                    handle, 1
                ):
                    self._library.NativeClose(handle)
                    raise _error(self._library, "CUDA transfer timing setup failed")
                self._contexts[count] = handle
            return self._contexts[count]

    def __call__(self, queries: jax.Array) -> RetrievedTemplates:
        if queries.dtype != jnp.float32 or queries.ndim not in (1, 2):
            raise ValueError("Queries must be rank-1 or rank-2 float32 JAX arrays")
        if queries.ndim == 1:
            queries = queries[None, :]
        count, dimension = queries.shape
        if (
            dimension != self.info["vector_dimension"]
            or not 1 <= count <= self.info["max_queries"]
        ):
            raise ValueError("Native query shape is outside the bound bank limits")
        handle = self._context(count)
        if self._output_layout == "direct":
            vector_shape = (count, self._matches, dimension)
            metadata_shape = (count, self._matches)
            outputs = jax.ffi.ffi_call(
                self._target + "_direct",
                (
                    jax.ShapeDtypeStruct(vector_shape, jnp.float32),
                    jax.ShapeDtypeStruct(vector_shape, jnp.float32),
                    jax.ShapeDtypeStruct(metadata_shape, jnp.float32),
                    jax.ShapeDtypeStruct(metadata_shape, jnp.int32),
                ),
                has_side_effect=True,
            )(jax.lax.stop_gradient(queries), context_id=np.int64(handle))
            return RetrievedTemplates(*outputs)
        vectors, metadata = count * self._matches * dimension, count * self._matches
        packed = jax.ffi.ffi_call(
            self._target,
            jax.ShapeDtypeStruct((2 * vectors + 2 * metadata,), jnp.float32),
            has_side_effect=True,
        )(jax.lax.stop_gradient(queries), context_id=np.int64(handle))
        return RetrievedTemplates(
            packed[:vectors].reshape(count, self._matches, dimension),
            packed[vectors : 2 * vectors].reshape(count, self._matches, dimension),
            packed[2 * vectors : 2 * vectors + metadata].reshape(count, self._matches),
            jax.lax.bitcast_convert_type(
                packed[2 * vectors + metadata :].reshape(count, self._matches),
                jnp.int32,
            ),
        )

    @contextmanager
    def capture_queries(self, *, max_calls: int = 64) -> Iterator[list[np.ndarray]]:
        """Capture on CPU in C++; fill the yielded list on exit, outside timed calls."""
        if (
            isinstance(max_calls, bool)
            or not isinstance(max_calls, int)
            or max_calls < 1
        ):
            raise ValueError("max_calls must be a positive integer")
        with self._lock:
            if self._closed or self._capturing:
                raise RuntimeError(
                    "Cannot capture on a closed retriever or nest capture"
                )
            if self._timing_transfers:
                raise RuntimeError("Cannot capture during transfer timing")
            jax.effects_barrier()
            if self._library.NativeCaptureStart(self._bank, max_calls):
                raise _error(self._library, "Native query capture failed")
            self._capturing = True
        captured: list[np.ndarray] = []
        try:
            yield captured
        finally:
            try:
                jax.effects_barrier()
            finally:
                with self._lock:
                    count = self._library.NativeCaptureStop(self._bank)
                    self._capturing = False
                    if count < 0:
                        raise RuntimeError("Native query capture failed or overflowed")
                    for i in range(count):
                        size = self._library.NativeCaptureSize(self._bank, i)
                        dim = self.info["vector_dimension"]
                        if size <= 0 or size % dim:
                            raise RuntimeError(
                                "Invalid captured native query dimensions"
                            )
                        value: np.ndarray = np.empty(
                            (size // dim, dim), dtype=np.float32
                        )
                        if self._library.NativeCaptureRead(self._bank, i, value, size):
                            raise RuntimeError("Native query capture read failed")
                        captured.append(value)

    @contextmanager
    def transfer_timing(self) -> Iterator[None]:
        """Enable GPU copy event intervals for an isolated profiling scope."""
        with self._lock:
            if self._closed or self._capturing or self._timing_transfers:
                raise RuntimeError("Transfer timing requires an open, idle retriever")
            jax.effects_barrier()
            enabled = []
            try:
                for handle in self._contexts.values():
                    if self._library.NativeSetTransferTiming(handle, 1):
                        raise _error(self._library, "CUDA transfer timing setup failed")
                    enabled.append(handle)
            except BaseException:
                for handle in enabled:
                    self._library.NativeSetTransferTiming(handle, 0)
                raise
            self._timing_transfers = True
        try:
            yield
        finally:
            try:
                jax.effects_barrier()
            finally:
                with self._lock:
                    self._timing_transfers = False
                    errors = [
                        handle
                        for handle in self._contexts.values()
                        if self._library.NativeSetTransferTiming(handle, 0)
                    ]
                    if errors:
                        raise _error(
                            self._library, "CUDA transfer timing cleanup failed"
                        )

    def transfer_stats(self, *, reset: bool = False) -> dict[str, float]:
        """Completed CUDA-event copy intervals; read outside timed model calls."""
        totals: np.ndarray = np.zeros(3, dtype=np.float64)
        with self._lock:
            if self._closed or not self._timing_transfers:
                raise RuntimeError("Read transfer stats inside transfer_timing()")
            for handle in self._contexts.values():
                values = (ctypes.c_double * 3)()
                if self._library.NativeTransferStats(handle, values, int(reset)):
                    raise _error(self._library, "CUDA transfer timing read failed")
                totals += np.asarray(values)
        return dict(
            zip(
                ("calls", "d2h_gpu_seconds", "h2d_gpu_seconds"),
                (float(value) for value in totals),
                strict=True,
            )
        )

    def stats(self, *, reset: bool = False) -> dict[str, float]:
        """Completed native-call counters, separate from external dispatch wall time."""
        totals: np.ndarray = np.zeros(5, dtype=np.float64)
        with self._lock:
            if self._closed:
                raise RuntimeError("Native retriever is closed")
            for handle in self._contexts.values():
                values = (ctypes.c_double * 5)()
                if self._library.NativeStats(handle, values, int(reset)):
                    raise RuntimeError("Native retrieval statistics context is closed")
                totals += np.asarray(values)
        return dict(
            zip(
                (
                    "calls",
                    "d2h_seconds",
                    "host_search_seconds",
                    "h2d_seconds",
                    "cpu_index_seconds",
                ),
                (float(v) for v in totals),
                strict=True,
            )
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._capturing:
                raise RuntimeError("Finish diagnostic capture before closing retrieval")
            if self._timing_transfers:
                raise RuntimeError("Finish transfer timing before closing retrieval")
            self._closed = True
        self._finalizer()

    def __enter__(self) -> NativeJaxRetriever:
        if self._closed:
            raise RuntimeError("Native retriever is closed")
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build",))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--faiss-prefix", type=Path)
    parser.add_argument(
        "--faiss-library",
        type=Path,
        help="Explicit matching shared library, e.g. libfaiss_avx2.so",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_native_library(
                args.output,
                compiler=args.compiler,
                faiss_prefix=args.faiss_prefix,
                faiss_library=args.faiss_library,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
