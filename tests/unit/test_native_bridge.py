"""Native snapshot/lifetime controls and opt-in compiled FAISS/CUDA parity."""

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from residual.symbolic_head.contracts.template_record import TemplateRecord
from residual.symbolic_head.retrieval.faiss_index import FaissTemplateIndex
from symbolic_memory import native_bridge


def make_index(backend="flat", count=2, dim=2):
    rng = np.random.default_rng(8)
    keys = (
        np.eye(2, dtype=np.float32)
        if count == dim == 2
        else rng.normal(size=(count, dim)).astype(np.float32)
    )
    index = FaissTemplateIndex(key_dim=dim, backend=backend)
    index.rebuild(
        [
            TemplateRecord(
                f"id-{i}", f"(T item{i})", key, (key * (i + 2)).astype(np.float32)
            )
            for i, key in enumerate(keys)
        ]
    )
    return index


class ControlLibrary:
    """Handle protocol only; does not simulate native search or CUDA."""

    def __init__(self):
        self.closed = []
        self.closed_banks = []
        self.valid = {}
        self.issued = 0
        self.banks = 0
        self.snapshot = None
        self.return_modes = []
        self.timing_calls = []

    def NativeCreateBank(self, serialized, _size, keys, values, mapping, *_args):  # noqa: N802
        self.banks += 1
        self.valid[self.banks] = True
        self.snapshot = tuple(v.copy() for v in (serialized, keys, values, mapping))
        return self.banks

    def NativeCreate(self, bank, *args):  # noqa: N802
        if not self.valid[bank]:
            return -1
        self.issued += 1
        return self.issued

    def NativeCreateWithReturnMode(self, bank, count, matches, device, mode):  # noqa: N802
        self.return_modes.append(mode)
        return self.NativeCreate(bank, count, matches, device)

    def NativeInvalidateBank(self, bank):  # noqa: N802
        self.valid[bank] = False
        return 0

    def NativeCloseBank(self, bank):  # noqa: N802
        self.closed_banks.append(bank)
        self.valid[bank] = False
        return 0

    def NativeClose(self, handle):  # noqa: N802
        self.closed.append(handle)
        return 0

    def NativeLastError(self):  # noqa: N802
        return b"bank generation changed"

    def NativeFaissCompileOptions(self):  # noqa: N802
        return b"unit-control-only"

    def NativeCaptureStart(self, *_args):  # noqa: N802
        return 0

    def NativeCaptureStop(self, *_args):  # noqa: N802
        return 0

    def NativeSetTransferTiming(self, handle, enabled):  # noqa: N802
        self.timing_calls.append((handle, enabled))
        return 0

    def NativeTransferStats(self, _handle, output, _reset):  # noqa: N802
        output[0] = output[1] = output[2] = 0
        return 0


@pytest.fixture
def bound(monkeypatch):
    index = make_index()
    library = ControlLibrary()
    monkeypatch.setattr(
        jax, "devices", lambda platform: [SimpleNamespace(platform="gpu", id=0)]
    )
    monkeypatch.setattr(
        native_bridge, "_library", lambda path: (library, "unit-control-only")
    )
    with native_bridge.NativeJaxRetriever(
        index,
        ["id-1", "id-0"],
        snapshot_sha256="a" * 64,
        library_path=Path("unused.so"),
        top_m=2,
    ) as retriever:
        yield retriever, index, library


def test_setup_exports_existing_index_and_canonical_registry(bound):
    import faiss

    retriever, index, library = bound
    serialized, keys, values, mapping = library.snapshot
    np.testing.assert_array_equal(serialized, index.native_snapshot()[2])
    np.testing.assert_array_equal(mapping, [1, 0])
    restored = faiss.deserialize_index(serialized)
    assert restored.ntotal == 2
    np.testing.assert_array_equal(keys, np.stack(index.key_store))
    np.testing.assert_array_equal(values, np.stack(index.val_store))
    assert not retriever.info["python_search_callback"]
    assert retriever.info["cpu_engine"] == "faiss-cpp"
    assert not retriever.info["mork_readback_verified"]
    assert retriever.info["output_layout"] == "packed"
    assert retriever.info["return_mode"] == "sync"


def test_direct_outputs_have_typed_shapes_without_unpacking(bound):
    _retriever, index, _library = bound
    with native_bridge.NativeJaxRetriever(
        index,
        ["id-1", "id-0"],
        snapshot_sha256="a" * 64,
        library_path=Path("unused.so"),
        top_m=2,
        output_layout="direct",
    ) as direct:
        query = jax.ShapeDtypeStruct((3, 2), jnp.float32)
        outputs = jax.eval_shape(direct, query)
        assert outputs.keys.shape == outputs.values.shape == (3, 2, 2)
        assert outputs.scores.shape == outputs.template_indices.shape == (3, 2)
        assert outputs.template_indices.dtype == jnp.int32
        assert outputs.keys.dtype == outputs.values.dtype == jnp.float32
        assert outputs.scores.dtype == jnp.float32
        primitives = [
            e.primitive.name for e in jax.make_jaxpr(direct)(query).jaxpr.eqns
        ]
        assert "slice" not in primitives and "bitcast_convert_type" not in primitives
        assert direct.info["output_layout"] == "direct"


def test_invalid_output_layout_is_rejected(bound):
    _retriever, index, _library = bound
    with pytest.raises(ValueError, match="output_layout"):
        native_bridge.NativeJaxRetriever(
            index,
            ["id-1", "id-0"],
            snapshot_sha256="a" * 64,
            library_path=Path("unused.so"),
            output_layout="invalid",
        )


def test_event_return_is_explicit_and_bound_to_each_shape(bound):
    _retriever, index, library = bound
    with native_bridge.NativeJaxRetriever(
        index,
        ["id-1", "id-0"],
        snapshot_sha256="a" * 64,
        library_path=Path("unused.so"),
        return_mode="event",
    ) as event:
        event._context(1)
        event._context(1)
        event._context(3)
        assert library.return_modes == [1, 1]
        assert event.info["return_mode"] == "event"
        assert event.info["h2d_counter_scope"] == "enqueue-and-event-record"


def test_invalid_return_mode_is_rejected(bound):
    _retriever, index, _library = bound
    with pytest.raises(ValueError, match="return_mode"):
        native_bridge.NativeJaxRetriever(
            index,
            ["id-1", "id-0"],
            snapshot_sha256="a" * 64,
            library_path=Path("unused.so"),
            return_mode="invalid",
        )


def test_shapes_are_distinct_and_close_releases_bank_last(bound):
    retriever, index, library = bound
    assert retriever._context(1) == retriever._context(1)
    assert retriever._context(3) != retriever._context(1)
    retriever.close()
    retriever.close()
    assert library.closed == [1, 2]
    assert library.closed_banks == [1]
    assert not index._native_listeners
    with pytest.raises(RuntimeError, match="closed"):
        retriever._context(1)


@pytest.mark.parametrize("mutation", ["clear", "rebuild", "add"])
def test_supported_mutations_invalidate_native_bank(bound, mutation):
    retriever, index, library = bound
    retriever._context(1)
    if mutation == "clear":
        index.clear()
    elif mutation == "rebuild":
        index.rebuild([])
    else:
        index.add_template(
            "new", "(New)", np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32)
        )
    assert not library.valid[retriever._bank]
    with pytest.raises(RuntimeError, match="generation changed"):
        retriever._context(3)


def test_capture_is_explicit_and_cannot_close_or_nest(bound, monkeypatch):
    retriever, _index, _library = bound
    monkeypatch.setattr(jax, "effects_barrier", lambda: None)
    with retriever.capture_queries() as captured:
        with pytest.raises(RuntimeError, match="nest"), retriever.capture_queries():
            pass
        with pytest.raises(RuntimeError, match="Finish"):
            retriever.close()
    assert captured == []
    assert not retriever._capturing


def test_transfer_timing_is_scoped_and_covers_new_shapes(bound, monkeypatch):
    retriever, _index, library = bound
    monkeypatch.setattr(jax, "effects_barrier", lambda: None)
    retriever._context(1)
    with retriever.transfer_timing():
        retriever._context(3)
        assert retriever.transfer_stats() == {
            "calls": 0,
            "d2h_gpu_seconds": 0,
            "h2d_gpu_seconds": 0,
        }
        with pytest.raises(RuntimeError, match="idle"), retriever.transfer_timing():
            pass
        with (
            pytest.raises(RuntimeError, match="transfer timing"),
            retriever.capture_queries(),
        ):
            pass
        with pytest.raises(RuntimeError, match="Finish transfer timing"):
            retriever.close()
    assert library.timing_calls == [(1, 1), (2, 1), (1, 0), (2, 0)]
    with pytest.raises(RuntimeError, match="inside transfer_timing"):
        retriever.transfer_stats()


def test_transfer_timing_setup_failure_rolls_back_enabled_handles(bound, monkeypatch):
    retriever, _index, library = bound
    monkeypatch.setattr(jax, "effects_barrier", lambda: None)
    retriever._context(1)
    retriever._context(3)
    original = library.NativeSetTransferTiming

    def fail_second(handle, enabled):
        return -1 if handle == 2 and enabled else original(handle, enabled)

    monkeypatch.setattr(library, "NativeSetTransferTiming", fail_second)
    with pytest.raises(RuntimeError, match="timing setup"), retriever.transfer_timing():
        pass
    assert library.timing_calls == [(1, 1), (1, 0)]
    assert not retriever._timing_transfers


def test_pinned_shapes_are_bounded(bound):
    retriever, _index, library = bound
    for count in range(1, 9):
        retriever._context(count)
    with pytest.raises(RuntimeError, match="shape limit"):
        retriever._context(9)
    assert library.issued == 8


def test_setup_race_closes_unbound_native_bank(bound, monkeypatch):
    _retriever, index, library = bound

    def fail(*args):
        raise RuntimeError("generation changed during setup")

    monkeypatch.setattr(index, "watch_native_generation", fail)
    with pytest.raises(RuntimeError, match="during setup"):
        native_bridge.NativeJaxRetriever(
            index,
            ["id-0", "id-1"],
            snapshot_sha256="b" * 64,
            library_path=Path("unused.so"),
        )
    assert library.closed_banks == [2]


@pytest.fixture
def compiled_library():
    path = os.environ.get("NSLLM_NATIVE_LIBRARY")
    if not path:
        pytest.skip(
            "Set NSLLM_NATIVE_LIBRARY to the built ABI-2 library for real C++/CUDA checks"
        )
    library = ctypes.CDLL(str(Path(path).resolve(strict=True)))
    native_bridge._configure_library(library)
    import faiss

    assert library.NativeAbiVersion() == 2
    assert library.NativeFaissVersion().decode() == faiss.__version__
    return library


@pytest.fixture
def cpu_bank(compiled_library):
    handles = []

    def create(index, ids=None):
        _generation, names, serialized, keys, values = index.native_snapshot()
        ids = tuple(reversed(names)) if ids is None else tuple(ids)
        mapping = np.array([ids.index(name) for name in names], dtype=np.int32)
        handle = compiled_library.NativeCreateBank(
            serialized,
            serialized.nbytes,
            keys,
            values,
            mapping,
            len(names),
            index.key_dim,
            2,
        )
        assert handle > 0, compiled_library.NativeLastError()
        handles.append(handle)
        return handle, ids

    yield compiled_library, create
    for handle in handles:
        assert compiled_library.NativeCloseBank(handle) == 0


def host_search(library, handle, query, top_m):
    result = np.empty(len(query) * top_m * (2 * query.shape[1] + 2), dtype=np.float32)
    elapsed = ctypes.c_double()
    code = library.NativeSearchHost(
        handle, len(query), top_m, query, result, ctypes.byref(elapsed)
    )
    return code, result, elapsed.value


@pytest.mark.parametrize("backend", ["flat", "hnsw"])
def test_compiled_cpu_search_matches_existing_index(cpu_bank, backend):
    library, create = cpu_bank
    index = make_index(backend, count=40, dim=16)
    handle, ids = create(index)
    query = np.random.default_rng(9).normal(size=(24, 16)).astype(np.float32)
    expected = index.query_top_k(query, 8)
    code, packed, elapsed = host_search(library, handle, query, 8)
    assert code == 0 and elapsed > 0, library.NativeLastError()
    vectors, metadata = len(query) * 8 * 16, len(query) * 8
    np.testing.assert_array_equal(packed[:vectors].reshape(24, 8, 16), expected.keys)
    np.testing.assert_array_equal(
        packed[vectors : 2 * vectors].reshape(24, 8, 16), expected.values
    )
    np.testing.assert_allclose(
        packed[2 * vectors : 2 * vectors + metadata].reshape(24, 8),
        expected.scores,
        rtol=2e-5,
        atol=2e-6,
    )
    expected_ids = np.array(
        [[ids.index(name) for name in row] for row in expected.template_ids]
    )
    np.testing.assert_array_equal(
        packed.view(np.int32)[-metadata:].reshape(24, 8), expected_ids
    )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_compiled_cpu_search_rejects_nonfinite_queries(cpu_bank, value):
    library, create = cpu_bank
    handle, _ids = create(make_index())
    code, _packed, _elapsed = host_search(
        library, handle, np.array([[value, 1]], dtype=np.float32), 2
    )
    assert code != 0
    assert b"finite" in library.NativeLastError()


def test_compiled_cpu_zero_query_and_invalidation(cpu_bank):
    library, create = cpu_bank
    handle, _ids = create(make_index())
    code, packed, _elapsed = host_search(
        library, handle, np.zeros((1, 2), dtype=np.float32), 2
    )
    assert code == 0 and np.isfinite(packed).all()
    np.testing.assert_array_equal(packed[8:10], [0, 0])
    assert library.NativeInvalidateBank(handle) == 0
    assert host_search(library, handle, np.ones((1, 2), dtype=np.float32), 2)[0] != 0


def test_compiled_capture_copies_runtime_query_without_python_callback(cpu_bank):
    library, create = cpu_bank
    handle, _ids = create(make_index())
    query = np.array([[0.3, 1.2]], dtype=np.float32)
    assert library.NativeCaptureStart(handle, 1) == 0
    assert host_search(library, handle, query, 2)[0] == 0
    assert library.NativeCaptureStop(handle) == 1
    assert library.NativeCaptureSize(handle, 0) == query.size
    captured = np.empty_like(query)
    assert library.NativeCaptureRead(handle, 0, captured, query.size) == 0
    np.testing.assert_array_equal(captured, query)
    assert library.NativeCaptureStart(handle, 1) == 0
    assert host_search(library, handle, query, 2)[0] == 0
    assert host_search(library, handle, query, 2)[0] != 0
    assert library.NativeCaptureStop(handle) == -1


def test_compiled_banks_keep_separate_registries(cpu_bank):
    library, create = cpu_bank
    index = make_index()
    reverse, _ = create(index)
    forward, _ = create(index, index.id_store)
    query = np.array([[1, 0]], dtype=np.float32)
    left = host_search(library, reverse, query, 2)
    right = host_search(library, forward, query, 2)
    assert left[0] == right[0] == 0
    np.testing.assert_array_equal(left[1].view(np.int32)[-2:], [1, 0])
    np.testing.assert_array_equal(right[1].view(np.int32)[-2:], [0, 1])
    library.NativeInvalidateBank(reverse)
    assert host_search(library, forward, query, 2)[0] == 0


def test_compiled_invalid_return_mode_is_rejected_before_allocation(cpu_bank):
    library, create = cpu_bank
    handle, _ = create(make_index())
    assert library.NativeCreateWithReturnMode(handle, 1, 2, 0, 2) == -1
    assert b"Invalid native context arguments" in library.NativeLastError()


@pytest.mark.parametrize("backend", ["flat", "hnsw"])
@pytest.mark.parametrize("output_layout", ["packed", "direct"])
@pytest.mark.parametrize("return_mode", ["sync", "event"])
def test_real_cuda_runtime_queries_capture_gradients_and_stale_handles(
    compiled_library, backend, output_layout, return_mode
):
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("Real JAX CUDA device required")
    index = make_index(backend, count=40, dim=16)
    path = Path(os.environ["NSLLM_NATIVE_LIBRARY"])
    ids = tuple(reversed(index.id_store))
    retriever = native_bridge.NativeJaxRetriever(
        index,
        ids,
        snapshot_sha256="a" * 64,
        library_path=path,
        top_m=8,
        output_layout=output_layout,
        return_mode=return_mode,
    )
    try:
        call = jax.jit(retriever)
        queries = [
            np.random.default_rng(seed).normal(size=(4, 16)).astype(np.float32)
            for seed in (4, 6)
        ]
        expected = [index.query_top_k(q, 8) for q in queries]
        for query, reference in zip(queries, expected, strict=True):
            gpu = jax.device_put(query)
            actual = jax.block_until_ready(call(gpu))
            assert actual.template_indices.dtype == jnp.int32
            np.testing.assert_array_equal(actual.keys, reference.keys)
            np.testing.assert_array_equal(actual.values, reference.values)
            np.testing.assert_allclose(
                actual.scores, reference.scores, rtol=2e-5, atol=2e-6
            )
            np.testing.assert_array_equal(
                actual.template_indices,
                [[ids.index(name) for name in row] for row in reference.template_ids],
            )
        with retriever.capture_queries() as captured:
            jax.block_until_ready(call(jax.device_put(queries[1])))
        assert len(captured) == 1
        np.testing.assert_array_equal(captured[0], queries[1])
        before = retriever.stats()
        gradient = jax.jit(
            jax.grad(lambda q: jnp.sum(retriever(q).keys * q[:, None, :]))
        )
        actual_grad = jax.block_until_ready(gradient(jax.device_put(queries[0])))
        np.testing.assert_allclose(
            actual_grad, expected[0].keys.sum(axis=1), rtol=2e-5, atol=2e-6
        )
        assert retriever.stats()["calls"] > before["calls"]

        def reject_stale_handles():
            # JAX retains failed effect tokens per dispatch thread. Keep expected
            # failures off pytest's main thread; assert both errors before exit.
            index.clear()
            with pytest.raises(Exception, match="generation changed"):
                jax.block_until_ready(call(jax.device_put(queries[0])))
            retriever.close()
            with pytest.raises(Exception, match="closed or unknown"):
                jax.block_until_ready(call(jax.device_put(queries[0])))

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(reject_stale_handles).result()
        # Successful calls on the main thread must still have a clean barrier.
        jax.effects_barrier()
    finally:
        retriever.close()


@pytest.mark.parametrize("output_layout", ["packed", "direct"])
def test_real_cuda_event_return_reuses_buffers_with_gpu_consumers(
    compiled_library, output_layout
):
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("Real JAX CUDA device required")
    index = make_index("flat", count=64, dim=256)
    ids = tuple(reversed(index.id_store))
    options = {
        "snapshot_sha256": "a" * 64,
        "library_path": Path(os.environ["NSLLM_NATIVE_LIBRARY"]),
        "top_m": 8,
        "output_layout": output_layout,
    }
    with (
        native_bridge.NativeJaxRetriever(
            index, ids, return_mode="event", **options
        ) as retriever,
        native_bridge.NativeJaxRetriever(
            index, ids, return_mode="sync", **options
        ) as synchronous,
    ):

        def consumer(retrieve):
            @jax.jit
            def consume(query):
                result = retrieve(query)
                scores = jnp.einsum(
                    "bd,bmd->bm",
                    query,
                    result.keys,
                    precision=jax.lax.Precision.HIGHEST,
                )
                weights = jax.nn.softmax(scores, axis=-1)
                context = jnp.einsum(
                    "bm,bmd->bd",
                    weights,
                    result.values,
                    precision=jax.lax.Precision.HIGHEST,
                )
                return result, context

            return consume

        queries = [
            np.random.default_rng(seed).normal(size=(128, 256)).astype(np.float32)
            for seed in range(4)
        ]
        references = [index.query_top_k(query, 8) for query in queries]
        gpu_queries = [jax.device_put(query) for query in queries]
        call = consumer(retriever).lower(gpu_queries[0]).compile()
        sync_call = consumer(synchronous).lower(gpu_queries[0]).compile()
        # Same GPU attention graph, changing only native return synchronization.
        sync_outputs = [jax.device_get(sync_call(query)) for query in gpu_queries]
        jax.block_until_ready(call(gpu_queries[0]))

        def submit(offset):
            # No per-call readiness barrier: exercise repeated and concurrent reuse.
            return [
                (i, call(gpu_queries[i])) for i in [(offset + j) % 4 for j in range(8)]
            ]

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(submit, offset) for offset in (0, 2)]
            outputs = [item for future in futures for item in future.result()]
        for i, output in outputs:
            result, context = jax.device_get(output)
            reference = references[i]
            np.testing.assert_array_equal(result.keys, reference.keys)
            np.testing.assert_array_equal(result.values, reference.values)
            np.testing.assert_allclose(
                result.scores, reference.scores, rtol=2e-5, atol=2e-6
            )
            np.testing.assert_array_equal(
                result.template_indices,
                [[ids.index(name) for name in row] for row in reference.template_ids],
            )
            sync_result, sync_context = sync_outputs[i]
            np.testing.assert_array_equal(sync_result.keys, reference.keys)
            np.testing.assert_array_equal(sync_result.values, reference.values)
            assert np.isfinite(context).all() and np.isfinite(sync_context).all()
            np.testing.assert_allclose(context, sync_context, rtol=2e-4, atol=2e-4)
        assert retriever.stats()["calls"] == 17
        jax.effects_barrier()


@pytest.mark.parametrize("return_mode", ["sync", "event"])
@pytest.mark.parametrize("output_layout", ["packed", "direct"])
def test_real_cuda_transfer_event_intervals_are_optional_and_counted_once(
    compiled_library, return_mode, output_layout
):
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("Real JAX CUDA device required")
    index = make_index("flat", count=64, dim=256)
    with native_bridge.NativeJaxRetriever(
        index,
        index.id_store,
        snapshot_sha256="a" * 64,
        library_path=Path(os.environ["NSLLM_NATIVE_LIBRARY"]),
        top_m=8,
        return_mode=return_mode,
        output_layout=output_layout,
    ) as retriever:
        call = jax.jit(retriever)
        query = np.random.default_rng(70).normal(size=(128, 256)).astype(np.float32)
        reference = index.query_top_k(query, 8)
        gpu = jax.device_put(query)
        jax.block_until_ready(call(gpu))
        with retriever.transfer_timing():
            for current in (gpu, gpu[:3], gpu):
                actual = jax.block_until_ready(call(current))
                np.testing.assert_array_equal(
                    actual.keys, reference.keys[: len(current)]
                )
                np.testing.assert_array_equal(
                    actual.values, reference.values[: len(current)]
                )
            stats = retriever.transfer_stats()
            assert stats["calls"] == 3
            assert stats["d2h_gpu_seconds"] > 0 and stats["h2d_gpu_seconds"] > 0
            assert retriever.transfer_stats() == stats
            assert retriever.transfer_stats(reset=True) == stats
            assert all(value == 0 for value in retriever.transfer_stats().values())

        # Disabled scopes must not collect events for ordinary production calls.
        jax.block_until_ready(call(gpu))
        values = (ctypes.c_double * 3)()
        assert (
            compiled_library.NativeTransferStats(retriever._context(128), values, 0)
            == 0
        )
        assert list(values) == [0, 0, 0]
        with retriever.transfer_timing():
            jax.block_until_ready(call(gpu))
            assert retriever.transfer_stats()["calls"] == 1
        jax.effects_barrier()
