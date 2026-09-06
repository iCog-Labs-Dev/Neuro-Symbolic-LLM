"""GPU memory monitoring and the 50% headroom safety rule.

Tests that require a real accelerator are skipped automatically on CPU-only
environments (e.g. CI) instead of failing.
"""

from __future__ import annotations

import jax
import pytest
from substrate import (
    MemoryStatus,
    check_memory_headroom,
    compute_memory_headroom,
    get_memory_status,
    maybe_reduce_batch_size,
)


def _has_gpu() -> bool:
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:  # pragma: no cover - platform dependent
        return False


_requires_gpu = pytest.mark.skipif(
    not _has_gpu(), reason="no GPU/accelerator visible to JAX on this machine"
)


def _fake_status(total: int, allocated: int) -> MemoryStatus:
    available = total - allocated
    return MemoryStatus(
        available=True,
        total_bytes=total,
        allocated_bytes=allocated,
        available_bytes=available,
        headroom_ratio=available / total,
        device="gpu:0",
        platform="cuda",
        diagnostic="ok",
        raw={},
    )


class TestGetMemoryStatus:
    def test_returns_status_object(self):
        status = get_memory_status()
        assert isinstance(status, MemoryStatus)

    @_requires_gpu
    def test_gpu_memory_is_available(self):
        status = get_memory_status()
        assert status.available is True
        assert status.platform == "gpu"
        assert status.total_bytes > 0
        assert status.available_bytes >= 0
        assert status.allocated_bytes >= 0

    @_requires_gpu
    def test_gpu_headroom_is_reported(self):
        status = get_memory_status()
        headroom = compute_memory_headroom(status)

        assert headroom is not None
        assert 0.0 <= headroom <= 1.0


class TestHeadroomRule:
    def test_safe_headroom_no_warning(self):
        status = _fake_status(total=1000, allocated=300)

        assert check_memory_headroom(status, min_headroom=0.5) == []
        assert compute_memory_headroom(status) == pytest.approx(0.7)

    def test_below_50_percent_warns(self):
        status = _fake_status(total=1000, allocated=600)

        warnings = check_memory_headroom(status, min_headroom=0.5)

        assert len(warnings) == 1
        assert "below" in warnings[0]
        assert "50%" in warnings[0]

    def test_configurable_threshold(self):
        status = _fake_status(total=1000, allocated=600)

        # Headroom = 0.4, safe under a 0.3 threshold.
        assert check_memory_headroom(status, min_headroom=0.3) == []

        # Headroom = 0.4, unsafe under a 0.5 threshold.
        assert len(check_memory_headroom(status, min_headroom=0.5)) == 1


class TestAutoBatchReduction:
    def test_no_reduction_when_safe(self):
        status = _fake_status(total=1000, allocated=300)

        batch, reduced, warnings = maybe_reduce_batch_size(
            status,
            8,
            auto_reduce=True,
        )

        assert batch == 8
        assert reduced is False
        assert warnings == []

    def test_warn_only_by_default(self):
        status = _fake_status(total=1000, allocated=900)

        batch, reduced, warnings = maybe_reduce_batch_size(
            status,
            8,
            auto_reduce=False,
        )

        # Configuration remains untouched when auto-reduction is disabled.
        assert batch == 8
        assert reduced is False
        assert any("WARNING" in w for w in warnings)

    def test_auto_reduce_reports_reduction(self):
        status = _fake_status(total=1000, allocated=900)

        batch, reduced, warnings = maybe_reduce_batch_size(
            status,
            8,
            auto_reduce=True,
        )

        assert reduced is True
        assert batch < 8
        assert any("automatically reduced" in w for w in warnings)

    def test_no_silent_change(self):
        status = _fake_status(total=1000, allocated=900)

        batch, reduced, warnings = maybe_reduce_batch_size(
            status,
            8,
            auto_reduce=False,
        )

        # Never silently changes the user's configuration.
        assert batch == 8
        assert reduced is False


class TestDrift:
    def test_kl_zero_for_identical(self):
        import jax.numpy as jnp
        from substrate import compute_kl_drift

        logits = jnp.array(
            [
                [1.0, 2.0, 3.0],
                [0.0, 0.0, 1.0],
            ]
        )

        result = compute_kl_drift(logits, logits)

        assert result["kl_divergence"] == 0.0

    def test_kl_positive_for_different(self):
        import jax.numpy as jnp
        from substrate import compute_kl_drift

        a = jnp.array([[1.0, 2.0, 3.0]])
        b = jnp.array([[3.0, 2.0, 1.0]])

        result = compute_kl_drift(a, b)

        assert result["kl_divergence"] > 0.0

    def test_kl_stable(self):
        # Extreme logits must not produce NaN.
        import jax.numpy as jnp
        from substrate import compute_kl_drift

        a = jnp.array([[1e10, -1e10, 0.0]])
        b = jnp.array([[-1e10, 1e10, 0.0]])

        result = compute_kl_drift(a, b)

        assert result["kl_divergence"] == result["kl_divergence"]
