"""Unit tests for residual compressibility. No model needed."""

from __future__ import annotations

import pytest

from metrics.representation import (
    compressibility_curve,
    loss_improvement,
    residual_compressibility,
)


def test_loss_improvement_sign() -> None:
    assert float(loss_improvement(3.0, 2.5)) == pytest.approx(0.5)
    assert float(loss_improvement(3.0, 3.2)) == pytest.approx(-0.2)


def test_residual_compressibility() -> None:
    assert float(residual_compressibility(0.4, 0.5)) == pytest.approx(0.8)
    # eps keeps a zero denominator finite.
    assert float(residual_compressibility(0.0, 0.0, eps=1e-3)) == 0.0


def test_negative_eps_raises() -> None:
    with pytest.raises(ValueError, match="eps"):
        residual_compressibility(0.1, 0.2, eps=-1.0)


def test_compressibility_curve_is_relative_to_largest_rank() -> None:
    curve = compressibility_curve({64: 0.50, 1: 0.10, 16: 0.45}, eps=0.0)
    assert list(curve) == [1, 16, 64]
    assert float(curve[1]) == pytest.approx(0.2)
    assert float(curve[16]) == pytest.approx(0.9)
    assert float(curve[64]) == pytest.approx(1.0)


@pytest.mark.parametrize("deltas", [{}, {0: 0.1, 4: 0.2}])
def test_compressibility_curve_rejects_bad_input(deltas: dict[int, float]) -> None:
    with pytest.raises(ValueError):
        compressibility_curve(deltas)
