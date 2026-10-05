"""Test U6: RelNaive, its cap and the seasonal naive against hand-computed examples (D21)."""
import numpy as np
import pytest

from src.metrics.rel_naive import rel_naive, seasonal_naive_forecast


def test_rel_naive_formula():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 5.0])
    naive = np.array([2.0, 2.0, 2.0])
    assert np.isclose(rel_naive(y, pred, naive), (2 / 3) / (2 / 3 + 1e-8))


def test_rel_naive_epsilon_and_clipping():
    y = np.array([1.0, 1.0])
    pred = np.array([100.0, 100.0])
    naive = np.array([1.0, 1.0])
    assert rel_naive(y, pred, naive) > 10
    assert rel_naive(y, pred, naive, clip=10) == 10


def test_u6_rel_naive_hand_computed():
    y = np.array([10.0, 12.0, 14.0])
    pred = np.array([11.0, 12.0, 12.0])   # MAE = (1 + 0 + 2) / 3 = 1
    naive = np.array([8.0, 8.0, 8.0])     # MAE = (2 + 4 + 6) / 3 = 4
    assert rel_naive(y, pred, naive) == pytest.approx(0.25, rel=1e-8)
    assert rel_naive(y, pred, naive, clip=10) == pytest.approx(0.25, rel=1e-8)  # below the cap: unchanged
    worse = np.array([60.0, 60.0, 60.0])  # MAE = 48 -> 12, capped at 10 (RelNaive^(10))
    assert rel_naive(y, worse, naive) == pytest.approx(12.0, rel=1e-8)
    assert rel_naive(y, worse, naive, clip=10) == 10


def test_u6_seasonal_naive_refuses_less_than_one_season():
    for short in ([], [1.0, 2.0, 3.0]):
        with pytest.raises(ValueError, match="needs 4 training values"):
            seasonal_naive_forecast(short, h=6, season_length=4)


def test_u6_seasonal_naive_repeats_the_last_season():
    assert seasonal_naive_forecast([1, 2, 3, 4, 5, 6, 7, 8], h=6, season_length=4).tolist() == [5, 6, 7, 8, 5, 6]
    monthly = np.arange(1, 37, dtype=float)
    expected = np.r_[np.arange(25, 37), np.arange(25, 31)]
    np.testing.assert_array_equal(seasonal_naive_forecast(monthly, h=18, season_length=12), expected)
