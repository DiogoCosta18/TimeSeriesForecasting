import numpy as np

from src.metrics.rel_naive import rel_naive


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

