from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)))


def wape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps))


def mase(y_true, y_pred, y_train, season_length: int, eps: float = 1e-8) -> float:
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) > season_length:
        denom = np.mean(np.abs(y_train[season_length:] - y_train[:-season_length]))
    elif len(y_train) > 1:
        denom = np.mean(np.abs(np.diff(y_train)))
    else:
        denom = 0.0
    return float(mae(y_true, y_pred) / (denom + eps))

