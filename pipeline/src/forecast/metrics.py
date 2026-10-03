"""Per-row accuracy measures (protocol D21, Section 5.1).

RelNaive and its cap of 10, MASE, sMAPE and MAE use the existing definitions
(src.metrics), which the earlier runs used. POCID follows the paper's equation: the
share of the h steps whose actual and forecast changes have the same sign, anchored on
the last training value (y_t stands in for the forecast at the first step); a zero
change on either side counts as no agreement.
"""
from __future__ import annotations

import numpy as np

from src.metrics.losses import mae, mase, smape
from src.metrics.rel_naive import rel_naive

RELNAIVE_CAP = 10.0


def pocid(last_train_value: float, y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    actual = np.diff(np.r_[last_train_value, y_true])
    predicted = np.diff(np.r_[last_train_value, y_pred])
    return float(np.mean(actual * predicted > 0))


def row_metrics(y_train, y_true, y_pred, y_naive, season_length: int) -> dict[str, float]:
    y_train = np.asarray(y_train, dtype=float)
    return {
        "relnaive_capped": rel_naive(y_true, y_pred, y_naive, clip=RELNAIVE_CAP),
        "relnaive": rel_naive(y_true, y_pred, y_naive),
        "mase": mase(y_true, y_pred, y_train, season_length),
        "smape": smape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "pocid": pocid(y_train[-1], y_true, y_pred),
    }
