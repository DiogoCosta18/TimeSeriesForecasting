from __future__ import annotations

import numpy as np

from .errors import FeatureUndefined
from .mstl_features import finite_array


def nonlinearity_feature(y, season_length: int) -> float:
    y = finite_array(y)
    max_lag = min(max(2, season_length), 12)
    # Guard: monthly 48 (4 x 12 lags); quarterly 20, lowered from 30 by protocol D5.
    if len(y) < max(20, 4 * max_lag):
        raise FeatureUndefined("too_short")
    if np.std(y) < 1e-10:
        raise FeatureUndefined("constant")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error

    X, target = [], []
    for t in range(max_lag, len(y)):
        X.append(y[t - max_lag : t])
        target.append(y[t])
    X = np.asarray(X)
    target = np.asarray(target)
    n = len(target)
    split_starts = [int(n * 0.55), int(n * 0.7), int(n * 0.85)]
    lin_err, nonlin_err = [], []
    for split in split_starts:
        if split <= max_lag or split >= n - 2:
            continue
        lin = Ridge(alpha=1.0).fit(X[:split], target[:split])
        rf = RandomForestRegressor(n_estimators=40, max_depth=4, random_state=17, n_jobs=1).fit(X[:split], target[:split])
        lin_err.append(mean_absolute_error(target[split:], lin.predict(X[split:])))
        nonlin_err.append(mean_absolute_error(target[split:], rf.predict(X[split:])))
    if not lin_err:
        raise FeatureUndefined("no_valid_split")
    mae_linear = float(np.mean(lin_err))
    mae_nonlinear = float(np.mean(nonlin_err))
    return float(np.clip(max(0.0, mae_linear - mae_nonlinear) / (mae_linear + 1e-8), 0, 2))
