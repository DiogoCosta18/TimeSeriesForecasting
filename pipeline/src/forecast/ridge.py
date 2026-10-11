"""Ridge of change log v2.7: standardised lags and a penalty per training row.

scikit-learn's ``Ridge`` minimises ||y - Xw||^2 + alpha ||w||^2, so its penalty competes with X'X,
which grows with the number of training rows and with the square of the level of the lags: on the
pooled M3/M4 series every alpha of the old range gave LinearRegression's forecasts (defect R1).
``ScaledRidge`` centres each lag and divides it by its standard deviation over the training rows
(a constant lag is left unscaled) and fits ``Ridge`` with penalty n * alpha on the n rows, i.e. it
minimises (1/n) ||y - Xw||^2 + alpha ||w||^2. Alpha then means the same for every level of the
series, target transform and pool size, and alpha -> 0 is exactly LinearRegression (scaling the
lags does not change least squares). The intercept is not penalised; the fit is deterministic.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge


class ScaledRidge(RegressorMixin, BaseEstimator):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        self.scale_ = np.where(scale > 0, scale, 1.0)
        self.ridge_ = Ridge(alpha=float(self.alpha) * len(X)).fit((X - self.mean_) / self.scale_, y)
        return self

    def predict(self, X):
        return self.ridge_.predict((np.asarray(X, dtype=float) - self.mean_) / self.scale_)
