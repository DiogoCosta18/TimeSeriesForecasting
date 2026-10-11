"""Validation loss of the neural and transformer studies (protocol change log v2.6 (a), defect T1).

neuralforecast 3.1.8's ``MASE`` receives the input windows of a validation batch as [Ws, L, 1],
averages them over time into a scale of shape [Ws, 1] and indexes it as ``scale[:, None, None]``.
Against errors of shape [Ws, h, 1] this broadcasts to [Ws, Ws, h, 1]: every window's errors are
divided by every other window's scale, so the loss weights series by their level instead of being
scale-free. ``WindowMASE`` computes one scale per window.

The scale is the mean absolute seasonal difference over the last min(L, 3m) points of the input
window. NeuralForecast pads the input of a series whose training part is shorter than L with zeros
at the start; every eligible series has at least 3m observed points before its validation block
(D3), so those points are always real values (tested in tests/test_neural_losses.py).
"""
from __future__ import annotations

import torch
from neuralforecast.losses.pytorch import MASE, _divide_no_nan, _weighted_mean


class WindowMASE(MASE):
    """MASE with one seasonal-naive scale per validation window."""

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, y_insample: torch.Tensor,
                 mask: torch.Tensor | None = None) -> torch.Tensor:
        m = self.seasonality
        window = y_insample.reshape(y_insample.shape[0], -1)          # [Ws, L]; univariate models only
        window = window[:, -min(window.shape[1], 3 * m):]
        scale = torch.mean(torch.abs(window[:, m:] - window[:, :-m]), dim=1)          # [Ws]
        delta = torch.abs(y - y_hat)                                                    # [Ws, h, ...]
        losses = _divide_no_nan(delta, scale.reshape(-1, *([1] * (delta.ndim - 1))))
        return _weighted_mean(losses=losses, weights=self._compute_weights(y=y, mask=mask))
