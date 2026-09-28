import numpy as np

from src.transforms.recomposition import recompose_components, recompose_nonseasonal
from src.transforms.stl_transform import STLTransform


def test_components_recompose_original_approximately():
    t = np.arange(72)
    y = 10 + 0.1 * t + np.sin(2 * np.pi * t / 12)
    tr = STLTransform(12).fit(y)
    c = tr.components()
    assert np.allclose(recompose_components(c["trend"], c["seasonal"], c["residual"]), y, atol=1e-5)


def test_forecast_recomposition_identities():
    nonseasonal = np.array([1.0, 2.0])
    seasonal = np.array([10.0, 20.0])
    assert np.allclose(recompose_nonseasonal(nonseasonal, seasonal), np.array([11.0, 22.0]))
    assert np.allclose(recompose_components([1, 2], [3, 4], [5, 6]), np.array([9, 12]))

