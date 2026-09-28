import numpy as np

from src.transforms.leakage_checks import assert_stl_fit_uses_training_only
from src.transforms.stl_transform import STLTransform


def test_stl_transform_fits_only_training_slice():
    y = np.arange(60, dtype=float)
    train = y[:40]
    tr = STLTransform(season_length=12).fit(train)
    assert tr.n_train_ == len(train)
    assert len(tr.components()["trend"]) == len(train)
    assert_stl_fit_uses_training_only(tr, len(train))

