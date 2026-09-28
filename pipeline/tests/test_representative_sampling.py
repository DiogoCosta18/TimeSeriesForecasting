import numpy as np
import pandas as pd

from src.data.sample_series import representative_sample


def _features(n=1000):
    return pd.DataFrame(
        {
            "unique_id": [f"s{i}" for i in range(n)],
            "source_dataset": ["M3_Monthly"] * n,
            "feature_non_normality": np.linspace(0, 1, n),
        }
    )


def test_representative_sampling_count_spread_deterministic_no_duplicates():
    f = _features()
    a = representative_sample(f, "feature_non_normality", "monthly", 750, 30, 123)
    b = representative_sample(f, "feature_non_normality", "monthly", 750, 30, 123)
    assert len(a) == 750
    assert a["unique_id"].is_unique
    assert set(a["unique_id"]) == set(b["unique_id"])
    assert a["selection_stratum"].nunique() >= 25

