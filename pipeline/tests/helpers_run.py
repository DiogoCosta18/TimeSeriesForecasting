"""A small synthetic frozen copy and configuration for the stage tests.

Test-only: the pipeline itself never generates data. Each source holds eight regular
series, one too short for L (SHORT) and one with a constant feature history (FLAT).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from helpers_frozen import write_frozen
from src.data.frozen import COLUMNS

COMMIT = "a" * 40
CONFIG = {
    "random_seed": 7,
    "seed_check_seeds": [8, 9],
    "frequencies": {
        "monthly": {"season_length": 12, "horizon": 18, "m3_group": "Monthly", "m4_group": "Monthly"},
        "quarterly": {"season_length": 4, "horizon": 8, "m3_group": "Quarterly", "m4_group": "Quarterly"},
    },
    "validation": {"n_windows": 3},
    "sampling": {"n_per_source": 6, "n_strata": 3},
    "buckets": {"min_share": 0.2},
    "tuning_set": {"n_per_source": 6},
}
SPEC = {"Monthly": (12, 120, 100), "Quarterly": (4, 50, 40)}  # m, regular length, short length
SOURCES = ["M3_Monthly", "M4_Monthly", "M3_Quarterly", "M4_Quarterly"]


def canonical(key: str, seed: int) -> pd.DataFrame:
    group = key.split("_")[1]
    m, n, n_short = SPEC[group]
    rng = np.random.default_rng(seed)
    series = {}
    for i in range(8):
        t = np.arange(n)
        series[f"R{i}"] = 100 + 0.2 * t + 6 * np.sin(2 * np.pi * t / m + i) + rng.normal(0, 1.5 + i / 4, n)
    series["SHORT"] = 100 + rng.normal(0, 2, n_short)       # history below L
    flat = 100 + rng.normal(0, 2, n)
    flat[: n - 3 * {12: 18, 4: 8}[m]] = 50.0                  # constant history: features undefined
    series["FLAT"] = flat
    rows = [
        {"unique_id": f"{key}_{sid}", "source_dataset": key, "frequency": group.lower(), "t": t, "y": float(v), "ds_source": str(t)}
        for sid, values in series.items() for t, v in enumerate(values, start=1)
    ]
    return pd.DataFrame(rows)[COLUMNS].astype({"t": "int64", "y": "float64"})


def synthetic_frozen_copy(root: Path) -> tuple[Path, Path]:
    """Write the synthetic frozen copy under ``root`` (the data directory); return (root, manifest_path)."""
    _, manifest_path = write_frozen(root, {key: canonical(key, i) for i, key in enumerate(SOURCES)})
    return root, manifest_path
