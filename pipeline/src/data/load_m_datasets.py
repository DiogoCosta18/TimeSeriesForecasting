from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _periods(freq: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range("1990-01-01", periods=n, freq="MS" if freq == "monthly" else "QS")


def _synthetic_source(source: str, frequency: str, n_series: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    season_length = 12 if frequency == "monthly" else 4
    n_obs = 96 if frequency == "monthly" else 56
    frames = []
    for i in range(n_series):
        uid = f"{source}_{frequency.title()}_S{i:04d}"
        ds = _periods(frequency, n_obs)
        t = np.arange(n_obs)
        amp = rng.uniform(2, 12)
        trend = rng.normal(0.04, 0.03) * t
        seasonal = amp * np.sin(2 * np.pi * t / season_length + rng.uniform(0, np.pi))
        nonlin = 0.03 * np.maximum(t - n_obs * rng.uniform(0.35, 0.75), 0) ** 1.25
        noise_scale = rng.uniform(0.5, 2.0) * (1 + 0.6 * np.sin(2 * np.pi * t / max(8, season_length * 3)) ** 2)
        y = 50 + trend + seasonal + nonlin + rng.normal(0, noise_scale)
        if i % 9 == 0:
            y[int(n_obs * 0.6) :] += rng.normal(8, 3)
        frames.append(pd.DataFrame({"unique_id": uid, "ds": ds, "y": y, "source_dataset": source}))
    return pd.concat(frames, ignore_index=True)


def _load_m4_local(root: Path, group: str, frequency: str) -> pd.DataFrame | None:
    path = root.parent / "m4" / "datasets" / f"{group}-train.csv"
    if not path.exists():
        return None
    raw = pd.read_csv(path)
    id_col = raw.columns[0]
    value_cols = [c for c in raw.columns if c != id_col]
    records = []
    for _, row in raw.iterrows():
        vals = pd.to_numeric(row[value_cols], errors="coerce").dropna().to_numpy(dtype=float)
        ds = _periods(frequency, len(vals))
        uid = f"M4_{group}_{row[id_col]}"
        records.append(pd.DataFrame({"unique_id": uid, "ds": ds, "y": vals, "source_dataset": f"M4_{group}"}))
    return pd.concat(records, ignore_index=True) if records else None


def _load_with_datasetsforecast(group: str, source: str, root: Path) -> pd.DataFrame | None:
    try:
        if source == "M3":
            from datasetsforecast.m3 import M3

            y_df, *_ = M3.load(directory=str(root / "datasetsforecast"), group=group)
        else:
            from datasetsforecast.m4 import M4

            y_df, *_ = M4.load(directory=str(root / "datasetsforecast"), group=group)
        y_df = y_df.rename(columns={"unique_id": "unique_id", "ds": "ds", "y": "y"})
        y_df["unique_id"] = f"{source}_{group}_" + y_df["unique_id"].astype(str)
        y_df["source_dataset"] = f"{source}_{group}"
        return y_df[["unique_id", "ds", "y", "source_dataset"]]
    except Exception:
        return None


def load_dataset_pair(frequency_cfg: dict, project_root: Path, seed: int, smoke_min: int = 40) -> pd.DataFrame:
    frequency = frequency_cfg["frequency"]
    group_m3 = frequency_cfg["m3_group"]
    group_m4 = frequency_cfg["m4_group"]
    parts: list[pd.DataFrame] = []
    force_synthetic = bool(frequency_cfg.get("force_synthetic", False))

    m3 = None if force_synthetic else _load_with_datasetsforecast(group_m3, "M3", project_root)
    if m3 is None:
        m3 = _synthetic_source(f"M3_{group_m3}", frequency, max(smoke_min, frequency_cfg.get("n_m3_requested", 750)), seed + 3)
    parts.append(m3)

    m4 = None if force_synthetic else _load_with_datasetsforecast(group_m4, "M4", project_root)
    if m4 is None:
        m4 = None if force_synthetic else _load_m4_local(project_root, group_m4, frequency)
    if m4 is None:
        m4 = _synthetic_source(f"M4_{group_m4}", frequency, max(smoke_min, frequency_cfg.get("n_m4_requested", 750)), seed + 7)
    parts.append(m4)

    out = pd.concat(parts, ignore_index=True)
    out["ds"] = pd.to_datetime(out["ds"])
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna(subset=["unique_id", "ds", "y"]).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return out
