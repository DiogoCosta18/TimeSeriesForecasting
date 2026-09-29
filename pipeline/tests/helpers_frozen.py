"""Builders for small synthetic frozen copies, shared by the data-layer tests.

Test-only: the pipeline itself never generates data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.frozen import COLUMNS, MANIFEST_FORMAT, content_sha256, sha256_file, validate_canonical


def make_canonical(source_dataset: str = "M3_Monthly", lengths=(5, 7, 6), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frequency = source_dataset.split("_")[1].lower()
    rows = []
    for i, n in enumerate(lengths, start=1):
        for t in range(1, n + 1):
            rows.append({
                "unique_id": f"{source_dataset}_S{i}",
                "source_dataset": source_dataset,
                "frequency": frequency,
                "t": t,
                "y": float(rng.uniform(10, 100)),
                "ds_source": str(t),
            })
    df = pd.DataFrame(rows)[COLUMNS]
    return df.astype({"t": "int64", "y": "float64"})


def spec(df: pd.DataFrame) -> dict:
    lengths = df.groupby("unique_id")["t"].count()
    return {
        "source_dataset": df["source_dataset"].iloc[0],
        "frequency": df["frequency"].iloc[0],
        "expected_series": int(lengths.size),
        "expected_min_length": int(lengths.min()),
    }


def write_frozen(tmp_path: Path, frames: dict[str, pd.DataFrame]) -> tuple[Path, Path]:
    """Write ``frames`` as a frozen copy under ``tmp_path``; return (frozen_dir, manifest_path)."""
    frozen_dir = tmp_path / "frozen"
    frozen_dir.mkdir()
    entries = {}
    for key, df in frames.items():
        path = frozen_dir / f"{key}.parquet"
        df.to_parquet(path, engine="pyarrow", index=False, compression="zstd")
        stats = validate_canonical(df, **spec(df))
        entries[key] = {
            "file": path.name,
            "frequency": df["frequency"].iloc[0],
            "horizon": 2,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "content_sha256": content_sha256(df),
            **stats,
        }
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps({"format": MANIFEST_FORMAT, "dataset_version": "test", "frozen_files": entries}),
        encoding="utf-8",
    )
    return frozen_dir, manifest_path
