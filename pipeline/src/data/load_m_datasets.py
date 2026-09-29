"""Load the M3 and M4 series of one frequency from the verified frozen copy (protocol D2).

This is the experiment's only entry point to the data. It reads through
``src.data.frozen``, which checks every file against the committed manifest, and
raises ``FrozenDataError`` on any problem. There is no download, no local-CSV
path and no synthetic substitute: if the frozen copy is absent or altered, the
run stops.

The returned ``ds`` column is a regular calendar rebuilt from the position ``t``
(same origin and frequency for M3 and M4), so it carries no information beyond
the order of the observations. The source time stamps stay in the frozen files
for traceability only.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.data.frozen import DEFAULT_MANIFEST, FrozenDataError, load_frequency, load_manifest, sha256_file

DATA_DIR_ENV = "RERUN_DATA_DIR"
# The origin is arbitrary; 1900 keeps the longest M4 monthly series (~2,800 months)
# plus any forecast horizon well inside pandas' timestamp range (year 2262).
CALENDAR_ORIGIN = pd.Timestamp("1900-01-31")
CALENDAR_FREQ = {"monthly": "ME", "quarterly": "QE"}
OUTPUT_COLUMNS = ["unique_id", "ds", "y", "source_dataset", "t"]


def resolve_data_dir(data_dir: Path | str | None = None) -> Path:
    """The folder that holds ``frozen/``: the explicit argument, else ``$RERUN_DATA_DIR``."""
    if data_dir is None:
        data_dir = os.environ.get(DATA_DIR_ENV) or None
    if data_dir is None:
        raise FrozenDataError(
            f"no frozen data location: pass --data-dir or set {DATA_DIR_ENV} "
            "to the folder that contains frozen/ and MANIFEST.json"
        )
    path = Path(data_dir)
    if not (path / "frozen").is_dir():
        raise FrozenDataError(f"{path} has no frozen/ folder")
    return path


def load_dataset_pair(
    frequency_cfg: dict,
    data_dir: Path | str | None = None,
    manifest: dict | None = None,
) -> pd.DataFrame:
    """All M3 and M4 series of ``frequency_cfg['frequency']``, verified, sorted by (unique_id, t)."""
    frequency = frequency_cfg["frequency"]
    if frequency not in CALENDAR_FREQ:
        raise FrozenDataError(f"unsupported frequency {frequency!r}")
    manifest = load_manifest() if manifest is None else manifest
    expected = {f"M3_{frequency_cfg['m3_group']}", f"M4_{frequency_cfg['m4_group']}"}
    listed = {k for k, e in manifest["frozen_files"].items() if e["frequency"] == frequency}
    if listed != expected:
        raise FrozenDataError(f"{frequency}: configuration expects sources {sorted(expected)}, manifest lists {sorted(listed)}")

    df = load_frequency(resolve_data_dir(data_dir) / "frozen", frequency, manifest)
    calendar = pd.date_range(CALENDAR_ORIGIN, periods=int(df["t"].max()), freq=CALENDAR_FREQ[frequency])
    df["ds"] = calendar[df["t"].to_numpy() - 1]
    return df[OUTPUT_COLUMNS]


def frozen_data_provenance(manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    """What a run records about the data it read: manifest identity and per-file checksums."""
    manifest = load_manifest(manifest_path)
    return {
        "manifest_file": Path(manifest_path).name,
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_version": manifest.get("dataset_version"),
        "frozen_files": {
            key: {"sha256": e["sha256"], "content_sha256": e["content_sha256"], "n_series": e["n_series"], "rows": e["rows"]}
            for key, e in sorted(manifest["frozen_files"].items())
        },
    }
