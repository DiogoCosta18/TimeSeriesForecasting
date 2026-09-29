"""Frozen M3/M4 data: canonical layout, content hashing, validation and verified loading.

Rerun protocol D2 / test U18 / gate G13: the experiment reads the M3 and M4 series
only from a frozen copy whose files are identified by SHA-256 checksums recorded in
a committed manifest. Loading verifies the bytes of every file before reading it and
refuses any mismatch; there is no fallback of any kind.

Canonical layout (one parquet file per source and frequency, e.g. ``M3_Monthly``):

    unique_id       str      "<source>_<Group>_<source id>", e.g. "M3_Monthly_M1"
    source_dataset  str      "M3_Monthly", "M3_Quarterly", "M4_Monthly", "M4_Quarterly"
    frequency       str      "monthly" | "quarterly"
    t               int64    1-based position of the observation within its series
    y               float64  observed value
    ds_source       str      time stamp as given by the source (ISO date for M3,
                             integer position for M4); kept for traceability only

The pipeline uses only the order of observations (``t``), never calendar dates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["unique_id", "source_dataset", "frequency", "t", "y", "ds_source"]
DTYPES = {
    "unique_id": "object",
    "source_dataset": "object",
    "frequency": "object",
    "t": "int64",
    "y": "float64",
    "ds_source": "object",
}
MANIFEST_FORMAT = "rerun-frozen-data-manifest/1"
CONTENT_HASH_TAG = b"rerun-m3m4-content-v1\n"
CONTENT_HASH_DEFINITION = (
    "sha256 over: the tag 'rerun-m3m4-content-v1\\n'; then, with rows sorted by "
    "(unique_id, t), the 8-byte little-endian length of the UTF-8 encoding of all "
    "unique_id values joined by '\\n', followed by that encoding; then all t values "
    "as little-endian int64; then all y values as little-endian IEEE-754 float64."
)
DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "data_manifest" / "m3m4_v1.json"


class FrozenDataError(RuntimeError):
    """The frozen data is missing, altered, or does not satisfy its specification."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha256(df: pd.DataFrame) -> str:
    """Format-independent hash of the series content (see CONTENT_HASH_DEFINITION)."""
    ordered = df.sort_values(["unique_id", "t"], kind="mergesort")
    digest = hashlib.sha256()
    digest.update(CONTENT_HASH_TAG)
    ids = "\n".join(ordered["unique_id"].astype(str).tolist()).encode("utf-8")
    digest.update(len(ids).to_bytes(8, "little"))
    digest.update(ids)
    digest.update(np.ascontiguousarray(ordered["t"].to_numpy(dtype="<i8")).tobytes())
    digest.update(np.ascontiguousarray(ordered["y"].to_numpy(dtype="<f8")).tobytes())
    return digest.hexdigest()


def validate_canonical(
    df: pd.DataFrame,
    *,
    source_dataset: str,
    frequency: str,
    expected_series: int,
    expected_min_length: int,
) -> dict:
    """Check a canonical frame against its specification; raise on any violation.

    Returns summary statistics that go into the manifest.
    """
    if list(df.columns) != COLUMNS:
        raise FrozenDataError(f"{source_dataset}: columns {list(df.columns)} != {COLUMNS}")
    for col, dtype in DTYPES.items():
        if str(df[col].dtype) != dtype:
            raise FrozenDataError(f"{source_dataset}: column {col} has dtype {df[col].dtype}, expected {dtype}")
    if df.empty:
        raise FrozenDataError(f"{source_dataset}: no rows")
    if set(df["source_dataset"].unique()) != {source_dataset}:
        raise FrozenDataError(f"{source_dataset}: unexpected source_dataset values {sorted(df['source_dataset'].unique())}")
    if set(df["frequency"].unique()) != {frequency}:
        raise FrozenDataError(f"{source_dataset}: unexpected frequency values {sorted(df['frequency'].unique())}")
    if not df["unique_id"].astype(str).str.startswith(source_dataset + "_").all():
        raise FrozenDataError(f"{source_dataset}: unique_id values must start with '{source_dataset}_'")
    if df.duplicated(["unique_id", "t"]).any():
        raise FrozenDataError(f"{source_dataset}: duplicated (unique_id, t) pairs")
    if not np.isfinite(df["y"].to_numpy()).all():
        raise FrozenDataError(f"{source_dataset}: non-finite values in y")
    by_series = df.groupby("unique_id", sort=False)["t"]
    t_min, t_max, n_obs = by_series.min(), by_series.max(), by_series.count()
    broken = t_min.ne(1) | t_max.ne(n_obs)
    if broken.any():
        raise FrozenDataError(
            f"{source_dataset}: {int(broken.sum())} series whose positions t are not exactly 1..n "
            f"(e.g. {broken[broken].index[:3].tolist()})"
        )
    n_series = int(n_obs.size)
    if n_series != expected_series:
        raise FrozenDataError(f"{source_dataset}: {n_series} series, expected {expected_series}")
    if int(n_obs.min()) != expected_min_length:
        raise FrozenDataError(
            f"{source_dataset}: shortest series has {int(n_obs.min())} observations, "
            f"expected exactly {expected_min_length} (published minimum training length + horizon)"
        )
    return {
        "rows": int(len(df)),
        "n_series": n_series,
        "length_min": int(n_obs.min()),
        "length_median": float(n_obs.median()),
        "length_max": int(n_obs.max()),
        "y_min": float(df["y"].min()),
        "y_max": float(df["y"].max()),
    }


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FrozenDataError(f"manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != MANIFEST_FORMAT:
        raise FrozenDataError(f"manifest {path} has format {manifest.get('format')!r}, expected {MANIFEST_FORMAT!r}")
    if not manifest.get("frozen_files"):
        raise FrozenDataError(f"manifest {path} lists no frozen files")
    return manifest


def _read_verified(frozen_dir: Path, manifest: dict, key: str) -> tuple[pd.DataFrame, dict]:
    entries = manifest["frozen_files"]
    if key not in entries:
        raise FrozenDataError(f"{key} is not listed in the manifest (listed: {sorted(entries)})")
    entry = entries[key]
    path = Path(frozen_dir) / entry["file"]
    if not path.is_file():
        raise FrozenDataError(f"{key}: frozen file missing: {path}")
    actual = sha256_file(path)
    if actual != entry["sha256"]:
        raise FrozenDataError(f"{key}: checksum mismatch for {path}: {actual} != manifest {entry['sha256']}")
    df = pd.read_parquet(path)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing or len(df.columns) != len(COLUMNS):
        raise FrozenDataError(f"{key}: columns {list(df.columns)} do not match the canonical layout {COLUMNS}")
    df = df[COLUMNS].astype({"unique_id": "object", "source_dataset": "object", "frequency": "object", "ds_source": "object"})
    if len(df) != entry["rows"]:
        raise FrozenDataError(f"{key}: {len(df)} rows, manifest says {entry['rows']}")
    return df, entry


def load_source(frozen_dir: Path, key: str, manifest: dict | None = None) -> pd.DataFrame:
    """Load one frozen source (e.g. 'M4_Quarterly') after verifying its checksum and specification."""
    manifest = load_manifest() if manifest is None else manifest
    df, entry = _read_verified(frozen_dir, manifest, key)
    validate_canonical(
        df,
        source_dataset=key,
        frequency=entry["frequency"],
        expected_series=entry["n_series"],
        expected_min_length=entry["length_min"],
    )
    return df


def load_frequency(frozen_dir: Path, frequency: str, manifest: dict | None = None) -> pd.DataFrame:
    """Load every frozen source of one frequency (M3 and M4), verified, in canonical order."""
    manifest = load_manifest() if manifest is None else manifest
    keys = sorted(k for k, e in manifest["frozen_files"].items() if e["frequency"] == frequency)
    if not keys:
        raise FrozenDataError(f"no frozen sources for frequency {frequency!r}")
    frames = [load_source(frozen_dir, key, manifest) for key in keys]
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["unique_id", "t"], kind="mergesort").reset_index(drop=True)


def verify_all(frozen_dir: Path, manifest: dict | None = None) -> dict:
    """Verify every frozen file: bytes (sha256), specification, and content hash."""
    manifest = load_manifest() if manifest is None else manifest
    report = {}
    for key, entry in sorted(manifest["frozen_files"].items()):
        df = load_source(frozen_dir, key, manifest)
        content = content_sha256(df)
        if content != entry["content_sha256"]:
            raise FrozenDataError(f"{key}: content hash {content} != manifest {entry['content_sha256']}")
        report[key] = {"rows": len(df), "n_series": int(df["unique_id"].nunique()), "sha256": entry["sha256"]}
    return report
