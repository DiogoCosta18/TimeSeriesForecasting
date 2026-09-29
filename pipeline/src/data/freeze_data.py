"""Create the frozen M3/M4 copy once, and verify it (rerun protocol D2, U18, G13).

    python -m src.data.freeze_data freeze --out DIR --code-commit SHA
    python -m src.data.freeze_data verify --data-dir DIR [--manifest PATH]

``freeze`` downloads the four sources through datasetsforecast 1.0.1 (the loader the
earlier runs used), converts them to the canonical layout of ``src.data.frozen``,
validates them, writes one parquet file per source plus ``MANIFEST.json`` (checksums
of every raw download and every frozen file), and finally re-reads everything
through the verified loader. It refuses to write into a non-empty directory, so a
frozen copy is never modified in place.

What the sources are (verified in datasetsforecast 1.0.1, not assumed):
- M3: the Monash Time Series Forecasting Archive files on Zenodo (Godahewa et al.,
  2021), complete series including the competition's test period; series ids are
  positional ("M1", "M2", ... in file order).
- M4: the official competition files (Mcompetitions/M4-methods, ``Dataset/Train``
  and ``Dataset/Test``); the loader appends each test row to its training row, so
  series are complete, including the competition's test period.
Both sources are therefore complete series, treated identically.

Validation that stops the freeze on any failure:
- exact series counts (M3: 1,428 monthly, 756 quarterly; M4: 48,000 and 24,000);
- positions t are exactly 1..n for every series, no duplicates, all values finite;
- the shortest series equals the published minimum training length plus the
  horizon (M3: 48+18 monthly, 16+8 quarterly, Makridakis & Hibon 2000; M4: 42+18
  monthly, 16+8 quarterly, M4 Competitor's Guide), which confirms complete series;
- M3: the Monash file declares no missing values and the expected horizon;
- M4: every training row is a gap-free prefix, every test row has exactly h values,
  and every frozen series equals its training row followed by its test row, exactly.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.frozen import (
    COLUMNS,
    CONTENT_HASH_DEFINITION,
    MANIFEST_FORMAT,
    FrozenDataError,
    content_sha256,
    load_manifest,
    sha256_file,
    validate_canonical,
    verify_all,
)

DATASET_VERSION = "m3m4-v1"
HORIZON = {"Monthly": 18, "Quarterly": 8}
EXPECTED_SERIES = {
    ("M3", "Monthly"): 1428,
    ("M3", "Quarterly"): 756,
    ("M4", "Monthly"): 48000,
    ("M4", "Quarterly"): 24000,
}
# Published minimum training length of each group; complete series are h longer.
MIN_TRAIN_LENGTH = {
    ("M3", "Monthly"): 48,
    ("M3", "Quarterly"): 16,
    ("M4", "Monthly"): 42,
    ("M4", "Quarterly"): 16,
}
SOURCES = sorted(EXPECTED_SERIES)
PACKAGES = ["datasetsforecast", "pandas", "numpy", "pyarrow", "utilsforecast"]


def _canonicalize(y_df: pd.DataFrame, source: str, group: str) -> pd.DataFrame:
    df = y_df[["unique_id", "ds", "y"]].copy()
    if source == "M4":
        df["ds"] = pd.to_numeric(df["ds"], errors="raise").astype("int64")
        ds_text = df["ds"].astype(str)
    else:
        if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
            raise FrozenDataError(f"{source}_{group}: expected datetime ds, got {df['ds'].dtype}")
        ds_text = df["ds"].dt.strftime("%Y-%m-%d")
    df["ds_source"] = ds_text.astype("object")
    df["unique_id"] = (f"{source}_{group}_" + df["unique_id"].astype(str)).astype("object")
    df = df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)
    df["t"] = (df.groupby("unique_id", sort=False).cumcount() + 1).astype("int64")
    if source == "M4" and not df["t"].eq(df["ds"]).all():
        raise FrozenDataError(f"M4_{group}: integer positions from the source are not 1..n")
    df["source_dataset"] = f"{source}_{group}"
    df["frequency"] = group.lower()
    df["y"] = pd.to_numeric(df["y"], errors="raise").astype("float64")
    return df[COLUMNS]


def _load_m3(raw_dir: Path, group: str) -> tuple[pd.DataFrame, dict]:
    from datasetsforecast.m3 import M3, M3Info
    from datasetsforecast.utils import convert_tsf_to_dataframe

    y_df, *_ = M3.load(directory=str(raw_dir), group=group)
    info = M3Info.get_group(group)
    tsf = raw_dir / "m3" / "datasets" / f"{info.file_name}.tsf"
    _, frequency, horizon, has_missing, equal_length = convert_tsf_to_dataframe(str(tsf))
    if has_missing:
        raise FrozenDataError(f"M3_{group}: the Monash file declares missing values")
    if horizon is not None and int(horizon) != HORIZON[group]:
        raise FrozenDataError(f"M3_{group}: file horizon {horizon} != {HORIZON[group]}")
    return y_df, {
        "source_url": info.source_url,
        "declared_frequency": frequency,
        "declared_horizon": horizon,
        "declared_missing_values": bool(has_missing),
        "declared_equal_length": bool(equal_length),
    }


def _load_m4(raw_dir: Path, group: str) -> tuple[pd.DataFrame, dict]:
    from datasetsforecast.m4 import M4

    y_df, *_ = M4.load(directory=str(raw_dir), group=group, cache=False)
    return y_df, {"source_urls": M4._download_urls(group)}


def check_m4_against_raw(canonical: pd.DataFrame, raw_dir: Path, group: str, horizon: int) -> None:
    """Every frozen M4 series must equal its official training row followed by its test row."""
    base = raw_dir / "m4" / "datasets"
    test = pd.read_csv(base / f"{group}-test.csv")
    test_vals = test.iloc[:, 1:].to_numpy(dtype=float)
    if test_vals.shape[1] != horizon or not np.isfinite(test_vals).all():
        raise FrozenDataError(f"M4_{group}: test file does not hold exactly {horizon} finite values per series")
    test_by_id = dict(zip(test.iloc[:, 0].astype(str), test_vals))
    frozen = {
        uid.split("_", 2)[2]: g.to_numpy()
        for uid, g in canonical.groupby("unique_id", sort=False)["y"]
    }
    seen = 0
    for chunk in pd.read_csv(base / f"{group}-train.csv", chunksize=2000):
        ids = chunk.iloc[:, 0].astype(str).to_numpy()
        vals = chunk.iloc[:, 1:].to_numpy(dtype=float)
        present = np.isfinite(vals)
        lengths = present.sum(axis=1)
        for i, sid in enumerate(ids):
            k = int(lengths[i])
            if not (present[i, :k].all() and not present[i, k:].any()):
                raise FrozenDataError(f"M4_{group} {sid}: training row has a gap (values are not a prefix)")
            if sid not in test_by_id or sid not in frozen:
                raise FrozenDataError(f"M4_{group} {sid}: missing from the test file or from the frozen data")
            expected = np.concatenate([vals[i, :k], test_by_id[sid]])
            if not np.array_equal(expected, frozen[sid]):
                raise FrozenDataError(f"M4_{group} {sid}: frozen series differs from training row + test row")
        seen += len(ids)
    if seen != len(frozen) or seen != len(test_by_id):
        raise FrozenDataError(f"M4_{group}: {seen} training rows, {len(test_by_id)} test rows, {len(frozen)} frozen series")


def _raw_file_records(raw_dir: Path) -> list[dict]:
    records = []
    for path in sorted(p for p in raw_dir.rglob("*") if p.is_file()):
        records.append({
            "path": path.relative_to(raw_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return records


def freeze(out_dir: Path, code_commit: str) -> dict:
    out_dir = Path(out_dir)
    if not re.fullmatch(r"[0-9a-f]{40}", code_commit):
        raise FrozenDataError("--code-commit must be a full 40-character git commit hash")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FrozenDataError(f"{out_dir} is not empty; a frozen copy is never overwritten")
    raw_dir, frozen_dir = out_dir / "raw", out_dir / "frozen"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frozen_dir.mkdir(parents=True, exist_ok=True)

    frozen_files: dict[str, dict] = {}
    for source, group in SOURCES:
        key = f"{source}_{group}"
        y_df, provenance = (_load_m3 if source == "M3" else _load_m4)(raw_dir, group)
        canonical = _canonicalize(y_df, source, group)
        stats = validate_canonical(
            canonical,
            source_dataset=key,
            frequency=group.lower(),
            expected_series=EXPECTED_SERIES[(source, group)],
            expected_min_length=MIN_TRAIN_LENGTH[(source, group)] + HORIZON[group],
        )
        if source == "M4":
            check_m4_against_raw(canonical, raw_dir, group, HORIZON[group])
        path = frozen_dir / f"{key}.parquet"
        canonical.to_parquet(path, engine="pyarrow", index=False, compression="zstd")
        frozen_files[key] = {
            "file": path.name,
            "frequency": group.lower(),
            "horizon": HORIZON[group],
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "content_sha256": content_sha256(canonical),
            **stats,
            "provenance": provenance,
        }
        print(f"{key}: {stats['n_series']} series, {stats['rows']} observations, "
              f"length {stats['length_min']}..{stats['length_max']}", flush=True)

    manifest = {
        "format": MANIFEST_FORMAT,
        "dataset_version": DATASET_VERSION,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "created_by": {
            "code_commit": code_commit,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {p: metadata.version(p) for p in PACKAGES},
        },
        "content_hash_definition": CONTENT_HASH_DEFINITION,
        "raw_files": _raw_file_records(raw_dir),
        "frozen_files": frozen_files,
    }
    manifest_path = out_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Self-check: read everything back through the verified loader.
    verify_all(frozen_dir, load_manifest(manifest_path))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.data.freeze_data")
    sub = parser.add_subparsers(dest="command", required=True)
    p_freeze = sub.add_parser("freeze", help="download, validate and freeze M3/M4 once")
    p_freeze.add_argument("--out", required=True, type=Path)
    p_freeze.add_argument("--code-commit", required=True)
    p_verify = sub.add_parser("verify", help="verify a frozen copy against a manifest")
    p_verify.add_argument("--data-dir", required=True, type=Path, help="directory holding the frozen/ folder")
    p_verify.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        if args.command == "freeze":
            freeze(args.out, args.code_commit)
            print(f"Frozen copy written to {args.out}; verified.")
        else:
            manifest = load_manifest(args.manifest) if args.manifest else load_manifest()
            report = verify_all(args.data_dir / "frozen", manifest)
            for key, entry in report.items():
                print(f"{key}: OK ({entry['n_series']} series, {entry['rows']} rows, sha256 {entry['sha256'][:16]})")
            print("All frozen files verified.")
    except FrozenDataError as exc:
        print(f"FROZEN DATA ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
