"""Create the frozen M3/M4 copy once, and verify it (rerun protocol D2, U18, G13).

    python -m src.data.freeze_data freeze --out DIR --code-commit SHA
    python -m src.data.freeze_data verify --data-dir DIR [--manifest PATH]

``freeze`` downloads the four sources, converts them to the canonical layout of
``src.data.frozen``, validates them, writes one parquet file per source plus
``MANIFEST.json`` (checksums of every raw download and every frozen file), and
finally re-reads everything through the verified loader. It refuses to write into a
non-empty directory, so a frozen copy is never modified in place.

Sources (protocol D2, amended in v1.3):
- M3: Mcomp 2.7 (CRAN archive, R. J. Hyndman), ``data/M3.rda``: the M3 competition
  data with training (x) and test (xx) values, official ids (N0646, ...) and start
  periods. The archive is refused unless its SHA-256 equals the pinned value.
  Independent cross-check: every series must equal, bit for bit, the Monash Time
  Series Forecasting Archive copy on Zenodo (the file the earlier runs used, read here
  at full precision), except for the documented differences in
  ``MONASH_M3_KNOWN_DIFFERENCES``; any other difference, or a documented one that is
  not found, stops the freeze. (datasetsforecast's own TSF reader stores values as
  float32 and is not used.)
- M4: the official competition files (Mcompetitions/M4-methods, ``Dataset/Train``
  and ``Dataset/Test``), fetched by datasetsforecast's ``M4.download``;
  ``read_m4_official`` appends each test row to its training row exactly as
  ``M4.load`` does.
Both sources are complete series, including the competition's test period, and are
stored identically: position t = 1..n and the source's time label as text.

Validation that stops the freeze on any failure:
- exact series counts (M3: 1,428 monthly, 756 quarterly; M4: 48,000 and 24,000);
- positions t are exactly 1..n for every series, no duplicates, all values finite;
- the shortest series equals the published minimum training length plus the
  horizon (M3: 48+18 monthly, 16+8 quarterly, Makridakis & Hibon 2000; M4: 42+18
  monthly, 16+8 quarterly, M4 Competitor's Guide), which confirms complete series;
- M3: n and h of every series match its x and xx; time labels are consecutive;
  the legacy positional ids (Mcomp field ``st``) follow the official-id order;
- M4: every training row is a gap-free prefix, every test row has exactly h finite
  values, and training and test ids match one-to-one without duplicates.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import re
import shutil
import sys
import tarfile
import urllib.request
import warnings
from datetime import datetime, timezone
from fractions import Fraction
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
PACKAGES = ["datasetsforecast", "pandas", "numpy", "pyarrow", "rdata", "utilsforecast"]

MCOMP_URL = "https://cran.r-project.org/src/contrib/Archive/Mcomp/Mcomp_2.7.tar.gz"
MCOMP_SHA256 = "2177b210a612a47a1e9885e2aea6d3f7f9154090fd3d8e629a5909cfc04768af"
MCOMP_MEMBER = "Mcomp/data/M3.rda"
M3_PERIOD = {"Monthly": ("MONTHLY", 12), "Quarterly": ("QUARTERLY", 4)}
# The only values in which the Monash copy of M3 differs from Mcomp, keyed by
# (group, M3 id, position t) -> (Mcomp value, Monash value). N2786 at t=84 (test
# period) carries a minus sign in Monash only; M3 series are positive (the
# competition was scored with sMAPE and MAPE) and +1200 continues the seasonal
# trough 3360, 1200, 1520.
MONASH_M3_KNOWN_DIFFERENCES = {("Monthly", "N2786", 84): (1200.0, -1200.0)}


def _shared_text(values: pd.Series, render) -> pd.Series:
    """``render`` applied once per distinct value; rows share the resulting str objects (memory)."""
    return values.map({v: render(v) for v in values.unique()}).astype("object")


def _canonicalize(y_df: pd.DataFrame, source: str, group: str) -> pd.DataFrame:
    """Loader output (unique_id, ds = position 1..n, y, optional ds_source) -> canonical layout."""
    df = y_df[["unique_id", "ds", "y"]].copy()
    df["ds"] = pd.to_numeric(df["ds"], errors="raise").astype("int64")
    if "ds_source" in y_df:
        df["ds_source"] = _shared_text(y_df["ds_source"].astype(str), str)
    else:
        df["ds_source"] = _shared_text(df["ds"], str)
    df["unique_id"] = _shared_text(df["unique_id"].astype(str), lambda u: f"{source}_{group}_{u}")
    df = df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)
    df["t"] = (df.groupby("unique_id", sort=False).cumcount() + 1).astype("int64")
    if not df["t"].eq(df["ds"]).all():
        raise FrozenDataError(f"{source}_{group}: positions from the source are not 1..n")
    df["source_dataset"] = f"{source}_{group}"
    df["frequency"] = group.lower()
    df["y"] = pd.to_numeric(df["y"], errors="raise").astype("float64")
    return df[COLUMNS]


def _fetch(url: str, dest: Path, expected_sha256: str | None = None) -> None:
    """Download ``url`` to ``dest`` unless present; refuse a checksum other than ``expected_sha256``."""
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        with urllib.request.urlopen(url, timeout=300) as response, open(part, "wb") as fh:
            shutil.copyfileobj(response, fh)
        part.replace(dest)
    if expected_sha256 is not None and sha256_file(dest) != expected_sha256:
        raise FrozenDataError(f"{dest.name}: sha256 {sha256_file(dest)} != pinned {expected_sha256}")


def _extract_member(archive: Path, member: str, dest: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        fh = tar.extractfile(member)
        if fh is None:
            raise FrozenDataError(f"{archive.name}: {member} is not a regular file")
        dest.write_bytes(fh.read())


def read_mcomp_m3(rda_path: Path) -> dict:
    """The ``M3`` object of Mcomp's ``M3.rda``: {M3 id: {sn, st, type, period, n, h, x, xx, ...}}."""
    import rdata

    with warnings.catch_warnings():
        # Mcomp's S3 classes have no Python counterpart; rdata returns the plain lists.
        warnings.filterwarnings("ignore", message=r'Missing constructor for R class "(Mcomp|Mdata)"')
        return rdata.conversion.convert(rdata.parser.parse_file(Path(rda_path)))["M3"]


def _one(value):
    arr = np.asarray(value).ravel()
    if arr.size != 1:
        raise FrozenDataError(f"expected a single value, got {arr.size}")
    return arr[0]


def _period_label(time: Fraction, freq: int) -> str:
    year = math.floor(time)
    step = (time - year) * freq
    if step.denominator != 1:
        raise FrozenDataError(f"time {time} is not on the {freq}-per-year grid")
    return f"{year}-{int(step) + 1:02d}" if freq == 12 else f"{year}-Q{int(step) + 1}"


def m3_frame_from_mcomp(m3: dict, group: str, horizon: int) -> pd.DataFrame:
    """Long frame (unique_id = official M3 id, ds = 1..n+h, y, ds_source) of one M3 group.

    Series are ordered by official id; the legacy positional id (field ``st``, e.g.
    "M1") must equal the group letter and the rank in that order, which makes the old
    ids ("M3_Monthly_M1") map one-to-one onto the official ones.
    """
    period, freq = M3_PERIOD[group]
    chosen = sorted(
        ((str(key), s) for key, s in m3.items() if str(_one(s["period"])).upper() == period),
        key=lambda kv: int(kv[0][1:]),
    )
    parts = []
    for rank, (key, s) in enumerate(chosen, start=1):
        sn, st = str(_one(s["sn"])), str(_one(s["st"]))
        if sn != key:
            raise FrozenDataError(f"M3_{group} {key}: field sn is {sn}")
        if st != f"{group[0]}{rank}":
            raise FrozenDataError(f"M3_{group} {key}: legacy id {st} is not {group[0]}{rank}")
        n, h = float(_one(s["n"])), float(_one(s["h"]))
        x, xx = s["x"], s["xx"]
        if h != horizon or len(xx) != horizon or n != len(x):
            raise FrozenDataError(f"M3_{group} {key}: n={n}, h={h}, but {len(x)} training and {len(xx)} test values")
        values = np.concatenate([x.to_numpy(dtype=float), xx.to_numpy(dtype=float)])
        if not np.isfinite(values).all():
            raise FrozenDataError(f"M3_{group} {key}: non-finite values")
        times = [Fraction(t) for t in list(x.index) + list(xx.index)]
        if any(b - a != Fraction(1, freq) for a, b in zip(times, times[1:])):
            raise FrozenDataError(f"M3_{group} {key}: time labels are not consecutive")
        parts.append(pd.DataFrame({
            "unique_id": key,
            "ds": np.arange(1, len(values) + 1, dtype="int64"),
            "y": values,
            "ds_source": [_period_label(t, freq) for t in times],
        }))
    if not parts:
        raise FrozenDataError(f"M3_{group}: no series with period {period}")
    return pd.concat(parts, ignore_index=True)


def read_monash_tsf(path: Path) -> tuple[dict, list[str], list[np.ndarray]]:
    """Header, series names and values (float64, full precision) of a Monash .tsf file.

    The files are Windows-1252 text (the M3 headers contain an en dash, byte 0x96), as
    in the Monash reference loader.
    """
    lines = Path(path).read_text(encoding="cp1252").splitlines()
    header: dict[str, str] = {}
    for i, line in enumerate(lines):
        text = line.strip()
        if text.lower() == "@data":
            break
        if text.startswith("@"):
            key, _, value = text[1:].partition(" ")
            header[key.lower()] = value.strip()
    else:
        raise FrozenDataError(f"{Path(path).name}: no @data section")
    names, series = [], []
    for line in lines[i + 1:]:
        if not line.strip():
            continue
        name, _, rest = line.partition(":")
        _, _, values = rest.partition(":")
        parts = values.split(",")
        if "?" in parts:
            raise FrozenDataError(f"{Path(path).name} {name}: missing values")
        names.append(name)
        series.append(np.array([float(v) for v in parts], dtype="float64"))
    return header, names, series


def cross_check_m3(frame: pd.DataFrame, names: list[str], monash: list[np.ndarray], group: str,
                   known: dict = MONASH_M3_KNOWN_DIFFERENCES) -> dict:
    """Compare every M3 series with the Monash copy (series k of Monash = legacy id k)."""
    by_id = {uid: g["y"].to_numpy() for uid, g in frame.groupby("unique_id", sort=False)}
    order = sorted(by_id, key=lambda u: int(u[1:]))
    if names != [f"T{k}" for k in range(1, len(names) + 1)] or len(names) != len(order):
        raise FrozenDataError(f"M3_{group}: Monash has {len(names)} series named {names[:2]}..., Mcomp {len(order)}")
    found, n_values = {}, 0
    for sn, other in zip(order, monash):
        ours = by_id[sn]
        if len(ours) != len(other):
            raise FrozenDataError(f"M3_{group} {sn}: {len(ours)} values, Monash {len(other)}")
        n_values += len(ours)
        for i in np.flatnonzero(ours != other):
            key = (group, sn, int(i) + 1)
            if known.get(key) != (float(ours[i]), float(other[i])):
                raise FrozenDataError(f"M3_{group} {sn} t={i + 1}: Mcomp {ours[i]!r}, Monash {other[i]!r} (undocumented)")
            found[key] = known[key]
    expected = {k for k in known if k[0] == group}
    if set(found) != expected:
        raise FrozenDataError(f"M3_{group}: documented differences not found: {sorted(expected - set(found))}")
    return {
        "series_compared": len(order),
        "values_compared": n_values,
        "values_identical": n_values - len(found),
        "differences": [
            {"series": sn, "t": t, "mcomp": v[0], "monash": v[1]} for (_, sn, t), v in sorted(found.items())
        ],
    }


def _load_m3(raw_dir: Path, group: str) -> tuple[pd.DataFrame, dict]:
    from datasetsforecast.m3 import M3, M3Info

    m3_dir = Path(raw_dir) / "m3"
    archive, rda = m3_dir / "Mcomp_2.7.tar.gz", m3_dir / "Mcomp_2.7_M3.rda"
    _fetch(MCOMP_URL, archive, MCOMP_SHA256)
    if not rda.exists():
        _extract_member(archive, MCOMP_MEMBER, rda)
    frame = m3_frame_from_mcomp(read_mcomp_m3(rda), group, HORIZON[group])

    info = M3Info.get_group(group)
    M3.download(str(raw_dir), info)
    header, names, monash = read_monash_tsf(m3_dir / "datasets" / f"{info.file_name}.tsf")
    if header.get("missing") != "false" or header.get("horizon") != str(HORIZON[group]):
        raise FrozenDataError(f"M3_{group}: Monash header {header}")
    check = cross_check_m3(frame, names, monash, group)
    return frame, {
        "source": "Mcomp 2.7 (CRAN), data/M3.rda",
        "source_url": MCOMP_URL,
        "archive_sha256": MCOMP_SHA256,
        "member": MCOMP_MEMBER,
        "member_sha256": sha256_file(rda),
        "series_ids": "official M3 ids (field sn); field st = legacy positional id (M1.. / Q1..) "
                      "of datasetsforecast and the earlier runs, checked to follow official-id order",
        "cross_check": {"reference": "Monash TSF archive (read at full precision)", "reference_url": info.source_url, **check},
    }


def read_m4_official(raw_dir: Path, group: str, horizon: int) -> pd.DataFrame:
    """Read the official M4 ``{group}-train.csv`` and ``{group}-test.csv`` into long form.

    Returns the frame datasetsforecast's ``M4.load`` returns (columns unique_id, ds, y;
    ds = 1-based position, test values appended after the training values), built
    row by row instead of melting the whole wide table: ``M4.load`` needs more than
    8 GB for M4 Monthly. Both use pandas' default CSV parser, so values are
    bit-identical (tests/test_frozen_data.py runs ``M4.load`` itself as the reference).

    Raises on anything ``M4.load`` would pass over silently: a training row whose
    values are not a gap-free prefix (its ``dropna`` would close the gap), a test row
    without exactly ``horizon`` finite values, duplicated ids, or ids not matched
    one-to-one between the training and test files.
    """
    base = Path(raw_dir) / "m4" / "datasets"
    test = pd.read_csv(base / f"{group}-test.csv")
    test_ids = test.iloc[:, 0].astype(str).to_numpy()
    test_vals = test.iloc[:, 1:].to_numpy(dtype=float)
    if test_vals.shape[1] != horizon or not np.isfinite(test_vals).all():
        raise FrozenDataError(f"M4_{group}: test file does not hold exactly {horizon} finite values per series")
    test_by_id = dict(zip(test_ids, test_vals))
    if len(test_by_id) != len(test_ids):
        raise FrozenDataError(f"M4_{group}: duplicated ids in the test file")

    ids, lengths, values = [], [], []
    seen: set[str] = set()
    for chunk in pd.read_csv(base / f"{group}-train.csv", chunksize=2000):
        chunk_ids = chunk.iloc[:, 0].astype(str).to_numpy()
        vals = chunk.iloc[:, 1:].to_numpy(dtype=float)
        present = np.isfinite(vals)
        for i, sid in enumerate(chunk_ids):
            k = int(present[i].sum())
            if k == 0 or not present[i, :k].all():
                raise FrozenDataError(f"M4_{group} {sid}: training values are not a gap-free prefix")
            if sid in seen:
                raise FrozenDataError(f"M4_{group} {sid}: duplicated id in the training file")
            if sid not in test_by_id:
                raise FrozenDataError(f"M4_{group} {sid}: no row in the test file")
            seen.add(sid)
            ids.append(sid)
            lengths.append(k + horizon)
            values.append(np.concatenate([vals[i, :k], test_by_id[sid]]))
    if seen != set(test_by_id):
        raise FrozenDataError(f"M4_{group}: {len(set(test_by_id) - seen)} test rows without a training row")

    lengths_arr = np.asarray(lengths, dtype="int64")
    return pd.DataFrame({
        "unique_id": np.repeat(np.asarray(ids, dtype=object), lengths_arr),
        "ds": np.concatenate([np.arange(1, n + 1, dtype="int64") for n in lengths_arr]),
        "y": np.concatenate(values),
    })


def _load_m4(raw_dir: Path, group: str) -> tuple[pd.DataFrame, dict]:
    from datasetsforecast.m4 import M4

    M4.download(str(raw_dir), group)
    return read_m4_official(raw_dir, group, HORIZON[group]), {
        "source_urls": M4._download_urls(group),
        "parser": "src.data.freeze_data.read_m4_official",
    }


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
        del y_df
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
