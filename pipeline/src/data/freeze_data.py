"""Create the frozen M3/M4 copy once, and verify it (rerun protocol D2, U18, G13).

    python -m src.data.freeze_data freeze --out DIR --code-commit SHA --m3c PATH/M3C.xls
    python -m src.data.freeze_data verify --data-dir DIR [--manifest PATH]

``freeze`` reads or downloads the four sources, converts them to the canonical
layout of ``src.data.frozen``, validates them, writes one parquet file per source plus
``MANIFEST.json`` (checksums of every raw file and every frozen file), and finally
re-reads everything through the verified loader. It refuses to write into a
non-empty directory, so a frozen copy is never modified in place.

Sources (protocol D2, amended in v1.4):
- M3: the original competition file ``M3C.xls`` of the International Institute of
  Forecasters (M3 competition page). The site refuses scripted downloads, so the file
  is supplied with ``--m3c``; it is refused unless its SHA-256 equals the pinned value,
  and it is copied into the frozen copy's raw folder. Two independent copies are
  compared with it value by value, and any difference other than those documented
  in ``KNOWN_DIFFERENCES`` (or a documented one that is absent) stops the freeze:
  the Monash Time Series Forecasting Archive files on Zenodo (the files the earlier
  runs used, read here at full precision; datasetsforecast's own reader stores values
  as float32 and is not used), which must be identical bit for bit; and Mcomp 2.7
  (CRAN archive, R. J. Hyndman), which differs in one documented value.
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
- M3: every row holds exactly N values followed by empty cells, NF = h, ids are
  unique and in official-id order (so the positional ids M1.., Q1.. of the earlier
  runs map one-to-one onto official ids, as Mcomp's field ``st`` confirms);
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
PACKAGES = ["datasetsforecast", "pandas", "numpy", "pyarrow", "rdata", "utilsforecast", "xlrd"]

M3C_PAGE = "https://forecasters.org/resources/time-series-data/m3-competition/"
M3C_SHA256 = "23bfbba215dad49eb9b0f315a3e38bcc541f78f4689b168616ef25a5e8f293a6"
M3C_SHEET = {"Monthly": ("M3Month", "Starting Month", 12), "Quarterly": ("M3Quart", "Starting Quarter", 4)}
MCOMP_URL = "https://cran.r-project.org/src/contrib/Archive/Mcomp/Mcomp_2.7.tar.gz"
MCOMP_SHA256 = "2177b210a612a47a1e9885e2aea6d3f7f9154090fd3d8e629a5909cfc04768af"
MCOMP_MEMBER = "Mcomp/data/M3.rda"
M3_PERIOD = {"Monthly": ("MONTHLY", 12), "Quarterly": ("QUARTERLY", 4)}
# The only values in which a copy of M3 may differ from the original M3C.xls, keyed by
# (group, M3 id, position t) -> (M3C.xls value, copy's value). Monash reproduces the
# original exactly. Mcomp has +1200 where the original has -1200 (N2786, t=84, a test
# value); its change log does not mention it.
KNOWN_DIFFERENCES = {
    "monash": {},
    "mcomp": {("Monthly", "N2786", 84): (-1200.0, 1200.0)},
}


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


def _copy_pinned(src: Path, dest: Path, expected_sha256: str) -> None:
    """Copy a supplied file into the raw folder; refuse it unless its SHA-256 is the pinned one."""
    src = Path(src)
    if not src.is_file():
        raise FrozenDataError(f"{src} does not exist")
    if sha256_file(src) != expected_sha256:
        raise FrozenDataError(f"{src.name}: sha256 {sha256_file(src)} != pinned {expected_sha256}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    if sha256_file(dest) != expected_sha256:
        raise FrozenDataError(f"{dest}: copy does not match the pinned checksum")


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


def _m3_id(raw) -> str:
    """'N 646' / 'N1402' (M3C.xls) -> 'N0646' / 'N1402'."""
    match = re.fullmatch(r"N\s*(\d{1,4})", str(raw).strip())
    if match is None:
        raise FrozenDataError(f"unexpected M3 series name {raw!r}")
    return f"N{int(match.group(1)):04d}"


def read_m3c_sheet(path: Path, group: str) -> pd.DataFrame:
    """The sheet of one group in M3C.xls, as read by pandas (xlrd), header row included."""
    return pd.read_excel(Path(path), sheet_name=M3C_SHEET[group][0], engine="xlrd")


def m3_frame_from_m3c(sheet: pd.DataFrame, group: str, horizon: int) -> tuple[pd.DataFrame, dict]:
    """Long frame (unique_id = official id, ds = 1..N, y, ds_source) of one M3C.xls sheet, plus notes.

    Each row is: Series, N (values including the test period), NF (horizon),
    Category, Starting Year, Starting Month/Quarter, then the N values and empty cells.
    Rows must be in official-id order, so the k-th row is the series the earlier
    runs called M{k} / Q{k}. Start labels are recorded as given; a start period
    beyond the year (N1071: quarter 9) is carried into the following years, and
    series without a start date (year and period 0) keep their positions as labels.
    """
    _, period_col, freq = M3C_SHEET[group]
    head = ["Series", "N", "NF", "Category", "Starting Year", period_col]
    if list(sheet.columns[:6]) != head or list(sheet.columns[6:]) != list(range(1, sheet.shape[1] - 5)):
        raise FrozenDataError(f"M3C.xls {group}: unexpected columns {list(sheet.columns[:8])}...")
    values_all = sheet.iloc[:, 6:].to_numpy(dtype=float)
    parts, ids = [], []
    rolled_over, no_start = [], []
    for row, values_row in zip(sheet.itertuples(index=False), values_all):
        sid = _m3_id(row[0])
        n, nf, year, start = int(row[1]), int(row[2]), int(row[4]), int(row[5])
        if nf != horizon:
            raise FrozenDataError(f"M3C.xls {group} {sid}: NF={nf}, expected {horizon}")
        present = np.isfinite(values_row)
        if n < 1 or not present[:n].all() or present[n:].any():
            raise FrozenDataError(f"M3C.xls {group} {sid}: N={n} but {int(present.sum())} values, or not a gap-free prefix")
        if year == 0 and start == 0:
            labels = [str(t) for t in range(1, n + 1)]
            no_start.append(sid)
        else:
            if not 1 <= start <= freq:
                rolled_over.append(sid)
            base = year * freq + start - 1
            labels = [
                f"{(base + k) // freq}-{(base + k) % freq + 1:02d}" if freq == 12 else f"{(base + k) // freq}-Q{(base + k) % freq + 1}"
                for k in range(n)
            ]
        ids.append(sid)
        parts.append(pd.DataFrame({
            "unique_id": sid,
            "ds": np.arange(1, n + 1, dtype="int64"),
            "y": values_row[:n],
            "ds_source": labels,
        }))
    if len(set(ids)) != len(ids):
        raise FrozenDataError(f"M3C.xls {group}: duplicated series names")
    if ids != sorted(ids, key=lambda s: int(s[1:])):
        raise FrozenDataError(f"M3C.xls {group}: rows are not in official-id order")
    if not parts:
        raise FrozenDataError(f"M3C.xls {group}: no series")
    notes = {"start_period_carried_over": rolled_over, "no_start_date_labels_are_positions": no_start}
    return pd.concat(parts, ignore_index=True), notes


def compare_with_copy(frame: pd.DataFrame, copy: dict[str, np.ndarray], group: str, known: dict) -> dict:
    """Compare every M3 series with another copy, value by value (exact equality).

    ``known`` maps (group, id, t) -> (our value, the copy's value). An undocumented
    difference, a documented one that is absent, or any mismatch in ids or lengths
    raises FrozenDataError.
    """
    ours = {uid: g["y"].to_numpy() for uid, g in frame.groupby("unique_id", sort=False)}
    if set(ours) != set(copy):
        raise FrozenDataError(f"M3_{group}: ids differ from the copy ({len(set(ours) ^ set(copy))} not shared)")
    found, n_values = {}, 0
    for sid in sorted(ours, key=lambda s: int(s[1:])):
        a, b = ours[sid], np.asarray(copy[sid], dtype=float)
        if len(a) != len(b):
            raise FrozenDataError(f"M3_{group} {sid}: {len(a)} values, copy {len(b)}")
        n_values += len(a)
        for i in np.flatnonzero(a != b):
            key = (group, sid, int(i) + 1)
            if known.get(key) != (float(a[i]), float(b[i])):
                raise FrozenDataError(f"M3_{group} {sid} t={i + 1}: M3C.xls {a[i]!r}, copy {b[i]!r} (undocumented)")
            found[key] = known[key]
    expected = {k for k in known if k[0] == group}
    if set(found) != expected:
        raise FrozenDataError(f"M3_{group}: documented differences not found: {sorted(expected - set(found))}")
    return {
        "series_compared": len(ours),
        "values_compared": n_values,
        "values_identical": n_values - len(found),
        "differences": [{"series": sid, "t": t, "m3c": v[0], "copy": v[1]} for (_, sid, t), v in sorted(found.items())],
    }


def monash_by_id(names: list[str], series: list[np.ndarray], ids_in_order: list[str]) -> dict[str, np.ndarray]:
    """Monash names series T1..Tn in file order, which is the official-id order."""
    if names != [f"T{k}" for k in range(1, len(names) + 1)] or len(names) != len(ids_in_order):
        raise FrozenDataError(f"Monash file has {len(names)} series named {names[:2]}..., expected {len(ids_in_order)}")
    return dict(zip(ids_in_order, series))


def _load_m3(raw_dir: Path, group: str, m3c_path: Path) -> tuple[pd.DataFrame, dict]:
    from datasetsforecast.m3 import M3, M3Info

    m3_dir = Path(raw_dir) / "m3"
    m3c = m3_dir / "M3C.xls"
    if not m3c.exists():
        _copy_pinned(m3c_path, m3c, M3C_SHA256)
    frame, notes = m3_frame_from_m3c(read_m3c_sheet(m3c, group), group, HORIZON[group])
    ids = sorted(frame["unique_id"].unique(), key=lambda s: int(s[1:]))

    info = M3Info.get_group(group)
    M3.download(str(raw_dir), info)
    header, names, monash = read_monash_tsf(m3_dir / "datasets" / f"{info.file_name}.tsf")
    if header.get("missing") != "false" or header.get("horizon") != str(HORIZON[group]):
        raise FrozenDataError(f"M3_{group}: Monash header {header}")
    monash_check = compare_with_copy(frame, monash_by_id(names, monash, ids), group, KNOWN_DIFFERENCES["monash"])

    archive, rda = m3_dir / "Mcomp_2.7.tar.gz", m3_dir / "Mcomp_2.7_M3.rda"
    _fetch(MCOMP_URL, archive, MCOMP_SHA256)
    if not rda.exists():
        _extract_member(archive, MCOMP_MEMBER, rda)
    mcomp = m3_frame_from_mcomp(read_mcomp_m3(rda), group, HORIZON[group])
    mcomp_values = {uid: g["y"].to_numpy() for uid, g in mcomp.groupby("unique_id", sort=False)}
    mcomp_check = compare_with_copy(frame, mcomp_values, group, KNOWN_DIFFERENCES["mcomp"])
    return frame, {
        "source": "M3C.xls, International Institute of Forecasters (supplied file; the site refuses scripted downloads)",
        "source_page": M3C_PAGE,
        "sheet": M3C_SHEET[group][0],
        "sha256": M3C_SHA256,
        "series_ids": "official M3 ids, zero-padded (N 646 -> N0646); row k is the positional id "
                      f"{group[0]}k of datasetsforecast and the earlier runs (rows checked to be in id order)",
        "start_labels": notes,
        "cross_checks": {
            "monash": {"reference_url": info.source_url, "read_at": "full precision", **monash_check},
            "mcomp": {"reference_url": MCOMP_URL, "archive_sha256": MCOMP_SHA256, **mcomp_check},
        },
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


def freeze(out_dir: Path, code_commit: str, m3c_path: Path) -> dict:
    out_dir = Path(out_dir)
    if not re.fullmatch(r"[0-9a-f]{40}", code_commit):
        raise FrozenDataError("--code-commit must be a full 40-character git commit hash")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FrozenDataError(f"{out_dir} is not empty; a frozen copy is never overwritten")
    if not Path(m3c_path).is_file() or sha256_file(m3c_path) != M3C_SHA256:
        raise FrozenDataError(f"{m3c_path}: not the pinned M3C.xls (sha256 must be {M3C_SHA256})")
    raw_dir, frozen_dir = out_dir / "raw", out_dir / "frozen"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frozen_dir.mkdir(parents=True, exist_ok=True)

    frozen_files: dict[str, dict] = {}
    for source, group in SOURCES:
        key = f"{source}_{group}"
        if source == "M3":
            y_df, provenance = _load_m3(raw_dir, group, m3c_path)
        else:
            y_df, provenance = _load_m4(raw_dir, group)
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
    p_freeze.add_argument("--m3c", required=True, type=Path, help="the original M3C.xls (pinned SHA-256)")
    p_verify = sub.add_parser("verify", help="verify a frozen copy against a manifest")
    p_verify.add_argument("--data-dir", required=True, type=Path, help="directory holding the frozen/ folder")
    p_verify.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        if args.command == "freeze":
            freeze(args.out, args.code_commit, args.m3c)
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
