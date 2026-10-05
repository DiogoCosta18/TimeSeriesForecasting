"""Regression test I2 (protocol Section 7.2; as amended in change log v1.9): the new code computes
exactly what the code of the 25 May strict run computed, for AutoARIMA and AutoETS Direct, on
every eligible row of that run.

    python scripts/regression_i2.py rows --may-metrics MAY/final/test_metrics.parquet \
        --legacy-data DATASETSFORECAST_FILES --prepare RUN/prepare --out DIR
    python scripts/regression_i2.py forecast --code old|new --src PIPELINE_ROOT --dir DIR [--jobs N]
    python scripts/regression_i2.py compare --dir DIR

``rows`` keeps the May rows whose series are eligible today (prepare bundle; M3 ids map
positionally to the official ids, M4 ids are the same) and stores the inputs exactly as the May
run read them (datasetsforecast 1.0.1, so M3 keeps its float32 rounding, defect m7); every
row's window is checked against rule D8 (n - (3 - w) h). ``forecast`` runs the old code
(``--src`` = a checkout of the baseline commit f1eb88b) or the new code on every row, in
chunks that a rerun skips, with the same numeric pins for both (set before numpy loads).
``compare`` passes I2 if and only if old and new forecasts are bit-identical on every row, and
reports how the new metrics compare with the values the May run stored: identical code and
inputs give different AutoARIMA models on different processors, so that comparison measures
the May machine, not the code.
"""
from __future__ import annotations

import os

PINS = {"OPENBLAS_CORETYPE": "Haswell", "NUMBA_CPU_NAME": "haswell", "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}  # as src/__init__.py, applied to the old code too
os.environ.update(PINS)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SPEC = {"Monthly": (12, 18), "Quarterly": (4, 8)}
MODELS = ["AutoARIMA", "AutoETS"]
COMPARED = {"rel_naive_unclipped": "relnaive", "mase": "mase", "smape": "smape", "mae": "mae"}
CHUNK = 2000
_forecast = None


def build_rows(args) -> int:
    from datasetsforecast.m3 import M3
    from datasetsforecast.m4 import M4

    may = pd.read_parquet(args.may_metrics, columns=["frequency", "decomposition_method", "model_name", "unique_id", "window",
                                                     "cutoff", *COMPARED])
    may = may[may["model_name"].isin(MODELS) & (may["decomposition_method"] == "without_stl")]
    if may.groupby(["model_name", "unique_id", "window"])[list(COMPARED)].nunique().max().max() != 1:
        raise SystemExit("May rows of one series and window differ across feature samples")
    may = may.drop_duplicates(["model_name", "unique_id", "window"]).reset_index(drop=True)
    elig = pd.read_parquet(args.prepare / "eligibility.parquet")
    eligible = set(elig.loc[elig["eligible"], "unique_id"])
    m3 = {g: sorted(elig.loc[elig["source_dataset"] == f"M3_{g}", "unique_id"], key=lambda s: int(s.rsplit("_N", 1)[1]))
          for g in SPEC}

    def frozen_id(legacy):
        source, group, uid = legacy.split("_", 2)
        return legacy if source == "M4" else m3[group][int(uid[1:]) - 1]

    may["frozen_id"] = may["unique_id"].map(frozen_id)
    rows = may[may["frozen_id"].isin(eligible)].reset_index(drop=True)

    inputs, dates = {}, {}
    for group in SPEC:
        for source, loader in (("M3", M3), ("M4", M4)):
            df = loader.load(directory=str(args.legacy_data), group=group)[0].sort_values(["unique_id", "ds"], kind="mergesort")
            for uid, g in df.groupby("unique_id", sort=False):
                key = f"{source}_{group}_{uid}"
                inputs[key] = g["y"].to_numpy(dtype=float)
                dates[key] = pd.DatetimeIndex(g["ds"]) if source == "M3" else None
    mismatched = 0
    for r in rows.itertuples():
        m, h = SPEC[r.unique_id.split("_")[1]]
        end = len(inputs[r.unique_id]) - (3 - int(r.window)) * h
        if dates[r.unique_id] is not None and pd.Timestamp(dates[r.unique_id][end - 1]) != pd.Timestamp(r.cutoff):
            mismatched += 1
    if mismatched:
        raise SystemExit(f"{mismatched} May rows have another window than rule D8")
    args.out.mkdir(parents=True, exist_ok=True)
    rows.drop(columns="cutoff").to_parquet(args.out / "rows.parquet", index=False)
    used = sorted(set(rows["unique_id"]))
    pd.DataFrame({"unique_id": used, "y": [inputs[u].tolist() for u in used]}).to_parquet(args.out / "inputs.parquet", index=False)
    print(json.dumps({"may_rows": len(may), "eligible_rows": len(rows), "ineligible_rows": len(may) - len(rows),
                      "series": len(used), "window_mismatches": 0}))
    return 0


def _init(code: str, src: str) -> None:
    global _forecast
    sys.path.insert(0, src)
    if code == "old":
        from src.models.statistical import forecast_statistical

        def forecast(model, y, h, m):
            result = forecast_statistical(model, y, h, m)
            if result.model_output_source != "trained_statsforecast":
                raise RuntimeError(f"old code did not train: {result.fallback_reason}")
            return np.asarray(result.yhat, dtype=float)
    else:
        from src.forecast.models import forecast_statistical

        def forecast(model, y, h, m):
            return forecast_statistical({"AutoARIMA": "ARIMA", "AutoETS": "ETS"}[model], y, h, m).yhat
    _forecast = forecast


def _work(item):
    model, y, window, m, h = item
    y = np.asarray(y, dtype=float)
    end = len(y) - (3 - window) * h
    return [float(v) for v in _forecast(model, y[:end], h, m)]


def _inputs(directory: Path) -> dict:
    df = pd.read_parquet(directory / "inputs.parquet")
    return dict(zip(df["unique_id"], df["y"]))


def run_forecasts(args) -> int:
    rows = pd.read_parquet(args.dir / "rows.parquet")
    inputs = _inputs(args.dir)
    items = [(r.model_name, inputs[r.unique_id], int(r.window), *SPEC[r.unique_id.split("_")[1]]) for r in rows.itertuples()]
    out = args.dir / f"forecasts_{args.code}"
    out.mkdir(exist_ok=True)
    with ProcessPoolExecutor(args.jobs, initializer=_init, initargs=(args.code, str(args.src))) as pool:
        for start in range(0, len(items), CHUNK):
            path = out / f"{start:06d}.parquet"
            if path.exists():
                continue
            yhat = list(pool.map(_work, items[start:start + CHUNK], chunksize=16))
            part = rows.iloc[start:start + len(yhat)][["model_name", "unique_id", "window"]].assign(yhat=yhat)
            part.to_parquet(path.with_suffix(".tmp"), index=False)
            path.with_suffix(".tmp").replace(path)
            print(f"{args.code}: {min(start + CHUNK, len(items))}/{len(items)}", flush=True)
    return 0


def _hash(values) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes()).hexdigest()


def compare(args) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.forecast.metrics import row_metrics
    from src.metrics.rel_naive import seasonal_naive_forecast
    from src.numeric_platform import platform_info

    rows = pd.read_parquet(args.dir / "rows.parquet")
    inputs = _inputs(args.dir)
    old = pd.concat([pd.read_parquet(p) for p in sorted((args.dir / "forecasts_old").glob("*.parquet"))], ignore_index=True)
    new = pd.concat([pd.read_parquet(p) for p in sorted((args.dir / "forecasts_new").glob("*.parquet"))], ignore_index=True)
    keys = ["model_name", "unique_id", "window"]
    if not (len(old) == len(new) == len(rows) and old[keys].equals(rows[keys]) and new[keys].equals(rows[keys])):
        raise SystemExit("forecasts incomplete or out of order; run forecast for both codes first")
    rows["identical"] = [_hash(a) == _hash(b) for a, b in zip(old["yhat"], new["yhat"])]
    metrics = []
    for r, yhat in zip(rows.itertuples(), new["yhat"]):
        m, h = SPEC[r.unique_id.split("_")[1]]
        y = np.asarray(inputs[r.unique_id], dtype=float)
        end = len(y) - (3 - int(r.window)) * h
        metrics.append(row_metrics(y[:end], y[end:end + h], np.asarray(yhat, dtype=float), seasonal_naive_forecast(y[:end], h, m), m))
    agree = np.ones(len(rows), dtype=bool)
    for col, metric in COMPARED.items():
        rows[f"new_{col}"] = [x[metric] for x in metrics]
        agree &= np.isclose(rows[f"new_{col}"], rows[col], rtol=1e-6, atol=0)
    rows["matches_may"] = agree
    report = {
        "test": "I2, code equivalence (change log v1.9)", "rows": int(len(rows)), "platform": platform_info(),
        "old_equals_new_bit_for_bit": int(rows["identical"].sum()), "passed": bool(rows["identical"].all()),
        "matches_may_values_rel_1e-6": {f"{k[0]}|{k[1]}": {"rows": int(len(g)), "matching": int(g["matches_may"].sum())}
                                        for k, g in rows.groupby(["model_name", "frequency"])},
    }
    rows.to_parquet(args.dir / "compared.parquet", index=False)
    (args.dir / "i2_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["passed"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="step", required=True)
    p = sub.add_parser("rows")
    p.add_argument("--may-metrics", required=True, type=Path)
    p.add_argument("--legacy-data", required=True, type=Path)
    p.add_argument("--prepare", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p = sub.add_parser("forecast")
    p.add_argument("--code", required=True, choices=["old", "new"])
    p.add_argument("--src", required=True, type=Path, help="pipeline folder holding src/ (a baseline checkout for old)")
    p.add_argument("--dir", required=True, type=Path)
    p.add_argument("--jobs", type=int, default=1)
    p = sub.add_parser("compare")
    p.add_argument("--dir", required=True, type=Path)
    args = ap.parse_args()
    return {"rows": build_rows, "forecast": run_forecasts, "compare": compare}[args.step](args)


if __name__ == "__main__":
    sys.exit(main())
