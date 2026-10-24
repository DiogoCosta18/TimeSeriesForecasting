"""A7 and A8: robustness of the metric and of the threshold, and breakdowns of Delta_STL.

- Q12 (A7, k = 32): Delta_STL as the log of the geometric mean of the relative MAE (rv.log_ratio_effect, the
  mean over series; threshold rv.LOG_DELTA, reported as GM = exp(-estimate)) and as a MASE difference
  (rv.mase_delta with the in-sample scale of each training window; median, threshold 0.01), for the eight cells
  of Table 4 and for STL-SN and STL-AC periodic against Direct (STL-AC from E1).
- thresholds (A7.4): every labelled question relabelled with thresholds of 0.005 and 0.02 from its intervals.
- source (A8.1, k = 32) and length (A8.2, terciles of the history per frequency, k = 48): families x {STL-SN,
  STL-AC} x {default, periodic}, the contrast restricted to the series of the stratum.
- spread (A8.4): the family median of each training pool, per frequency the minimum and maximum over its six
  pools, beside those over the three seeds of Table C3.

Run:  python -m src.analysis.v2.robust --tables TABLES --data-dir DATA --questions QDIR --out OUT [--workers N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.v2 import infer
from src.analysis.v2 import revisao_v2 as rv
from src.analysis.v2.a0 import arm, contrast, main_seed
from src.analysis.v2.tables import instances, load, windows
from src.stages import sensitivity4 as s4

CELLS = [("stl_sn", "default"), ("stl_ac", "default"), ("stl_sn", "periodic")]
THRESHOLDS = (0.005, 0.02)


def paired(fc: pd.DataFrame, inst: pd.DataFrame) -> pd.DataFrame:
    """Direct against each STL cell, per forecast, with the two MAEs and the seasonal naive MAE."""
    c = instances(main_seed(fc[(fc["scope"] == "cohort") & (fc["setting"] != "default_m")
                               & fc["source"].isin(["run", "s3", "s4"])]), inst)
    parts = []
    for strategy, setting in CELLS:
        sub = c[c["model"].isin(s4.models_of(setting))]
        a = arm(sub)
        pair = ("direct/none", f"{strategy}/{setting}", ["strategy", "setting"])
        d = contrast(sub, a, *pair, keep=True, value="mae").rename(columns={"first": "mae_direct", "second": "mae_stl"})
        d["rn_delta"] = contrast(sub, a, *pair)["delta"].to_numpy()
        naive = contrast(sub, a, *pair, keep=True, value="mae_naive")
        if not np.allclose(naive["first"], naive["second"]):
            raise ValueError("paired forecasts with different seasonal naive errors")
        d["mae_naive"] = naive["first"].to_numpy()
        parts.append(d.drop(columns="delta").assign(strategy=strategy, setting=setting))
    return pd.concat(parts, ignore_index=True)


def q12(d: pd.DataFrame, inst: pd.DataFrame, win: pd.DataFrame, workers: int) -> pd.DataFrame:
    scale = {(s, int(w)): rv.insample_scale(y, m) for s, w, y, m in zip(win["series_id"], win["window"], win["y_train"], win["m"])}
    d = d.assign(gm=rv.log_ratio_effect(d["mae_direct"], d["mae_stl"]),
                 mase=rv.mase_delta(d["mae_direct"], d["mae_stl"], d["mae_naive"],
                                    [scale[(s, int(w))] for s, w in zip(d["series_id"].astype(str), d["window"])]))
    n_zero = int(d["gm"].isna().sum())
    gm = d.rename(columns={"gm": "delta"})
    X, labels = infer.instance_matrix(gm, inst, ["family", "strategy", "setting"])
    a = infer.summary(inst, X, labels, 32, "Q12", workers, stat="mean", delta=rv.LOG_DELTA).assign(
        metric="gm_relmae", n_zero=n_zero)
    a["gm"] = np.exp(-a["estimate"])
    X, labels = infer.instance_matrix(d.rename(columns={"mase": "delta"}), inst, ["family", "strategy", "setting"])
    b = infer.summary(inst, X, labels, 32, "Q12", workers).assign(metric="mase")
    return pd.concat([a, b], ignore_index=True)


def strata(d: pd.DataFrame, inst: pd.DataFrame, meta: pd.DataFrame, workers: int) -> dict[str, pd.DataFrame]:
    d = d.rename(columns={"rn_delta": "delta"})
    source = d.assign(stratum=d["series_id"].astype(str).str.split("_").str[0])
    X, labels = infer.instance_matrix(source, inst, ["family", "strategy", "setting", "stratum"])
    out = {"source": infer.summary(inst, X, labels, 32, "A8-source", workers)}
    cut = meta.groupby("freq")["n_hist"].transform(lambda v: pd.qcut(v, 3, labels=["Short", "Medium", "Long"]).astype(str))
    length = d.assign(stratum=d["series_id"].astype(str).map(dict(zip(meta["series_id"], cut))))
    X, labels = infer.instance_matrix(length, inst, ["family", "strategy", "setting", "stratum"])
    out["length"] = infer.summary(inst, X, labels, 48, "A8-length", workers)
    cuts = meta.groupby("freq")["n_hist"].quantile([1 / 3, 2 / 3]).rename("cut").reset_index()
    out["length_cuts"] = cuts
    return out


def spread(d: pd.DataFrame) -> pd.DataFrame:
    d = d.rename(columns={"rn_delta": "delta"})
    v = d.groupby(["series_id", "feature_sample", "freq", "family", "strategy", "setting"], observed=True)["delta"].mean()
    pools = v.groupby(level=["feature_sample", "freq", "family", "strategy", "setting"], observed=True).median().reset_index()
    return (pools.groupby(["freq", "family", "strategy", "setting"])["delta"].agg(pool_min="min", pool_max="max")
            .reset_index())


def thresholds(questions: Path) -> pd.DataFrame:
    """Every labelled cell of the questions' CSVs relabelled with the other thresholds (the same intervals)."""
    rows = []
    for path in sorted(Path(questions).glob("*.csv")):
        q = pd.read_csv(path)
        if not {"sim_lo", "sim_hi", "label"} <= set(q.columns):
            continue
        if "metric" in q.columns:                       # Q12: the log-ratio cells have their own threshold
            q = q[q["metric"] != "gm_relmae"].reset_index(drop=True)
        for delta in THRESHOLDS:
            new = [rv.label(lo, hi, delta) for lo, hi in zip(q["sim_lo"], q["sim_hi"])]
            changed = q.assign(label_new=new)[np.array(new) != q["label"].to_numpy()]
            rows.append({"table": path.stem, "delta": delta, "cells": len(q), "changed": len(changed),
                         "which": "; ".join(f"{'|'.join(str(r[c]) for c in q.columns[:q.columns.get_loc('question')])}: "
                                            f"{r['label']} -> {r['label_new']}" for _, r in changed.iterrows())})
    return pd.DataFrame(rows)


def run(tables: Path, data_dir: Path, questions: Path, out: Path, workers: int) -> dict:
    tables, out = Path(tables), Path(out)
    fc, meta, inst = load(tables)
    d = paired(fc, inst)
    out.mkdir(parents=True, exist_ok=True)
    results = {"Q12": q12(d, inst, windows(tables.parent, data_dir), workers), **strata(d, inst, meta, workers),
               "spread": spread(d), "thresholds": thresholds(questions)}
    for name, df in results.items():
        df.to_csv(out / f"A7A8_{name}.csv" if name != "Q12" else out / "Q12.csv", index=False, float_format="%.10g")
    return {k: len(v) for k, v in results.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.analysis.v2.robust", description=__doc__.splitlines()[0])
    parser.add_argument("--tables", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    a = parser.parse_args(argv)
    print(run(a.tables, a.data_dir, a.questions, a.out, a.workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
