"""A5: a diagnostic of the add-back. STL-SN continues the seasonal component S by its last cycle; here it is
continued by the mean of its last k cycles (k = 2, 3), with the non-seasonal forecast N unchanged.

The default STL is deterministic, so it is fitted again on each training window and S_last and S_k taken from it;
N = (stored STL-SN forecast) - S_last, and the forecast with k cycles is N + S_k. Every STL-SN default forecast of
the main specification (11 models, cohort scope, main seed) gives two new forecasts (strategies stl_sn_k2, stl_sn_k3).

Q11 (k = 16): per family, Direct minus each k (Delta_STL) and STL-SN default minus each k. The recovered share,
Rec_f = median(STL-SN default - k) / median(STL-SN default - STL-SN periodic), takes its denominator from Table 8.

Run:  python -m src.analysis.v2.addback --tables TABLES --data-dir DATA --out OUT [--workers N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.v2 import infer
from src.analysis.v2 import revisao_v2 as rv
from src.analysis.v2.a0 import MAIN_SEED, arm, contrast, main_seed
from src.analysis.v2.tables import RELNAIVE_EPS, TEXT, instances, load, windows
from src.forecast.targets import component_targets
from src.stages import sensitivity as s1
from src.stages import sensitivity3 as s3

KS = (2, 3)


def stored_sn(run_dir: Path) -> pd.DataFrame:
    """The STL-SN default forecasts of the main specification: the run's global cohort rows of the main seed
    and S3's ARIMA and ETS rows (season length 1)."""
    cols = ["unique_id", "feature_name", "frequency", "window", "family", "model", "seed", "y_test", "yhat"]
    frames = []
    for p in sorted((run_dir / "evaluate").glob(f"*/tasks/eval__*__stl_sn__*__cohort__seed{MAIN_SEED}.parquet")):
        frames.append(pd.read_parquet(p, columns=cols))
    prepare = run_dir / "prepare"
    tasks = [t for t in s3.eval_tasks(pd.read_parquet(prepare / "bucket_summary.parquet"), pd.read_parquet(prepare / "samples.parquet"))
             if t["task_id"].startswith("s3stat|") and t["strategy"] == "stl_sn"]
    s3_rows, facts = s1._s1_rows(run_dir, tasks, s3.ROOT, cols)
    if facts["missing"] or facts["changed"]:
        raise ValueError("S3's STL-SN rows are missing or changed")
    return pd.concat([*frames, s3_rows], ignore_index=True)


def k_forecasts(run_dir: Path, data_dir: Path) -> pd.DataFrame:
    """One row per stored STL-SN forecast and k, in the columns of fc_errors (strategy stl_sn_k<k>)."""
    win = windows(run_dir, data_dir)
    shift = {}
    for s, w, y, m, h in zip(win["series_id"], win["window"], win["y_train"], win["m"], win["h"]):
        seasonal = component_targets(y, m)["seasonal"]
        last = rv.seasonal_naive(seasonal, m, h)
        shift[(s, int(w))] = {k: rv.mean_last_cycles(seasonal, m, h, k) - last for k in KS}
    naive = {(s, int(w)): np.mean(np.abs(yt - rv.seasonal_naive(y, m, h))) + RELNAIVE_EPS
             for s, w, y, yt, m, h in zip(win["series_id"], win["window"], win["y_train"], win["y_test"], win["m"], win["h"])}
    sn = stored_sn(run_dir)
    out = []
    for k in KS:
        mae = np.array([np.mean(np.abs(np.asarray(yt) - (np.asarray(yh) + shift[(u, int(w))][k])))
                        for u, w, yt, yh in zip(sn["unique_id"], sn["window"], sn["y_test"], sn["yhat"])])
        mae_naive = np.array([naive[(u, int(w))] for u, w in zip(sn["unique_id"], sn["window"])])
        stat = (sn["family"] == "statistical").to_numpy()
        out.append(pd.DataFrame({
            "source": "a5", "series_id": sn["unique_id"].to_numpy(), "freq": sn["frequency"].to_numpy(),
            "feature_sample": np.where(stat, "none", sn["feature_name"].astype(object)),
            "window": sn["window"].to_numpy(), "model": sn["model"].to_numpy(), "family": sn["family"].to_numpy(),
            "strategy": f"stl_sn_k{k}", "setting": "default", "scope": "cohort", "group": "none",
            "seed": sn["seed"].astype("Int64").fillna(-1).astype("int64").to_numpy(), "scale": "raw",
            "mae": mae, "mae_naive": mae_naive, "rn": rv.relnaive_capped(mae, mae_naive)}))
    return pd.concat(out, ignore_index=True)


def run(tables: Path, data_dir: Path, out: Path, workers: int) -> pd.DataFrame:
    tables, out = Path(tables), Path(out)
    fc, _, inst = load(tables)
    kf = k_forecasts(tables.parent, data_dir)
    base = main_seed(fc[(fc["scope"] == "cohort") & (fc["setting"] != "default_m") & fc["source"].isin(["run", "s3"])
                        & fc["strategy"].isin(["direct", "stl_sn"])])
    base = base[~((base["model"] == "SARIMA"))]
    both = pd.concat([base.astype({c: str for c in TEXT}), kf.astype({c: str for c in TEXT})], ignore_index=True)
    for c in TEXT:
        both[c] = both[c].astype("category")
    c = instances(both, inst)
    a = arm(c)
    parts = []
    for k in KS:
        for name, first in (("vs direct", "direct/none"), ("vs stl_sn default", "stl_sn/default")):
            parts.append(contrast(c, a, first, f"stl_sn_k{k}/default", ["strategy", "setting"]).assign(k=k, comparison=name))
    X, labels = infer.instance_matrix(pd.concat(parts, ignore_index=True), inst, ["family", "k", "comparison"])
    q11 = infer.summary(inst, X, labels, 16, "Q11", workers)
    out.mkdir(parents=True, exist_ok=True)
    q11.to_csv(out / "Q11.csv", index=False, float_format="%.10g")
    return q11


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.analysis.v2.addback", description=__doc__.splitlines()[0])
    parser.add_argument("--tables", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    a = parser.parse_args(argv)
    print(run(a.tables, a.data_dir, a.out, a.workers)[["family", "k", "comparison", "estimate", "label"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
