"""A8.3 and A8.5, from the stored per-step forecasts (y_test, yhat, yhat_naive of every task file).

- Horizon (A8.3): each forecast's RelNaive over a block of steps (monthly 1-6, 7-12, 13-18; quarterly 1-4, 5-8;
  the model's MAE over the block / (the seasonal naive's + the run's guard), capped at 10) and Direct minus STL per
  block, for STL-SN default, STL-SN periodic and STL-AC default; cohort scope, main seed, the main specification;
  k = the 60 cells.
- Seed ensemble (A8.5): on the evolving-seasonality sample, the mean of the three seeds' forecasts step by step
  (the two deterministic global models are the same under every seed), and Direct minus STL-SN (default,
  periodic) and minus STL-AC (default) of the ensembles per global family; one pool per frequency, so the series
  are resampled; k = 9.

Run:  python -m src.analysis.v2.steps --tables TABLES --out OUT [--workers N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.v2 import infer
from src.analysis.v2.a0 import MAIN_SEED, SEEDS, contrast
from src.analysis.v2.tables import CAP, RELNAIVE_EPS, instances, load
from src.stages import sensitivity as s1
from src.stages import sensitivity3 as s3
from src.stages import sensitivity4 as s4
from src.stages import sensitivity4s as s4s
from src.stages.tasks import SEED_CHECK_FEATURE, STOCHASTIC_MODELS

BLOCKS = {"monthly": [(1, 6), (7, 12), (13, 18)], "quarterly": [(1, 4), (5, 8)]}
COLS = ["unique_id", "feature_name", "frequency", "window", "family", "model", "strategy", "scope", "seed", "bucket",
        "y_test", "yhat", "yhat_naive"]


def blocks(rows: pd.DataFrame, setting: str) -> pd.DataFrame:
    """One row per forecast and block with its RelNaive, in the key columns of fc_errors."""
    out = []
    for freq, g in rows.groupby("frequency", sort=False):
        yt, yh, yn = (np.stack(g[c].to_numpy()) for c in ("y_test", "yhat", "yhat_naive"))
        stat = (g["family"] == "statistical").to_numpy()
        base = pd.DataFrame({"series_id": g["unique_id"].to_numpy(), "freq": freq,
                             "feature_sample": np.where(stat, "none", g["feature_name"].astype(object)),
                             "window": g["window"].to_numpy(), "model": g["model"].to_numpy(), "family": g["family"].to_numpy(),
                             "strategy": g["strategy"].to_numpy(), "setting": np.where(g["strategy"] == "direct", "none", setting),
                             "scope": "cohort", "group": "none",
                             "seed": g["seed"].astype("Int64").fillna(-1).astype("int64").to_numpy()})
        for a, b in BLOCKS[freq]:
            sl = slice(a - 1, b)
            mae = np.abs(yt[:, sl] - yh[:, sl]).mean(axis=1)
            naive = np.abs(yt[:, sl] - yn[:, sl]).mean(axis=1) + RELNAIVE_EPS
            out.append(base.assign(block=f"{a}-{b}", rn=np.minimum(mae / naive, CAP)))
    return pd.concat(out, ignore_index=True)


def _files(run_dir: Path):
    """(rows, setting) of the forecasts A8.3 needs, one file at a time: the run's cohort rows of the main seed
    (Direct, and STL of the global models), S3's statistical STL rows, S4's periodic STL-SN rows."""
    ev = run_dir / "evaluate"
    for p in sorted(ev.glob(f"*/tasks/eval__*__cohort__seed{MAIN_SEED}.parquet")):
        yield pd.read_parquet(p, columns=COLS), "default"
    for p in sorted(ev.glob("statistical-*/tasks/stat__*__direct__chunk*.parquet")):
        r = pd.read_parquet(p, columns=COLS)
        yield r[r["model"] != "SARIMA"], "default"
    prepare = run_dir / "prepare"
    bs, sm = pd.read_parquet(prepare / "bucket_summary.parquet"), pd.read_parquet(prepare / "samples.parquet")
    tasks = [t for t in s3.eval_tasks(bs, sm) if t["task_id"].startswith("s3stat|")]
    yield s1._s1_rows(run_dir, tasks, s3.ROOT, COLS)[0], "default"
    tasks = [t for t in s4.eval_tasks(bs, sm) if (t["task_id"].startswith("s4eval|") and t["variant"] == "periodic"
                                                  and t["entry"] == s4.RUN_ENTRY)
             or (t["task_id"].startswith("s4stat|") and t["variant"] == "periodic_p1")]
    yield s1._s1_rows(run_dir, tasks, s4.ROOT, COLS)[0], "periodic"


def horizon(tables: Path, inst: pd.DataFrame, workers: int) -> pd.DataFrame:
    run_dir = Path(tables).parent
    b = pd.concat([blocks(r, s) for r, s in _files(run_dir) if len(r)], ignore_index=True)   # SARIMA's files are empty here
    c = instances(b.assign(source="steps", dataset=b["series_id"].str.split("_").str[0], pool_id="", bucket=None,
                           scale="raw", mae=np.nan, mae_naive=np.nan), inst)
    a = c["strategy"].astype(str) + "/" + c["setting"].astype(str) + "/" + c["block"].astype(str)
    parts = []
    for strategy, setting in (("stl_sn", "default"), ("stl_sn", "periodic"), ("stl_ac", "default")):
        models = s4.models_of("periodic") if setting == "periodic" else None
        sub = c if models is None else c[c["model"].isin(models)]
        for blk in sorted(set(c["block"].astype(str))):
            d = contrast(sub, a[sub.index], f"direct/none/{blk}", f"{strategy}/{setting}/{blk}", ["strategy", "setting"])
            parts.append(d.assign(strategy=strategy, setting=setting, block=blk))
    d = pd.concat(parts, ignore_index=True)
    X, labels = infer.instance_matrix(d, inst, ["family", "strategy", "setting", "freq", "block"])
    if X.shape[1] != 60:
        raise ValueError(f"A8.3 has {X.shape[1]} cells, expected 60")
    return infer.summary(inst, X, labels, 60, "A8-horizon", workers)


def ensemble(tables: Path, inst: pd.DataFrame, workers: int) -> pd.DataFrame:
    run_dir = Path(tables).parent
    ev = run_dir / "evaluate"
    frames = [pd.read_parquet(p, columns=COLS) for s in SEEDS
              for p in sorted(ev.glob(f"*/tasks/eval__{SEED_CHECK_FEATURE}__*__cohort__seed{s}.parquet"))]
    rows = pd.concat(frames, ignore_index=True).assign(setting="default")
    rows.loc[rows["strategy"] == "direct", "setting"] = "none"
    prepare = run_dir / "prepare"
    bs, sm = pd.read_parquet(prepare / "bucket_summary.parquet"), pd.read_parquet(prepare / "samples.parquet")
    s4_tasks = [t for t in s4.eval_tasks(bs, sm) if t["task_id"].startswith("s4eval|") and t["feature_name"] == SEED_CHECK_FEATURE
                and t["variant"] == "periodic" and t["entry"] == s4.RUN_ENTRY]
    cfg = {"random_seed": SEEDS[0], "seed_check_seeds": SEEDS[1:]}
    s4s_tasks = [t for t in s4s.study(cfg).eval_tasks(bs, sm) if t["task_id"].startswith("s4seval|")]
    per = pd.concat([s1._s1_rows(run_dir, s4_tasks, s4.ROOT, COLS)[0], s1._s1_rows(run_dir, s4s_tasks, s4s.ROOT, COLS)[0]],
                    ignore_index=True).assign(setting="periodic")
    rows = pd.concat([rows, per], ignore_index=True)
    rows = rows[rows["family"] != "statistical"]
    det = ~rows["model"].isin(STOCHASTIC_MODELS)
    rows = rows[~det | (rows["seed"] == MAIN_SEED)]                     # deterministic models: one copy, any seed
    keys = ["unique_id", "frequency", "window", "family", "model", "strategy", "setting"]
    n = rows.groupby(keys)["seed"].nunique()
    if not ((n == 3) | n.index.get_level_values("model").isin(rows.loc[det, "model"].unique())).all():
        raise ValueError("a stochastic forecast without its three seeds")
    ens = rows.groupby(keys, sort=False).agg(y_test=("y_test", "first"), yhat_naive=("yhat_naive", "first"),
                                             yhat=("yhat", lambda v: list(np.mean(np.stack(v.to_numpy()), axis=0)))).reset_index()
    rn = np.array([min(np.mean(np.abs(np.asarray(t) - np.asarray(f))) / (np.mean(np.abs(np.asarray(t) - np.asarray(n))) + RELNAIVE_EPS), CAP)
                   for t, f, n in zip(ens["y_test"], ens["yhat"], ens["yhat_naive"])])     # horizons differ by frequency
    e = pd.DataFrame({"series_id": ens["unique_id"], "freq": ens["frequency"], "feature_sample": SEED_CHECK_FEATURE,
                      "window": ens["window"], "family": ens["family"], "model": ens["model"], "strategy": ens["strategy"],
                      "setting": ens["setting"], "scope": "cohort", "group": "none", "seed": 0, "rn": rn})
    a = e["strategy"].astype(str) + "/" + e["setting"].astype(str)
    parts = [contrast(e, a, "direct/none", f"{s}/{x}", ["strategy", "setting"]).assign(strategy=s, setting=x)
             for s, x in (("stl_sn", "default"), ("stl_sn", "periodic"), ("stl_ac", "default"))]
    sub = inst[inst["feature_sample"] == SEED_CHECK_FEATURE].reset_index(drop=True)
    X, labels = infer.instance_matrix(pd.concat(parts, ignore_index=True), sub, ["family", "strategy", "setting"])
    return infer.summary(sub, X, labels, 9, "A8-ensemble", workers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.analysis.v2.steps", description=__doc__.splitlines()[0])
    parser.add_argument("--tables", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    a = parser.parse_args(argv)
    _, _, inst = load(a.tables, ["run"])
    a.out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("A8_ensemble", ensemble), ("A8_horizon", horizon)):
        df = fn(a.tables, inst, a.workers)
        df.to_csv(a.out / f"{name}.csv", index=False, float_format="%.10g")
        print(name, len(df), df["label"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
