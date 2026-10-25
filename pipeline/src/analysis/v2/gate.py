"""A2: decompose only the series that pass a seasonality test (the 90% Bartlett test at lag m of the M4
benchmarks, rv.seasonality_test), on the raw training part of each window; Direct otherwise.

Outputs in OUT: ``gate.csv`` (series_id, freq, window, passes), ``gate_description.csv``, and the questions
Q7c (Delta_STL of the default setting on the series that pass and fail in the first window; k = 16), Q7 (the
gated policy against Direct, per family and setting; k = 24) and Q7b (the best model of each policy, with and
without the test, against the best Direct configuration; k = 12), STL-AC under the periodic and span-25 settings
from E1.

Run:  python -m src.analysis.v2.gate --tables TABLES --data-dir DATA --out OUT [--workers N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import review
from src.analysis.v2 import infer
from src.analysis.v2 import revisao_v2 as rv
from src.analysis.v2.a0 import arm, contrast, main_seed
from src.analysis.v2.tables import E7, RELNAIVE_EPS, instances, load, windows
from src.stages import sensitivity4 as s4

SETTINGS = [(s, x) for s in ("stl_sn", "stl_ac") for x in ("default", "periodic", "s25")]
RANK1 = "PatchTST"                      # rank1-config: PatchTST Direct


def gate_table(win: pd.DataFrame) -> pd.DataFrame:
    return win.assign(passes=[rv.seasonality_test(y, m) for y, m in zip(win["y_train"], win["m"])])[
        ["series_id", "freq", "window", "passes"]]


def check_windows(win: pd.DataFrame, fc: pd.DataFrame) -> int:
    """The windows are those of the forecasts: the seasonal naive MAE from them equals every stored one (which
    carries the run's RelNaive guard, + RELNAIVE_EPS)."""
    naive = {(s, w): float(np.mean(np.abs(yt - rv.seasonal_naive(y, m, h)))) + RELNAIVE_EPS
             for s, w, y, yt, m, h in zip(win["series_id"], win["window"], win["y_train"], win["y_test"], win["m"], win["h"])}
    d = fc[fc["source"] == "run"].drop_duplicates(["series_id", "window"])
    expected = np.array([naive[(s, int(w))] for s, w in zip(d["series_id"], d["window"])])
    if not np.allclose(expected, d["mae_naive"].to_numpy(), rtol=1e-9, atol=0):
        raise ValueError("the windows do not give the stored seasonal naive MAE")
    return len(d)


def describe(gate: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    first = gate[gate["window"] == 0].assign(dataset=lambda g: g["series_id"].str.split("_").str[0])
    rows = [{"what": "pass, first window", "freq": f, "dataset": s, "share": g["passes"].mean(), "n": len(g)}
            for (f, s), g in first.groupby(["freq", "dataset"])]
    rows += [{"what": "pass, first window", "freq": f, "dataset": "all", "share": g["passes"].mean(), "n": len(g)}
             for f, g in first.groupby("freq")]
    rows.append({"what": "pass, series x window", "freq": "all", "dataset": "all", "share": gate["passes"].mean(), "n": len(gate)})
    same = gate.groupby("series_id")["passes"].agg(lambda p: p.nunique() == 1)
    rows.append({"what": "same result in the three windows", "freq": "all", "dataset": "all", "share": same.mean(), "n": len(same)})
    t = first.merge(x[["unique_id", "fs_tercile"]], left_on="series_id", right_on="unique_id", how="left", validate="one_to_one")
    rows += [{"what": f"pass, first window, phi7 {b}", "freq": f, "dataset": "all", "share": g["passes"].mean(), "n": len(g)}
             for (f, b), g in t.groupby(["freq", "fs_tercile"])]
    return pd.DataFrame(rows)


def run(tables: Path, data_dir: Path, out: Path, workers: int) -> dict:
    tables, out = Path(tables), Path(out)
    fc, _, inst = load(tables)
    run_dir = tables.parent
    win = windows(run_dir, data_dir)
    n_checked = check_windows(win, fc)
    gate = gate_table(win)
    x, _ = review.cells(pd.read_parquet(run_dir / "prepare" / "features.parquet"),
                        pd.read_parquet(run_dir / "prepare" / "bucket_summary.parquet"), pd.read_csv(run_dir / E7))
    out.mkdir(parents=True, exist_ok=True)
    gate.to_csv(out / "gate.csv", index=False)
    describe(gate, x).to_csv(out / "gate_description.csv", index=False, float_format="%.10g")

    c = instances(main_seed(fc[(fc["scope"] == "cohort") & (fc["setting"] != "default_m") & (fc["scale"] == "raw")
                               & fc["source"].isin(["run", "s3", "s4", "s6"])]), inst)
    passes = gate.set_index(["series_id", "window"])["passes"]
    first = gate[gate["window"] == 0].set_index("series_id")["passes"]
    parts = []
    for strategy, setting in SETTINGS:
        sub = c[c["model"].isin(s4.models_of(setting))]
        d = contrast(sub, arm(sub), "direct/none", f"{strategy}/{setting}", ["strategy", "setting"], keep=True)
        d["passes"] = passes.reindex(pd.MultiIndex.from_arrays([d["series_id"].astype(str), d["window"]])).to_numpy()
        if d["passes"].isna().any():
            raise ValueError("forecasts without a test result")
        parts.append(d.assign(strategy=strategy, setting=setting))
    d = pd.concat(parts, ignore_index=True)
    d["passes"] = d["passes"].astype(bool)
    d["gated"] = np.where(d["passes"], d["delta"], 0.0)                       # Direct where the test fails
    d["group"] = np.where(d["series_id"].astype(str).map(first), "pass", "fail")

    results = {}
    q7c = d[d["setting"] == "default"]
    X, labels = infer.instance_matrix(q7c, inst, ["family", "strategy", "group"])
    results["Q7c"] = infer.summary(inst, X, labels, 16, "Q7c", workers)
    X, labels = infer.instance_matrix(d, inst, ["family", "strategy", "setting"], value="gated")
    results["Q7"] = infer.summary(inst, X, labels, 24, "Q7", workers)

    # Q7b: per policy, the model with the lowest mean RelNaive over distinct forecasts (as Table 3), against PatchTST Direct
    distinct = ~d.duplicated(["series_id", "window", "model", "strategy", "setting"]) | (d["family"] != "statistical")
    rank1 = c[(c["model"] == RANK1) & (c["strategy"] == "direct")][["series_id", "feature_sample", "window", "rn"]]
    rank1 = rank1.astype({"series_id": str, "feature_sample": str})
    parts, chosen = [], []
    for (strategy, setting), g in d.groupby(["strategy", "setting"], sort=True):
        for policy in ("no test", "test"):
            rn = g["second"] if policy == "no test" else g["first"] - g["gated"]
            best = rn[distinct.loc[g.index]].groupby(g.loc[distinct.loc[g.index], "model"].astype(str)).mean().idxmin()
            mean = float(rn[distinct.loc[g.index] & (g["model"] == best)].mean())
            b = g[g["model"] == best].assign(rn_policy=rn[g["model"] == best])
            b = b[["series_id", "feature_sample", "window", "rn_policy"]].astype({"series_id": str, "feature_sample": str})
            j = b.merge(rank1, on=["series_id", "feature_sample", "window"], how="inner", validate="one_to_one")
            j = j.assign(delta=j["rn"] - j["rn_policy"], family="all", strategy=strategy, setting=setting, policy=policy,
                         model=best)
            parts.append(j)
            chosen.append({"strategy": strategy, "setting": setting, "policy": policy, "model": best, "mean_rn": mean})
    j = pd.concat(parts, ignore_index=True)
    X, labels = infer.instance_matrix(j, inst, ["strategy", "setting", "policy", "model"])
    results["Q7b"] = infer.summary(inst, X, labels, 12, "Q7b", workers).merge(
        pd.DataFrame(chosen), on=["strategy", "setting", "policy", "model"], validate="one_to_one")
    for name, df in results.items():
        df.to_csv(out / f"{name}.csv", index=False, float_format="%.10g")
    return {"windows_checked": n_checked, **{k: len(v) for k, v in results.items()}}


def acts(tables: Path, data_dir: Path, out: Path, workers: int) -> dict:
    """Round 2: where the policy acts. Many series fail the test in every window, so their policy delta is exactly zero
    and the median of Q7 sits on that atom. Q7p: the delta over the forecasts whose window passes the test (k = 24);
    Q7m: the mean over series of the policy delta, zero where the test fails (k = 24); Q7bm: the same mean for the
    best model of each policy against PatchTST Direct (k = 12); Q7r: the best re-tuned periodic STL-SN model of each
    global family against PatchTST Direct and ETS Direct (k = 6); and the shares of series and windows decomposed."""
    tables, out = Path(tables), Path(out)
    fc, _, inst = load(tables)
    gate = gate_table(windows(tables.parent, data_dir))
    c = instances(main_seed(fc[(fc["scope"] == "cohort") & (fc["setting"] != "default_m") & (fc["scale"] == "raw")
                               & fc["source"].isin(["run", "s3", "s4", "s6", "s7"])]), inst)
    passes = gate.set_index(["series_id", "window"])["passes"]
    parts = []
    for strategy, setting in SETTINGS:
        sub = c[c["model"].isin(s4.models_of(setting))]
        d = contrast(sub, arm(sub), "direct/none", f"{strategy}/{setting}", ["strategy", "setting"], keep=True)
        d["passes"] = passes.reindex(pd.MultiIndex.from_arrays([d["series_id"].astype(str), d["window"]])).to_numpy().astype(bool)
        parts.append(d.assign(strategy=strategy, setting=setting))
    d = pd.concat(parts, ignore_index=True)
    d["gated"] = np.where(d["passes"], d["delta"], 0.0)
    results = {}
    X, labels = infer.instance_matrix(d[d["passes"]], inst, ["family", "strategy", "setting"])
    results["Q7p"] = infer.summary(inst, X, labels, 24, "Q7p", workers)
    X, labels = infer.instance_matrix(d, inst, ["family", "strategy", "setting"], value="gated")
    results["Q7m"] = infer.summary(inst, X, labels, 24, "Q7m", workers, stat="mean")
    # Q7bm: the policies of Q7b (their chosen models) by the mean
    chosen = pd.read_csv(out.parent / "final" / "Q7b.csv")[["strategy", "setting", "policy", "model"]]
    rank1 = c[(c["model"] == RANK1) & (c["strategy"] == "direct")][["series_id", "feature_sample", "window", "rn"]]
    rank1 = rank1.astype({"series_id": str, "feature_sample": str})
    jparts = []
    for r in chosen.itertuples():
        g = d[(d["strategy"] == r.strategy) & (d["setting"] == r.setting) & (d["model"] == r.model)]
        rn = g["second"] if r.policy == "no test" else g["first"] - g["gated"]
        b = g.assign(rn_policy=rn)[["series_id", "feature_sample", "window", "rn_policy"]].astype({"series_id": str, "feature_sample": str})
        jj = b.merge(rank1, on=["series_id", "feature_sample", "window"], how="inner", validate="one_to_one")
        jparts.append(jj.assign(delta=jj["rn"] - jj["rn_policy"], family="all", strategy=r.strategy, setting=r.setting,
                                policy=r.policy, model=r.model))
    X, labels = infer.instance_matrix(pd.concat(jparts, ignore_index=True), inst, ["strategy", "setting", "policy", "model"])
    results["Q7bm"] = infer.summary(inst, X, labels, 12, "Q7bm", workers, stat="mean")
    # Q7r: the re-tuned periodic STL-SN against the best Direct configurations
    rt = c[(c["strategy"] == "stl_sn") & (c["setting"] == "periodic_tuned")]
    distinct = rt[~rt.duplicated(["series_id", "window", "model"])]
    rparts = []
    for fam in ("ml", "neural", "transformer"):
        g = distinct[distinct["family"] == fam]
        best = g.groupby(g["model"].astype(str))["rn"].mean().idxmin()
        b = rt[rt["model"] == best][["series_id", "feature_sample", "window", "rn"]].rename(columns={"rn": "rn_rt"})
        b = b.astype({"series_id": str, "feature_sample": str})
        for comp in ("PatchTST", "ETS"):
            ref = c[(c["model"] == comp) & (c["strategy"] == "direct")][["series_id", "feature_sample", "window", "rn"]]
            jj = b.merge(ref.astype({"series_id": str, "feature_sample": str}), on=["series_id", "feature_sample", "window"],
                         how="inner", validate="one_to_one")
            rparts.append(jj.assign(delta=jj["rn"] - jj["rn_rt"], family=fam, model=best, comparator=comp,
                                    mean_rn=float(g[g["model"] == best]["rn"].mean())))
    r = pd.concat(rparts, ignore_index=True)
    X, labels = infer.instance_matrix(r, inst, ["family", "model", "comparator"])
    results["Q7r"] = infer.summary(inst, X, labels, 6, "Q7r", workers).merge(
        r.drop_duplicates(["family", "model", "comparator"])[["family", "model", "comparator", "mean_rn"]],
        on=["family", "model", "comparator"], validate="one_to_one")
    per = gate.groupby(["freq", "series_id"])["passes"].any().reset_index()
    shares = [{"what": "series decomposed in at least one window", "freq": f, "share": g["passes"].mean(), "n": len(g)}
              for f, g in per.groupby("freq")]
    shares.append({"what": "series decomposed in at least one window", "freq": "all", "share": per["passes"].mean(), "n": len(per)})
    shares.append({"what": "series never decomposed", "freq": "all", "share": 1 - per["passes"].mean(), "n": len(per)})
    shares.append({"what": "series-window pairs decomposed", "freq": "all", "share": gate["passes"].mean(), "n": len(gate)})
    results["gate_shares"] = pd.DataFrame(shares)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_csv(out / f"{name}.csv", index=False, float_format="%.10g")
    return {k: len(v) for k, v in results.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.analysis.v2.gate", description=__doc__.splitlines()[0])
    parser.add_argument("--tables", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--acts", action="store_true", help="round 2: Q7p, Q7m, Q7bm, Q7r and the shares decomposed")
    a = parser.parse_args(argv)
    print((acts if a.acts else run)(a.tables, a.data_dir, a.out, a.workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
