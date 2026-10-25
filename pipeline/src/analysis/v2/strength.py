"""A6: seasonal strength measured without the default STL, and the dose-response curve.

Every sampled series' history before the first test window (the data of phi7, E7) gives three new measures:
phi7_per (the strength formula with periodic STL), phi7_cls (the same with a classical additive decomposition,
statsmodels seasonal_decompose, the ends without values left out; the measure without STL) and phi7_acf (the
autocorrelation at lag m, as in the seasonality test). The default STL on the same history must reproduce E7's
phi7, which checks the history.

Outputs in OUT:
- ``strength_measures.csv``: the measures per series;
- Q17 (k = 12): per frequency and measure (phi4, phi7, phi4', phi7_per, phi7_cls, phi7_acf), the median STL-SN
  default delta (families pooled with equal weight) in each tercile and High minus Low with its simultaneous
  interval; ``strength_tests.csv``: the Jonckheere-Terpstra test (Holm over the 12 rows), Spearman's rho and the
  partial rho with its 95% interval (strength measures given phi4, phi4 and phi4' given phi7);
- Q18 (k = 16): the median STL-SN default delta of each family in the High tercile of phi7 and the Low tercile of
  phi4, per frequency;
- ``A14.csv``: the share of series with phi2 = 0 per tercile of trend strength (1 - Var(R) / Var(T + R) of the
  same default STL), per frequency;
- ``dose.csv`` and ``dose_crossings.csv``: LOWESS of the per-series delta on phi7 and phi7_cls per frequency, with a
  95% band and the strength at which the curve stays above 0 and 0.01 from 1,000 replicates of the two-level
  bootstrap; per global family on monthly series the curves only.

Run:  python -m src.analysis.v2.strength --tables TABLES --data-dir DATA --out OUT [--workers N]
"""
from __future__ import annotations

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import review, stats
from src.analysis.v2 import infer
from src.analysis.v2 import revisao_v2 as rv
from src.analysis.v2.a0 import MAIN_SEED
from src.analysis.v2.questions import spec_cohort, stl_contrasts
from src.analysis.v2.tables import E7, load, windows
from src.features.mstl_features import decompose_series, seasonal_strength

MEASURES = ["phi4", "phi7", "phi4p", "phi7_per", "phi7_cls", "phi7_acf"]
GIVEN = {"phi4": "phi7", "phi4p": "phi7"}                       # partial rho given phi7; the others given phi4
CURVES = ["phi7", "phi7_cls"]
N_BAND = 1000


def strength(seasonal, residual) -> float:
    try:
        return seasonal_strength(np.asarray(seasonal, float), np.asarray(residual, float))
    except Exception:
        return np.nan


def measures(run_dir: Path, data_dir: Path) -> pd.DataFrame:
    from statsmodels.tsa.seasonal import seasonal_decompose

    win = windows(run_dir, data_dir)
    rows = []
    first = win[win["window"] == 0]
    for s, f, y, m in zip(first["series_id"], first["freq"], first["y_train"], first["m"]):
        d = decompose_series(y, m)
        p = decompose_series(y, m, periodic=True)
        c = seasonal_decompose(y, model="additive", period=m, two_sided=True)
        ok = ~np.isnan(np.asarray(c.resid))
        rows.append({"series_id": s, "freq": f, "phi7_check": strength(d["seasonal"], d["residual"]),
                     "trend_strength": strength(d["trend"], d["residual"]),
                     "phi7_per": strength(p["seasonal"], p["residual"]),
                     "phi7_cls": strength(np.asarray(c.seasonal)[ok], np.asarray(c.resid)[ok]),
                     "phi7_acf": float(rv.acf(y, m)[m])})
    return pd.DataFrame(rows)


def terciles(x: pd.Series, freq: pd.Series, cuts: dict | None = None) -> np.ndarray:
    out = np.full(len(x), None, dtype=object)
    for f in freq.unique():
        sel = (freq == f).to_numpy() & x.notna().to_numpy()
        lo, hi = cuts[f] if cuts else np.quantile(x[sel], [1 / 3, 2 / 3])
        v = x[sel].to_numpy()
        out[sel] = np.where(v <= lo, "Low", np.where(v <= hi, "Medium", "High"))
    return out


def _band(args):
    xs, vals = args
    return rv.lowess_curve(xs, vals)


def run(tables: Path, data_dir: Path, out: Path, workers: int) -> dict:
    tables, out = Path(tables), Path(out)
    fc, meta, inst = load(tables)
    run_dir = tables.parent
    m = measures(run_dir, data_dir)
    e7 = pd.read_csv(run_dir / E7).rename(columns={"unique_id": "series_id", "frequency": "freq"})
    chk = m.merge(e7[["series_id", "freq", "seasonal_strength"]], on=["series_id", "freq"], validate="one_to_one")
    if not np.allclose(chk["phi7_check"], chk["seasonal_strength"], rtol=0, atol=1e-12, equal_nan=True):
        raise ValueError("the default STL on the history does not reproduce E7's phi7")
    x = meta[["series_id", "freq", "feature_evolving_seasonality", "phi7", "phi4p"]].rename(
        columns={"feature_evolving_seasonality": "phi4"}).merge(m.drop(columns="phi7_check"), on=["series_id", "freq"])
    summary = pd.read_parquet(run_dir / "prepare" / "bucket_summary.parquet")
    cut = summary[summary["feature_name"] == "feature_evolving_seasonality"].set_index("frequency")
    for name in MEASURES:
        cuts = {f: (cut.loc[f, "tercile_cut_1"], cut.loc[f, "tercile_cut_2"]) for f in cut.index} if name == "phi4" else None
        x[f"{name}_t"] = terciles(x[name], x["freq"], cuts)
    out.mkdir(parents=True, exist_ok=True)
    x.to_csv(out / "strength_measures.csv", index=False, float_format="%.10g")

    d = stl_contrasts(spec_cohort(fc, inst))
    d = d[d["strategy"] == "stl_sn"].astype({"series_id": str})
    del fc                                       # the band workers start empty (spawn); keep this process small
    hb = rv.HierarchicalBootstrap(inst)

    # Q17: High - Low per measure and frequency (one column per tercile; shared replicates)
    parts = []
    for name in MEASURES:
        t = dict(zip(x["series_id"], x[f"{name}_t"]))
        parts.append(d.assign(measure=name, tercile=d["series_id"].map(t)).dropna(subset=["tercile"]))
    X, labels = infer.instance_matrix(pd.concat(parts, ignore_index=True), inst, ["measure", "freq", "tercile"], balanced=True)
    reps = infer.replicates(inst, X, rv.n_boot(12), workers=workers)
    point = hb.point(X)
    rows, tests = [], []
    per_series = (d.groupby(["series_id", "freq", "family"], observed=True)["delta"].mean()
                  .groupby(level=["series_id", "freq"], observed=True).mean().rename("delta").reset_index())
    ps = per_series.merge(x, on=["series_id", "freq"])
    for (name, f), g in labels.groupby(["measure", "freq"], sort=False):
        b = dict(zip(g["tercile"], g.index))
        cell = pd.DataFrame({"measure": [name], "freq": [f], **{f"median_{k.lower()}": [point[b[k]]] for k in ("Low", "Medium", "High")}})
        rows.append(infer.labelled(point[[b["High"]]] - point[[b["Low"]]], reps[:, [b["High"]]] - reps[:, [b["Low"]]], cell, 12, "Q17"))
        h = ps[(ps["freq"] == f)].dropna(subset=[name])
        groups = [h.loc[h[f"{name}_t"] == k, "delta"].to_numpy() for k in ("Low", "Medium", "High")]
        jt = stats.jonckheere_terpstra(groups)
        sp = stats.spearman(h[name].to_numpy(), h["delta"].to_numpy(), MAIN_SEED, f"A6|{f}|{name}")
        given = GIVEN.get(name, "phi4")
        hh = h.dropna(subset=[given])
        u, v, w = hh[name].to_numpy(), hh["delta"].to_numpy(), hh[given].to_numpy()
        lo, hi = review.partial_ci(u, v, w, MAIN_SEED, f"A6|{f}|{name}|partial")
        tests.append({"measure": name, "freq": f, "n_series": len(h), "jt_z": jt["z"], "jt_p": jt["p"], "rho": sp["rho"],
                      "given": given, "partial": review.partial_rho(u, v, w), "partial_lo": lo, "partial_hi": hi})
    q17 = pd.concat(rows, ignore_index=True)
    tests = pd.DataFrame(tests)
    tests["jt_p_holm"] = stats.holm(tests["jt_p"].to_numpy())

    # Q18: per family, High phi7 and Low phi4
    parts = []
    for name, t in (("phi7", "High"), ("phi4", "Low")):
        ids = set(x.loc[x[f"{name}_t"] == t, "series_id"])
        parts.append(d[d["series_id"].isin(ids)].assign(subset=f"{name} {t}"))
    X18, l18 = infer.instance_matrix(pd.concat(parts, ignore_index=True), inst, ["family", "freq", "subset"])
    q18 = infer.summary(inst, X18, l18, 16, "Q18", workers)

    # dose-response: the per-series delta (pooled) on phi7 and phi7_cls, per frequency, with bands
    curves, crossings = [], []
    xs_of = x.set_index("series_id")
    Xf, lf = infer.instance_matrix(d, inst, ["freq"], balanced=True)
    for j, f in enumerate(lf["freq"]):
        series = list(rv.series_values_replicates(hb, Xf[:, j], N_BAND))
        for name in CURVES:
            xv = xs_of[name].reindex(hb.series_ids).to_numpy()
            g = ps[ps["freq"] == f].dropna(subset=[name])
            curve = rv.lowess_curve(g[name].to_numpy(), g["delta"].to_numpy())
            jobs = [(xv[idx], vals) for idx, vals in series]
            jobs = [(a[~np.isnan(a)], b[~np.isnan(a)]) for a, b in jobs]
            with ProcessPoolExecutor(max(1, workers), mp_context=multiprocessing.get_context("spawn")) as pool:
                band = np.array(list(pool.map(_band, jobs, chunksize=20)))
            cross = {lvl: np.array([rv.upper_crossing(rv.GRID, c, lvl) for c in band]) for lvl in (0.0, 0.01)}
            curves.append(pd.DataFrame({"freq": f, "measure": name, "family": "pooled", "x": rv.GRID, "curve": curve,
                                        "band_lo": np.nanquantile(band, 0.025, axis=0), "band_hi": np.nanquantile(band, 0.975, axis=0)}))
            for lvl, c in cross.items():
                ok = c[~np.isnan(c)]
                crossings.append({"freq": f, "measure": name, "level": lvl, "point": rv.upper_crossing(rv.GRID, curve, lvl),
                                  "median": float(np.median(ok)) if len(ok) else np.nan,
                                  "lo": float(np.quantile(ok, 0.025)) if len(ok) else np.nan,
                                  "hi": float(np.quantile(ok, 0.975)) if len(ok) else np.nan, "share_crossing": len(ok) / len(c)})
    fam = (d[d["freq"] == "monthly"].groupby(["series_id", "family"], observed=True)["delta"].mean().reset_index()
           .merge(x[x["freq"] == "monthly"], on="series_id"))
    for (name, f_), g in [((n, fa), fam[fam["family"] == fa]) for n in CURVES for fa in ("ml", "neural", "transformer")]:
        g = g.dropna(subset=[name])
        curves.append(pd.DataFrame({"freq": "monthly", "measure": name, "family": f_, "x": rv.GRID,
                                    "curve": rv.lowess_curve(g[name].to_numpy(), g["delta"].to_numpy())}))
    # A14: the share of series with phi2 = 0 in each tercile of trend strength, per frequency
    tr = meta[["series_id", "freq", "feature_nonlinearity"]].merge(m[["series_id", "freq", "trend_strength"]], on=["series_id", "freq"])
    tr["tercile"] = terciles(tr["trend_strength"], tr["freq"])
    a14 = (tr.assign(zero=tr["feature_nonlinearity"] == 0).groupby(["freq", "tercile"])["zero"].agg(["mean", "size"])
           .rename(columns={"mean": "share_zero", "size": "n"}).reset_index())
    results = {"A14": a14, "Q17": q17, "strength_tests": tests, "Q18": q18, "dose": pd.concat(curves, ignore_index=True),
               "dose_crossings": pd.DataFrame(crossings)}
    for k, v in results.items():
        v.to_csv(out / f"{k}.csv", index=False, float_format="%.10g")
    return {k: len(v) for k, v in results.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.analysis.v2.strength", description=__doc__.splitlines()[0])
    parser.add_argument("--tables", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    a = parser.parse_args(argv)
    print(run(a.tables, a.data_dir, a.out, a.workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
