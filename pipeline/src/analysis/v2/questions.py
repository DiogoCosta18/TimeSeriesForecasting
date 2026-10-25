"""The question families of the revision analyses on the frozen forecasts (A1): each writes
``OUT/<question>.csv`` with one row per cell (infer.summary).

    Q1   Table 4: Direct minus STL (default) per family and strategy                       k = 8
    Q2   Table C2: the same per frequency                                                  k = 16
    Q6a  Table 11: cohort minus tercile model per family and strategy                      k = 9
    Q6b  Table 12: Delta_FT, Delta_rand and random minus tercile; a third of the steps     k = 39
    Q14  Tables C3 and C7: the estimand over all seeds (seeds, then series, resampled)     k = 16
    Q16  Table 5: High minus Low per feature and strategy                                  k = 12
    C4   Table C4: period 1 and period m of ARIMA and ETS                                  k = 12
    Q3   Table 8 (families): settings, STL-SN and STL-AC; E2's re-tuning                   k = 48
    Q4   Table C6 (models): the same per model                                             k = 84
    Q5   Tables 7 and C5: the best STL-SN model against the best Direct configurations     k = 108
    Q8   E3: the log scale for ARIMA and ETS                                               k = 27
    Q10  E1: the mechanism, variant (ii)                                                   k = 24
    Q13  A9: the families without linear regression and without NLinear                   k = 12
    Q6c  A10: Table 12 on the evolving-seasonality pools                                   k = 39

Run:  python -m src.analysis.v2.questions --tables TABLES --out OUT [--only Q1,Q2] [--workers N]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.v2 import infer
from src.analysis.v2 import revisao_v2 as rv
from src.analysis.v2.a0 import MAIN_SEED, SEEDS, arm, contrast, main_seed
from src.analysis.review import BEST_DIRECT
from src.analysis.v2.tables import E7, instances, load
from src.forecast.registry import family
from src.stages import sensitivity4 as s4
from src.stages.sensitivity5 import REDUCED
from src.stages.tasks import SEED_CHECK_FEATURE, STOCHASTIC_MODELS

GLOBAL = ["ml", "neural", "transformer"]


def spec_cohort(fc: pd.DataFrame, inst: pd.DataFrame) -> pd.DataFrame:
    """The main specification's cohort forecasts of the main seed, each in every feature sample of its series."""
    c = main_seed(fc[(fc["scope"] == "cohort") & (fc["setting"] != "default_m") & fc["source"].isin(["run", "s3"])])
    return instances(c, inst)


def stl_contrasts(c: pd.DataFrame, setting: str = "default") -> pd.DataFrame:
    """Direct minus STL-SN and STL-AC of one setting, per forecast, with the STL strategy in ``strategy``."""
    a = arm(c)
    return pd.concat([contrast(c, a, "direct/none", f"{s}/{setting}", ["strategy", "setting"]).assign(strategy=s)
                      for s in ("stl_sn", "stl_ac")], ignore_index=True)


def pool_contrasts(fc: pd.DataFrame, inst: pd.DataFrame) -> pd.DataFrame:
    """Delta_FT, Delta_rand and random minus tercile per forecast (quantity), full and a third of the steps."""
    p = main_seed(fc[fc["source"].isin(["run", "s5"]) & fc["family"].isin(GLOBAL)])
    sc = p["scope"].astype(str)
    parts = []
    for quantity, first, second, models in [("delta_ft", "cohort", "tercile", None), ("delta_rand", "cohort", "random", None),
                                            ("contrast", "random", "tercile", None),
                                            ("delta_ft_third", "cohort", "tercile_third", REDUCED),
                                            ("delta_rand_third", "cohort", "random_third", REDUCED)]:
        sub = p if models is None else p[p["model"].isin(models)]
        parts.append(contrast(sub, sc.loc[sub.index], first, second, ["scope", "group"]).assign(quantity=quantity))
    return pd.concat(parts, ignore_index=True)


def q1(fc, inst, workers):
    X, labels = infer.instance_matrix(stl_contrasts(spec_cohort(fc, inst)), inst, ["family", "strategy"])
    return infer.summary(inst, X, labels, 8, "Q1", workers)


def q2(fc, inst, workers):
    X, labels = infer.instance_matrix(stl_contrasts(spec_cohort(fc, inst)), inst, ["freq", "family", "strategy"])
    return infer.summary(inst, X, labels, 16, "Q2", workers)


def q6(fc, inst, workers):
    d = pool_contrasts(fc, inst)
    X, labels = infer.instance_matrix(d, inst, ["quantity", "family", "strategy"])
    q6b = infer.summary(inst, X, labels, 39, "Q6b", workers)
    if len(q6b) != 39:
        raise ValueError(f"Q6b has {len(q6b)} cells, expected 39")
    ft = d[d["quantity"] == "delta_ft"]
    X, labels = infer.instance_matrix(ft, inst, ["family", "strategy"])
    return {"Q6a": infer.summary(inst, X, labels, 9, "Q6a", workers), "Q6b": q6b}


def q16(fc, inst, workers, tables: Path):
    """Table 5: High minus Low of the median per-series delta (families pooled) per feature and strategy; the
    degenerate feature-frequency pairs left out, as in H4. k = 12 counts the excluded nonlinearity cells too.
    A feature sample has one pool per frequency, so its bootstrap resamples series only (inst restricted to it)."""
    summary = pd.read_parquet(Path(tables).parent / "prepare" / "bucket_summary.parquet")
    degenerate = set(map(tuple, summary.loc[summary["degenerate"], ["feature_name", "frequency"]].to_numpy()))
    d = stl_contrasts(spec_cohort(fc, inst))
    d = d.merge(inst[["series_id", "feature_sample", "bucket"]], on=["series_id", "feature_sample"], how="left",
                validate="many_to_one")
    d = d[[(f, q) not in degenerate for f, q in zip(d["feature_sample"].astype(str), d["freq"].astype(str))]]
    d = d.assign(feature=d["feature_sample"])
    out = []
    for feature, df in d.groupby("feature", sort=True, observed=True):
        sub = inst[inst["feature_sample"] == feature].reset_index(drop=True)      # one pool per frequency: series only
        X, labels = infer.instance_matrix(df, sub, ["feature", "strategy", "bucket"], balanced=True)
        reps = infer.replicates(sub, X, rv.n_boot(12), workers=workers)
        point = rv.HierarchicalBootstrap(sub).point(X)
        for strategy, g in labels.groupby("strategy", sort=True):
            b = dict(zip(g["bucket"], g.index))
            if not {"High", "Low"} <= set(b):
                continue
            cell = pd.DataFrame({"feature": [feature], "strategy": [strategy]})
            out.append(infer.labelled(point[[b["High"]]] - point[[b["Low"]]], reps[:, [b["High"]]] - reps[:, [b["Low"]]],
                                      cell, 12, "Q16"))
    return pd.concat(out, ignore_index=True)


def seed_view(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The forecasts of one seed: the stochastic models from that seed, the others from the main seed (A15)."""
    stoch = df["model"].isin(STOCHASTIC_MODELS)
    return df[(~stoch & df["seed"].isin([MAIN_SEED, -1])) | (stoch & (df["seed"] == seed))]


def _seed_cell(args):
    values, freq, source, B = args
    return rv.seed_bootstrap(values, freq, source, B)


def q14(fc, inst, workers):
    """Tables C3 and C7, the estimand over all seeds: per series, the mean of its deltas under the three seeds,
    then the median; seeds and then series resampled (rv.seed_bootstrap). k = 16 counts E1b's four cells."""
    ev = raw(fc[(fc["scope"] == "cohort") & fc["feature_sample"].isin([SEED_CHECK_FEATURE, "none"])])
    c3 = ev[(ev["setting"] != "default_m") & ev["source"].isin(["run", "s3"])]
    direct = (ev["strategy"] == "direct") & (ev["source"] == "run")
    c7 = ev[direct | ((ev["setting"] == "periodic") & (ev["strategy"] == "stl_sn") & ev["source"].isin(["s4", "s4s"]))]
    c7ac = ev[direct | ((ev["setting"] == "periodic") & (ev["strategy"] == "stl_ac") & (ev["source"] == "s6"))]

    def periodic(strategy):
        return lambda c: contrast(c, c["strategy"].astype(str), "direct", strategy, ["strategy", "setting"]).assign(
            strategy=f"{strategy}_periodic")

    per_seed = []
    for table, base, contrasts in [("C3", c3, lambda c: stl_contrasts(c)), ("C7", c7, periodic("stl_sn")),
                                   ("C7", c7ac, periodic("stl_ac"))]:
        for sd in SEEDS:
            v = instances(seed_view(base, sd), inst)
            d = contrasts(v[v["feature_sample"] == SEED_CHECK_FEATURE])
            s = d.groupby(["series_id", "freq", "family", "strategy"], observed=True)["delta"].mean()
            per_seed.append(s.rename(sd).reset_index().assign(table=table))
    long = pd.concat(per_seed, ignore_index=True)
    wide = long.groupby(["table", "family", "strategy", "series_id", "freq"], observed=True)[SEEDS].first().reset_index()
    if wide[SEEDS].isna().any().any():
        raise ValueError("a series without a delta under every seed")
    source = dict(zip(inst["series_id"], inst["source"]))
    cells = list(wide.groupby(["table", "family", "strategy"], sort=True))
    B = rv.n_boot(16)
    jobs = [(g[SEEDS].to_numpy(), g["freq"].to_numpy(), g["series_id"].map(source).to_numpy(), B) for _, g in cells]
    with ProcessPoolExecutor(max(1, workers)) as pool:
        reps = list(pool.map(_seed_cell, jobs))
    out = []
    for (key, g), r in zip(cells, reps):
        cell = pd.DataFrame([dict(zip(["table", "family", "strategy"], key))])
        row = infer.labelled(np.array([np.median(g[SEEDS].mean(axis=1))]), r[:, None], cell, 16, "Q14")
        row["win_series"] = float((g[SEEDS].mean(axis=1) > 0).mean())
        row["n_series"] = len(g)
        out.append(row)
    return pd.concat(out, ignore_index=True)


def c4(fc, inst, workers):
    """Table C4: ARIMA and ETS with period 1 (the main specification) and period m, against Direct and against
    each other (positive: period 1 better), per model and pooled; k = the 12 labelled cells."""
    c = main_seed(fc[(fc["scope"] == "cohort") & fc["model"].isin(["ARIMA", "ETS"])])
    c = instances(c, inst)
    a = arm(c)
    parts = []
    for s in ("stl_sn", "stl_ac"):
        for setting, period in (("default", "1"), ("default_m", "m")):
            parts.append(contrast(c, a, "direct/none", f"{s}/{setting}", ["strategy", "setting"])
                         .assign(strategy=s, comparison=f"period {period} vs direct"))
        parts.append(contrast(c, a, f"{s}/default_m", f"{s}/default", ["strategy", "setting"])
                     .assign(strategy=s, comparison="period 1 vs period m"))
    d = pd.concat(parts, ignore_index=True)
    pooled = d.assign(model="ARIMA+ETS")
    pooled = pooled[pooled["comparison"] != "period 1 vs period m"]
    X1, l1 = infer.instance_matrix(d, inst, ["model", "strategy", "comparison"])
    keep = ~((l1["comparison"] == "period m vs direct")).to_numpy()            # Table C4 shows period m per model
    X2, l2 = infer.instance_matrix(pooled, inst, ["model", "strategy", "comparison"])
    X, labels = np.concatenate([X1[:, keep], X2], axis=1), pd.concat([l1[keep], l2], ignore_index=True)
    if X.shape[1] != 12:
        raise ValueError(f"Table C4 has {X.shape[1]} labelled cells, expected 12")
    return infer.summary(inst, X, labels, 12, "C4", workers)


def raw(fc: pd.DataFrame) -> pd.DataFrame:
    """The forecasts on the original scale (E3's log-scale forecasts are Q8's alone)."""
    return fc[fc["scale"] == "raw"]


def retuned(c: pd.DataFrame) -> pd.DataFrame:
    """E2's arm, setting periodic_rt: the global models re-tuned for periodic STL-SN (S4's three and S7's six;
    setting periodic_tuned) and ARIMA and ETS under periodic STL-SN, which have no configuration to tune."""
    g = c[(c["strategy"] == "stl_sn") & (c["setting"] == "periodic_tuned")]
    s = c[(c["strategy"] == "stl_sn") & (c["setting"] == "periodic") & (c["family"] == "statistical")]
    rt = pd.concat([g, s], ignore_index=True).assign(setting="periodic_rt")
    out = pd.concat([c.astype({"setting": str}), rt.astype({"setting": str})], ignore_index=True)
    out["setting"] = out["setting"].astype("category")
    return out


def setting_contrasts(fc: pd.DataFrame, inst: pd.DataFrame) -> pd.DataFrame:
    """STL-SN and STL-AC under each setting against Direct and against the same strategy with the default (S4's
    and E1's models of each setting), E2's re-tuned periodic STL-SN against Direct and against the frozen periodic
    arm, and S4's re-tuning of three models (A12); ``strategy``, ``setting`` and ``comparison`` name the cell."""
    c = instances(main_seed(raw(fc[(fc["scope"] == "cohort") & (fc["setting"] != "default_m")
                                   & fc["source"].isin(["run", "s3", "s4", "s6", "s7"])])), inst)
    parts = []
    for strategy in ["stl_sn", "stl_ac"]:
        for label in ["default", "periodic", "s25"]:
            sub = c[c["model"].isin(s4.models_of(label))]
            a = arm(sub)
            parts.append(contrast(sub, a, "direct/none", f"{strategy}/{label}", ["strategy", "setting"])
                         .assign(strategy=strategy, setting=label, comparison="vs direct"))
            if label != "default":
                parts.append(contrast(sub, a, f"{strategy}/default", f"{strategy}/{label}", ["strategy", "setting"])
                             .assign(strategy=strategy, setting=label, comparison="vs default"))
    rt = retuned(c[c["model"].isin(s4.models_of("periodic"))])
    a = arm(rt)
    for first, comparison in (("direct/none", "vs direct"), ("stl_sn/periodic", "vs periodic")):
        parts.append(contrast(rt, a, first, "stl_sn/periodic_rt", ["strategy", "setting"])
                     .assign(strategy="stl_sn", setting="periodic_rt", comparison=comparison))
    sub = c[c["model"].isin(s4.models_of("periodic_tuned")) & c["source"].isin(["run", "s4"])]
    parts.append(contrast(sub, arm(sub), "stl_sn/periodic", "stl_sn/periodic_tuned", ["strategy", "setting"])
                 .assign(strategy="stl_sn", setting="periodic_tuned", comparison="vs periodic"))
    return pd.concat(parts, ignore_index=True)


def q3q4(fc, inst, workers):
    """Q3 (families; k = 48): STL-SN and STL-AC periodic and span 25 against Direct and the default, and E2's
    re-tuned STL-SN against Direct and the frozen periodic arm (40 cells; E2b's 8 did not run). Q4 (models;
    k = 84): STL-SN and STL-AC under the three settings against Direct and E2's re-tuned STL-SN of the nine global
    models (75 cells). A12: S4's re-tuning of three models, re-tuned minus frozen periodic, k = 3."""
    d = setting_contrasts(fc, inst)
    fam = d[d["setting"] != "default"]
    fam = fam[fam["setting"] != "periodic_tuned"]
    X, labels = infer.instance_matrix(fam, inst, ["family", "strategy", "setting", "comparison"])
    q3 = infer.summary(inst, X, labels, 48, "Q3", workers)
    mod = d[(d["comparison"] == "vs direct") & ~((d["setting"] == "periodic_rt") & (d["family"] == "statistical"))]
    X, labels = infer.instance_matrix(mod, inst, ["model", "strategy", "setting", "comparison"])
    q4 = infer.summary(inst, X, labels, 84, "Q4", workers)
    X, labels = infer.instance_matrix(d[d["setting"] == "periodic_tuned"], inst, ["model", "setting", "comparison"])
    a12 = infer.summary(inst, X, labels, 3, "A12", workers)
    for name, df, n in (("Q3", q3, 40), ("Q4", q4, 75), ("A12", a12, 3)):
        if len(df) != n:
            raise ValueError(f"{name} has {len(df)} cells, expected {n}")
    return {"Q3": q3, "Q4": q4, "A12": a12}


def q5(fc, inst, workers, tables: Path):
    """Tables 7 and C5: on the series with stable (phi4 below its first cut) or strong (phi7 in its High tercile)
    seasonality, the best STL-SN model of each global family (lowest mean RelNaive on those series, R2) against the
    three best Direct configurations, under the three settings; per frequency. k = 108. The chosen models are
    those of the frozen tables (checked) and stay fixed in the bootstrap."""
    from src.analysis import review

    run = Path(tables).parent
    x, _ = review.cells(pd.read_parquet(run / "prepare" / "features.parquet"),
                        pd.read_parquet(run / "prepare" / "bucket_summary.parquet"), pd.read_csv(run / E7))
    frozen = pd.read_csv(run / "sensitivity_s4" / "analysis" / "tables" / "s4_r2.csv")
    c = instances(main_seed(fc[(fc["scope"] == "cohort") & (fc["setting"] != "default_m")
                               & fc["source"].isin(["run", "s3", "s4"])]), inst)
    means = c.groupby(["series_id", "model", "strategy", "setting"], observed=True)["rn"].mean()
    a = arm(c).to_numpy()
    model_arm = c["model"].astype(str).to_numpy() + "/" + a
    parts = []
    for setting in ["default", "periodic", "s25"]:
        for cell in ["stable", "strong"]:
            ids = set(x.loc[x[cell], "unique_id"])
            for fam in GLOBAL:
                models = [m for m in s4.models_of(setting) if family(m) == fam]
                score = {m: means.xs((m, "stl_sn", setting), level=["model", "strategy", "setting"])
                         .loc[lambda s: s.index.isin(ids)].mean() for m in models}
                best = min(models, key=score.get)
                want = frozen[(frozen["variant"] == setting) & (frozen["cell"] == cell) & (frozen["family"] == fam)]
                if set(want["stl_sn_model"]) != {best}:
                    raise ValueError(f"{setting} {cell} {fam}: best STL-SN model {best}, frozen {set(want['stl_sn_model'])}")
                for comp in BEST_DIRECT:
                    sub = c[c["series_id"].isin(ids)]
                    d = contrast(sub, model_arm[sub.index], f"{comp}/direct/none", f"{best}/stl_sn/{setting}",
                                 ["family", "model", "strategy", "setting", "seed", "scope", "group"])
                    parts.append(d.assign(setting=setting, cell=cell, family=fam, comparator=comp, stl_sn_model=best))
    d = pd.concat(parts, ignore_index=True)
    X, labels = infer.instance_matrix(d, inst, ["setting", "cell", "freq", "family", "comparator", "stl_sn_model"])
    if X.shape[1] != 108:
        raise ValueError(f"Q5 has {X.shape[1]} cells, expected 108")
    out = infer.summary(inst, X, labels, 108, "Q5", workers)
    out["mean_series"] = rv.HierarchicalBootstrap(inst).point(X, "mean")
    return out


SUBFAMILIES = {"ml-3": ["Ridge", "XGBoost", "RandomForest"], "neural-2": ["NHITS", "LSTM"]}


def q13(fc, inst, workers):
    """A9: the machine learning family without linear regression and the neural family without NLinear, Direct
    minus STL-SN and STL-AC under the three settings; k = 12."""
    c = instances(main_seed(raw(fc[(fc["scope"] == "cohort") & (fc["source"].isin(["run", "s4", "s6"]))])), inst)
    parts = []
    for name, models in SUBFAMILIES.items():
        sub = c[c["model"].isin(models)]
        a = arm(sub)
        for strategy, setting in [(s, x) for s in ("stl_sn", "stl_ac") for x in ("default", "periodic", "s25")]:
            parts.append(contrast(sub, a, "direct/none", f"{strategy}/{setting}", ["strategy", "setting"])
                         .assign(subfamily=name, strategy=strategy, setting=setting))
    X, labels = infer.instance_matrix(pd.concat(parts, ignore_index=True), inst, ["subfamily", "strategy", "setting"])
    return infer.summary(inst, X, labels, 12, "Q13", workers)


def q6c(fc, inst, workers):
    """A10: Table 12 on the evolving-seasonality pools only (one pool per frequency: series resampled within
    frequency and source; the interval depends on the models trained); k = 39."""
    d = pool_contrasts(fc, inst)
    d = d[d["feature_sample"] == SEED_CHECK_FEATURE]
    sub = inst[inst["feature_sample"] == SEED_CHECK_FEATURE].reset_index(drop=True)
    X, labels = infer.instance_matrix(d, sub, ["quantity", "family", "strategy"])
    if X.shape[1] != 39:
        raise ValueError(f"Q6c has {X.shape[1]} cells, expected 39")
    return infer.summary(sub, X, labels, 39, "Q6c", workers)


def q8(fc, inst, workers):
    """E3: ARIMA and ETS on the log scale. STL-SN and STL-AC on log(y) (default and periodic STL) against the
    frozen Direct on the raw series and against Direct on log(y), and Direct on log(y) against Direct on the raw
    series; per model and for the two together (the statistical family); k = 27."""
    c = main_seed(fc[(fc["scope"] == "cohort") & fc["model"].isin(["ARIMA", "ETS"])
                     & (((fc["source"] == "run") & (fc["strategy"] == "direct")) | ((fc["source"] == "s6") & (fc["scale"] == "log")))])
    c = instances(c, inst)
    a = (arm(c) + "/" + c["scale"].astype(str)).to_numpy()
    vary = ["strategy", "setting"]
    parts = []
    for s in ("stl_sn", "stl_ac"):
        for setting in ("default", "periodic"):
            for first, comparison in (("direct/none/raw", "vs direct raw"), ("direct/none/log", "vs direct log")):
                parts.append(contrast(c, a, first, f"{s}/{setting}/log", vary).assign(strategy=f"{s}_log", setting=setting,
                                                                                    comparison=comparison))
    parts.append(contrast(c, a, "direct/none/raw", "direct/none/log", vary).assign(strategy="direct_log", setting="none",
                                                                                 comparison="vs direct raw"))
    d = pd.concat(parts, ignore_index=True)
    d = pd.concat([d, d.assign(model="ARIMA+ETS")], ignore_index=True)
    X, labels = infer.instance_matrix(d, inst, ["model", "strategy", "setting", "comparison"])
    if X.shape[1] != 27:
        raise ValueError(f"Q8 has {X.shape[1]} cells, expected 27")
    return infer.summary(inst, X, labels, 27, "Q8", workers)


def q10(fc, inst, workers):
    """E1, the mechanism: per family and setting (periodic and span 25; the default's component forecasts were
    not stored), the cost of modelling S instead of repeating its last cycle (RN_AC - RN_ii) and the cost of
    modelling T and R apart (RN_ii - RN_SN), positive values a cost; k = 24 counts the default's cells."""
    c = instances(main_seed(raw(fc[(fc["scope"] == "cohort") & fc["source"].isin(["s4", "s6"])
                                   & fc["setting"].isin(["periodic", "s25"])])), inst)
    a = arm(c)
    parts = []
    for setting in ("periodic", "s25"):
        for first, second, parcel in ((f"stl_ac/{setting}", f"stl_ac_ii/{setting}", "model S"),
                                      (f"stl_ac_ii/{setting}", f"stl_sn/{setting}", "split T and R")):
            parts.append(contrast(c, a, first, second, ["strategy", "setting"]).assign(setting=setting, parcel=parcel))
    X, labels = infer.instance_matrix(pd.concat(parts, ignore_index=True), inst, ["family", "setting", "parcel"])
    if X.shape[1] != 16:
        raise ValueError(f"Q10 has {X.shape[1]} cells, expected 16")
    return infer.summary(inst, X, labels, 24, "Q10", workers)


def q24(fc, inst, workers):
    """Round 2: the gap of Section 4.2, non-linear minus linear Delta_FT per strategy (families pooled with equal
    weight within each type, as H6), with simultaneous intervals over the three strategies; k = 3."""
    from src.forecast.registry import LINEARITY

    d = pool_contrasts(fc, inst)
    d = d[d["quantity"] == "delta_ft"]
    d = d.assign(linearity=d["model"].astype(str).map(LINEARITY))
    v = (d.groupby(["series_id", "feature_sample", "strategy", "linearity", "family"], observed=True)["delta"].mean()
         .groupby(level=["series_id", "feature_sample", "strategy", "linearity"], observed=True).mean().unstack("linearity"))
    g = (v["nonlinear"] - v["linear"]).rename("delta").reset_index().assign(family="all")
    X, labels = infer.instance_matrix(g, inst, ["strategy"])
    return infer.summary(inst, X, labels, 3, "Q24", workers)


def q25(fc, inst, workers):
    """Round 2: the per-seed cells of Tables C3 and C7 with the labels of the revision (one pool per frequency on the
    evolving-seasonality sample, so the series are resampled); k = 36 (C3: 8 cells x 3 seeds; C7: 4 x 3)."""
    ev = raw(fc[(fc["scope"] == "cohort") & fc["feature_sample"].isin([SEED_CHECK_FEATURE, "none"])])
    c3 = ev[(ev["setting"] != "default_m") & ev["source"].isin(["run", "s3"])]
    direct = (ev["strategy"] == "direct") & (ev["source"] == "run")
    c7 = ev[direct | ((ev["setting"] == "periodic") & (ev["strategy"] == "stl_sn") & ev["source"].isin(["s4", "s4s"]))]
    sub = inst[inst["feature_sample"] == SEED_CHECK_FEATURE].reset_index(drop=True)
    parts = []
    for sd in SEEDS:
        v = instances(seed_view(c3, sd), inst)
        parts.append(stl_contrasts(v[v["feature_sample"] == SEED_CHECK_FEATURE]).assign(table="C3", seed=sd))
        v = instances(seed_view(c7, sd), inst)
        v = v[v["feature_sample"] == SEED_CHECK_FEATURE]
        parts.append(contrast(v, v["strategy"].astype(str), "direct", "stl_sn", ["strategy", "setting"])
                     .assign(strategy="stl_sn_periodic", table="C7", seed=sd))
    d = pd.concat(parts, ignore_index=True).rename(columns={"seed": "run_seed"})
    d = d.assign(seed_cell=d["run_seed"].astype(str))
    X, labels = infer.instance_matrix(d, sub, ["table", "family", "strategy", "seed_cell"])
    if X.shape[1] != 36:
        raise ValueError(f"Q25 has {X.shape[1]} cells, expected 36")
    return infer.summary(sub, X, labels, 36, "Q25", workers)


QUESTIONS = {"Q1": q1, "Q2": q2, "Q6": q6, "Q14": q14, "Q16": q16, "C4": c4, "Q3Q4": q3q4, "Q5": q5, "Q13": q13,
             "Q6c": q6c, "Q8": q8, "Q10": q10, "Q24": q24, "Q25": q25}
NEEDS_TABLES = {"Q16", "Q5"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.analysis.v2.questions", description=__doc__.splitlines()[0])
    parser.add_argument("--tables", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--only", default=",".join(QUESTIONS))
    parser.add_argument("--workers", type=int, default=1)
    a = parser.parse_args(argv)
    fc, _, inst = load(a.tables)
    a.out.mkdir(parents=True, exist_ok=True)
    for name in a.only.split(","):
        result = QUESTIONS[name](fc, inst, a.workers, *([a.tables] if name in NEEDS_TABLES else []))
        for q, df in (result.items() if isinstance(result, dict) else [(name, result)]):
            path = a.out / f"{q}.csv"
            if path.exists():
                raise FileExistsError(f"{path} exists; question outputs are never overwritten")
            df.to_csv(path, index=False, float_format="%.10g")
            print(q, len(df), "cells:", df["label"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
