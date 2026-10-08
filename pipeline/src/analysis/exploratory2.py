"""Exploratory analyses E7-E9 of a gated run: NOT pre-registered (protocol change log v2.4).

They answer a review of the paper draft (8 October 2026). The confirmatory plan is protocol
Section 5; nothing here tests one of its hypotheses again or changes one of its labels. Per-series
values follow the run's aggregation rules (D27: a series' value is its mean over the instances of
a group; cohort scope, main seed; families pooled with equal weight).

- E7 seasonal strength as a moderator. The strength F_s = max(0, 1 - Var(R) / Var(S + R)) of one
  robust STL fit (D9) on each sampled series' history before the first test window: its Spearman
  correlation with per-series Delta_STL beside that of evolving seasonality; Delta_STL by strength
  tercile with the Jonckheere-Terpstra trend test; the table of evolving-seasonality tercile (the
  prepare bundle's cuts) by strength tercile (cut per frequency over the sampled series); a median
  regression of per-series Delta_STL on evolving seasonality, strength, log length and source; tests
  inside the Low evolving-seasonality tercile and inside the cell of Low evolving seasonality and High
  strength; the correlation of every feature with series length.
- E8 the test behind each label. For every Wilcoxon test of H1, H2, H3, H5, H6, A13 (STL per
  model), S1 and S2, on the per-series values those analyses test: the sign test and the
  Hodges-Lehmann estimate, Holm within each family, and the labels the pre-specified rule would give
  with (a) the sign test and the median and (b) the Wilcoxon test and the Hodges-Lehmann estimate.
- E9 Direct + STL-SN. The equal-weight average of a model's Direct and STL-SN forecasts of the same
  series and window, scored like any forecast, against Direct and against STL-SN; shares of RelNaive
  above 2.

    python -m src.analysis.exploratory2 --config C --run RUN --data-dir D --out OUT
"""
from __future__ import annotations

import argparse
import contextlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from src.analysis import hypotheses as hy
from src.analysis import secondary
from src.analysis import stats
from src.analysis.data import AnalysisError, build_instances, load_gated, per_series, strategy_contrast
from src.analysis.exploratory import FAMILY_ORDER, series_by_strategy
from src.data.frozen import DEFAULT_MANIFEST, load_manifest, sha256_file
from src.data.load_m_datasets import load_dataset_pair
from src.features.errors import FeatureUndefined
from src.features.mstl_features import decompose_series, seasonal_strength
from src.forecast.registry import FAMILIES
from src.metrics.rel_naive import rel_naive
from src.stages import sensitivity as s1mod
from src.stages import sensitivity2 as s2mod
from src.stages.io import code_commit, sha256_json, utc_now, write_json

STATUS = "exploratory: not pre-registered (protocol change log v2.4; Section 5 is the confirmatory plan)"
RULE_FEATURE = "feature_evolving_seasonality"
CAP = 10.0
TERCILES = ["Low", "Medium", "High"]
E8_FAMILIES = ("H1", "H2", "H3", "H5", "H6", "A13_STL_PER_MODEL", "S1", "S2-vs-direct", "S2-vs-run-stl-sn")


# --- statistics -----------------------------------------------------------------------------------------

def hodges_lehmann(x) -> float:
    """One-sample Hodges-Lehmann estimate: the median of the Walsh averages (x_i + x_j) / 2, i <= j.
    The k-th smallest average is found by bisection on how many averages lie below a value, which
    takes O(n log n) per step instead of forming the n (n + 1) / 2 averages."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0:
        return float("nan")
    total = n * (n + 1) // 2
    first = np.arange(n)

    def count_le(v: float) -> int:          # averages (x_i + x_j) / 2 <= v with i <= j
        return int(np.maximum(np.searchsorted(x, 2 * v - x, side="right") - first, 0).sum())

    def kth(k: int) -> float:
        lo, hi = x[0], x[-1]
        if count_le(lo) >= k:
            return float(lo)
        for _ in range(200):
            mid = lo + (hi - lo) / 2
            if mid <= lo or mid >= hi:
                break
            if count_le(mid) >= k:
                hi = mid
            else:
                lo = mid
        return float(hi)

    if total % 2:
        return kth((total + 1) // 2)
    return (kth(total // 2) + kth(total // 2 + 1)) / 2


def sign_test(x) -> dict:
    """Two-sided sign test of per-series values against zero (zeros dropped)."""
    x = np.asarray(x, dtype=float)
    nonzero = x[x != 0]
    k, n = int((nonzero > 0).sum()), len(nonzero)
    return {"n_positive": k, "n_negative": n - k, "p_sign": float(sps.binomtest(k, n, 0.5).pvalue) if n else 1.0}


# --- E8 -------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def capture_tests(store: dict):
    """Record the values every Wilcoxon test of the hypotheses module receives, keyed by contrast id,
    while skipping the tests themselves (the confirmatory tables already hold them)."""
    original = hy._test

    def record(x, base_seed, contrast_id):
        x = np.asarray(x, dtype=float)
        store[contrast_id] = x.copy()
        middle = float(np.median(x)) if len(x) else float("nan")
        return {"n_series": len(x), "n_nonzero": int((x != 0).sum()), "statistic": np.nan, "p": 1.0, "rank_biserial": 0.0,
                "median": middle, "mean": float(np.mean(x)) if len(x) else float("nan"), "ci_low": np.nan, "ci_high": np.nan}

    hy._test = record
    try:
        yield store
    finally:
        hy._test = original


def confirmatory_series(main: pd.DataFrame, base_seed: int) -> dict[str, np.ndarray]:
    """The per-series values behind H1, H2, H3, H5, H6 and the per-model STL table (A13)."""
    store: dict[str, np.ndarray] = {}
    with capture_tests(store):
        hy.h1(main, base_seed)
        hy.h2(main, base_seed)
        hy.h3(main, base_seed)
        hy.h5(main, base_seed)
        hy.h6(main, base_seed)
        secondary.a13_per_model(main, base_seed)
    return {k: v for k, v in store.items() if k.split("|")[0] in E8_FAMILIES}


def sensitivity_series(run_dir: Path, main: pd.DataFrame, tables: dict, main_seed: int) -> dict[str, np.ndarray]:
    """The per-series values behind the S1 and S2 tests, built as their analyses build them."""
    samples, summary, buckets = tables["samples"], tables["bucket_summary"], tables["buckets"]
    specs = [("S1", s1mod.ROOT, s1mod.eval_tasks(summary, samples),
              [(m, v) for m in s1mod.S1_MODELS.values() for v in s1mod.VARIANTS], {"direct": "S1"}),
             ("S2", s2mod.ROOT, s2mod.eval_tasks(summary, samples), s2mod.CONTRASTS,
              {"direct": "S2-vs-direct", "stl_sn": "S2-vs-run-stl-sn"})]
    out = {}
    for name, root, tasks, contrasts, families in specs:
        rows, facts = s1mod._s1_rows(run_dir, tasks, root)
        if facts["missing"] or facts["changed"] or facts["listed_failures"]:
            raise AnalysisError(f"{name}: incomplete or changed task outputs")
        rows = rows.assign(strategy="stl_sn_" + rows["stl_variant"].astype(str))
        variant_inst = build_instances(rows, rows.iloc[0:0], buckets, main_seed)
        variant_inst = variant_inst[variant_inst["main_seed"]]
        models = sorted({m for m, _ in contrasts})
        base = main[(main["scope"].astype(str) == "cohort") & main["model"].astype(str).isin(models)
                    & main["strategy"].astype(str).isin(["direct", "stl_sn"])]
        inst = pd.concat([base.astype({"strategy": str}), variant_inst[base.columns].astype({"strategy": str})], ignore_index=True)
        for first, family_id in families.items():
            for model, variant in contrasts:
                d = strategy_contrast(inst[inst["model"].astype(str) == model], first, f"stl_sn_{variant}")
                out[f"{family_id}|{model}|{variant}"] = per_series(d, ["model"])["delta"].to_numpy()
    return out


def e8_table(series: dict[str, np.ndarray]) -> pd.DataFrame:
    """E8: per contrast the Wilcoxon and sign tests, median and Hodges-Lehmann estimate; Holm within
    each family; the pre-specified label and the labels under the two alternatives."""
    rows = []
    for contrast_id, x in series.items():
        family_id = contrast_id.split("|")[0]
        rows.append({"test_family": family_id, "contrast": contrast_id[len(family_id) + 1:], "n_series": len(x),
                     **{k: v for k, v in stats.signed_rank(x).items() if k in ("n_nonzero", "p", "rank_biserial")},
                     **sign_test(x), "median": float(np.median(x)), "hodges_lehmann": hodges_lehmann(x),
                     "mean": float(np.mean(x))})
    df = pd.DataFrame(rows).rename(columns={"p": "p_wilcoxon"})
    parts = []
    for _, g in df.groupby("test_family", sort=False):
        g = g.copy()
        g["p_holm_wilcoxon"] = stats.holm(g["p_wilcoxon"].to_numpy())
        g["p_holm_sign"] = stats.holm(g["p_sign"].to_numpy())
        g["n_tests_in_family"] = len(g)
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)
    df["label"] = [stats.d26_label(p, e) for p, e in zip(df["p_holm_wilcoxon"], df["median"])]
    df["label_sign_median"] = [stats.d26_label(p, e) for p, e in zip(df["p_holm_sign"], df["median"])]
    df["label_wilcoxon_hl"] = [stats.d26_label(p, e) for p, e in zip(df["p_holm_wilcoxon"], df["hodges_lehmann"])]
    df["differs"] = (df["label"] != df["label_sign_median"]) | (df["label"] != df["label_wilcoxon_hl"])
    df["signs_disagree"] = np.sign(df["median"]) != np.sign(df["hodges_lehmann"])
    return df


# --- E7 -------------------------------------------------------------------------------------------------

def strength_table(config: dict, data_dir: Path, features: pd.DataFrame, ids: pd.DataFrame,
                   manifest_path: Path = DEFAULT_MANIFEST) -> pd.DataFrame:
    """F_s of one robust STL fit (D9) on each series' history before the first test window."""
    ends = features.set_index(["unique_id", "frequency"])["history_end_t"]
    out = []
    for frequency, want in ids.groupby("frequency"):
        fcfg = config["frequencies"][frequency]
        m = int(fcfg["season_length"])
        data = load_dataset_pair({"frequency": frequency, **fcfg}, data_dir, load_manifest(manifest_path))
        data = data[data["unique_id"].isin(set(want["unique_id"]))]
        for uid, g in data.groupby("unique_id", sort=True):
            y = g.sort_values("t")["y"].to_numpy(dtype=float)[: int(ends.loc[(uid, frequency)])]
            dec = decompose_series(y, m)
            try:
                fs = seasonal_strength(dec["seasonal"], dec["residual"])
            except FeatureUndefined:
                fs = np.nan
            out.append({"unique_id": uid, "frequency": frequency, "seasonal_strength": fs, "history_length": len(y)})
    return pd.DataFrame(out)


def series_deltas(table: pd.DataFrame) -> pd.DataFrame:
    """Per-series Delta_STL per family and pooled (families weighted equally), long format."""
    t = table.assign(stl_sn=table["direct"] - table["stl_sn"], stl_ac=table["direct"] - table["stl_ac"])
    fam = t.groupby(["unique_id", "frequency", "family"])[["stl_sn", "stl_ac"]].mean().reset_index()
    pooled = fam.groupby(["unique_id", "frequency"])[["stl_sn", "stl_ac"]].mean().reset_index().assign(family="pooled")
    both = pd.concat([fam, pooled], ignore_index=True)
    return both.melt(id_vars=["unique_id", "frequency", "family"], value_vars=["stl_sn", "stl_ac"],
                     var_name="strategy", value_name="delta")


def _family_order(df: pd.DataFrame) -> pd.DataFrame:
    order = {"pooled": -1, **FAMILY_ORDER}
    return df.sort_values(["family"], key=lambda s: s.map(order), kind="mergesort").reset_index(drop=True)


def moderator_frame(deltas: pd.DataFrame, strength: pd.DataFrame, features: pd.DataFrame,
                    bucket_summary: pd.DataFrame) -> pd.DataFrame:
    f = features[["unique_id", "frequency", "source_dataset", "series_length", RULE_FEATURE]]
    d = deltas.merge(strength, on=["unique_id", "frequency"]).merge(f, on=["unique_id", "frequency"])
    cuts = bucket_summary[bucket_summary["feature_name"] == RULE_FEATURE].set_index("frequency")
    c1, c2 = d["frequency"].map(cuts["tercile_cut_1"]), d["frequency"].map(cuts["tercile_cut_2"])
    d["evolving_tercile"] = np.select([d[RULE_FEATURE] <= c1, d[RULE_FEATURE] <= c2], TERCILES[:2], TERCILES[2])
    one = strength.dropna(subset=["seasonal_strength"])
    cut = {fq: np.quantile(g["seasonal_strength"], [1 / 3, 2 / 3]) for fq, g in one.groupby("frequency")}
    lo, hi = d["frequency"].map(lambda q: cut[q][0]), d["frequency"].map(lambda q: cut[q][1])
    d["strength_tercile"] = np.select([d["seasonal_strength"] <= lo, d["seasonal_strength"] <= hi], TERCILES[:2], TERCILES[2])
    d.loc[d["seasonal_strength"].isna(), "strength_tercile"] = None
    return d


def _frequencies(d: pd.DataFrame):
    yield "all", d
    for fq, g in d.groupby("frequency", sort=True):
        yield fq, g


def correlations(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, strategy), g0 in d.groupby(["family", "strategy"], sort=False):
        for fq, g in _frequencies(g0):
            g = g.dropna(subset=["seasonal_strength"])
            rows.append({"family": family, "strategy": strategy, "frequency": fq, "n_series": len(g),
                         "rho_strength": float(sps.spearmanr(g["seasonal_strength"], g["delta"])[0]),
                         "rho_evolving": float(sps.spearmanr(g[RULE_FEATURE], g["delta"])[0]),
                         "rho_evolving_strength": float(sps.spearmanr(g[RULE_FEATURE], g["seasonal_strength"])[0]),
                         "rho_evolving_length": float(sps.spearmanr(g[RULE_FEATURE], g["series_length"])[0])})
    return _family_order(pd.DataFrame(rows))


def strength_terciles(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, strategy), g in d.dropna(subset=["strength_tercile"]).groupby(["family", "strategy"], sort=False):
        groups = [g.loc[g["strength_tercile"] == t, "delta"].to_numpy() for t in TERCILES]
        if min(len(x) for x in groups) < 2:
            continue
        jt = stats.jonckheere_terpstra(groups)
        rows.append({"family": family, "strategy": strategy, **{f"n_{t.lower()}": len(x) for t, x in zip(TERCILES, groups)},
                     **{f"median_{t.lower()}": float(np.median(x)) for t, x in zip(TERCILES, groups)},
                     "high_minus_low": float(np.median(groups[2]) - np.median(groups[0])), "z": jt["z"], "p": jt["p"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["p_holm"] = stats.holm(df["p"].to_numpy())
    df["label"] = [stats.d26_label(p, e) for p, e in zip(df["p_holm"], df["high_minus_low"])]
    return _family_order(df)


def cross_table(d: pd.DataFrame) -> pd.DataFrame:
    g = d.dropna(subset=["strength_tercile"])
    rows = []
    for (family, strategy), h in g.groupby(["family", "strategy"], sort=False):
        for fq, k in _frequencies(h):
            for (ev, st), cell in k.groupby(["evolving_tercile", "strength_tercile"]):
                rows.append({"family": family, "strategy": strategy, "frequency": fq, "evolving_tercile": ev,
                             "strength_tercile": st, "n_series": len(cell), "median": float(cell["delta"].median()),
                             "share_positive": float((cell["delta"] > 0).mean())})
    return _family_order(pd.DataFrame(rows))


def median_regression(d: pd.DataFrame) -> pd.DataFrame:
    """Median regression of pooled per-series Delta_STL on standardised predictors, per frequency and
    strategy: evolving seasonality alone, strength alone, and both with log length and source."""
    import statsmodels.api as sm

    rows = []
    g0 = d[(d["family"] == "pooled")].dropna(subset=["seasonal_strength"])
    for (strategy, fq), g in g0.groupby(["strategy", "frequency"], sort=True):
        if len(g) < 20:                     # too few series for a regression with four predictors
            continue
        x = pd.DataFrame({"evolving_seasonality": g[RULE_FEATURE], "seasonal_strength": g["seasonal_strength"],
                          "log_length": np.log(g["series_length"]),
                          "m3": g["source_dataset"].astype(str).str.startswith("M3").astype(float)})
        z = (x - x.mean()) / x.std(ddof=0)
        z["m3"] = x["m3"]
        for model, cols in (("evolving only", ["evolving_seasonality"]), ("strength only", ["seasonal_strength"]),
                            ("all", ["evolving_seasonality", "seasonal_strength", "log_length", "m3"])):
            fit = sm.QuantReg(g["delta"].to_numpy(), sm.add_constant(z[cols])).fit(q=0.5)
            for c in cols:
                rows.append({"strategy": strategy, "frequency": fq, "model": model, "predictor": c,
                             "coefficient": float(fit.params[c]), "se": float(fit.bse[c]), "p": float(fit.pvalues[c]),
                             "n_series": len(g)})
    return pd.DataFrame(rows)


def cell_tests(d: pd.DataFrame, base_seed: int) -> pd.DataFrame:
    """Tests inside the Low evolving-seasonality tercile and inside the cell of Low evolving
    seasonality and High strength, per family and strategy (Holm over the table)."""
    rows = []
    for (family, strategy), g in d.groupby(["family", "strategy"], sort=False):
        for cell, mask in (("stable", g["evolving_tercile"] == "Low"),
                           ("stable and strong", (g["evolving_tercile"] == "Low") & (g["strength_tercile"] == "High"))):
            x = g.loc[mask, "delta"].to_numpy()
            if len(x) < 2:
                continue
            lo, hi = stats.median_ci(x, base_seed, f"E7|{family}|{strategy}|{cell}")
            rows.append({"family": family, "strategy": strategy, "cell": cell, "n_series": len(x),
                         "median": float(np.median(x)), "ci_low": lo, "ci_high": hi, "hodges_lehmann": hodges_lehmann(x),
                         "share_positive": float((x > 0).mean()), "p_wilcoxon": stats.signed_rank(x)["p"], **sign_test(x)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["p_holm_wilcoxon"] = stats.holm(df["p_wilcoxon"].to_numpy())
    df["p_holm_sign"] = stats.holm(df["p_sign"].to_numpy())
    df["label"] = [stats.d26_label(p, e) for p, e in zip(df["p_holm_wilcoxon"], df["median"])]
    df["label_sign_median"] = [stats.d26_label(p, e) for p, e in zip(df["p_holm_sign"], df["median"])]
    return _family_order(df)


def length_correlations(features: pd.DataFrame) -> pd.DataFrame:
    names = [c for c in features.columns if c.startswith("feature_")]
    rows = []
    for fq, g in features[features["eligible"]].groupby("frequency", sort=True):
        for name in names:
            rows.append({"frequency": fq, "feature": name, "n_series": int(g[name].notna().sum()),
                         "rho_length": float(sps.spearmanr(g[name], g["series_length"], nan_policy="omit")[0])})
    return pd.DataFrame(rows)


# --- E9 -------------------------------------------------------------------------------------------------

def average_pair(direct: pd.DataFrame, stl_sn: pd.DataFrame) -> pd.DataFrame:
    """RelNaive of Direct, STL-SN and their equal-weight average for the same series and windows
    (rows with y_test, yhat and yhat_naive); capped and uncapped."""
    keys = ["unique_id", "window"]
    j = direct[keys + ["y_test", "yhat", "yhat_naive"]].merge(stl_sn[keys + ["yhat"]], on=keys, suffixes=("", "_stl_sn"),
                                                              validate="one_to_one")
    out = []
    for r in j.itertuples(index=False):
        y, naive = np.asarray(r.y_test, float), np.asarray(r.yhat_naive, float)
        a, b = np.asarray(r.yhat, float), np.asarray(r.yhat_stl_sn, float)
        avg = (a + b) / 2
        out.append({"unique_id": r.unique_id, "window": r.window,
                    **{f"rn_{arm}": rel_naive(y, f, naive, clip=CAP) for arm, f in (("direct", a), ("stl_sn", b), ("average", avg))},
                    **{f"rn_raw_{arm}": rel_naive(y, f, naive) for arm, f in (("direct", a), ("stl_sn", b), ("average", avg))}})
    return pd.DataFrame(out)


def _task_files(record: dict, run_dir: Path, main_seed: int) -> dict[tuple, dict[str, Path]]:
    """Direct and STL-SN task outputs of the cohort scope and main seed, paired by everything else,
    each checked against the hash in the merge record."""
    pairs: dict[tuple, dict[str, Path]] = {}
    pattern_global = re.compile(r"^eval\|(?P<feature>[^|]+)\|(?P<freq>[^|]+)\|(?P<strategy>direct|stl_sn)\|(?P<model>[^|]+)\|cohort\|seed(?P<seed>\d+)$")
    pattern_stat = re.compile(r"^stat\|(?P<model>[^|]+)\|(?P<freq>[^|]+)\|(?P<strategy>direct|stl_sn)\|(?P<chunk>chunk\d+)$")
    for task_id, entry in record["tasks"].items():
        m = pattern_global.match(task_id)
        if m and int(m["seed"]) == main_seed:
            key = (m["model"], m["freq"], m["feature"])
        else:
            m = pattern_stat.match(task_id)
            if not m:
                continue
            key = (m["model"], m["freq"], m["chunk"])
        path = Path(run_dir) / entry["file"]
        if sha256_file(path) != entry["sha256"]:
            raise AnalysisError(f"{task_id}: output does not match the merge record")
        pairs.setdefault(key, {})[m["strategy"]] = path
    incomplete = [k for k, v in pairs.items() if set(v) != {"direct", "stl_sn"}]
    if incomplete:
        raise AnalysisError(f"unpaired Direct / STL-SN task outputs: {incomplete[:3]}")
    return pairs


def averaged_forecasts(record: dict, run_dir: Path, main_seed: int) -> pd.DataFrame:
    cols = ["unique_id", "window", "family", "model", "frequency", "status", "y_test", "yhat", "yhat_naive"]
    out = []
    for (model, freq, part), files in sorted(_task_files(record, run_dir, main_seed).items()):
        d, s = (pd.read_parquet(files[k], columns=cols) for k in ("direct", "stl_sn"))
        d, s = d[d["status"] == "trained"], s[s["status"] == "trained"]
        scored = average_pair(d, s)
        out.append(scored.assign(family=d["family"].iloc[0], model=model, frequency=freq, part=part))
    return pd.concat(out, ignore_index=True)


def average_tables(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per model and per family: per-series means, tests of Direct - average and STL-SN - average
    (> 0: the average is better), shares of uncapped RelNaive above 2 (forecast level)."""
    s = scored.assign(vs_direct=scored["rn_direct"] - scored["rn_average"], vs_stl_sn=scored["rn_stl_sn"] - scored["rn_average"])

    def summarise(g: pd.DataFrame, per_series_values: pd.DataFrame) -> dict:
        row = {f"share_rn_gt2_{arm}": float((g[f"rn_raw_{arm}"] > 2).mean()) for arm in ("direct", "stl_sn", "average")}
        row.update({f"mean_rn_{arm}": float(g[f"rn_{arm}"].mean()) for arm in ("direct", "stl_sn", "average")})
        row["forecasts"] = len(g)
        for col in ("vs_direct", "vs_stl_sn"):
            x = per_series_values[col].to_numpy()
            row.update({f"{col}_median": float(np.median(x)), f"{col}_hl": hodges_lehmann(x), f"{col}_mean": float(np.mean(x)),
                        f"{col}_p_wilcoxon": stats.signed_rank(x)["p"], f"{col}_p_sign": sign_test(x)["p_sign"],
                        f"{col}_share_better": float((x > 0).mean())})
        row["n_series"] = len(per_series_values)
        return row

    by_model = []
    for (fam, model), g in s.groupby(["family", "model"], sort=False):
        ps = g.groupby("unique_id")[["vs_direct", "vs_stl_sn"]].mean()
        by_model.append({"family": fam, "model": model, **summarise(g, ps)})
    by_family = []
    for fam, g in s.groupby("family", sort=False):
        ps = g.groupby(["unique_id", "model"])[["vs_direct", "vs_stl_sn"]].mean().groupby("unique_id").mean()
        by_family.append({"family": fam, **summarise(g, ps)})
    out = []
    for df in (pd.DataFrame(by_model), pd.DataFrame(by_family)):
        for col in ("vs_direct", "vs_stl_sn"):
            df[f"{col}_p_holm"] = stats.holm(df[f"{col}_p_wilcoxon"].to_numpy())
            df[f"{col}_label"] = [stats.d26_label(p, e) for p, e in zip(df[f"{col}_p_holm"], df[f"{col}_median"])]
        out.append(df.sort_values("family", key=lambda c: c.map(FAMILY_ORDER), kind="mergesort").reset_index(drop=True))
    return out[0], out[1]


# --- run ------------------------------------------------------------------------------------------------

def run_exploratory2(run_dir: Path, config: dict, data_dir: Path, out: Path, manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise AnalysisError(f"{out} is not empty; exploratory outputs are never overwritten")
    commit = code_commit()
    data = load_gated(run_dir, manifest_path)
    s0 = data.main_seed
    tables = data.tables
    features = tables["features"]
    ids = tables["samples"][["unique_id", "frequency"]].drop_duplicates()

    strength = strength_table(config, data_dir, features, ids, manifest_path)
    d = moderator_frame(series_deltas(series_by_strategy(data.main)), strength, features, tables["bucket_summary"])
    series = {**confirmatory_series(data.main, s0), **sensitivity_series(run_dir, data.main, tables, s0)}
    e9_model, e9_family = average_tables(averaged_forecasts(data.record, run_dir, s0))
    outputs = {
        "e7_strength": strength,
        "e7_correlations": correlations(d),
        "e7_strength_terciles": strength_terciles(d),
        "e7_cross": cross_table(d),
        "e7_median_regression": median_regression(d),
        "e7_cell_tests": cell_tests(d, s0),
        "e7_length_correlations": length_correlations(features),
        "e8_tests": e8_table(series),
        "e9_average_by_model": e9_model,
        "e9_average_by_family": e9_family,
    }
    (out / "tables").mkdir(parents=True)
    for name, df in outputs.items():
        df.to_csv(out / "tables" / f"{name}.csv", index=False)
    record = {"status": STATUS, "code_commit": commit, "created_at_utc": utc_now(),
              "merge_result_sha256": data.record["result_sha256"], "gates_created_at_utc": data.gates["created_at_utc"],
              "tables": {name: sha256_json(df.to_dict(orient="list")) for name, df in outputs.items()}}
    write_json(out / "exploratory2.json", record)
    return {"out": str(out), "tables": len(outputs), "contrasts": len(series)}


def main(argv: list[str] | None = None) -> int:
    from src.utils import read_yaml

    parser = argparse.ArgumentParser(prog="python -m src.analysis.exploratory2", description=STATUS)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    print(run_exploratory2(args.run, read_yaml(args.config), args.data_dir, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
