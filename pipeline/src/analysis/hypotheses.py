"""Confirmatory hypotheses H1-H7 (protocol Section 5.4; D23, D26, D27).

Every test is two-sided at alpha = 0.05 with Holm's correction within the stated family;
each result carries the median per-series delta with a 95% bootstrap CI (2,000 resamples
of series), the matched-pairs rank-biserial correlation where the test is paired, and
its D26 label. ``family_id`` names the correction family and seeds the bootstraps, so a
rerun gives identical intervals.

Orientation: a positive delta means the treatment (STL; specialisation) helps; for H3 a
positive delta RelNaive(first) - RelNaive(second) favours the second strategy; for H2 it
means a larger STL effect in the first family; for H6 a larger specialisation effect in
the non-linear models.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from src.analysis import stats
from src.analysis.data import PRIMARY, describe_groups, ft_deltas, per_series, stl_deltas, strategy_contrast
from src.forecast.registry import FAMILIES, GLOBAL_MODELS, LINEARITY, MODELS, STRATEGIES, family

BUCKETS = ["Low", "Medium", "High"]
STRATEGY_PAIRS = [("direct", "stl_sn"), ("direct", "stl_ac"), ("stl_sn", "stl_ac")]


def _finish(df: pd.DataFrame, family_id: str, effect: str = "median", p: str = "p") -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["p_holm"] = stats.holm(df[p].to_numpy())
    df["label"] = [stats.d26_label(ph, e) for ph, e in zip(df["p_holm"], df[effect])]
    df.insert(0, "test_family", family_id)
    df["n_tests_in_family"] = len(df)
    return df


def _test(x: np.ndarray, base_seed: int, contrast_id: str) -> dict:
    lo, hi = stats.median_ci(x, base_seed, contrast_id)
    return {**stats.signed_rank(x), "median": float(np.median(x)), "mean": float(np.mean(x)), "ci_low": lo, "ci_high": hi}


def one_sample(ps: pd.DataFrame, by: list[str], base_seed: int, family_id: str, value: str = "delta") -> pd.DataFrame:
    """Wilcoxon of the per-series values of each group against zero; Holm over the groups."""
    out = []
    for key, g in ps.groupby(by, observed=True, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        out.append({**dict(zip(by, key)), **_test(g[value].to_numpy(), base_seed, "|".join([family_id, *map(str, key)]))})
    return _finish(pd.DataFrame(out), family_id)


def _with_descriptives(tests: pd.DataFrame, deltas: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    desc = describe_groups(deltas, by).rename(columns=lambda c: c if c in by else f"instance_{c}")
    return tests.merge(desc, on=by, how="left") if not tests.empty else tests


def h1(inst: pd.DataFrame, base_seed: int, family_id: str = "H1", metric: str = PRIMARY, how: str = "mean") -> pd.DataFrame:
    """Delta_STL per family and STL strategy, cohort scope (Holm over 4 x 2)."""
    d = stl_deltas(inst[inst["scope"] == "cohort"], metric)
    tests = one_sample(per_series(d, ["family", "strategy"], how=how), ["family", "strategy"], base_seed, family_id)
    return _with_descriptives(tests, d, ["family", "strategy"])


def h2(inst: pd.DataFrame, base_seed: int, family_id: str = "H2") -> pd.DataFrame:
    """Delta_STL between families on the same series (Holm over 6 pairs x 2 strategies)."""
    ps = per_series(stl_deltas(inst[inst["scope"] == "cohort"]), ["family", "strategy"])
    out = []
    for strategy in ("stl_sn", "stl_ac"):
        s = ps[ps["strategy"] == strategy].pivot(index="unique_id", columns="family", values="delta")
        for a, b in combinations([f for f in FAMILIES if f in s.columns], 2):
            diff = (s[a] - s[b]).dropna().to_numpy()
            out.append({"strategy": strategy, "family_a": a, "family_b": b,
                        **_test(diff, base_seed, f"{family_id}|{strategy}|{a}|{b}")})
    df = _finish(pd.DataFrame(out), family_id)
    df["larger_stl_effect"] = np.where(df["label"].isin(["gain", "loss"]), np.where(df["median"] > 0, df["family_a"], df["family_b"]), "")
    return df


def h3(inst: pd.DataFrame, base_seed: int, family_id: str = "H3") -> pd.DataFrame:
    """Strategies within each family on RelNaive (Holm over 3 pairs x 4 families); the MCM cells."""
    cohort = inst[inst["scope"] == "cohort"]
    out = []
    for first, second in STRATEGY_PAIRS:
        d = strategy_contrast(cohort, first, second)
        ps = per_series(d, ["family"])
        for fam, g in ps.groupby("family", observed=True, sort=True):
            x = g["delta"].to_numpy()
            out.append({"family": fam, "first": first, "second": second,
                        "series_second_better": int((x > 0).sum()), "series_tied": int((x == 0).sum()),
                        "series_first_better": int((x < 0).sum()),
                        "n_excluded": int(d.loc[d["family"] == fam, "delta"].isna().sum()),
                        **_test(x, base_seed, f"{family_id}|{fam}|{first}|{second}")})
    df = _finish(pd.DataFrame(out), family_id)
    df["favours"] = np.where(df["label"].isin(["gain", "loss"]), np.where(df["median"] > 0, df["second"], df["first"]), "")
    return df


def _degenerate(tables: dict) -> dict[str, set[str]]:
    s = tables["bucket_summary"]
    return {f: set(s.loc[(s["feature_name"] == f) & s["degenerate"], "frequency"]) for f in s["feature_name"].unique()}


def _median_difference_ci(high: np.ndarray, low: np.ndarray, base_seed: int, contrast_id: str) -> tuple[float, float]:
    """Bootstrap CI of median(High) - median(Low), resampling series within each tercile."""
    rng = stats.bootstrap_rng(base_seed, contrast_id)
    diffs = np.empty(stats.N_BOOTSTRAP)
    for i in range(stats.N_BOOTSTRAP):
        diffs[i] = np.median(rng.choice(high, len(high))) - np.median(rng.choice(low, len(low)))
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def h4(inst: pd.DataFrame, tables: dict, base_seed: int, family_id: str = "H4", families: list[str] | None = None,
       series: set | None = None, features: list[str] | None = None) -> pd.DataFrame:
    """Delta_STL across the terciles of each feature: Jonckheere-Terpstra (degenerate
    frequencies excluded, D7) and Spearman with the feature value; Holm over feature x
    strategy within each test type. Families pooled with equal weights unless
    ``families`` restricts them; ``series`` restricts the series (A7)."""
    cohort = inst[inst["scope"] == "cohort"]
    if families is not None:
        cohort = cohort[cohort["family"].isin(families)]
    d = stl_deltas(cohort)
    if series is not None:
        d = d[d["unique_id"].isin(series)]
    values = tables["samples"][["frequency", "feature_name", "unique_id", "feature_value"]].astype({"frequency": str, "feature_name": str})
    degenerate = _degenerate(tables)
    jt_rows, sp_rows, excluded = [], [], []
    for feature in [f for f in tables["bucket_summary"]["feature_name"].unique() if features is None or f in features]:
        for strategy in ("stl_sn", "stl_ac"):
            sub = d[(d["feature_name"] == feature) & (d["strategy"] == strategy)]
            if sub.empty:
                continue
            ps = per_series(sub, ["frequency", "bucket"], family_balanced=True).astype({"frequency": str, "bucket": str})
            key = {"feature_name": feature, "strategy": strategy}
            cid = f"{family_id}|{feature}|{strategy}"
            usable = ps[~ps["frequency"].isin(degenerate.get(feature, set()))]
            groups = [usable.loc[usable["bucket"] == b, "delta"].to_numpy() for b in BUCKETS]
            if sum(len(g) > 0 for g in groups) >= 2 and len(groups[0]) and len(groups[2]):
                jt = stats.jonckheere_terpstra(groups)
                contrast = float(np.median(groups[2]) - np.median(groups[0]))
                lo, hi = _median_difference_ci(groups[2], groups[0], base_seed, cid + "|jt")
                jt_rows.append({**key, "n_series": len(usable), "excluded_frequencies": ",".join(sorted(degenerate.get(feature, set()))),
                                **{f"n_{b.lower()}": len(g) for b, g in zip(BUCKETS, groups)},
                                **{f"median_{b.lower()}": (float(np.median(g)) if len(g) else np.nan) for b, g in zip(BUCKETS, groups)},
                                "high_minus_low": contrast, "ci_low": lo, "ci_high": hi, "statistic": jt["statistic"], "z": jt["z"], "p": jt["p"]})
            else:
                excluded.append({**key, "excluded_frequencies": ",".join(sorted(degenerate.get(feature, set()))),
                                 "label": "excluded (D7)", "test_family": f"{family_id}a-JT"})
            merged = ps.merge(values[values["feature_name"] == feature], on=["frequency", "unique_id"], how="left")
            if merged["feature_value"].isna().any():
                raise ValueError(f"{feature}: per-series deltas without a feature value")
            sp = stats.spearman(merged["feature_value"].to_numpy(), merged["delta"].to_numpy(), base_seed, cid + "|spearman")
            high, low = merged.loc[merged["bucket"] == "High", "delta"], merged.loc[merged["bucket"] == "Low", "delta"]
            sp_rows.append({**key, **sp, "high_minus_low": float(np.median(high) - np.median(low)) if len(high) and len(low) else np.nan})
    jt = _finish(pd.DataFrame(jt_rows), f"{family_id}a-JT", effect="high_minus_low")
    sp = _finish(pd.DataFrame(sp_rows), f"{family_id}b-Spearman", effect="high_minus_low")
    for df in (jt, sp):
        if not df.empty:
            df["direction"] = np.where(df["label"].isin(["gain", "loss"]), np.where(df["high_minus_low"] > 0, "increasing", "decreasing"), "")
    return pd.concat([jt, pd.DataFrame(excluded), sp], ignore_index=True)


def h4_by_family(inst: pd.DataFrame, tables: dict, base_seed: int) -> pd.DataFrame:
    """H4 per family (secondary), Holm within each family and test type."""
    return pd.concat([h4(inst, tables, base_seed, family_id=f"H4[{fam}]", families=[fam]).assign(family=fam)
                      for fam in FAMILIES], ignore_index=True)


def h5(inst: pd.DataFrame, base_seed: int, family_id: str = "H5", metric: str = PRIMARY, how: str = "mean") -> pd.DataFrame:
    """Delta_FT per global family and strategy (Holm over 3 x 3)."""
    d = ft_deltas(inst, metric)
    tests = one_sample(per_series(d, ["family", "strategy"], how=how), ["family", "strategy"], base_seed, family_id)
    return _with_descriptives(tests, d, ["family", "strategy"])


def h6(inst: pd.DataFrame, base_seed: int, family_id: str = "H6") -> pd.DataFrame:
    """Non-linear minus linear models' per-series Delta_FT, families weighted equally within
    each type (Holm over the 3 strategies)."""
    d = ft_deltas(inst)
    d["linearity"] = d["model"].astype(str).map(LINEARITY)
    ps = per_series(d, ["strategy", "linearity"], family_balanced=True)
    out = []
    for strategy, g in ps.groupby("strategy", observed=True, sort=True):
        w = g.pivot(index="unique_id", columns="linearity", values="delta").dropna()
        x = (w["nonlinear"] - w["linear"]).to_numpy()
        out.append({"strategy": strategy, "median_nonlinear": float(w["nonlinear"].median()), "median_linear": float(w["linear"].median()),
                    **_test(x, base_seed, f"{family_id}|{strategy}")})
    return _finish(pd.DataFrame(out), family_id)


def h7(inst: pd.DataFrame, scope: str, frequency: str | None = None) -> dict:
    """Friedman over configurations (model x strategy) with distinct series as blocks,
    Nemenyi's critical difference and the pairwise table; ``scope`` cohort or tercile."""
    sub = inst[(inst["scope"] == "cohort") if scope == "cohort" else (inst["scope"] != "cohort")]
    if frequency is not None:
        sub = sub[sub["frequency"] == frequency]
    sub = sub.assign(configuration=sub["model"].astype(str) + "|" + sub["strategy"].astype(str))
    models = MODELS if scope == "cohort" else GLOBAL_MODELS
    expected = [f"{m}|{s}" for m in models for s in STRATEGIES]
    keys = ["feature_name", "frequency", "unique_id", "window"]
    wide = sub[keys + ["configuration", PRIMARY]].astype({k: str for k in keys}).set_index(keys + ["configuration"])[PRIMARY]
    if wide.index.duplicated().any():
        raise ValueError("instances are not unique per configuration")
    wide = wide.unstack("configuration")
    missing = sorted(set(expected) - set(wide.columns))
    if missing:
        raise ValueError(f"configurations without rows: {missing}")
    wide = wide[expected]
    complete = wide.dropna()
    ps = complete.groupby(level="unique_id").mean()
    fr = stats.friedman(ps.to_numpy())
    k, n = len(expected), len(ps)
    cd = stats.nemenyi_cd(k, n)
    ranks = pd.DataFrame({"configuration": expected, "mean_rank": fr["mean_ranks"],
                          "model": [c.split("|")[0] for c in expected], "strategy": [c.split("|")[1] for c in expected]})
    ranks["family"] = ranks["model"].map(family)
    ranks = ranks.sort_values("mean_rank", kind="mergesort").reset_index(drop=True)
    pairs = []
    mean_rank = dict(zip(ranks["configuration"], ranks["mean_rank"]))
    values = ps.to_numpy()
    for i, j in combinations(range(k), 2):
        a, b = expected[i], expected[j]
        diff = float(np.median(values[:, i] - values[:, j]))
        separated = abs(mean_rank[a] - mean_rank[b]) > cd
        pairs.append({"first": a, "second": b, "rank_difference": mean_rank[a] - mean_rank[b], "separated": separated,
                      "median_relnaive_difference": diff,
                      "practically_meaningful": bool(separated and abs(diff) >= stats.PRACTICAL_THRESHOLD)})
    summary = {"scope": scope, "frequency": frequency or "all", "k": k, "n_series": n, "statistic": fr["statistic"],
               "p": fr["p"], "critical_difference": cd, "n_excluded_instances": int(len(wide) - len(complete))}
    return {"summary": summary, "ranks": ranks, "pairs": pd.DataFrame(pairs)}
