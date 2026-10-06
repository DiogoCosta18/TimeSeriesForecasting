"""Secondary and robustness analyses A1-A15 and the descriptive outputs of Sections 2-4
(protocol Sections 5.5-5.6, Table 8).

Descriptives are at instance level, tests at series level (D27). A2, A3 and A5 repeat
confirmatory tests on subsets or other metrics, each with its own Holm family named in
``test_family``; A13 is exploratory (Holm within each table).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from scipy import stats as sps

from src.analysis import hypotheses as hy
from src.analysis import stats
from src.analysis.data import PRIMARY, AnalysisData, AnalysisError, describe_groups, ft_deltas, per_series, stl_deltas
from src.data.schemas import FEATURE_NAMES
from src.forecast import spaces
from src.forecast.registry import BACKEND, FAMILIES, FAMILY, GLOBAL_MODELS, LINEARITY, MODELS, STATISTICAL_MODELS, family
from src.stages.io import safe_name
from src.stages.tasks import STOCHASTIC_MODELS, tuning_tasks

GUARD_HISTORY = {"feature_nonlinearity": 30, "feature_arch_stat": 24}   # A7 (D5)
CAPS = [5, 10, 20]                                                       # A6


def _source(inst: pd.DataFrame) -> pd.Series:
    return inst["source_dataset"].astype(str).str.split("_").str[0]


def _reholm(df: pd.DataFrame) -> pd.DataFrame:
    """Holm over the tested rows of each test family again (after concatenating subsets);
    rows without a test (exclusions) are kept unchanged."""
    if df.empty or "p" not in df:
        return df
    df = df.copy()
    for _, g in df[df["p"].notna()].groupby("test_family", sort=False):
        df.loc[g.index, "p_holm"] = stats.holm(g["p"].to_numpy())
        effect = "high_minus_low" if "high_minus_low" in g and g["high_minus_low"].notna().all() else "median"
        df.loc[g.index, "label"] = [stats.d26_label(p, e) for p, e in zip(df.loc[g.index, "p_holm"], g[effect])]
        df.loc[g.index, "n_tests_in_family"] = len(g)
    return df


# --- Section 2 ------------------------------------------------------------------------

def data_counts(d: AnalysisData) -> pd.DataFrame:
    rows = []
    samples = d.tables["samples"]
    for frequency, sources in d.bundle["counts"].items():
        for source, c in sources.items():
            s = samples[(samples["frequency"] == frequency) & (samples["source_dataset"] == source)]
            rows.append({"frequency": frequency, "source": source, "pool": c["pool"], "length_eligible": c["length_eligible"],
                         "eligible": c["eligible"], "sampled_distinct": int(s["unique_id"].nunique()),
                         "per_feature_sample": int(s.groupby("feature_name").size().max())})
    return pd.DataFrame(rows)


def model_table(d: AnalysisData) -> pd.DataFrame:
    versions = d.rows.groupby("model", observed=True)["backend_version"].unique()
    rows = []
    for model in MODELS:
        fam = FAMILY[model]
        if fam == "statistical":
            from src.forecast.models import statistical_model
            cls = type(statistical_model(model, 12)).__name__
        elif fam == "ml":
            cls = type(spaces.ml_estimator(model, spaces.sample_ml(model, _RecordingTrial()), 0)).__name__
        else:
            cls = spaces.neural_class(model).__name__
        rows.append({"family": fam, "model": model, "linearity": LINEARITY.get(model, "n/a (local)"), "backend": BACKEND[fam],
                     "backend_version": ",".join(sorted(map(str, versions.get(model, [])))), "class": cls,
                     "scopes": "cohort" if fam == "statistical" else "cohort, tercile"})
    return pd.DataFrame(rows)


class _RecordingTrial:
    """Stands in for an Optuna trial and records each suggestion (the search space as coded)."""

    def __init__(self):
        self.space = []

    def suggest_categorical(self, name, choices):
        self.space.append({"parameter": name, "kind": "categorical", "values": json.dumps(list(choices))})
        return choices[0]

    def suggest_float(self, name, low, high, log=False):
        self.space.append({"parameter": name, "kind": "float, log" if log else "float", "values": f"[{low}, {high}]"})
        return low

    def suggest_int(self, name, low, high):
        self.space.append({"parameter": name, "kind": "int", "values": f"[{low}, {high}]"})
        return low


def search_spaces() -> pd.DataFrame:
    rows = []
    for model in GLOBAL_MODELS:
        if FAMILY[model] == "ml":
            trial = _RecordingTrial()
            spaces.sample_ml(model, trial)
            rows += [{"model": model, "frequency": "both", **r} for r in trial.space]
        else:
            for frequency, m in (("monthly", 12), ("quarterly", 4)):
                trial = _RecordingTrial()
                spaces.sample_neural(model, trial, m)
                rows += [{"model": model, "frequency": frequency, **r} for r in trial.space]
    return pd.DataFrame(rows)


def a10_tuning(d: AnalysisData) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Frozen configurations; best vs first trial per study; the trial curves."""
    configs, studies, curves = [], [], []
    for key, e in sorted(d.frozen["entries"].items()):
        configs.append({"config_key": key, "model": e["model"], "frequency": e["frequency"], "target": e["target"],
                        "params": json.dumps(e["params"], sort_keys=True), "trained_steps": e.get("trained_steps"),
                        "best_validation_mase": e.get("best_validation_mase")})
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for task in tuning_tasks():
        storage = d.run_dir / "tune" / task.shard / "studies.sqlite"
        if not storage.exists():
            raise AnalysisError(f"{storage} is missing; A10 needs every archived study")
        name = safe_name(f"{task.model}|{task.frequency}|{task.target}")
        trials = [t for t in optuna.load_study(study_name=name, storage=f"sqlite:///{storage.resolve()}").trials
                  if t.value is not None]
        values = np.array([t.value for t in trials])
        best = int(np.argmin(values))
        studies.append({"model": task.model, "frequency": task.frequency, "target": task.target, "n_trials": len(values),
                        "first_trial_mase": values[0], "best_trial_mase": values[best], "best_trial_number": best,
                        "improvement_over_first": values[0] - values[best],
                        "total_duration_seconds": float(sum(t.user_attrs.get("duration_seconds", np.nan) for t in trials))})
        curves += [{"model": task.model, "frequency": task.frequency, "target": task.target, "trial": i,
                    "best_so_far": float(np.min(values[: i + 1]))} for i in range(len(values))]
    return pd.DataFrame(configs), pd.DataFrame(studies), pd.DataFrame(curves)


def a11_ks(tables: dict) -> pd.DataFrame:
    features = tables["features"][tables["features"]["eligible"]]
    rows = []
    for (frequency, feature, source), s in tables["samples"].groupby(["frequency", "feature_name", "source_dataset"], sort=True):
        pool = features[(features["frequency"] == frequency) & (features["source_dataset"] == source)][feature]
        res = sps.ks_2samp(s["feature_value"], pool)
        rows.append({"frequency": frequency, "feature": feature, "source": source, "n_sample": len(s), "n_pool": len(pool),
                     "ks_distance": float(res.statistic), "p": float(res.pvalue)})
    return pd.DataFrame(rows)


def a12_buckets(tables: dict) -> pd.DataFrame:
    return tables["bucket_summary"].copy()


def a14_correlations(tables: dict) -> pd.DataFrame:
    f = tables["features"][tables["features"]["eligible"]]
    rows = []
    for frequency, g in f.groupby("frequency", sort=True):
        rho = g[FEATURE_NAMES].corr(method="spearman")
        rows += [{"frequency": frequency, "feature_a": a, "feature_b": b, "spearman_rho": float(rho.loc[a, b]), "n_series": len(g)}
                 for i, a in enumerate(FEATURE_NAMES) for b in FEATURE_NAMES[i + 1:]]
    return pd.DataFrame(rows)


# --- Section 3.1 (A1) and Section 4 descriptives --------------------------------------

def _trained_weighted_mean(g: pd.DataFrame) -> float:
    """Mean RelNaive over a configuration's scopes, weighted by their trained instances."""
    weights = (g["n"] - g["n_failed"]).to_numpy(dtype=float)
    return float(np.average(g["relnaive_mean"], weights=weights)) if weights.sum() > 0 else np.nan


def a1_standing(main: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = main.groupby(["family", "model", "strategy", "scope"], observed=True, sort=True)
    standing = g.agg(n=(PRIMARY, "size"), n_failed=(PRIMARY, lambda x: int(x.isna().sum())),
                     relnaive_mean=(PRIMARY, "mean"), relnaive_median=(PRIMARY, "median"),
                     mase_mean=("mase", "mean"), smape_mean=("smape", "mean"), pocid_mean=("pocid", "mean")).reset_index()
    kind = np.where(standing["scope"].astype(str) == "cohort", "cohort", "tercile")
    top = (standing.assign(scope_kind=kind).groupby(["scope_kind", "family", "model", "strategy"], observed=True)
           .apply(_trained_weighted_mean, include_groups=False)
           .rename("relnaive_mean").reset_index().sort_values(["scope_kind", "relnaive_mean"], kind="mergesort"))
    top["rank"] = top.groupby("scope_kind").cumcount() + 1
    return standing, top


def stl_feature_tercile(main: pd.DataFrame, base_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Delta_STL by feature x tercile x strategy (families pooled with equal weights) and per family."""
    d = stl_deltas(main[main["scope"] == "cohort"])
    pooled = describe_groups(d, ["feature_name", "bucket", "strategy"], family_balanced=True)
    ps = per_series(d, ["feature_name", "bucket", "strategy"], family_balanced=True)
    med = ps.groupby(["feature_name", "bucket", "strategy"], observed=True)["delta"].agg(["median", "size"]).reset_index()
    pooled = pooled.merge(med.rename(columns={"median": "median_per_series", "size": "n_series"}),
                          on=["feature_name", "bucket", "strategy"], how="left")
    per_family = per_series(d, ["family", "feature_name", "bucket", "strategy"]).groupby(
        ["family", "feature_name", "bucket", "strategy"], observed=True)["delta"].median().rename("median_per_series").reset_index()
    return pooled, per_family


def ft_descriptives(main: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = ft_deltas(main)
    d["linearity"] = d["model"].astype(str).map(LINEARITY)
    return {
        "ft_overall": describe_groups(d, [], family_balanced=True),
        "ft_by_bucket": describe_groups(d, ["bucket"], family_balanced=True),
        "ft_by_family_strategy": describe_groups(d, ["family", "strategy"]),
        "ft_by_model": describe_groups(d, ["family", "model", "linearity"]),
        "ft_by_linearity": describe_groups(d, ["linearity"], family_balanced=True),
    }


def a13_per_model(main: pd.DataFrame, base_seed: int) -> dict[str, pd.DataFrame]:
    """Per-model tables (exploratory; Holm within each table) and the significance counts."""
    stl = stl_deltas(main[main["scope"] == "cohort"])
    ft = ft_deltas(main)
    out = {}
    for name, d, by in (("a13_stl_per_model", stl, ["model", "strategy"]), ("a13_ft_per_model", ft, ["model", "strategy"]),
                        ("a13_ft_per_model_bucket", ft, ["model", "strategy", "bucket"])):
        tests = hy.one_sample(per_series(d, by), by, base_seed, name.upper())
        tests.insert(1, "family", tests["model"].astype(str).map(family))
        out[name] = tests.merge(describe_groups(d, by).rename(columns=lambda c: c if c in by else f"instance_{c}"), on=by, how="left")
    counts = out["a13_ft_per_model_bucket"].groupby(["family", "label"]).size().unstack("label", fill_value=0)
    out["ft_significance_counts"] = counts.reindex(columns=list(stats.LABELS), fill_value=0).reset_index()
    return out


# --- Appendix --------------------------------------------------------------------------

def a2_a3_splits(main: pd.DataFrame, tables: dict, base_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    freq, src = [], []
    for frequency in sorted(main["frequency"].astype(str).unique()):
        sub = main[main["frequency"] == frequency]
        tag = f"A2[{frequency}]"
        freq += [hy.h1(sub, base_seed, f"{tag}-H1"), hy.h4(sub, tables, base_seed, f"{tag}-H4"), hy.h5(sub, base_seed, f"{tag}-H5")]
    source = _source(main)
    for s in sorted(source.unique()):
        sub = main[source == s]
        tag = f"A3[{s}]"
        src += [hy.h1(sub, base_seed, f"{tag}-H1"), hy.h4(sub, tables, base_seed, f"{tag}-H4"), hy.h5(sub, base_seed, f"{tag}-H5")]
    return pd.concat(freq, ignore_index=True), pd.concat(src, ignore_index=True)


def a4_windows(main: pd.DataFrame) -> pd.DataFrame:
    stl = describe_groups(stl_deltas(main[main["scope"] == "cohort"]), ["window", "family", "strategy"]).assign(delta="STL")
    ft = describe_groups(ft_deltas(main), ["window", "family", "strategy"]).assign(delta="FT")
    return pd.concat([stl, ft], ignore_index=True)


def a5_metrics(main: pd.DataFrame, base_seed: int) -> pd.DataFrame:
    out = []
    for metric, how in (("mase", "mean"), ("smape", "mean"), ("relnaive", "median")):
        out += [hy.h1(main, base_seed, f"A5[{metric}]-H1", metric=metric, how=how).assign(metric=metric, aggregation=how),
                hy.h5(main, base_seed, f"A5[{metric}]-H5", metric=metric, how=how).assign(metric=metric, aggregation=how)]
    return pd.concat(out, ignore_index=True)


def a6_caps(main: pd.DataFrame) -> pd.DataFrame:
    inst = main.copy()
    for cap in CAPS:
        inst[f"cap{cap}"] = inst["relnaive"].clip(upper=cap)
    out = []
    cohort = inst[inst["scope"] == "cohort"]
    for name, fn, src in (("STL", stl_deltas, cohort), ("FT", ft_deltas, inst)):
        frames = {cap: fn(src, f"cap{cap}") for cap in CAPS}
        raw = fn(src, "relnaive")
        keys = ["family", "strategy"]
        table = raw.dropna().groupby(keys, observed=True)["delta"].apply(lambda x: sps.trim_mean(x, 0.05)).rename("trimmed_mean_5pct_unclipped")
        for cap, f in frames.items():
            table = pd.concat([table, f.groupby(keys, observed=True)["delta"].mean().rename(f"mean_cap{cap}")], axis=1)
        out.append(table.reset_index().assign(delta=name))
    return pd.concat(out, ignore_index=True)


def a7_guard(main: pd.DataFrame, tables: dict, base_seed: int) -> pd.DataFrame:
    elig = tables["eligibility"]
    quarterly = main[main["frequency"] == "quarterly"]
    parts = []
    for feature, minimum in GUARD_HISTORY.items():
        if feature not in set(tables["bucket_summary"]["feature_name"]):
            continue
        keep = set(elig.loc[(elig["frequency"] == "quarterly") & elig["eligible"] & (elig["history_length"] >= minimum), "unique_id"])
        part = hy.h4(quarterly, tables, base_seed, "A7", series=keep, features=[feature])
        if part.empty:
            part = pd.DataFrame([{"feature_name": feature, "n_series": 0, "label": "no series meet the guard"}])
        parts.append(part.assign(min_history=minimum, test_family=part.get("test_family", "A7")))
    return _reholm(pd.concat(parts, ignore_index=True)) if parts else pd.DataFrame()


def a8_failures(d: AnalysisData) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = d.failed.astype({"model": str, "strategy": str})
    stat = d.rows[d.rows["family"] == "statistical"].astype({"model": str, "strategy": str})
    by_model = stat.groupby(["model", "strategy"]).size().rename("trained").reset_index()
    by_model = by_model.merge(f.groupby(["model", "strategy"]).size().rename("failed").reset_index(), on=["model", "strategy"], how="left")
    by_model["failed"] = by_model["failed"].fillna(0).astype(int)
    by_model["failed_share"] = by_model["failed"] / (by_model["trained"] + by_model["failed"])
    reasons = (f.assign(reason=f["failure"].str.split(":").str[0]).groupby(["model", "strategy", "reason"]).size()
               .rename("rows").reset_index()) if len(f) else pd.DataFrame(columns=["model", "strategy", "reason", "rows"])
    by_model = by_model.merge(reasons.groupby(["model", "strategy"])["reason"].agg(lambda x: ", ".join(sorted(set(x)))).reset_index(),
                              on=["model", "strategy"], how="left") if len(reasons) else by_model.assign(reason="")
    tasks = []
    for tid, e in d.record["tasks"].items():
        for attempt in e.get("failed_attempts", []):
            tasks.append({"task_id": tid, "kind": "retried within the run (succeeded on retry)", "detail": attempt})
    for tid, e in d.record["failures"]["resolved"].items():
        tasks.append({"task_id": tid, "kind": f"listed, resolved by commit {e['resolved_by_commit']}", "detail": "; ".join(e["attempts"])})
    for tid, e in d.record["d16_reruns"].items():
        tasks.append({"task_id": tid, "kind": "D16 rerun", "detail": f"{e['main_commit']} -> {e['fix_commit']}"})
    for tid in d.record["failures"]["listed"]:
        tasks.append({"task_id": tid, "kind": "listed, unresolved", "detail": ""})
    return by_model, pd.DataFrame(tasks, columns=["task_id", "kind", "detail"])


def a9_compute(rows: pd.DataFrame) -> pd.DataFrame:
    glob = rows[rows["family"] != "statistical"].drop_duplicates(["task_id", "window"])
    stat = rows[rows["family"] == "statistical"]
    both = pd.concat([glob, stat], ignore_index=True)
    kind = np.where(both["scope"].astype(str) == "cohort", "cohort", "tercile")
    g = both.assign(scope_kind=kind).groupby(["family", "strategy", "scope_kind"], observed=True)
    return g.agg(n_fits=("fit_seconds", "size"), total_fit_hours=("fit_seconds", lambda x: x.sum() / 3600),
                 median_fit_seconds=("fit_seconds", "median"), total_predict_hours=("predict_seconds", lambda x: x.sum() / 3600)
                 ).reset_index()


def a9_stl_time_ratio(rows: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    """STL-AC / STL-SN training time per global model and frequency (median over paired task
    windows), and for neural and transformer models divided by the ratio of trained steps
    (T + S + R steps over non-seasonal steps): near 1 when time follows the frozen steps."""
    from src.forecast.registry import config_key

    glob = rows[rows["family"] != "statistical"]
    keys = ["feature_name", "frequency", "family", "model", "scope", "seed", "window"]
    per_window = (glob.astype({k: str for k in keys}).drop_duplicates(["task_id", "window"])
                  .pivot_table(index=keys, columns="strategy", values="fit_seconds", aggfunc="first", observed=True))
    ratio = (per_window["stl_ac"] / per_window["stl_sn"]).dropna().rename("time_ratio").reset_index()
    table = ratio.groupby(["family", "model", "frequency"])["time_ratio"].median().reset_index()

    def steps(model, frequency, target):
        return frozen["entries"][config_key(model, frequency, target)].get("trained_steps")

    step_ratio = []
    for r in table.itertuples():
        if r.family == "ml":
            step_ratio.append(np.nan)
        else:
            step_ratio.append(sum(steps(r.model, r.frequency, t) for t in ("trend", "seasonal", "residual"))
                              / steps(r.model, r.frequency, "nonseasonal"))
    table["step_ratio"] = step_ratio
    table["time_ratio_per_step_ratio"] = table["time_ratio"] / table["step_ratio"]
    return table


def a15_seeds(d: AnalysisData) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Seed check: per-series spread across seeds (table and per series), and whether H1 or
    H3 labels change when another seed is used."""
    extra = list(d.config["seed_check_seeds"])
    if not extra:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    s0 = d.main_seed
    inst = d.instances[(d.instances["feature_name"] == "feature_evolving_seasonality") & (d.instances["scope"] == "cohort")]
    stochastic = inst["model"].isin(STOCHASTIC_MODELS)
    spread_parts = []
    for seed in [s0, *extra]:
        sub = inst[stochastic & (inst["seed_key"] == seed)]
        spread_parts.append(per_series(sub.rename(columns={PRIMARY: "value"}), ["model", "strategy"], value="value").assign(seed=seed))
    per_seed = pd.concat(spread_parts).pivot_table(index=["unique_id", "model", "strategy"], columns="seed", values="value", observed=True)
    per_seed = per_seed.dropna()
    spread = pd.DataFrame({"range": per_seed.max(axis=1) - per_seed.min(axis=1), "sd": per_seed.std(axis=1, ddof=1)}).reset_index()
    spread_table = spread.groupby(["model", "strategy"], observed=True).agg(
        n_series=("range", "size"), median_range=("range", "median"), p90_range=("range", lambda x: x.quantile(0.9)),
        mean_sd=("sd", "mean")).reset_index()
    labels = []
    for seed in [s0, *extra]:
        view = pd.concat([inst[~stochastic & inst["main_seed"]], inst[stochastic & (inst["seed_key"] == seed)]], ignore_index=True)
        view["seed_key"] = np.where(view["seed_key"] == -1, -1, s0)
        for name, table in (("H1", hy.h1(view, s0, f"A15[{seed}]-H1")), ("H3", hy.h3(view, s0, f"A15[{seed}]-H3"))):
            keys = ["family", "strategy"] if name == "H1" else ["family", "first", "second"]
            labels.append(table[keys + ["median", "p_holm", "label"]].assign(hypothesis=name, seed=seed))
    labels = pd.concat(labels, ignore_index=True)
    labels["contrast"] = labels.apply(lambda r: "|".join(str(r[k]) for k in ("family", "strategy", "first", "second")
                                                         if k in r and isinstance(r[k], str)), axis=1)
    wide = labels.pivot_table(index=["hypothesis", "contrast"], columns="seed", values="label", aggfunc="first")
    wide.columns = [f"label_seed_{c}" for c in wide.columns]
    wide["label_changes"] = wide.nunique(axis=1) > 1
    return spread_table, wide.reset_index(), spread
