"""Exploratory analyses of a gated run: NOT pre-registered.

The confirmatory plan is protocol Section 5 (H1-H7, A1-A15, Table 8); nothing here tests a
hypothesis or changes a label of that plan. These analyses answer questions the earlier paper
raised, on the same gated instances and with the same aggregation rules (D27: a series' value
is its mean over the instances of a group; cohort scope, main seed):

- E1 oracle: for each series and model, the strategy with the lowest RelNaive (ties go to the
  simpler strategy: Direct, then STL-SN). How often each strategy is best, and what a perfect
  per-series choice would gain over always forecasting Direct: the ceiling of any rule that
  decides per series whether to decompose.
- E2 feature rule: STL-SN for a series whose evolving seasonality is below its frequency's
  first tercile cut (the cut of the prepare bundle), Direct otherwise. The rule was chosen
  after seeing H4 on the same data, so its gain is in-sample and only illustrative; an honest
  estimate needs out-of-sample selection.
- E3 POCID: mean and median RelNaive and mean POCID per model and strategy (the earlier
  paper's RelNaive-POCID trade-off).
- E4 honest oracle: the strategy of each series and model chosen on every window but the last
  and judged on the last, beside the in-sample oracle of the last window: how much of E1 a
  choice from a series' own past keeps.
- E5 feature cells: per family and STL strategy, the median per-series STL delta in each
  tercile of one feature and in each pair of terciles of two features (terciles cut as the
  prepare bundle cuts them). Hundreds of cells are scanned on the same data: hypotheses for a
  later study, not findings.
- E6 Ridge: the ML family's per-series STL deltas with all four models, without Ridge and
  without LinearRegression (Ridge's tuned penalty is negligible at the data's scale, change
  log v2.1).

    python -m src.analysis.exploratory --run RUN --out OUT
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.data import PRIMARY, AnalysisError, load_gated
from src.forecast.registry import FAMILIES, STRATEGIES
from src.stages.io import code_commit, sha256_json, utc_now, write_json

STATUS = "exploratory: not pre-registered (protocol Section 5 is the confirmatory plan)"
RULE_FEATURE = "feature_evolving_seasonality"
FAMILY_ORDER = {f: i for i, f in enumerate(FAMILIES)}


def series_by_strategy(inst: pd.DataFrame, metric: str = PRIMARY) -> pd.DataFrame:
    """One row per (series, model) with its mean metric under each strategy (cohort scope, main
    seed); rows missing a strategy are dropped (pairwise deletion)."""
    cohort = inst[(inst["scope"].astype(str) == "cohort") & inst["main_seed"]]
    keys = ["unique_id", "frequency", "family", "model"]
    means = (cohort.dropna(subset=[metric]).astype({k: str for k in keys + ["strategy"]})
             .groupby(keys + ["strategy"], sort=True)[metric].mean().unstack("strategy"))
    return means.reindex(columns=STRATEGIES).dropna().reset_index()


def oracle(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """E1. Per model and per family: share of series where each strategy is best, and the
    oracle's gain over Direct (RelNaive(Direct) - RelNaive(best)); a family's per-series gain is
    the mean over its models."""
    values = table[STRATEGIES]
    t = table.assign(best=values.idxmin(axis=1), gain=table["direct"] - values.min(axis=1))
    shares = pd.crosstab([t["family"], t["model"]], t["best"], normalize="index").reindex(columns=STRATEGIES, fill_value=0.0)
    by_model = shares.add_prefix("best_share_").join(
        t.groupby(["family", "model"])["gain"].agg(gain_median="median", gain_mean="mean")).reset_index()
    rows = []
    for fam, g in t.groupby("family"):
        per_series = g.groupby("unique_id")["gain"].mean()
        share = g["best"].value_counts(normalize=True).reindex(STRATEGIES, fill_value=0.0)
        rows.append({"family": fam, "n_series": int(per_series.size), "n_series_models": int(len(g)),
                     **{f"best_share_{s}": float(share[s]) for s in STRATEGIES},
                     "gain_median": float(per_series.median()), "gain_mean": float(per_series.mean()),
                     "share_series_gain_ge_0.01": float((per_series >= 0.01).mean())})
    by_family = pd.DataFrame(rows).sort_values("family", key=lambda s: s.map(FAMILY_ORDER)).reset_index(drop=True)
    return by_family, by_model.sort_values(["family", "model"], key=lambda s: s.map(FAMILY_ORDER) if s.name == "family" else s)


def feature_rule(table: pd.DataFrame, features: pd.DataFrame, bucket_summary: pd.DataFrame) -> pd.DataFrame:
    """E2. Gain over Direct per family (per-series mean over its models) of: always STL-SN, the
    evolving-seasonality rule, and the oracle; the rule's share of decomposed series."""
    cuts = bucket_summary[bucket_summary["feature_name"] == RULE_FEATURE].set_index("frequency")["tercile_cut_1"]
    value = features.set_index(["unique_id", "frequency"])[RULE_FEATURE]
    t = table.join(value, on=["unique_id", "frequency"])
    if t[RULE_FEATURE].isna().any():
        raise AnalysisError(f"series without {RULE_FEATURE}")
    decompose = t[RULE_FEATURE] < t["frequency"].map(cuts)
    t = t.assign(decompose=decompose,
                 always_stl_sn=t["direct"] - t["stl_sn"],
                 rule=t["direct"] - t["stl_sn"].where(decompose, t["direct"]),
                 oracle=t["direct"] - t[STRATEGIES].min(axis=1))
    rows = []
    for fam, g in t.groupby("family"):
        per_series = g.groupby("unique_id")[["always_stl_sn", "rule", "oracle", "decompose"]].mean()
        row = {"family": fam, "n_series": int(len(per_series)), "share_decomposed": float(per_series["decompose"].mean())}
        for name in ("always_stl_sn", "rule", "oracle"):
            row[f"{name}_median"] = float(per_series[name].median())
            row[f"{name}_mean"] = float(per_series[name].mean())
            row[f"{name}_share_improved"] = float((per_series[name] > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("family", key=lambda s: s.map(FAMILY_ORDER)).reset_index(drop=True)


def series_by_window(inst: pd.DataFrame, metric: str = PRIMARY) -> pd.DataFrame:
    """One row per (series, model, window) with its mean metric under each strategy over the
    feature samples (cohort scope, main seed); rows missing a strategy are dropped."""
    cohort = inst[(inst["scope"].astype(str) == "cohort") & inst["main_seed"]]
    keys = ["unique_id", "frequency", "family", "model", "window"]
    means = (cohort.dropna(subset=[metric]).astype({k: str for k in keys[:-1] + ["strategy"]})
             .groupby(keys + ["strategy"], sort=True)[metric].mean().unstack("strategy"))
    return means.reindex(columns=STRATEGIES).dropna().reset_index()


def honest_oracle(window_table: pd.DataFrame) -> pd.DataFrame:
    """E4. Choose each (series, model)'s strategy on every window but the last (ties go to the
    simpler strategy), judge it on the last; beside it the in-sample oracle of the last window."""
    keys = ["unique_id", "frequency", "family", "model"]
    last = window_table["window"].max()
    select = window_table[window_table["window"] < last].groupby(keys)[STRATEGIES].mean()
    judge = window_table[window_table["window"] == last].set_index(keys)[STRATEGIES]
    j = select.join(judge, lsuffix="_select", how="inner")
    choice = j[[f"{s}_select" for s in STRATEGIES]].to_numpy().argmin(axis=1)
    test = j[STRATEGIES].to_numpy()
    j = j.assign(chose_stl=choice > 0, honest=test[:, 0] - test[np.arange(len(test)), choice],
                 oracle_last=test[:, 0] - test.min(axis=1)).reset_index()
    rows = []
    for fam, g in j.groupby("family"):
        ps = g.groupby("unique_id")[["chose_stl", "honest", "oracle_last"]].mean()
        rows.append({"family": fam, "n_series": int(len(ps)), "selection_windows": int(last),
                     "share_chose_stl": float(ps["chose_stl"].mean()),
                     "honest_median": float(ps["honest"].median()), "honest_mean": float(ps["honest"].mean()),
                     "honest_share_improved": float((ps["honest"] > 0).mean()),
                     "honest_share_worse": float((ps["honest"] < 0).mean()),
                     "oracle_last_median": float(ps["oracle_last"].median()), "oracle_last_mean": float(ps["oracle_last"].mean())})
    return pd.DataFrame(rows).sort_values("family", key=lambda c: c.map(FAMILY_ORDER)).reset_index(drop=True)


def _terciles(features: pd.DataFrame, bucket_summary: pd.DataFrame) -> pd.DataFrame:
    """Each eligible series' tercile (Low / Medium / High) of every feature, cut as the prepare
    bundle cuts it (value <= first cut: Low; <= second cut: Medium)."""
    names = [c for c in features.columns if c.startswith("feature_")]
    f = features[features["eligible"]][["unique_id", "frequency", *names]].copy()
    for name in names:
        cuts = bucket_summary[bucket_summary["feature_name"] == name].set_index("frequency")
        c1, c2 = f["frequency"].map(cuts["tercile_cut_1"]), f["frequency"].map(cuts["tercile_cut_2"])
        f[name] = np.select([f[name] <= c1, f[name] <= c2], ["Low", "Medium"], "High")
    return f


def feature_cells(table: pd.DataFrame, features: pd.DataFrame,
                  bucket_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """E5. Median per-series STL delta per family and strategy in each feature tercile (one
    feature) and each pair of terciles (two features); a family's value is its models' mean."""
    t = table.assign(stl_sn=table["direct"] - table["stl_sn"], stl_ac=table["direct"] - table["stl_ac"])
    fam = t.groupby(["unique_id", "frequency", "family"])[["stl_sn", "stl_ac"]].mean().reset_index()
    terc = _terciles(features, bucket_summary)
    names = [c for c in terc.columns if c.startswith("feature_")]
    fs = fam.merge(terc, on=["unique_id", "frequency"], how="left")
    if fs[names].isna().any().any():
        raise AnalysisError("series without feature terciles")

    def cells(by):
        long = fs.melt(id_vars=["unique_id", "family", *by], value_vars=["stl_sn", "stl_ac"],
                       var_name="strategy", value_name="delta")
        return (long.groupby(["family", "strategy", *by])["delta"]
                .agg(n_series="size", median="median", share_positive=lambda x: float((x > 0).mean())).reset_index())

    single = pd.concat([cells([n]).rename(columns={n: "tercile"}).assign(feature=n[8:]) for n in names], ignore_index=True)
    pairs = pd.concat([cells([a, b]).rename(columns={a: "tercile_a", b: "tercile_b"}).assign(feature_a=a[8:], feature_b=b[8:])
                       for a, b in combinations(names, 2)], ignore_index=True)
    return (single.sort_values("median", ascending=False).reset_index(drop=True),
            pairs.sort_values("median", ascending=False).reset_index(drop=True))


def ridge_sensitivity(table: pd.DataFrame) -> pd.DataFrame:
    """E6. The ML family's per-series STL deltas with all four models, without Ridge and
    without LinearRegression."""
    ml = table[table["family"] == "ml"]
    ml = ml.assign(stl_sn=ml["direct"] - ml["stl_sn"], stl_ac=ml["direct"] - ml["stl_ac"])
    rows = []
    for label, drop in (("all four models", None), ("without Ridge", "Ridge"), ("without LinearRegression", "LinearRegression")):
        sub = ml if drop is None else ml[ml["model"] != drop]
        ps = sub.groupby("unique_id")[["stl_sn", "stl_ac"]].mean()
        rows.append({"ml_family": label, "models": int(sub["model"].nunique()), "n_series": int(len(ps)),
                     **{f"{s}_{stat}": float(getattr(ps[s], stat)()) for s in ("stl_sn", "stl_ac") for stat in ("median", "mean")}})
    return pd.DataFrame(rows)


def pocid_table(inst: pd.DataFrame) -> pd.DataFrame:
    """E3. Per model and strategy (cohort scope, main seed): mean and median RelNaive (capped),
    mean POCID (share of correctly predicted directions)."""
    cohort = inst[(inst["scope"].astype(str) == "cohort") & inst["main_seed"]]
    keys = ["family", "model", "strategy"]
    g = cohort.astype({k: str for k in keys}).groupby(keys, sort=True)
    out = g.agg(n=(PRIMARY, "count"), relnaive_mean=(PRIMARY, "mean"), relnaive_median=(PRIMARY, "median"),
                pocid_mean=("pocid", "mean")).reset_index()
    return out.sort_values(["family", "model", "strategy"], key=lambda s: s.map(FAMILY_ORDER) if s.name == "family" else s)


def pocid_figure(table: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.analysis.figures import COLORS, STRATEGY_LABEL, _save

    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy in STRATEGIES:
        sub = table[table["strategy"] == strategy]
        ax.scatter(sub["relnaive_median"], sub["pocid_mean"], color=COLORS[strategy], label=STRATEGY_LABEL[strategy], s=28)
        for r in sub.itertuples():
            ax.annotate(r.model, (r.relnaive_median, r.pocid_mean), xytext=(3, 2), textcoords="offset points", fontsize=6)
    ax.set_xlabel("median RelNaive (cohort scope; lower is better)")
    ax.set_ylabel("mean POCID (higher is better)")
    ax.set_title("Exploratory, not pre-registered", fontsize=8)
    ax.legend(fontsize=7)
    _save(fig, path)


def run_exploratory(run_dir: Path, out: Path) -> dict:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise AnalysisError(f"{out} is not empty; exploratory outputs are never overwritten")
    commit = code_commit()
    data = load_gated(run_dir)
    table = series_by_strategy(data.main)
    by_family, by_model = oracle(table)
    single, pairs = feature_cells(table, data.tables["features"], data.tables["bucket_summary"])
    outputs = {
        "e1_oracle_by_family": by_family,
        "e1_oracle_by_model": by_model,
        "e2_evolving_seasonality_rule": feature_rule(table, data.tables["features"], data.tables["bucket_summary"]),
        "e3_pocid": pocid_table(data.main),
        "e4_honest_oracle": honest_oracle(series_by_window(data.main)),
        "e5_feature_single": single,
        "e5_feature_pairs": pairs,
        "e6_ridge": ridge_sensitivity(table),
    }
    (out / "tables").mkdir(parents=True)
    for name, df in outputs.items():
        df.to_csv(out / "tables" / f"{name}.csv", index=False)
    pocid_figure(outputs["e3_pocid"], out / "figures" / "e3_pocid_tradeoff.png")
    record = {"status": STATUS, "code_commit": commit, "created_at_utc": utc_now(),
              "merge_result_sha256": data.record["result_sha256"], "gates_created_at_utc": data.gates["created_at_utc"],
              "tables": {name: sha256_json(df.to_dict(orient="list")) for name, df in outputs.items()}}
    write_json(out / "exploratory.json", record)
    return {"out": str(out), "tables": len(outputs)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.analysis.exploratory", description=STATUS)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    print(run_exploratory(args.run, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
