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

    python -m src.analysis.exploratory --run RUN --out OUT
"""
from __future__ import annotations

import argparse
from pathlib import Path

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
    outputs = {
        "e1_oracle_by_family": by_family,
        "e1_oracle_by_model": by_model,
        "e2_evolving_seasonality_rule": feature_rule(table, data.tables["features"], data.tables["bucket_summary"]),
        "e3_pocid": pocid_table(data.main),
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
