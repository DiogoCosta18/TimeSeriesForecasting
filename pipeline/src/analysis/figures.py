"""Figures of the analysis (protocol Table 6). Each function draws from analysis tables only
and writes one PNG; nothing is recomputed here."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.forecast.registry import FAMILIES, GLOBAL_MODELS, LINEARITY, STRATEGIES  # noqa: E402

STRATEGY_LABEL = {"direct": "Direct", "stl_sn": "STL-SN", "stl_ac": "STL-AC"}
COLORS = {"direct": "#4C72B0", "stl_sn": "#DD8452", "stl_ac": "#55A868"}
LABEL_MARK = {"gain": "+", "loss": "−", "negligible": "≈", "no evidence": ""}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def _box(ax, data, positions, color):
    """Boxes: interquartile range and median; whiskers 5th-95th percentile; diamond: mean."""
    data = [np.asarray(d, dtype=float) for d in data]
    ax.boxplot(data, positions=positions, widths=0.6, whis=(5, 95), showfliers=False, patch_artist=True,
               boxprops={"facecolor": color, "alpha": 0.6}, medianprops={"color": "black"})
    ax.scatter(positions, [d.mean() if len(d) else np.nan for d in data], marker="D", color="black", s=12, zorder=3)


def sampling_illustration(tables: dict, path: Path) -> None:
    features = tables["features"][tables["features"]["eligible"]]
    samples = tables["samples"]
    names = sorted(samples["feature_name"].unique())
    freqs = sorted(samples["frequency"].unique())
    fig, axes = plt.subplots(len(names), len(freqs), figsize=(4.5 * len(freqs), 2.4 * len(names)), squeeze=False)
    for i, feature in enumerate(names):
        for j, frequency in enumerate(freqs):
            ax = axes[i, j]
            for source, color in zip(sorted(samples["source_dataset"][samples["frequency"] == frequency].unique()), ["C0", "C3"]):
                pool = np.sort(features[(features["frequency"] == frequency) & (features["source_dataset"] == source)][feature])
                s = np.sort(samples[(samples["frequency"] == frequency) & (samples["feature_name"] == feature)
                                    & (samples["source_dataset"] == source)]["feature_value"])
                ax.plot(pool, np.arange(1, len(pool) + 1) / len(pool), color=color, label=f"{source} pool")
                ax.plot(s, np.arange(1, len(s) + 1) / len(s), color=color, ls="--", label=f"{source} sample")
            ax.set_title(f"{feature.replace('feature_', '')} | {frequency}", fontsize=9)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=6)
    _save(fig, path)


def feature_distributions(tables: dict, path: Path) -> None:
    f = tables["features"][tables["features"]["eligible"]]
    names = [c for c in f.columns if c.startswith("feature_")]
    freqs = sorted(f["frequency"].unique())
    fig, axes = plt.subplots(len(names), len(freqs), figsize=(4.5 * len(freqs), 2.0 * len(names)), squeeze=False)
    for i, feature in enumerate(names):
        for j, frequency in enumerate(freqs):
            ax = axes[i, j]
            ax.hist(f.loc[f["frequency"] == frequency, feature], bins=50, color="C0")
            ax.set_title(f"{feature.replace('feature_', '')} | {frequency}", fontsize=9)
            ax.tick_params(labelsize=7)
    _save(fig, path)


def relnaive_distributions(main: pd.DataFrame, path: Path) -> None:
    cohort = main[main["scope"] == "cohort"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, strategy in enumerate(STRATEGIES):
        data = [cohort.loc[(cohort["family"] == fam) & (cohort["strategy"] == strategy), "relnaive_capped"].dropna()
                for fam in FAMILIES]
        _box(ax, data, [i * 4 + k for i in range(len(FAMILIES))], COLORS[strategy])
    ax.set_xticks([i * 4 + 1 for i in range(len(FAMILIES))], FAMILIES)
    ax.set_ylabel("RelNaive (cap 10)")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=COLORS[s], alpha=0.6) for s in STRATEGIES], [STRATEGY_LABEL[s] for s in STRATEGIES])
    _save(fig, path)


def cd_diagram(result: dict, path: Path) -> None:
    """Critical-difference diagram (Demsar 2006): mean ranks, CD bar, cliques of configurations
    whose ranks differ by at most the critical difference."""
    ranks = result["ranks"]
    cd, k = result["summary"]["critical_difference"], result["summary"]["k"]
    names, r = ranks["configuration"].tolist(), ranks["mean_rank"].to_numpy()
    half = (k + 1) // 2
    height = 1.2 + 0.22 * half
    fig, ax = plt.subplots(figsize=(10, height))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-0.25 * half - 0.6, 1.0)
    ax.axis("off")
    ax.hlines(0, 1, k, color="black")
    for t in range(1, k + 1):
        ax.vlines(t, 0, 0.08, color="black")
        if t == 1 or t == k or t % 5 == 0:
            ax.text(t, 0.15, str(t), ha="center", fontsize=7)
    ax.hlines(0.6, 1, 1 + cd, color="black", lw=2)
    ax.text(1 + cd / 2, 0.7, f"CD = {cd:.2f}", ha="center", fontsize=7)
    for i, (name, rank) in enumerate(zip(names, r)):
        left = i < half
        level = -(0.25 * (i if left else k - 1 - i)) - 0.4
        x_text = 0.6 if left else k + 0.4
        ax.plot([rank, rank, x_text], [0, level, level], color="gray", lw=0.6)
        ax.text(x_text, level, f"{name} ({rank:.1f})", ha="right" if left else "left", va="center", fontsize=6)
    cliques, last_end = [], -1
    for i in range(k):
        j = i
        while j + 1 < k and r[j + 1] - r[i] <= cd:
            j += 1
        if j > i and j > last_end:
            cliques.append((i, j))
            last_end = j
    for c, (i, j) in enumerate(cliques):
        y = -0.1 - 0.06 * (c % 4)
        ax.hlines(y, r[i] - 0.05, r[j] + 0.05, color="black", lw=2.5)
    s = result["summary"]
    ax.set_title(f"{s['scope']} scope, {s['frequency']}: {s['k']} configurations, {s['n_series']} series; "
                 f"Friedman p = {s['p']:.2g}", fontsize=8)
    _save(fig, path)


def _bars(table: pd.DataFrame, path: Path, families: list[str], strategies: list[str], ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    width = 0.8 / len(strategies)
    for k, strategy in enumerate(strategies):
        for i, fam in enumerate(families):
            row = table[(table["family"] == fam) & (table["strategy"] == strategy)]
            if row.empty:
                continue
            row = row.iloc[0]
            x = i + (k - (len(strategies) - 1) / 2) * width
            ax.bar(x, row["median"], width=width, color=COLORS[strategy], alpha=0.8,
                   label=STRATEGY_LABEL[strategy] if i == 0 else None)
            ax.errorbar(x, row["median"], yerr=[[row["median"] - row["ci_low"]], [row["ci_high"] - row["median"]]], color="black",
                        capsize=2, lw=0.8)
            ax.text(x, row["ci_high"], LABEL_MARK.get(row["label"], ""), ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(len(families)), families)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)
    if table["label"].astype(str).str.len().any():
        ax.text(1.0, -0.12, "D26: + gain, − loss, ≈ negligible, none: no evidence", transform=ax.transAxes,
                ha="right", fontsize=6)
    _save(fig, path)


def stl_delta_bars(h1: pd.DataFrame, path: Path) -> None:
    _bars(h1, path, FAMILIES, ["stl_sn", "stl_ac"], "median per-series ΔSTL (95% CI)")


def ft_bars(h5: pd.DataFrame, path: Path) -> None:
    _bars(h5, path, [f for f in FAMILIES if f != "statistical"], STRATEGIES, "median per-series ΔFT (95% CI)")


def mcm(h3: pd.DataFrame, path: Path) -> None:
    """Multiple comparison matrices per family: median per-series RelNaive difference
    (row - column), series won / tied / lost by the column strategy, Holm p."""
    fams = [f for f in FAMILIES if f in set(h3["family"])]
    fig, axes = plt.subplots(1, len(fams), figsize=(4.2 * len(fams), 4), squeeze=False)
    vmax = max(float(h3["median"].abs().max()), 1e-9)
    for ax, fam in zip(axes[0], fams):
        grid = np.full((3, 3), np.nan)
        text = [["" for _ in range(3)] for _ in range(3)]
        for _, r in h3[h3["family"] == fam].iterrows():
            i, j = STRATEGIES.index(r["first"]), STRATEGIES.index(r["second"])
            grid[i, j], grid[j, i] = r["median"], -r["median"]
            text[i][j] = (f"{r['median']:+.3f}\n{r['series_second_better']}/{r['series_tied']}/{r['series_first_better']}"
                          f"\np={r['p_holm']:.1g}\n{r['label']}")
            text[j][i] = (f"{-r['median']:+.3f}\n{r['series_first_better']}/{r['series_tied']}/{r['series_second_better']}"
                          f"\np={r['p_holm']:.1g}\n{r['label'].replace('gain', 'tmp').replace('loss', 'gain').replace('tmp', 'loss')}")
        ax.imshow(grid, cmap="RdBu", vmin=-vmax, vmax=vmax)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, text[i][j], ha="center", va="center", fontsize=6)
        ax.set_xticks(range(3), [STRATEGY_LABEL[s] for s in STRATEGIES], fontsize=7)
        ax.set_yticks(range(3), [STRATEGY_LABEL[s] for s in STRATEGIES], fontsize=7)
        ax.set_title(fam, fontsize=9)
    fig.text(0.5, -0.02, "Cell: median per-series RelNaive(row) − RelNaive(column) (> 0: column better); "
             "series where the column is better / tied / worse; Holm p; D26 label for the column.", ha="center", fontsize=7)
    _save(fig, path)


def stl_feature_tercile(pooled: pd.DataFrame, path: Path) -> None:
    features = sorted(pooled["feature_name"].astype(str).unique())
    fig, axes = plt.subplots(2, len(features), figsize=(2.6 * len(features), 5), squeeze=False, sharey="row")
    for i, strategy in enumerate(["stl_sn", "stl_ac"]):
        for j, feature in enumerate(features):
            ax = axes[i, j]
            sub = pooled[(pooled["feature_name"] == feature) & (pooled["strategy"] == strategy)]
            sub = sub.set_index(sub["bucket"].astype(str)).reindex(["Low", "Medium", "High"])
            ax.bar(range(3), sub["median_per_series"], color=COLORS[strategy])
            ax.axhline(0, color="black", lw=0.6)
            ax.set_xticks(range(3), ["L", "M", "H"], fontsize=7)
            ax.set_title(f"{feature.replace('feature_', '')}\n{STRATEGY_LABEL[strategy]}", fontsize=7)
    _save(fig, path)


def stl_family_heatmap(per_family: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3), squeeze=False)
    for ax, strategy in zip(axes[0], ["stl_sn", "stl_ac"]):
        sub = per_family[per_family["strategy"] == strategy].astype({"feature_name": str, "bucket": str, "family": str})
        sub = sub.assign(column=sub["feature_name"].str.replace("feature_", "") + "|" + sub["bucket"].str[0])
        grid = sub.pivot(index="family", columns="column", values="median_per_series").reindex(FAMILIES)
        v = max(float(np.nanmax(np.abs(grid.to_numpy()))), 1e-9)
        ax.imshow(grid.to_numpy(), cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
        ax.set_yticks(range(len(grid.index)), grid.index, fontsize=7)
        ax.set_xticks(range(len(grid.columns)), grid.columns, rotation=90, fontsize=6)
        ax.set_title(STRATEGY_LABEL[strategy], fontsize=9)
    _save(fig, path)


def ft_linear_nonlinear(ft: pd.DataFrame, path: Path) -> None:
    models = [m for m in GLOBAL_MODELS if m in set(ft["model"].astype(str))]
    groups = {"nonlinear": [m for m in models if LINEARITY[m] == "nonlinear"], "linear": [m for m in models if LINEARITY[m] == "linear"]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, gridspec_kw={"width_ratios": [len(groups["nonlinear"]) or 1, len(groups["linear"]) or 1]})
    for ax, (kind, ms) in zip(axes, groups.items()):
        _box(ax, [ft.loc[ft["model"] == m, "delta"].dropna() for m in ms], list(range(len(ms))), "C0" if kind == "nonlinear" else "C3")
        ax.set_xticks(range(len(ms)), ms, rotation=30, fontsize=7)
        ax.axhline(0, color="black", ls="--", lw=0.6)
        ax.set_title(kind, fontsize=9)
    axes[0].set_ylabel("ΔFT (instances)")
    _save(fig, path)


def seed_spread(spread: pd.DataFrame, path: Path) -> None:
    models = [m for m in GLOBAL_MODELS if m in set(spread["model"].astype(str))]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for k, strategy in enumerate(STRATEGIES):
        data = [spread.loc[(spread["model"] == m) & (spread["strategy"] == strategy), "range"] for m in models]
        _box(ax, data, [i * 4 + k for i in range(len(models))], COLORS[strategy])
    ax.set_xticks([i * 4 + 1 for i in range(len(models))], models, fontsize=7)
    ax.set_ylabel("per-series RelNaive range across seeds")
    _save(fig, path)


def tuning_curves(curves: pd.DataFrame, path: Path) -> None:
    models = [m for m in GLOBAL_MODELS if m in set(curves["model"])]
    fig, axes = plt.subplots(1, len(models), figsize=(2.4 * len(models), 2.6), squeeze=False)
    for ax, model in zip(axes[0], models):
        for (frequency, target), g in curves[curves["model"] == model].groupby(["frequency", "target"]):
            ax.plot(g["trial"], g["best_so_far"], lw=0.8, ls="-" if frequency == "monthly" else "--")
        ax.set_title(model, fontsize=8)
        ax.tick_params(labelsize=6)
    axes[0, 0].set_ylabel("best validation MASE", fontsize=7)
    _save(fig, path)


def compute_cost(a9: pd.DataFrame, path: Path) -> None:
    g = a9.groupby(["family", "strategy"], observed=True)["total_fit_hours"].sum().reset_index()
    _bars(g.assign(median=g["total_fit_hours"], ci_low=g["total_fit_hours"], ci_high=g["total_fit_hours"], label=""),
          path, FAMILIES, STRATEGIES, "total training time (hours)")
