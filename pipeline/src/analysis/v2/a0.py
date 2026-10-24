"""Step A0: the canonical tables reproduce the published numbers before any revision analysis reads them.

Each value is recomputed from fc_errors with the definitions of the frozen analysis (an instance's delta is its mean
over windows and models; a series' delta the mean over its instances; the estimand the median over the series) and
compared with numbers.tex, and with Table 3 (tab_top10.tex), at the printed precision. The report lists every value;
one mismatch fails the step.

Run:  python -m src.analysis.v2.a0 --tables OUT_DIR --numbers numbers.tex --top10 tab_top10.tex --report A0_check.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecast.registry import GLOBAL_MODELS
from src.stages import sensitivity4 as s4
from src.stages.tasks import SEED_CHECK_FEATURE, STOCHASTIC_MODELS

from src.analysis.v2.tables import instances, load

MAIN_SEED = 20260521
SEEDS = [20260521, 20260522, 20260523]
FAMILIES = ["statistical", "ml", "neural", "transformer"]
GLOBAL = ["ml", "neural", "transformer"]
BASE = ["series_id", "freq", "feature_sample", "window", "family", "model", "strategy", "setting", "scope", "group", "seed"]


def contrast(df: pd.DataFrame, arm: pd.Series, first: str, second: str, vary: list[str]) -> pd.DataFrame:
    """Per forecast rn[first] - rn[second], paired on every key of BASE but ``vary``; unpaired forecasts are dropped."""
    keys = [k for k in BASE if k not in vary]
    d = df.assign(arm=np.asarray(arm, dtype=object))
    d = d[d["arm"].isin([first, second])]
    s = pd.Series(d["rn"].to_numpy(), index=pd.MultiIndex.from_frame(d[keys + ["arm"]]))
    if s.index.duplicated().any():
        raise ValueError(f"forecasts not unique on {keys} for {first} - {second}")
    wide = s.unstack("arm")
    out = wide.index.to_frame(index=False)
    out["delta"] = (wide[first] - wide[second]).to_numpy()
    return out.dropna(subset=["delta"])


def estimand(d: pd.DataFrame, by: list[str]) -> pd.Series:
    """Median over series of the series' mean over its instances of the instance's mean over windows and models."""
    inst = d.groupby(["series_id", "feature_sample", *by], observed=True)["delta"].mean().reset_index()
    series = inst.groupby(["series_id", *by], observed=True)["delta"].mean().reset_index()
    return series.groupby(by, observed=True)["delta"].median()


def arm(df: pd.DataFrame) -> pd.Series:
    return df["strategy"].astype(str) + "/" + df["setting"].astype(str)


def main_seed(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["seed"].isin([MAIN_SEED, -1])]


def compute(fc: pd.DataFrame, inst: pd.DataFrame, counts: dict) -> dict[str, float | str]:
    """Every recomputed value, keyed as in numbers.tex (Table 3 as tab3-<rank>-<statistic>)."""
    v: dict[str, float | str] = {"forecasts-total": counts["run"]}
    cohort = main_seed(fc[(fc["scope"] == "cohort") & (fc["setting"] != "default_m")])
    spec = cohort[cohort["source"].isin(["run", "s3"])]

    # Table 3: mean and median RelNaive over the distinct forecasts of each configuration
    top = spec.groupby(["model", "strategy"], observed=True)["rn"].agg(["mean", "median"]).sort_values("mean").head(3)
    for i, ((model, strategy), r) in enumerate(top.iterrows(), 1):
        v[f"tab3-{i}-config"] = f"{model} {strategy}"
        v[f"tab3-{i}-mean"], v[f"tab3-{i}-median"] = r["mean"], r["median"]

    # Table 4 (H1): Direct minus STL, the default setting
    spec_inst = instances(spec, inst)
    for s in ["stl_sn", "stl_ac"]:
        pair = ("direct/none", f"{s}/default", ["strategy", "setting"])
        fcst = contrast(spec, arm(spec), *pair)                        # over distinct forecasts
        per = estimand(contrast(spec_inst, arm(spec_inst), *pair), ["family"])
        for f in FAMILIES:
            d = fcst[fcst["family"] == f]["delta"]
            v[f"h1-{f}-{s}-mean"], v[f"h1-{f}-{s}-fmedian"] = d.mean(), d.median()
            v[f"h1-{f}-{s}-win"] = (d > 0).mean()
            v[f"h1-{f}-{s}-median"] = per[f]

    # Table 8 (S4): each setting against Direct and against the default; the re-tuning check
    s4rows = instances(cohort[cohort["source"].isin(["run", "s3", "s4"])], inst)
    for label in ["periodic", "s25", "periodic_tuned"]:
        sub = s4rows[s4rows["model"].isin(s4.models_of(label))]
        a = arm(sub)
        if label == "periodic_tuned":
            d = contrast(sub, a, "stl_sn/periodic", "stl_sn/periodic_tuned", ["strategy", "setting"])
            for m, x in estimand(d, ["model"]).items():
                v[f"s4-tuned-{m.lower()}"] = x
            continue
        direct = contrast(sub, a, "direct/none", f"stl_sn/{label}", ["strategy", "setting"])
        default = contrast(sub, a, "stl_sn/default", f"stl_sn/{label}", ["strategy", "setting"])
        for f, x in estimand(direct, ["family"]).items():
            v[f"s4-{f}-{label}-direct"] = x
        for f, x in estimand(default, ["family"]).items():
            v[f"s4-{f}-{label}-default"] = x
        if label == "periodic":
            for m, x in estimand(direct, ["model"]).items():
                v[f"s4-{m.lower()}-periodic"] = x

    # Table 11 (H5) and Table 12 (S5): cohort against tercile and random-group models, full steps
    pools = main_seed(fc[fc["source"].isin(["run", "s5"]) & fc["family"].isin(GLOBAL)
                         & fc["scope"].isin(["cohort", "tercile", "random"])])
    sc = pools["scope"].astype(str)
    for name, first, second in [("deltaft", "cohort", "tercile"), ("deltarand", "cohort", "random"),
                                ("contrast", "random", "tercile")]:
        d = estimand(contrast(pools, sc, first, second, ["scope", "group"]), ["family", "strategy"])
        for (f, s), x in d.items():
            v[f"s5-{name}-{f}-{s}"] = x
            if name == "deltaft":
                v[f"h5-{f}-{s}-median"] = x

    # Table C7 (S4s): the periodic setting under each seed, on the evolving-seasonality sample
    ev = fc[(fc["scope"] == "cohort") & fc["model"].isin([*GLOBAL_MODELS, *s4.STATISTICAL])]
    ev = ev[(ev["feature_sample"].isin([SEED_CHECK_FEATURE, "none"]))]
    stoch = ev["model"].isin(STOCHASTIC_MODELS)
    for sd in SEEDS:
        direct = ev[(ev["strategy"] == "direct") & ev["source"].eq("run")
                    & ((~stoch & ev["seed"].isin([MAIN_SEED, -1])) | (stoch & (ev["seed"] == sd)))]
        periodic = ev[(ev["setting"] == "periodic") & ev["source"].isin(["s4", "s4s"])
                      & ((~stoch & ev["seed"].isin([MAIN_SEED, -1])) | (stoch & (ev["seed"] == sd)))]
        both = instances(pd.concat([direct, periodic], ignore_index=True), inst)
        both = both[both["feature_sample"] == SEED_CHECK_FEATURE]
        d = contrast(both, both["strategy"].astype(str), "direct", "stl_sn", ["strategy", "setting"])
        for f, x in estimand(d, ["family"]).items():
            v[f"s4s-{f}-{sd}"] = x
    return v


def published(numbers: Path, top10: Path) -> dict[str, str]:
    """The printed values: numbers.tex macros and the first three rows of Table 3."""
    out = {}
    for line in Path(numbers).read_text(encoding="utf-8").splitlines():
        m = re.match(r"\\@namedef\{R@([^}]*)\}\{(.*)\}$", line)
        if m:
            out[m.group(1)] = m.group(2)
    strat = {"Direct": "direct", "STL-SN": "stl_sn", "STL-AC": "stl_ac"}
    for line in Path(top10).read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
        if len(cells) == 7 and cells[0] in ("1", "2", "3"):
            out[f"tab3-{cells[0]}-config"] = f"{cells[2]} {strat[cells[3]]}"
            out[f"tab3-{cells[0]}-mean"], out[f"tab3-{cells[0]}-median"] = cells[4], cells[5]
    return out


def compare(value, printed: str) -> tuple[str, bool]:
    """The value as printed with the printed precision, and whether it is the printed value."""
    if isinstance(value, str):
        return value, value == printed
    text = printed.replace(r"\ensuremath{", "").rstrip("}").replace("{,}", "")
    scale = 100.0 if text.endswith(r"\%") else 1.0
    text = text.removesuffix(r"\%")
    decimals = len(text.split(".")[1]) if "." in text else 0
    got = round(float(value) * scale, decimals)
    return f"{got:.{decimals}f}", abs(got - float(text)) < 10 ** -(decimals + 3)


def run(tables: Path, numbers: Path, top10: Path, report: Path) -> pd.DataFrame:
    from src.stages.io import read_json

    fc, _, inst = load(tables)
    values = compute(fc, inst, read_json(Path(tables) / "tables.json")["rows"])
    printed = published(numbers, top10)
    rows = []
    for key, value in values.items():
        if key not in printed:
            continue                                   # recomputed but not printed (e.g. a model outside Table C6)
        got, ok = compare(value, printed[key])
        rows.append({"key": key, "expected": printed[key], "obtained": got,
                     "value": value if not isinstance(value, str) else np.nan, "match": ok})
    out = pd.DataFrame(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(report, index=False, float_format="%.10g")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.analysis.v2.a0", description=__doc__.splitlines()[0])
    parser.add_argument("--tables", required=True, type=Path)
    parser.add_argument("--numbers", required=True, type=Path)
    parser.add_argument("--top10", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    a = parser.parse_args(argv)
    out = run(a.tables, a.numbers, a.top10, a.report)
    bad = out[~out["match"]]
    print(f"A0: {len(out)} values, {len(bad)} differ")
    if len(bad):
        print(bad.to_string(index=False))
    return 1 if len(bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
