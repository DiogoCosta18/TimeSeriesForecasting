"""Analyses R1 to R3 of the supervisor's review (protocol change log v2.12 (b)): NOT pre-registered.

First run from the paper's tools (``review_analyses.py``, outputs ``full3_20261011/review_20261020``);
here so that S4 (v2.12 (f)) and E10 (v2.12 (g)) reuse them. On the same run this module gives
the same tables. Per-series values follow the run's rules (D27 aggregation, Wilcoxon with Holm,
the D26 label, bootstrap streams keyed by contrast name).

- R1: H1 under each seed on the views of A15 (evolving-seasonality sample, cohort scope; the
  stochastic models from the seed, the rest from the main seed); under the run's specification the
  labels must equal ``a15_seed_labels.csv``.
- R2: on the series with stable seasonality (evolving seasonality below its first tercile cut) and
  on those with strong seasonality (seasonal strength of E7 in its highest tercile, cut per
  frequency), the best STL-SN model of each global family (lowest mean per-series RelNaive(10) in
  that set) against the three best Direct configurations; per series Direct minus STL-SN, Holm over
  the table. ``stl`` names another STL-SN arm (an S4 variant).
- R3: evolving seasonality (phi4), seasonal strength (phi7) and, with E10, evolving seasonality under
  periodic STL (phi4'): Delta_STL by tercile with the Jonckheere-Terpstra test (Holm over the
  table), Spearman correlations and partial Spearman correlations with percentile bootstrap intervals.

    python -m src.analysis.review --run RUN --strength E7_TABLE --out OUT [--statistical-stl p1] [--e10 E10_TABLE]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from src.analysis import hypotheses as hy
from src.analysis import stats
from src.analysis.data import PRIMARY, AnalysisError, load_gated, per_series, stl_deltas
from src.data.frozen import DEFAULT_MANIFEST
from src.stages.io import code_commit, sha256_json, utc_now, write_json
from src.stages.tasks import STOCHASTIC_MODELS

STATUS = "exploratory: not pre-registered (protocol change log v2.12 (b)); answers the supervisor's review of 2026-10-19"
EV = "feature_evolving_seasonality"
EVP = "evolving_seasonality_periodic"                 # phi4' of E10
TERCILES = ["Low", "Medium", "High"]
GLOBAL = ["ml", "neural", "transformer"]
BEST_DIRECT = ["PatchTST", "ETS", "NHITS"]           # Table 3 of the paper: the three best configurations, all Direct


def _tercile(values: pd.Series, frequency: pd.Series, cuts: dict) -> np.ndarray:
    out = np.select([values <= frequency.map(lambda k: cuts[k][0]), values <= frequency.map(lambda k: cuts[k][1])],
                    TERCILES[:2], TERCILES[2]).astype(object)
    out[values.isna().to_numpy()] = None
    return out


def cells(features: pd.DataFrame, bucket_summary: pd.DataFrame, strength: pd.DataFrame,
          periodic: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each sampled series' frequency, phi4, phi7 (and phi4'), their terciles and the stable and
    strong flags; phi4 is cut by the prepare bundle's cuts, phi7 and phi4' per frequency over the
    sampled series."""
    f = features[["unique_id", "frequency", EV]].astype({"unique_id": str, "frequency": str})
    cut = bucket_summary[bucket_summary["feature_name"] == EV].astype({"frequency": str}).set_index("frequency")
    x = f.merge(strength[["unique_id", "frequency", "seasonal_strength"]].astype({"unique_id": str, "frequency": str}),
                on=["unique_id", "frequency"])
    ev_cuts = {fq: (cut.loc[fq, "tercile_cut_1"], cut.loc[fq, "tercile_cut_2"]) for fq in cut.index}
    x["ev_tercile"] = _tercile(x[EV], x["frequency"], ev_cuts)
    one = x.dropna(subset=["seasonal_strength"])
    q = {fq: tuple(np.quantile(g["seasonal_strength"], [1 / 3, 2 / 3])) for fq, g in one.groupby("frequency")}
    x["fs_tercile"] = _tercile(x["seasonal_strength"], x["frequency"], q)
    x["stable"] = x[EV] < x["frequency"].map(cut["tercile_cut_1"])        # E2's rule: below the first cut
    x["strong"] = x["fs_tercile"] == "High"
    rows = [{"frequency": fq, "fs_cut_1": v[0], "fs_cut_2": v[1], "ev_cut_1": ev_cuts[fq][0], "ev_cut_2": ev_cuts[fq][1]}
            for fq, v in sorted(q.items())]
    if periodic is not None:
        x = x.merge(periodic[["unique_id", "frequency", EVP]].astype({"unique_id": str, "frequency": str}),
                    on=["unique_id", "frequency"], how="left", validate="one_to_one")
        p = {fq: tuple(np.quantile(g[EVP], [1 / 3, 2 / 3])) for fq, g in x.dropna(subset=[EVP]).groupby("frequency")}
        x["evp_tercile"] = _tercile(x[EVP], x["frequency"], p)
        for r in rows:
            r.update(evp_cut_1=p[r["frequency"]][0], evp_cut_2=p[r["frequency"]][1])
    return x, pd.DataFrame(rows)


# --- R1 ------------------------------------------------------------------------------------------------

def r1_seeds(instances: pd.DataFrame, seeds: list[int], s0: int) -> pd.DataFrame:
    inst = instances[(instances["feature_name"] == EV) & (instances["scope"] == "cohort")]
    stochastic = inst["model"].isin(STOCHASTIC_MODELS)
    out = []
    for seed in [s0, *seeds]:
        view = pd.concat([inst[~stochastic & inst["main_seed"]], inst[stochastic & (inst["seed_key"] == seed)]],
                         ignore_index=True)
        view["seed_key"] = np.where(view["seed_key"] == -1, -1, s0)
        t = hy.h1(view, s0, f"A15[{seed}]-H1")
        out.append(t[["family", "strategy", "n_series", "median", "ci_low", "ci_high", "p_holm", "label"]].assign(seed=seed))
    return pd.concat(out, ignore_index=True)


def r1_check(r1: pd.DataFrame, a15_labels: pd.DataFrame, families: list[str] | None = None) -> list[str]:
    """Rows of R1 whose label differs from A15's (only ``families``, when given)."""
    a15 = a15_labels[a15_labels["hypothesis"] == "H1"].set_index("contrast")
    bad = []
    for _, r in r1.iterrows():
        if families is not None and r.family not in families:
            continue
        want = a15.loc[f"{r.family}|{r.strategy}", f"label_seed_{r.seed}"]
        if want != r.label:
            bad.append(f"{r.family} {r.strategy} seed {r.seed}: {r.label} vs {want}")
    return bad


# --- R2 ------------------------------------------------------------------------------------------------

def series_model_means(main: pd.DataFrame) -> pd.DataFrame:
    """Per (series, model, strategy): mean RelNaive(10) over windows and feature samples (cohort, main seed)."""
    c = main[main["scope"].astype(str) == "cohort"].dropna(subset=[PRIMARY])
    keys = ["unique_id", "frequency", "family", "model", "strategy"]
    c = c.astype({k: str for k in keys})
    return c.groupby(keys, sort=True)[PRIMARY].mean().reset_index()


def r2_best_direct(means: pd.DataFrame, x: pd.DataFrame, s0: int, stl: str = "stl_sn", family_id: str = "R2") -> pd.DataFrame:
    wide = means.pivot_table(index=["unique_id", "frequency"], columns=["model", "strategy"], values=PRIMARY)
    fam_of = means.drop_duplicates("model").set_index("model")["family"]
    rows = []
    for cell in ["stable", "strong"]:
        ids = x.loc[x[cell], ["unique_id", "frequency"]]
        w = wide.loc[pd.MultiIndex.from_frame(ids).intersection(wide.index)]
        for fam in GLOBAL:
            models = [m for m in fam_of.index if fam_of[m] == fam and (m, stl) in w.columns]
            if not models:
                continue
            best = min(models, key=lambda m: w[(m, stl)].mean())
            for comp in BEST_DIRECT:
                delta = (w[(comp, "direct")] - w[(best, stl)]).dropna()
                df = delta.rename("delta").reset_index()
                rows.append(df.assign(cell=cell, family=fam, stl_sn_model=best, comparator=comp))
    ps = pd.concat(rows, ignore_index=True)
    by = ["cell", "frequency", "family", "stl_sn_model", "comparator"]
    tests = hy.one_sample(ps, by, s0, family_id)
    desc = ps.groupby(by, sort=True)["delta"].agg(win=lambda v: float((v > 0).mean())).reset_index()
    return tests.merge(desc, on=by)


# --- R3 ------------------------------------------------------------------------------------------------

def partial_rho(a, b, c) -> float:
    """Spearman correlation of a and b given c: the partial correlation of their ranks."""
    ra, rb, rc = (sps.rankdata(v) for v in (a, b, c))
    r = np.corrcoef([ra, rb, rc])
    ab, ac, bc = r[0, 1], r[0, 2], r[1, 2]
    return float((ab - ac * bc) / np.sqrt((1 - ac ** 2) * (1 - bc ** 2)))


def partial_ci(a, b, c, base_seed: int, cid: str, n: int = stats.N_BOOTSTRAP) -> tuple[float, float]:
    rng = stats.bootstrap_rng(base_seed, cid)
    vals = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(a), size=len(a))
        vals[i] = partial_rho(a[idx], b[idx], c[idx])
    tail = (1 - stats.CI_LEVEL) / 2
    return float(np.quantile(vals, tail)), float(np.quantile(vals, 1 - tail))


def gradients(pooled: pd.DataFrame, x: pd.DataFrame, s0: int, by: list[str], family_id: str = "R3") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Delta by tercile of each feature with the Jonckheere-Terpstra test (Holm over the table) and the
    correlations, per frequency and ``by`` group; ``pooled`` holds one per-series delta per group."""
    feats = [("evolving_seasonality", EV, "ev_tercile"), ("seasonal_strength", "seasonal_strength", "fs_tercile")]
    if EVP in x.columns:
        feats.append(("evolving_seasonality_periodic", EVP, "evp_tercile"))
    g = pooled.merge(x, on=["unique_id", "frequency"]).dropna(subset=["seasonal_strength"])
    terc, corr = [], []
    for key, h in g.groupby(["frequency", *by], sort=True):
        fq, *rest = key
        head = {"frequency": fq, **dict(zip(by, rest))}
        for name, col, tcol in feats:
            groups = [h.loc[h[tcol] == t, "delta"].to_numpy() for t in TERCILES]
            jt = stats.jonckheere_terpstra(groups)
            terc.append({**head, "feature": name, **{f"n_{t.lower()}": len(v) for t, v in zip(TERCILES, groups)},
                         **{f"median_{t.lower()}": float(np.median(v)) for t, v in zip(TERCILES, groups)},
                         "high_minus_low": float(np.median(groups[2]) - np.median(groups[0])), "z": jt["z"], "p": jt["p"]})
        tag = "|".join([fq, *map(str, rest)])
        a, b, dl = h[EV].to_numpy(), h["seasonal_strength"].to_numpy(), h["delta"].to_numpy()
        row = {**head, "n_series": len(h)}
        pairs = {"evolving": (a, dl), "strength": (b, dl), "evolving_strength": (a, b)}
        partials = {"evolving_given_strength": (a, dl, b), "strength_given_evolving": (b, dl, a)}
        if EVP in x.columns:
            hp = h.dropna(subset=[EVP])
            p, bp, dp, ap = hp[EVP].to_numpy(), hp["seasonal_strength"].to_numpy(), hp["delta"].to_numpy(), hp[EV].to_numpy()
            row["n_series_periodic"] = len(hp)
            pairs.update({"evolving_periodic": (p, dp), "evolving_periodic_strength": (p, bp), "evolving_evolving_periodic": (ap, p)})
            partials.update({"evolving_periodic_given_strength": (p, dp, bp)})
        for k, (u, v) in pairs.items():
            sp = stats.spearman(u, v, s0, f"{family_id}|{tag}|{k}")
            row.update({f"rho_{k}": sp["rho"], f"rho_{k}_ci_low": sp["rho_ci_low"], f"rho_{k}_ci_high": sp["rho_ci_high"]})
        for k, (u, v, w) in partials.items():
            lo, hi = partial_ci(u, v, w, s0, f"{family_id}|{tag}|partial|{k}")
            row.update({f"partial_{k}": partial_rho(u, v, w), f"partial_{k}_ci_low": lo, f"partial_{k}_ci_high": hi})
        corr.append(row)
    terc = pd.DataFrame(terc)
    terc["p_holm"] = stats.holm(terc["p"].to_numpy())
    terc["label"] = [stats.d26_label(p, e) for p, e in zip(terc["p_holm"], terc["high_minus_low"])]
    return terc, pd.DataFrame(corr)


def family_terciles(fam: pd.DataFrame, x: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Median per-series delta by family, ``by`` group and tercile of each feature; all and per frequency."""
    gf = fam.merge(x, on=["unique_id", "frequency"]).dropna(subset=["fs_tercile"])
    feats = [("fs_tercile", "seasonal_strength"), ("ev_tercile", "evolving_seasonality")]
    if "evp_tercile" in x.columns:
        feats.append(("evp_tercile", "evolving_seasonality_periodic"))
    parts = []
    for tcol, feat in feats:
        for keys, fq in ((["family", *by], "all"), (["family", *by, "frequency"], None)):
            t = gf.dropna(subset=[tcol]).groupby(keys + [tcol])["delta"].median().unstack(tcol).reindex(columns=TERCILES).reset_index()
            parts.append(t.assign(feature=feat, **({"frequency": fq} if fq else {})))
    return pd.concat(parts, ignore_index=True)


def r3(main: pd.DataFrame, x: pd.DataFrame, s0: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """R3 on Delta_STL of the run's strategies (cohort scope), families pooled with equal weight."""
    deltas = stl_deltas(main[main["scope"].astype(str) == "cohort"])
    pooled = per_series(deltas, ["frequency", "strategy"], family_balanced=True).astype({"unique_id": str, "frequency": str})
    fam = per_series(deltas, ["frequency", "strategy", "family"]).astype({"unique_id": str, "frequency": str, "family": str})
    terc, corr = gradients(pooled.astype({"strategy": str}), x, s0, ["strategy"])
    return terc, corr, family_terciles(fam.astype({"strategy": str}), x, ["strategy"])


# --- stage ---------------------------------------------------------------------------------------------

def run_review(run_dir: Path, strength_path: Path, out: Path, statistical_stl: str = "run",
               e10_path: Path | None = None, manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise AnalysisError(f"{out} is not empty; exploratory outputs are never overwritten")
    d = load_gated(run_dir, manifest_path)
    if statistical_stl == "p1":
        from src.stages.sensitivity3 import main_specification
        d = main_specification(d)
    s0 = d.main_seed
    strength = pd.read_csv(strength_path)
    periodic = pd.read_csv(e10_path) if e10_path is not None else None
    x, cuts = cells(d.tables["features"], d.tables["bucket_summary"], strength, periodic)
    r1 = r1_seeds(d.instances, list(d.config["seed_check_seeds"]), s0)
    a15 = pd.read_csv(Path(run_dir) / "analysis" / "tables" / "a15_seed_labels.csv")
    bad = r1_check(r1, a15, None if statistical_stl == "run" else GLOBAL)
    if bad:
        raise AnalysisError(f"R1 does not reproduce A15: {bad}")
    terc, corr, fam_terc = r3(d.main, x, s0)
    outputs = {"r1_seed_h1": r1, "r2_stl_sn_vs_best_direct": r2_best_direct(series_model_means(d.main), x, s0),
               "r3_terciles": terc, "r3_correlations": corr, "r3_family_terciles": fam_terc, "cuts": cuts}
    out.mkdir(parents=True, exist_ok=True)
    for name, df in outputs.items():
        df.to_csv(out / f"{name}.csv", index=False)
    record = {"status": STATUS, "code_commit": code_commit(), "created_at_utc": utc_now(),
              "statistical_stl": statistical_stl, "strength_table": str(strength_path), "e10_table": str(e10_path),
              "merge_result_sha256": d.record["result_sha256"], "main_seed": s0,
              "n_stable": int(x["stable"].sum()), "n_strong": int(x["strong"].sum()), "r1_reproduces_a15": True,
              "tables": {name: sha256_json(df.to_dict(orient="list")) for name, df in outputs.items()}}
    write_json(out / "review.json", record)
    return {"out": str(out), "tables": len(outputs)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.analysis.review", description=STATUS)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--strength", required=True, type=Path, help="E7's e7_strength.csv")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--statistical-stl", choices=["run", "p1"], default="run")
    parser.add_argument("--e10", type=Path, help="E10's e10_phi4.csv (adds phi4')")
    args = parser.parse_args(argv)
    print(run_review(args.run, args.strength, args.out, args.statistical_stl, args.e10))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
