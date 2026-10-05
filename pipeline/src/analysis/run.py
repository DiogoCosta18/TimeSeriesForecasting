"""Stage R5, analyse (protocol Section 5; Table 6; test I3).

``run_analysis`` reads a gated result bundle and writes every table (CSV) and figure (PNG)
of Table 6 into ``out``, plus ``analysis.json`` with the merge and gate hashes, the
analysis code commit, the SHA-256 of every output and the map from paper section to
outputs. It refuses a bundle whose gates did not pass and an ``out`` that is not empty;
the same bundle and code give the same tables.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis import figures as fg
from src.analysis import hypotheses as hy
from src.analysis import secondary as sc
from src.analysis.data import AnalysisError, ft_deltas, load_gated
from src.data.frozen import DEFAULT_MANIFEST, sha256_file
from src.stages.io import code_commit, utc_now, write_json

ANALYSIS_FORMAT = "rerun-v2-analysis-1"
# Table 6 of the protocol: paper section -> outputs (tables/*.csv, figures/*.png).
OUTPUTS = {
    "S2 Data": ["tables/data_counts.csv", "figures/sampling_illustration.png", "tables/a11_ks.csv"],
    "S2 Models": ["tables/model_table.csv", "tables/search_spaces.csv", "tables/a10_frozen_configurations.csv"],
    "S2 Features": ["figures/feature_distributions.png", "tables/a12_bucket_support.csv", "tables/a14_feature_correlations.csv"],
    "S3.1": ["tables/a1_standing.csv", "tables/a1_top_configurations.csv", "figures/relnaive_distributions.png",
             "tables/h7_summary.csv", "tables/h7_ranks.csv", "tables/h7_pairs.csv",
             "figures/cd_cohort_all.png", "figures/cd_cohort_monthly.png", "figures/cd_cohort_quarterly.png"],
    "S3.2": ["tables/h1.csv", "figures/stl_delta_bars.png", "tables/h2.csv", "tables/h3_mcm.csv", "figures/mcm.png"],
    "S3.3": ["tables/stl_feature_tercile.csv", "tables/stl_feature_tercile_by_family.csv", "tables/h4.csv",
             "tables/h4_by_family.csv", "figures/stl_feature_tercile.png", "figures/stl_family_heatmap.png", "tables/a7_guard.csv"],
    "S4": ["tables/ft_overall.csv", "tables/ft_by_bucket.csv", "tables/ft_by_family_strategy.csv", "tables/ft_by_model.csv",
           "tables/ft_by_linearity.csv", "tables/h5.csv", "figures/ft_bars.png", "tables/h6.csv", "figures/ft_linear_nonlinear.png",
           "tables/a13_stl_per_model.csv", "tables/a13_ft_per_model.csv", "tables/a13_ft_per_model_bucket.csv",
           "tables/ft_significance_counts.csv",
           "figures/cd_tercile_all.png", "figures/cd_tercile_monthly.png", "figures/cd_tercile_quarterly.png"],
    "Appendix": ["tables/a2_frequency.csv", "tables/a3_source.csv", "tables/a4_window.csv", "tables/a5_metric.csv",
                 "tables/a6_cap.csv", "tables/a8_failures_by_model.csv", "tables/a8_task_failures.csv", "tables/a9_compute.csv",
                 "figures/compute_cost.png", "tables/a10_studies.csv", "figures/tuning_curves.png",
                 "tables/a15_seed_spread.csv", "tables/a15_seed_labels.csv", "figures/seed_spread.png"],
}


def run_analysis(run_dir: Path, out: Path, manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise AnalysisError(f"{out} is not empty; analysis outputs are never overwritten")
    d = load_gated(run_dir, manifest_path)
    main, s0, tables = d.main, d.main_seed, d.tables
    t = {}

    def table(name: str, df: pd.DataFrame) -> None:
        t[name] = df
        path = out / "tables" / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, float_format="%.10g")

    table("data_counts", sc.data_counts(d))
    table("a11_ks", sc.a11_ks(tables))
    table("model_table", sc.model_table(d))
    table("search_spaces", sc.search_spaces())
    configs, studies, curves = sc.a10_tuning(d)
    table("a10_frozen_configurations", configs)
    table("a10_studies", studies)
    table("a12_bucket_support", sc.a12_buckets(tables))
    table("a14_feature_correlations", sc.a14_correlations(tables))

    standing, top = sc.a1_standing(main)
    table("a1_standing", standing)
    table("a1_top_configurations", top)
    h7 = {(scope, f or "all"): hy.h7(main, scope, f) for scope in ("cohort", "tercile") for f in (None, "monthly", "quarterly")}
    table("h7_summary", pd.DataFrame([r["summary"] for r in h7.values()]))
    table("h7_ranks", pd.concat([r["ranks"].assign(scope=k[0], frequency=k[1]) for k, r in h7.items()], ignore_index=True))
    table("h7_pairs", pd.concat([r["pairs"].assign(scope=k[0], frequency=k[1]) for k, r in h7.items()], ignore_index=True))

    table("h1", hy.h1(main, s0))
    table("h2", hy.h2(main, s0))
    table("h3_mcm", hy.h3(main, s0))
    pooled, per_family = sc.stl_feature_tercile(main, s0)
    table("stl_feature_tercile", pooled)
    table("stl_feature_tercile_by_family", per_family)
    table("h4", hy.h4(main, tables, s0))
    table("h4_by_family", hy.h4_by_family(main, tables, s0))
    table("a7_guard", sc.a7_guard(main, tables, s0))

    for name, df in sc.ft_descriptives(main).items():
        table(name, df)
    table("h5", hy.h5(main, s0))
    table("h6", hy.h6(main, s0))
    for name, df in sc.a13_per_model(main, s0).items():
        table(name, df)

    a2, a3 = sc.a2_a3_splits(main, tables, s0)
    table("a2_frequency", a2)
    table("a3_source", a3)
    table("a4_window", sc.a4_windows(main))
    table("a5_metric", sc.a5_metrics(main, s0))
    table("a6_cap", sc.a6_caps(main))
    by_model, task_failures = sc.a8_failures(d)
    table("a8_failures_by_model", by_model)
    table("a8_task_failures", task_failures)
    table("a9_compute", sc.a9_compute(d.rows))
    spread, labels, spread_long = sc.a15_seeds(d)
    table("a15_seed_spread", spread)
    table("a15_seed_labels", labels)

    figs = out / "figures"
    fg.sampling_illustration(tables, figs / "sampling_illustration.png")
    fg.feature_distributions(tables, figs / "feature_distributions.png")
    fg.relnaive_distributions(main, figs / "relnaive_distributions.png")
    for (scope, f), r in h7.items():
        fg.cd_diagram(r, figs / f"cd_{scope}_{f}.png")
    fg.stl_delta_bars(t["h1"], figs / "stl_delta_bars.png")
    fg.mcm(t["h3_mcm"], figs / "mcm.png")
    fg.stl_feature_tercile(pooled, figs / "stl_feature_tercile.png")
    fg.stl_family_heatmap(per_family, figs / "stl_family_heatmap.png")
    fg.ft_bars(t["h5"], figs / "ft_bars.png")
    fg.ft_linear_nonlinear(ft_deltas(main), figs / "ft_linear_nonlinear.png")
    fg.compute_cost(t["a9_compute"], figs / "compute_cost.png")
    fg.tuning_curves(curves, figs / "tuning_curves.png")
    if len(spread_long):
        fg.seed_spread(spread_long, figs / "seed_spread.png")

    expected = [p for paths in OUTPUTS.values() for p in paths]
    produced = sorted(str(p.relative_to(out)).replace("\\", "/") for p in out.rglob("*") if p.is_file())
    absent = sorted(set(expected) - set(produced))
    if absent and not (set(absent) <= {"figures/seed_spread.png"} and not d.config["seed_check_seeds"]):
        raise AnalysisError(f"outputs of Table 6 not produced: {absent}")
    record = {"format": ANALYSIS_FORMAT, "created_at_utc": utc_now(), "analysis_code_commit": code_commit(),
              "merge_result_sha256": d.record["result_sha256"], "gates_created_at_utc": d.gates["created_at_utc"],
              "provenance": d.record["provenance"], "outputs_by_section": OUTPUTS,
              "files": {p: sha256_file(out / p) for p in produced}}
    write_json(out / "analysis.json", record)
    return {"tables": len(t), "files": len(produced), "out": str(out)}
