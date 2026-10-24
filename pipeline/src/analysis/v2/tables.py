"""The canonical tables of the revision analyses, built read-only from a frozen run and its sensitivity studies.

- ``fc_errors/`` (parquet, one file per source): one row per distinct forecast. A statistical forecast does not
  depend on the training pool, so it is stored once, with feature sample and pool "none"; ``instances`` gives it to
  every feature sample that holds its series, with the series' bucket there.
- ``series_meta.parquet``: one row per sampled series.
- ``inst.parquet``: one row per instance (series x feature sample); its order is the HierarchicalBootstrap's index.

Columns of fc_errors: ``source`` is the study the forecast comes from (run, s3, s4, s4s, s5) and ``dataset`` M3 or
M4. ``setting`` is the STL setting: "none" under Direct; "default" in the main specification (the run's default
STL; ARIMA and ETS with S3's season length 1); "default_m" the run's statistical STL forecasts with season length
m (Table C4, SARIMA among them); "periodic", "s25" and "periodic_tuned" S4's (statistical models with season
length 1). ``scope`` is cohort, tercile, random, tercile_third or random_third, and ``group`` the pool of a tercile
or random model (Low, Medium, High; "none" for the cohort). ``seed`` is -1 for the statistical models.

Run:  python -m src.analysis.v2.tables --run RUN_DIR --out OUT_DIR
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.data import STATISTICAL_SEED
from src.data.freeze_data import HORIZON
from src.data.schemas import FEATURE_NAMES
from src.stages import sensitivity as s1
from src.stages import sensitivity3 as s3
from src.stages import sensitivity4 as s4
from src.stages import sensitivity4s as s4s
from src.stages import sensitivity5 as s5
from src.stages.io import StageError, code_commit, read_json, sha256_file, utc_now, write_json
from src.stages.tasks import BUCKET_SCOPE
from src.validation.gates import load_merged

from src.analysis.v2.revisao_v2 import CAP

FORMAT = "rerun-v2-tables/1"
READ = ["task_id", "feature_name", "frequency", "unique_id", "source_dataset", "window", "family", "model", "strategy",
        "scope", "bucket", "seed", "stl_variant", "status", "mae", "relnaive", "relnaive_capped"]
KEYS = ["series_id", "freq", "feature_sample", "window", "model", "strategy", "setting", "scope", "group", "seed", "scale"]
TEXT = ["source", "series_id", "dataset", "freq", "feature_sample", "pool_id", "model", "family", "strategy", "setting",
        "scope", "group", "bucket", "scale"]
SCOPE_BUCKET = {v: k for k, v in BUCKET_SCOPE.items()}
E7 = "exploratory2_977f022/tables/e7_strength.csv"      # phi7
E10 = "exploratory3_642d834/e10_phi4.csv"               # phi4'


def canonical(rows: pd.DataFrame, source: str, setting) -> pd.DataFrame:
    """The study's rows in the columns of fc_errors; ``setting`` is the STL setting of every STL row, or a
    function of the rows giving it row by row."""
    if (rows["status"] != "trained").any():
        raise StageError(f"{source}: rows that did not train")
    stat = (rows["family"] == "statistical").to_numpy()
    scope = rows["scope"].astype(str)
    base = scope.str.removesuffix(s5.THIRD)
    third = scope.str.endswith(s5.THIRD).to_numpy()
    random = base.str.startswith("r").to_numpy()
    cohort = (base == "cohort").to_numpy()
    kind = np.where(cohort, "cohort", np.where(random, "random", "tercile"))
    kind = np.where(third, np.char.add(kind.astype(str), "_third"), kind)
    group = np.where(cohort, "none", np.where(random, base.str[1:], base).astype(object))
    group = pd.Series(group).map(lambda g: SCOPE_BUCKET.get(g, g)).to_numpy()
    stl = setting(rows) if callable(setting) else np.full(len(rows), setting, dtype=object)
    mae_naive = rows["mae"].to_numpy() / rows["relnaive"].to_numpy()
    rn = rows["relnaive_capped"].to_numpy()
    if not np.isfinite(mae_naive).all() or (mae_naive <= 0).any():
        raise StageError(f"{source}: a seasonal naive MAE that is not positive and finite")
    if not np.allclose(np.minimum(rows["mae"].to_numpy() / mae_naive, CAP), rn, rtol=1e-12, atol=0):
        raise StageError(f"{source}: RelNaive(10) is not min(MAE / MAE of the seasonal naive, 10)")
    feature = np.where(stat, "none", rows["feature_name"].astype(object))
    freq = rows["frequency"].astype(str).to_numpy()
    out = pd.DataFrame({
        "source": source, "series_id": rows["unique_id"].to_numpy(),
        "dataset": rows["source_dataset"].astype(str).str.split("_").str[0].to_numpy(), "freq": freq,
        "feature_sample": feature, "pool_id": np.where(stat, "none", np.char.add(np.char.add(feature.astype(str), "|"), freq)),
        "window": rows["window"].to_numpy(), "model": rows["model"].to_numpy(), "family": rows["family"].to_numpy(),
        "strategy": rows["strategy"].to_numpy(), "setting": np.where(rows["strategy"] == "direct", "none", stl),
        "scope": kind, "group": group, "bucket": np.where(stat, None, rows["bucket"].astype(object)),
        "seed": rows["seed"].astype("Int64").fillna(STATISTICAL_SEED).astype("int64").to_numpy(), "scale": "raw",
        "mae": rows["mae"].to_numpy(), "mae_naive": mae_naive, "rn": rn})
    for c in TEXT:
        out[c] = out[c].astype("category")
    return out


def _study(run_dir: Path, tasks: list[dict], root: str, control: str) -> pd.DataFrame:
    tasks = [t for t in tasks if not t["task_id"].startswith(control)]
    rows, facts = s1._s1_rows(run_dir, tasks, root, READ)
    if facts["missing"] or facts["changed"]:
        raise StageError(f"{root}: task files missing or changed: {facts['missing'] + facts['changed']}")
    return rows


def sources(run_dir: Path, config: dict):
    """(source, canonical rows) for the run and each study, one at a time (the largest, S5, alone fills GBs)."""
    prepare = run_dir / "prepare"
    samples, summary = pd.read_parquet(prepare / "samples.parquet"), pd.read_parquet(prepare / "bucket_summary.parquet")
    gates = read_json(run_dir / "merged" / "gates.json")
    record, rows, failed = load_merged(run_dir, [c for c in READ if c != "stl_variant"])
    if not gates["passed"] or gates["merge_result_sha256"] != record["result_sha256"] or len(failed):
        raise StageError("the run's gates did not pass on this merge, or it has failed rows")
    stat_stl = (rows["family"] == "statistical") & (rows["strategy"] != "direct")
    yield "run", canonical(rows, "run", lambda r: np.where(stat_stl.loc[r.index], "default_m", "default")), record
    del rows
    yield "s3", canonical(_study(run_dir, s3.eval_tasks(summary, samples), s3.ROOT, s3.CONTROL_PREFIX), "s3", "default"), None
    yield "s4", canonical(_study(run_dir, s4.eval_tasks(summary, samples), s4.ROOT, s4.CONTROL_PREFIX), "s4",
                          lambda r: s4._label(r).to_numpy()), None
    study = s4s.study(config)
    yield "s4s", canonical(_study(run_dir, study.eval_tasks(summary, samples), s4s.ROOT, s4s.CONTROL_PREFIX), "s4s",
                           "periodic"), None
    yield "s5", canonical(_study(run_dir, s5.eval_tasks(summary, samples), s5.ROOT, s5.CONTROL_PREFIX), "s5",
                          "default"), None


def series_meta(run_dir: Path) -> pd.DataFrame:
    prepare = run_dir / "prepare"
    samples = pd.read_parquet(prepare / "samples.parquet")
    features = pd.read_parquet(prepare / "features.parquet")
    member = (samples.sort_values("feature_name").groupby(["unique_id", "frequency"])["feature_name"]
              .agg(";".join).rename("feature_samples").reset_index())
    meta = member.merge(features, on=["unique_id", "frequency"], how="left", validate="one_to_one")
    phi7 = pd.read_csv(run_dir / E7)[["unique_id", "frequency", "seasonal_strength"]].rename(columns={"seasonal_strength": "phi7"})
    phi4p = pd.read_csv(run_dir / E10)[["unique_id", "frequency", "evolving_seasonality_periodic", "periodic_flag"]].rename(
        columns={"evolving_seasonality_periodic": "phi4p", "periodic_flag": "phi4p_flag"})
    meta = meta.merge(phi7, on=["unique_id", "frequency"], how="left", validate="one_to_one").merge(
        phi4p, on=["unique_id", "frequency"], how="left", validate="one_to_one")
    if meta[["series_length", "phi7", "phi4p"]].isna().any().any():
        raise StageError("a sampled series without its length, phi7 or phi4'")
    h = meta["frequency"].map({f.lower(): v for f, v in HORIZON.items()})
    meta = meta.rename(columns={"unique_id": "series_id", "frequency": "freq", "series_length": "n_obs"})
    meta.insert(1, "dataset", meta["source_dataset"].str.split("_").str[0])
    meta["n_hist"] = meta["n_obs"] - 3 * h
    return meta[["series_id", "dataset", "freq", "n_obs", "n_hist", *FEATURE_NAMES, "feature_samples", "phi7", "phi4p",
                 "phi4p_flag"]].sort_values(["freq", "series_id"], ignore_index=True)


def inst_table(run_dir: Path) -> pd.DataFrame:
    """The instances of the cohort scope, ordered by frequency, feature sample, dataset and series."""
    buckets = pd.read_parquet(run_dir / "prepare" / "buckets.parquet")
    out = pd.DataFrame({"series_id": buckets["unique_id"], "freq": buckets["frequency"],
                        "feature_sample": buckets["feature_name"], "pool_id": buckets["feature_name"] + "|" + buckets["frequency"],
                        "source": buckets["source_dataset"].str.split("_").str[0], "bucket": buckets["feature_tercile_bucket"]})
    return out.sort_values(["freq", "feature_sample", "source", "series_id"], ignore_index=True)


def instances(fc: pd.DataFrame, inst: pd.DataFrame) -> pd.DataFrame:
    """Every forecast in each feature sample that holds its series: the statistical rows (feature sample "none")
    are repeated for each of the series' samples, with the series' bucket there."""
    stat = (fc["feature_sample"] == "none").to_numpy()
    views = fc[stat].drop(columns=["feature_sample", "pool_id", "bucket"]).merge(
        inst[["series_id", "freq", "feature_sample", "pool_id", "bucket"]], on=["series_id", "freq"], how="inner")
    if len(views) < stat.sum():
        raise StageError("statistical forecasts of a series outside every feature sample")
    out = pd.concat([fc[~stat], views[fc.columns]], ignore_index=True)
    for c in TEXT:
        out[c] = out[c].astype("category")
    return out


def build(run_dir: Path, out: Path) -> dict:
    run_dir, out = Path(run_dir), Path(out)
    if out.exists() and any(out.iterdir()):
        raise StageError(f"{out} is not empty; the canonical tables are never overwritten")
    commit = code_commit()
    config = read_json(run_dir / "prepare" / "bundle.json")["config"]
    (out / "fc_errors").mkdir(parents=True)
    # every forecast once: unique on KEYS within a source, and no cell (the keys without series and window) in two sources
    counts, cells, record = {}, {}, None
    for name, df, rec in sources(run_dir, config):
        record = rec or record
        duplicated = int(df.duplicated(KEYS).sum())
        if duplicated:
            raise StageError(f"{name}: {duplicated} forecasts appear twice on {KEYS}")
        cell = [k for k in KEYS if k not in ("series_id", "window")]
        cells[name] = set(df[cell].drop_duplicates().astype(str).itertuples(index=False, name=None))
        df.to_parquet(out / "fc_errors" / f"{name}.parquet", index=False)
        counts[name] = len(df)
        del df
    names = list(cells)
    shared = {f"{a}&{b}": len(cells[a] & cells[b]) for i, a in enumerate(names) for b in names[i + 1:] if cells[a] & cells[b]}
    if shared:
        raise StageError(f"cells in two sources: {shared}")
    meta, inst = series_meta(run_dir), inst_table(run_dir)
    meta.to_parquet(out / "series_meta.parquet", index=False)
    inst.to_parquet(out / "inst.parquet", index=False)
    files = sorted(str(p.relative_to(out)).replace("\\", "/") for p in out.rglob("*.parquet"))
    summary = {"format": FORMAT, "created_at_utc": utc_now(), "code_commit": commit, "run": str(run_dir),
               "merge_result_sha256": record["result_sha256"], "rows": counts, "series": len(meta), "instances": len(inst),
               "inputs": {p: sha256_file(run_dir / p) for p in (E7, E10)},
               "files": {p: sha256_file(out / p) for p in files}}
    write_json(out / "tables.json", summary)
    return summary


def load(out: Path, sources_: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """fc_errors (only the ``sources_`` given, if any), series_meta and inst, after checking their hashes."""
    out = Path(out)
    summary = read_json(out / "tables.json")
    names = sources_ or list(summary["rows"])
    for p in [f"fc_errors/{n}.parquet" for n in names] + ["series_meta.parquet", "inst.parquet"]:
        if sha256_file(out / p) != summary["files"][p]:
            raise StageError(f"{p} changed since it was built")
    fc = pd.concat([pd.read_parquet(out / "fc_errors" / f"{n}.parquet") for n in names], ignore_index=True)
    for c in TEXT:
        fc[c] = fc[c].astype("category")
    return fc, pd.read_parquet(out / "series_meta.parquet"), pd.read_parquet(out / "inst.parquet")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.analysis.v2.tables", description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    a = parser.parse_args(argv)
    print(build(a.run, a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
