"""Analysis data (protocol Sections 5.1-5.2; D18, D27).

``load_gated`` opens a result bundle only if its gate report passed and still matches the
merge, and the prepare bundle and frozen configurations are unchanged.

The unit of analysis is the instance (feature sample, series, window, model, strategy,
scope, seed). Statistical forecasts are pool-independent (D18): every statistical row
enters each feature sample that holds its series, with the series' bucket in that
sample. Failed rows stay as instances without values, so contrasts delete them pairwise
and every table reports how many instances it excluded.

- Delta_STL = RelNaive(Direct) - RelNaive(STL), paired on everything but the strategy;
- Delta_FT = RelNaive(cohort) - RelNaive(tercile), paired on everything but the scope;
positive values mean the treatment helps. Inference uses per-series means over the
instances of a reporting group; pooled families are weighted equally (D27).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.stats import describe
from src.data.frozen import DEFAULT_MANIFEST
from src.forecast.engine import METRICS
from src.stages.io import StageError, load_bundle, read_json
from src.stages.merge import run_provenance
from src.validation.gates import load_merged

STATISTICAL_SEED = -1
CATEGORIES = ["feature_name", "frequency", "unique_id", "source_dataset", "family", "model", "strategy", "scope", "bucket"]
VALUES = METRICS + ["fit_seconds", "predict_seconds"]
STL_KEYS = ["feature_name", "frequency", "unique_id", "source_dataset", "window", "family", "model", "scope", "bucket", "seed_key"]
FT_KEYS = ["feature_name", "frequency", "unique_id", "source_dataset", "window", "family", "model", "strategy", "bucket", "seed_key"]
PRIMARY = "relnaive_capped"
# the merged-row columns the analysis reads; forecast hashes, components and provenance hashes are
# checked by the merge and the gates, and leaving them out lets the analysis run in a few GB
ROW_COLUMNS = ["task_id", *[k for k in STL_KEYS if k != "seed_key"], "strategy", "seed", "status", *VALUES,
               "failure", "backend_version"]


class AnalysisError(StageError):
    """The analysis cannot run on this bundle without breaking the protocol."""


@dataclass
class AnalysisData:
    run_dir: Path
    config: dict
    bundle: dict
    record: dict
    gates: dict
    frozen: dict
    tables: dict
    rows: pd.DataFrame
    failed: pd.DataFrame
    instances: pd.DataFrame

    @property
    def main_seed(self) -> int:
        return int(self.config["random_seed"])

    @property
    def main(self) -> pd.DataFrame:
        """Instances of the main seed (H1-H7); the extra seeds serve A15 only."""
        return self.instances[self.instances["main_seed"]]


def build_instances(rows: pd.DataFrame, failed: pd.DataFrame, buckets: pd.DataFrame, main_seed: int) -> pd.DataFrame:
    columns = ["task_id", *[k for k in STL_KEYS if k != "seed_key"], "strategy", "seed", "status", *VALUES]
    both = pd.concat([rows[columns], failed[columns]], ignore_index=True)
    both["seed_key"] = both["seed"].astype("Int64").fillna(STATISTICAL_SEED).astype("int64")
    both = both.drop(columns="seed")
    glob = both[both["family"] != "statistical"]
    stat = both[both["family"] == "statistical"].drop(columns=["feature_name", "bucket"])
    membership = buckets[["frequency", "feature_name", "unique_id", "feature_tercile_bucket"]].rename(
        columns={"feature_tercile_bucket": "bucket"})
    views = stat.merge(membership, on=["frequency", "unique_id"], how="left", validate="many_to_many")
    if views["feature_name"].isna().any():
        raise AnalysisError("statistical rows of a series outside every feature sample")
    inst = pd.concat([glob, views[glob.columns]], ignore_index=True)
    if glob["bucket"].isna().any():
        raise AnalysisError("global rows without a bucket")
    inst["main_seed"] = inst["seed_key"].isin([main_seed, STATISTICAL_SEED])
    for column in CATEGORIES:
        inst[column] = inst[column].astype("category")
    return inst


def load_gated(run_dir: Path, manifest_path: Path = DEFAULT_MANIFEST) -> AnalysisData:
    run_dir = Path(run_dir)
    record, rows, failed = load_merged(run_dir, ROW_COLUMNS)
    gates_path = run_dir / "merged" / "gates.json"
    if not gates_path.exists():
        raise AnalysisError("the gates have not been run on this bundle")
    report = read_json(gates_path)
    if not report["passed"]:
        raise AnalysisError(f"gates failed: {sorted(k for k, v in report['gates'].items() if not v['passed'])}")
    if report["merge_result_sha256"] != record["result_sha256"]:
        raise AnalysisError("the gate report belongs to another merge")
    config = read_json(run_dir / "prepare" / "bundle.json")["config"]
    bundle = load_bundle(run_dir, config, manifest_path)
    main, frozen = run_provenance(run_dir, bundle, manifest_path)
    if main != record["provenance"]:
        raise AnalysisError("the merged bundle carries other provenance than the run's")
    tables = {name: pd.read_parquet(run_dir / "prepare" / entry["file"]) for name, entry in bundle["files"].items()}
    instances = build_instances(rows, failed, tables["buckets"], int(config["random_seed"]))
    return AnalysisData(run_dir, config, bundle, record, report, frozen, tables, rows, failed, instances)


def _wide(df: pd.DataFrame, keys: list[str], column: str, value: str) -> pd.DataFrame:
    sub = df[keys + [column, value]].copy()
    for k in keys + [column]:
        if isinstance(sub[k].dtype, pd.CategoricalDtype):
            sub[k] = sub[k].astype(str)
    indexed = sub.set_index(keys + [column])[value]
    if indexed.index.duplicated().any():
        raise AnalysisError(f"instances are not unique on {keys + [column]}")
    return indexed.unstack(column)


def strategy_contrast(inst: pd.DataFrame, first: str, second: str, metric: str = PRIMARY) -> pd.DataFrame:
    """Instance-level metric[first] - metric[second], paired on everything but the strategy (NaN when an arm failed)."""
    wide = _wide(inst[inst["strategy"].isin([first, second])], STL_KEYS, "strategy", metric)
    out = wide.index.to_frame(index=False)
    out["delta"] = (wide.get(first) - wide.get(second)).to_numpy()
    return out


def stl_deltas(inst: pd.DataFrame, metric: str = PRIMARY) -> pd.DataFrame:
    """Delta_STL for STL-SN and STL-AC (column ``strategy``)."""
    return pd.concat([strategy_contrast(inst, "direct", s, metric).assign(strategy=s) for s in ("stl_sn", "stl_ac")],
                     ignore_index=True)


def ft_deltas(inst: pd.DataFrame, metric: str = PRIMARY) -> pd.DataFrame:
    """Delta_FT: cohort minus the tercile model of the instance's bucket (global families)."""
    glob = inst[inst["family"] != "statistical"]
    arm = np.where(glob["scope"].astype(str) == "cohort", "cohort", "tercile")
    wide = _wide(glob.assign(arm=arm), FT_KEYS, "arm", metric)
    out = wide.index.to_frame(index=False)
    out["delta"] = (wide.get("cohort") - wide.get("tercile")).to_numpy()
    return out


def per_series(df: pd.DataFrame, by: list[str], value: str = "delta", family_balanced: bool = False,
               how: str = "mean") -> pd.DataFrame:
    """A series' value per reporting group: its mean over the group's instances (D27);
    with ``family_balanced``, the mean of its per-family means."""
    d = df.dropna(subset=[value])
    if family_balanced:
        d = d.groupby(["unique_id", *by, "family"], observed=True, sort=True)[value].agg(how).reset_index()
        how = "mean"
    return d.groupby(["unique_id", *by], observed=True, sort=True)[value].agg(how).reset_index()


def describe_groups(df: pd.DataFrame, by: list[str], value: str = "delta", family_balanced: bool = False) -> pd.DataFrame:
    """Instance-level descriptives per group, with the number of excluded (unpaired) instances."""
    out = []
    groups = df.groupby(by, observed=True, sort=True) if by else [((), df)]
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        valid = g[value].notna().to_numpy()
        values = g[value].to_numpy()[valid]
        weights = None
        if family_balanced:
            fam = g["family"].astype(str).to_numpy()[valid]
            counts = pd.Series(fam).map(pd.Series(fam).value_counts()).to_numpy()
            weights = 1.0 / counts
        out.append({**dict(zip(by, key)), **describe(values, weights), "n_excluded": int((~valid).sum())})
    return pd.DataFrame(out)
