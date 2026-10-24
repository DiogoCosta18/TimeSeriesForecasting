"""Inference of the revision analyses: instance values, the two-level bootstrap and the labels.

A question is a family of contrasts (cells) whose intervals are simultaneous: Bonferroni over its k cells, with
rv.n_boot(k) replicates. ``instance_matrix`` turns per-forecast deltas into the matrix of instance values that
rv.HierarchicalBootstrap takes (one row per instance of ``inst``, one column per cell, NaN where the instance is
not in the cell); ``replicates`` runs the bootstrap with the columns split over processes (each process draws the
same resamples, so the result is that of one run); ``summary`` gives the estimate, the intervals, the label and the
win rate over series.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.analysis.v2 import revisao_v2 as rv


def instance_matrix(d: pd.DataFrame, inst: pd.DataFrame, cells: list[str], balanced: bool = False,
                    value: str = "delta") -> tuple[np.ndarray, pd.DataFrame]:
    """X and its cells: an instance's value in a cell is its mean over windows and models (with ``balanced``,
    families pooled: the mean of its family means). ``d`` has one row per paired forecast, with series_id,
    feature_sample (statistical rows already given to each sample, tables.instances) and family."""
    keys = ["series_id", "feature_sample", *cells]
    d = d.dropna(subset=[value]).astype({k: str for k in keys})
    if balanced:
        v = d.groupby([*keys, "family"], observed=True)[value].mean().groupby(level=keys, observed=True).mean()
    else:
        v = d.groupby(keys, observed=True)[value].mean()
    wide = v.unstack(cells) if cells else v.to_frame("all")
    rows = pd.MultiIndex.from_frame(inst[["series_id", "feature_sample"]].astype(str))
    extra = wide.index.difference(rows)
    if len(extra):
        raise ValueError(f"{len(extra)} instances outside inst, e.g. {list(extra[:3])}")
    wide = wide.reindex(rows)
    labels = wide.columns.to_frame(index=False) if cells else pd.DataFrame({"cell": ["all"]})
    if cells and len(cells) == 1:
        labels.columns = cells
    return wide.to_numpy(dtype=float), labels


def _run(args) -> np.ndarray:
    inst, X, B, stat = args
    return rv.HierarchicalBootstrap(inst).run(X, B, stat=stat)


def replicates(inst: pd.DataFrame, X: np.ndarray, B: int, stat: str = "median", workers: int = 1) -> np.ndarray:
    """B x C replicates of the cells' statistic; the same for any ``workers``."""
    cols = np.array_split(np.arange(X.shape[1]), max(1, min(workers, X.shape[1])))
    jobs = [(inst, X[:, c], B, stat) for c in cols]
    if len(jobs) == 1:
        reps = _run(jobs[0])
    else:
        with ProcessPoolExecutor(len(jobs)) as pool:
            reps = np.concatenate(list(pool.map(_run, jobs)), axis=1)
    if not np.isfinite(reps).all():      # a cell whose pools a replicate can miss (one pool per frequency): restrict inst
        raise ValueError(f"replicates that are not finite in {int((~np.isfinite(reps).all(axis=0)).sum())} cells")
    return reps


def summary(inst: pd.DataFrame, X: np.ndarray, labels: pd.DataFrame, k: int, question: str,
            workers: int = 1, stat: str = "median", delta: float = rv.DELTA) -> pd.DataFrame:
    """One row per cell: estimate (the published estimand, or the mean over series with ``stat``), 95% and
    simultaneous intervals over the question's k cells, label (threshold ``delta``), win rate over series and
    the number of series."""
    hb = rv.HierarchicalBootstrap(inst)
    reps = replicates(inst, X, rv.n_boot(k), stat=stat, workers=workers)
    out = labelled(hb.point(X, stat), reps, labels, k, question, delta)
    out["win_series"] = hb.point(X, "winrate")
    out["n_series"] = (hb.A @ (~np.isnan(X)).astype(float) > 0).sum(axis=0)
    return out


def labelled(point: np.ndarray, reps: np.ndarray, labels: pd.DataFrame, k: int, question: str,
             delta: float = rv.DELTA) -> pd.DataFrame:
    """rv.summarise with the cells and the question; for contrasts built from cells (e.g. High - Low of two
    columns' replicates), whose point and replicates the caller derives."""
    if len(reps) != rv.n_boot(k):
        raise ValueError(f"{len(reps)} replicates, expected {rv.n_boot(k)} for k = {k}")
    out = rv.summarise(point, reps, k, delta=delta)
    out.insert(0, "question", question)
    out.insert(1, "k", k)
    return pd.concat([labels.reset_index(drop=True), out], axis=1)
