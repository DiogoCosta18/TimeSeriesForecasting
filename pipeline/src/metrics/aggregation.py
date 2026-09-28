from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_metrics(metrics: pd.DataFrame) -> dict:
    if metrics.empty:
        return {}
    rel = metrics["rel_naive_unclipped"].astype(float)
    rel_clip = metrics["rel_naive_clipped"].astype(float)
    return {
        "rel_naive_mean_clipped": float(rel_clip.mean()),
        "rel_naive_mean_unclipped": float(rel.mean()),
        "rel_naive_median": float(rel.median()),
        "rel_naive_p75": float(rel.quantile(0.75)),
        "rel_naive_p90": float(rel.quantile(0.90)),
        "win_rate_vs_naive": float((rel < 1.0).mean()),
        "mae": float(metrics["mae"].mean()),
        "smape": float(metrics["smape"].mean()),
        "mase": float(metrics["mase"].mean()),
        "wape": float(metrics["wape"].mean()),
        "n_series": int(metrics["unique_id"].nunique()),
        "n_windows": int(metrics["window"].nunique()) if "window" in metrics else 0,
    }

