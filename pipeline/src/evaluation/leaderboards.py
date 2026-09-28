from __future__ import annotations

import logging
import pandas as pd

from src.metrics.aggregation import aggregate_metrics
from src.training.strict_mode import StrictModeViolation, is_strict_mode

_log = logging.getLogger(__name__)


def _group_model_output_source(group: pd.DataFrame) -> str:
    if "model_output_source" not in group.columns:
        return "unknown"
    sources = group["model_output_source"].dropna().unique().tolist()
    return sources[0] if len(sources) == 1 else ";".join(sorted(str(s) for s in sources))


def _group_benchmark_valid(group: pd.DataFrame) -> bool:
    if "model_output_source" not in group.columns:
        return False
    return all(str(s).startswith("trained_") for s in group["model_output_source"].dropna())


def overall_leaderboard(
    metrics: pd.DataFrame,
    training_times: dict[str, float],
    inference_times: dict[str, float],
    num_samples: int,
    strict_mode: bool | None = None,
) -> pd.DataFrame:
    if strict_mode is None:
        strict_mode = is_strict_mode()
    rows = []
    group_cols = ["feature_name", "frequency", "model_family", "model_name", "decomposition_method", "finetuning_mode"]
    for key, g in metrics.groupby(group_cols, dropna=False):
        # Reconstruct task_id matching make_tasks format: feature|freq|decomp|family|model|ft
        # groupby key order: feature[0], freq[1], family[2], model[3], decomp[4], ft[5]
        task_id = f"{key[0]}|{key[1]}|{key[4]}|{key[2]}|{key[3]}|{key[5]}"
        t_time = training_times.get(task_id)
        i_time = inference_times.get(task_id)
        if strict_mode:
            # Timing values of zero are acceptable (e.g. very fast models loaded from
            # checkpoint as "already_done"). Only raise for missing *metric* data, not timing.
            if not t_time or float(t_time) == 0.0:
                _log.debug(
                    "training_time_seconds is missing or zero for task %r — using 0.0",
                    task_id,
                )
            if not i_time or float(i_time) == 0.0:
                _log.debug(
                    "inference_time_seconds is missing or zero for task %r — using 0.0",
                    task_id,
                )
        agg = aggregate_metrics(g)
        rows.append(
            {
                "feature_name": key[0],
                "frequency": key[1],
                "dataset_set": f"m3_m4_{key[1]}",
                "model_family": key[2],
                "model_name": key[3],
                "decomposition_method": key[4],
                "finetuning_mode": key[5],
                "uses_static_features": key[2] in {"mlforecast", "neuralforecast", "transformers"},
                "uses_stl": key[4] != "without_stl",
                **agg,
                "num_samples": num_samples,
                "training_time_seconds": float(t_time) if t_time is not None else 0.0,
                "inference_time_seconds": float(i_time) if i_time is not None else 0.0,
                "model_output_source": _group_model_output_source(g),
                "benchmark_valid": _group_benchmark_valid(g),
                "status": "completed",
            }
        )
    return pd.DataFrame(rows).sort_values(["rel_naive_mean_unclipped", "rel_naive_median"], na_position="last")


def bucket_leaderboard(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = ["feature_name", "feature_bucket", "frequency", "model_family", "model_name", "decomposition_method", "finetuning_mode"]
    for key, g in metrics.groupby(cols, dropna=False):
        agg = aggregate_metrics(g)
        rows.append(
            {
                "feature_name": key[0],
                "feature_bucket": key[1],
                "frequency": key[2],
                "model_family": key[3],
                "model_name": key[4],
                "decomposition_method": key[5],
                "finetuning_mode": key[6],
                "rel_naive_mean_unclipped": agg["rel_naive_mean_unclipped"],
                "rel_naive_median": agg["rel_naive_median"],
                "win_rate_vs_naive": agg["win_rate_vs_naive"],
                "n_series": agg["n_series"],
            }
        )
    return pd.DataFrame(rows).sort_values(["rel_naive_mean_unclipped"], na_position="last")


def decomposition_leaderboard(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = ["decomposition_method", "frequency", "feature_name", "model_family", "model_name", "finetuning_mode"]
    for key, g in metrics.groupby(cols, dropna=False):
        agg = aggregate_metrics(g)
        rows.append(
            {
                "decomposition_method": key[0],
                "frequency": key[1],
                "feature_name": key[2],
                "model_family": key[3],
                "model_name": key[4],
                "finetuning_mode": key[5],
                "rel_naive_mean_unclipped": agg["rel_naive_mean_unclipped"],
                "rel_naive_median": agg["rel_naive_median"],
                "win_rate_vs_naive": agg["win_rate_vs_naive"],
            }
        )
    return pd.DataFrame(rows).sort_values(["rel_naive_mean_unclipped"], na_position="last")


def finetuning_leaderboard(metrics: pd.DataFrame, training_times: dict[str, float]) -> pd.DataFrame:
    rows = []
    cols = ["finetuning_mode", "feature_name", "feature_bucket", "frequency", "model_family", "model_name", "decomposition_method"]
    for key, g in metrics.groupby(cols, dropna=False):
        agg = aggregate_metrics(g)
        rows.append(
            {
                "finetuning_mode": key[0],
                "feature_name": key[1],
                "feature_bucket": key[2],
                "frequency": key[3],
                "model_family": key[4],
                "model_name": key[5],
                "decomposition_method": key[6],
                "rel_naive_mean_unclipped": agg["rel_naive_mean_unclipped"],
                "rel_naive_median": agg["rel_naive_median"],
                "win_rate_vs_naive": agg["win_rate_vs_naive"],
                "training_time_seconds": training_times.get("|".join(map(str, key)), 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values(["rel_naive_mean_unclipped"], na_position="last")
