from __future__ import annotations

import pandas as pd


def finetune_manifest_from_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    ft = metrics[metrics["finetuning_mode"] != "no_finetune"].copy()
    if ft.empty:
        return pd.DataFrame(columns=["feature_name", "frequency", "feature_bucket", "model_family", "model_name", "finetuning_mode", "implementation"])
    ft["implementation"] = ft.apply(
        lambda r: "series_refit_from_global_config"
        if r["model_family"] == "mlforecast" and r["finetuning_mode"] == "finetune_by_series"
        else "bucket_refit_from_global_config"
        if r["model_family"] == "mlforecast"
        else "checkpoint_continue_training",
        axis=1,
    )
    return ft[["feature_name", "frequency", "feature_bucket", "model_family", "model_name", "finetuning_mode", "implementation"]].drop_duplicates()

