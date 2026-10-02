from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.cli import CliArgs
from src.data.frozen import load_manifest
from src.data.load_m_datasets import frozen_data_provenance, load_dataset_pair
from src.data.sample_series import assign_feature_buckets, representative_sample
from src.data.schemas import FEATURE_NAMES, MODEL_FAMILIES
from src.data.validation import make_cutoffs
from src.evaluation.leaderboards import bucket_leaderboard, decomposition_leaderboard, finetuning_leaderboard, overall_leaderboard
from src.evaluation.summarize import write_summary
from src.features.compute_all import compute_feature_table
from src.features.scaling import fit_transform_features
from src.training.checkpointing import CheckpointManager
from src.training.finetune import finetune_manifest_from_metrics
from src.training.logging_utils import setup_logging
from src.training.resources import default_feature_compute_workers, forecast_parallel_config_from_env
from src.training.scheduler import run_parallel_schedule
from src.training.task_graph import BUDGETS, budget_config, make_tasks
from src.training.vast_lifecycle import vast_config_from_env
from src.utils import atomic_write_json, ensure_dirs, environment_snapshot, git_snapshot, set_seed, utc_now_iso, write_dataframe


def _truthy_env(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _run_id() -> str:
    return "run_" + utc_now_iso().replace(":", "").replace("-", "").replace("Z", "")


def _resolve_run_root(base: Path, resume: bool) -> Path:
    latest = base / "LATEST_RUN.txt"
    if resume and latest.exists():
        rid = latest.read_text(encoding="utf-8").strip()
        if rid and (base / rid).exists():
            return base / rid
    rid = _run_id()
    root = base / rid
    base.mkdir(parents=True, exist_ok=True)
    latest.write_text(rid, encoding="utf-8")
    return root


def _frequency_cfgs(cfg: dict, requested: list[str] | None) -> dict[str, dict]:
    out = {"monthly": cfg["monthly"], "quarterly": cfg["quarterly"]}
    if requested:
        out = {k: v for k, v in out.items() if k in requested}
    return out


def _apply_budget_to_frequency(freq_cfg: dict, budget: str) -> dict:
    out = dict(freq_cfg)
    b = BUDGETS[budget]
    out["n_m3_requested"] = min(out.get("n_m3_requested", 750), b["n_per_source"])
    out["n_m4_requested"] = min(out.get("n_m4_requested", 750), b["n_per_source"])
    out["n_windows"] = min(out.get("n_windows", 3), b["n_windows"])
    return out


def _run_config(run_id: str, cfg: dict, freq_cfgs: dict[str, dict], budget: str) -> dict:
    return {
        "run_id": run_id,
        "created_at_utc": utc_now_iso(),
        "random_seed": cfg["random_seed"],
        "budget": budget,
        "metric_to_optimize": cfg["metric_to_optimize"],
        "naive_denominator": cfg["naive_denominator"],
        "features": cfg["features"],
        "dataset_sets": [v["dataset_set"] for v in freq_cfgs.values()],
        "sampling": cfg["representative_sampling"],
        "decomposition_methods": cfg["decomposition_methods"],
        "model_families": MODEL_FAMILIES,
        "mlforecast_auto": {
            "num_samples": cfg["mlforecast_auto"]["num_samples"],
            "n_windows": max(v["n_windows"] for v in freq_cfgs.values()),
            "step_size": "h",
            "input_size": cfg["mlforecast_auto"].get("input_size"),
            "refit": cfg["mlforecast_auto"].get("refit", 1),
            "random_seed": cfg["random_seed"],
        },
        "neuralforecast_auto": {
            "backend": cfg.get("neuralforecast_auto", {}).get("backend", "optuna"),
            "num_samples": cfg["neuralforecast_auto"]["num_samples"],
            "n_windows": max(v["n_windows"] for v in freq_cfgs.values()),
            "random_seed": cfg["random_seed"],
        },
        "transformer_auto": {
            "backend": cfg.get("transformer_auto", {}).get("backend", "optuna"),
            "num_samples": cfg["transformer_auto"]["num_samples"],
            "n_windows": max(v["n_windows"] for v in freq_cfgs.values()),
            "random_seed": cfg["random_seed"],
        },
        "finetuning": cfg["finetuning"],
        "feature_compute": cfg.get("feature_compute", {}),
        "parallel": _parallel_settings(budget),
        "vast": {
            **vast_config_from_env(),
            "sync_outputs_before_destroy": cfg.get("vast", {}).get("sync_outputs_before_destroy", True),
            "checkpoint_interval_seconds": cfg.get("vast", {}).get("checkpoint_interval_seconds", 300),
            "status_interval_seconds": cfg.get("vast", {}).get("status_interval_seconds", 60),
        },
    }


def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    parts = [p for p in parts if p is not None and not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _feature_compute_settings(cfg: dict, budget: str) -> dict:
    fc = cfg.get("feature_compute", {}) or {}
    env_jobs = os.environ.get("FEATURE_COMPUTE_N_JOBS")
    raw_jobs = env_jobs if env_jobs not in {None, ""} else fc.get("n_jobs", "auto")
    if str(raw_jobs).lower() == "auto":
        n_jobs = default_feature_compute_workers()
    else:
        n_jobs = max(1, int(raw_jobs))
    max_candidate_cfg = fc.get("max_candidate_series_per_source", {}) or {}
    return {
        "n_jobs": n_jobs,
        "checkpoint_every_series": int(fc.get("checkpoint_every_series", 100)),
        "progress_log_every_series": int(fc.get("progress_log_every_series", 25)),
        "max_candidate_series_per_source": max_candidate_cfg.get(budget),
    }


def _parallel_settings(budget: str = "smoke") -> dict:
    settings = forecast_parallel_config_from_env()
    # Auto-enable parallel for any budget that isn't smoke.
    auto_parallel = budget not in {"smoke"}
    enabled = _truthy_env("FORECAST_PARALLEL_ENABLED", auto_parallel)
    effective = float(settings.get("effective_cpu_cores", 1.0))
    # Default CPU workers: leave 2 cores free for OS + main process.
    default_cpu_workers = max(1, min(8, int(effective) - 2))
    return {
        "forecast_parallel_enabled": enabled,
        "effective_cpu_cores": effective,
        "cpu_max": settings.get("cpu_max"),
        "cpu_period": settings.get("cpu_period"),
        "forecast_cpu_task_workers": int(settings.get("forecast_cpu_task_workers", default_cpu_workers)),
        "forecast_gpu_task_workers": int(settings.get("forecast_gpu_task_workers", 1)),
        "forecast_statistical_workers": int(settings.get("forecast_statistical_workers", default_cpu_workers)),
        "forecast_mlforecast_workers": int(settings.get("forecast_mlforecast_workers", default_cpu_workers)),
        "forecast_neural_workers": int(settings.get("forecast_neural_workers", 1)),
        "forecast_transformer_workers": int(settings.get("forecast_transformer_workers", 1)),
        "forecast_threads_per_cpu_task": int(settings.get("forecast_threads_per_cpu_task", 1)),
        "forecast_threads_per_gpu_task": int(settings.get("forecast_threads_per_gpu_task", 2)),
        "forecast_conservative_mode": bool(settings.get("forecast_conservative_mode", True)),
    }


def _limit_candidate_series_for_feature_compute(
    data: pd.DataFrame,
    max_per_source: int | None,
    seed: int,
    frequency: str,
    logger,
) -> pd.DataFrame:
    if max_per_source in {None, "", "null"}:
        n_series = int(data["unique_id"].nunique())
        if n_series > 5000:
            logger.warning(
                "Feature computation for frequency=%s will scan %s candidate series because max_candidate_series_per_source is null. "
                "This can be very expensive on large M4 pools.",
                frequency,
                n_series,
            )
        return data
    max_per_source = int(max_per_source)
    rng = np.random.default_rng(seed)
    keep_ids: set[str] = set()
    for source, g in data[["unique_id", "source_dataset"]].drop_duplicates().groupby("source_dataset", sort=True):
        ids = sorted(g["unique_id"].astype(str).tolist())
        if len(ids) <= max_per_source:
            chosen = ids
        else:
            chosen = sorted(rng.choice(ids, size=max_per_source, replace=False).tolist())
            logger.info(
                "Limiting feature candidate pool frequency=%s source=%s from %s to %s series",
                frequency,
                source,
                len(ids),
                len(chosen),
            )
        keep_ids.update(chosen)
    return data[data["unique_id"].astype(str).isin(keep_ids)].copy()


def run_experiment(cfg_raw: dict, args: CliArgs) -> Path:
    # Strict mode: auto-enable for production budgets unless explicitly disabled.
    # None = auto (True for pilot/full/large), True = explicit enable, False = explicit disable.
    strict_setting = getattr(args, "strict_model_mode", None)
    if strict_setting is True or (strict_setting is None and args.budget in {"pilot", "full", "large"}):
        os.environ["STRICT_MODEL_MODE"] = "1"
    elif strict_setting is False:
        os.environ.pop("STRICT_MODEL_MODE", None)
    cfg = budget_config(cfg_raw, args.budget)
    if args.features:
        cfg["features"] = [f for f in cfg["features"] if f in args.features]
    if args.decomposition_methods:
        cfg["decomposition_methods"] = [d for d in cfg["decomposition_methods"] if d in args.decomposition_methods]
    set_seed(int(cfg["random_seed"]))
    run_root = _resolve_run_root(args.run_root, args.resume)
    run_id = run_root.name
    ensure_dirs(run_root)
    logger = setup_logging(run_root, args.debug)
    logger.info("Starting run %s in %s", run_id, run_root)
    freq_cfgs = {k: _apply_budget_to_frequency(v, args.budget) for k, v in _frequency_cfgs(cfg, args.frequencies).items()}
    feature_compute = _feature_compute_settings(cfg, args.budget)
    logger.info("Feature compute settings: %s", feature_compute)
    atomic_write_json(run_root / "run_config.json", _run_config(run_id, cfg, freq_cfgs, args.budget))
    atomic_write_json(run_root / "environment.json", environment_snapshot())
    atomic_write_json(run_root / "git_info.json", git_snapshot(Path.cwd()))

    cp = CheckpointManager(run_root, run_id)
    data_by_freq: dict[str, pd.DataFrame] = {}
    feature_tables = []
    flag_tables = []
    sample_parts = []
    candidate_parts = []
    cutoff_parts = []
    frozen_manifest = load_manifest()
    data_manifest = {"run_id": run_id, "frozen_data": frozen_data_provenance(), "datasets": []}

    for freq, fcfg in freq_cfgs.items():
        cp.status({"stage": f"load_features_{freq}", "frequency": freq}, 1)
        data = load_dataset_pair(fcfg, args.data_dir, frozen_manifest)
        data = _limit_candidate_series_for_feature_compute(
            data,
            feature_compute["max_candidate_series_per_source"],
            int(cfg["random_seed"]),
            freq,
            logger,
        )
        data_by_freq[freq] = data
        last_feature_status = {"time": 0.0}

        def feature_status_callback(payload: dict) -> None:
            now = time.time()
            if now - last_feature_status["time"] >= 60 or payload.get("feature_completed_series") == payload.get("feature_total_series"):
                last_feature_status["time"] = now
                cp.status(payload, 1)

        raw_features, flags = compute_feature_table(
            data,
            int(fcfg["season_length"]),
            int(fcfg["horizon"]),
            n_windows=int(fcfg["n_windows"]),
            n_jobs=feature_compute["n_jobs"],
            cache_dir=run_root / "features" / "cache",
            frequency=freq,
            logger=logger,
            status_callback=feature_status_callback,
            checkpoint_every_series=feature_compute["checkpoint_every_series"],
            progress_log_every_series=feature_compute["progress_log_every_series"],
        )
        raw_features["frequency"] = freq
        flags["frequency"] = freq
        feature_tables.append(raw_features)
        flag_tables.append(flags)
        candidate_parts.append(raw_features.assign(frequency=freq))
        for source, g in raw_features.groupby("source_dataset"):
            data_manifest["datasets"].append(
                {
                    "frequency": freq,
                    "dataset_set": fcfg["dataset_set"],
                    "source_dataset": source,
                    "available_n": int(g["unique_id"].nunique()),
                    "requested_n": int(fcfg["n_m3_requested"] if source.startswith("M3") else fcfg["n_m4_requested"]),
                    "season_length": int(fcfg["season_length"]),
                    "horizon": int(fcfg["horizon"]),
                    "shortfall": max(0, int(fcfg["n_m3_requested"] if source.startswith("M3") else fcfg["n_m4_requested"]) - int(g["unique_id"].nunique())),
                }
            )
        for feature in cfg["features"]:
            requested = int(fcfg["n_m3_requested"])
            sample_parts.append(
                representative_sample(
                    raw_features,
                    feature,
                    freq,
                    requested_n=requested,
                    n_strata=int(cfg["representative_sampling"]["n_strata"]),
                    seed=int(cfg["random_seed"]) + FEATURE_NAMES.index(feature),
                )
            )
        all_selected_ids = set(_concat(sample_parts).query("frequency == @freq")["unique_id"]) if sample_parts else set()
        cutoffs = make_cutoffs(data[data["unique_id"].isin(all_selected_ids)], int(fcfg["horizon"]), int(fcfg["n_windows"]), int(fcfg["step_size"]))
        cutoffs["frequency"] = freq
        cutoff_parts.append(cutoffs)

    raw_all = _concat(feature_tables)
    flags_all = _concat(flag_tables)
    scaled_all, scaler = fit_transform_features(raw_all)
    sample_manifest = _concat(sample_parts)
    bucket_manifest = assign_feature_buckets(sample_manifest)
    cutoffs_all = _concat(cutoff_parts)
    atomic_write_json(run_root / "data_manifest.json", data_manifest)
    write_dataframe(run_root / "sampling" / "candidate_feature_values.parquet", _concat(candidate_parts))
    write_dataframe(run_root / "sampling" / "feature_sample_manifest.parquet", sample_manifest)
    write_dataframe(run_root / "sampling" / "feature_bucket_manifest.parquet", bucket_manifest)
    write_dataframe(run_root / "features" / "feature_table_raw.parquet", raw_all)
    write_dataframe(run_root / "features" / "feature_table_scaled.parquet", scaled_all)
    write_dataframe(run_root / "features" / "feature_quality_flags.parquet", flags_all)
    atomic_write_json(run_root / "features" / "feature_scaler.json", scaler)
    write_dataframe(run_root / "features" / "fold_feature_values.parquet", scaled_all)
    write_dataframe(run_root / "cv" / "cutoffs.parquet", cutoffs_all)

    tasks = make_tasks(cfg, args.budget, list(freq_cfgs), args.models, args.decomposition_methods, args.finetuning_modes)
    parallel = _parallel_settings(args.budget)
    task_run = run_parallel_schedule(
        run_root=run_root,
        tasks=tasks,
        data_by_frequency=data_by_freq,
        cutoffs_by_frequency={freq: cutoffs_all[cutoffs_all["frequency"] == freq] for freq in freq_cfgs},
        bucket_manifest=bucket_manifest,
        season_length_by_frequency={freq: int(freq_cfgs[freq]["season_length"]) for freq in freq_cfgs},
        horizon_by_frequency={freq: int(freq_cfgs[freq]["horizon"]) for freq in freq_cfgs},
        parallel_config=parallel,
        optuna_num_samples=int(cfg["mlforecast_auto"]["num_samples"]),
        cp=cp,
        logger=logger,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        cp.status(
            {
                "stage": "dry_run",
                "parallel_enabled": parallel["forecast_parallel_enabled"],
                "effective_cpu_cores": parallel["effective_cpu_cores"],
                "cpu_task_workers": parallel["forecast_cpu_task_workers"],
                "gpu_task_workers": parallel["forecast_gpu_task_workers"],
                "running_tasks": [],
                "queued_tasks": task_run["plan"]["queued_tasks"],
                "resource_class_counts": task_run["plan"]["resource_class_counts"],
                "recent_completed_task_ids": [],
                "recent_failed_task_ids": [],
            },
            len(tasks),
        )
        atomic_write_json(run_root / "dry_run_summary.json", task_run["plan"])
        logger.info("Dry run complete: %s tasks planned", task_run["plan"]["total_tasks"])
        return run_root

    metrics = task_run["metrics"]
    forecasts = task_run["forecasts"]
    naive_forecasts = task_run["naive"]
    training_times = task_run.get("training_times", {})
    inference_times = task_run.get("inference_times", {})
    write_dataframe(run_root / "final" / "model_artifacts" / "forecasts.parquet", forecasts)
    write_dataframe(run_root / "final" / "forecasts.parquet", forecasts)
    write_dataframe(run_root / "final" / "test_metrics.parquet", metrics)

    ft_manifest = finetune_manifest_from_metrics(metrics)
    write_dataframe(run_root / "finetuning" / "finetune_manifest.parquet", ft_manifest)
    write_dataframe(run_root / "finetuning" / "series_finetune_metrics.parquet", metrics[metrics.get("finetuning_mode", "") == "finetune_by_individual_series"] if not metrics.empty else pd.DataFrame())
    _bucket_ft_modes = {"finetune_by_feature_bucket", "finetune_bucket_low", "finetune_bucket_medium", "finetune_bucket_high"}
    _bfm_mask = metrics["finetuning_mode"].isin(_bucket_ft_modes) if not metrics.empty and "finetuning_mode" in metrics.columns else pd.Series(dtype=bool)
    write_dataframe(run_root / "finetuning" / "bucket_finetune_metrics.parquet", metrics[_bfm_mask] if not metrics.empty else pd.DataFrame())
    write_dataframe(run_root / "finetuning" / "finetuned_forecasts.parquet", forecasts[forecasts.get("finetuning_mode", "") != "no_finetune"] if not forecasts.empty else pd.DataFrame())

    num_samples = int(cfg["mlforecast_auto"]["num_samples"])
    overall = overall_leaderboard(metrics, training_times, inference_times, num_samples) if not metrics.empty else pd.DataFrame()
    by_bucket = bucket_leaderboard(metrics) if not metrics.empty else pd.DataFrame()
    by_decomp = decomposition_leaderboard(metrics) if not metrics.empty else pd.DataFrame()
    by_ft = finetuning_leaderboard(metrics, training_times) if not metrics.empty else pd.DataFrame()
    by_feature = overall.sort_values(["feature_name", "rel_naive_mean_unclipped"]) if not overall.empty else pd.DataFrame()
    by_horizon = metrics.groupby(["feature_name", "frequency", "model_family", "model_name", "decomposition_method", "finetuning_mode"], as_index=False).agg(rel_naive_mean_unclipped=("rel_naive_unclipped", "mean")) if not metrics.empty else pd.DataFrame()
    by_dataset = metrics.groupby(["source_dataset", "frequency", "model_family", "model_name"], as_index=False).agg(rel_naive_mean_unclipped=("rel_naive_unclipped", "mean")) if not metrics.empty else pd.DataFrame()
    write_dataframe(run_root / "reports" / "leaderboard_overall.csv", overall)
    write_dataframe(run_root / "reports" / "leaderboard_by_feature.csv", by_feature)
    write_dataframe(run_root / "reports" / "leaderboard_by_feature_bucket.csv", by_bucket)
    write_dataframe(run_root / "reports" / "leaderboard_by_decomposition_method.csv", by_decomp)
    write_dataframe(run_root / "reports" / "leaderboard_by_finetuning_mode.csv", by_ft)
    write_dataframe(run_root / "reports" / "leaderboard_by_horizon.csv", by_horizon)
    write_dataframe(run_root / "reports" / "leaderboard_by_dataset.csv", by_dataset)

    notes = []
    env = environment_snapshot()
    for pkg, status in env["packages"].items():
        if str(status).startswith("unavailable"):
            notes.append(f"{pkg}: {status}")
    fallback_counts: dict[str, int] = {}
    if not metrics.empty and "model_output_source" in metrics.columns:
        fallback_rows = metrics[~metrics["model_output_source"].str.startswith("trained_", na=False)]
        if not fallback_rows.empty and "model_family" in fallback_rows.columns:
            for fam, g in fallback_rows.groupby("model_family"):
                fallback_counts[str(fam)] = len(g)
    benchmark_valid = len(fallback_counts) == 0
    write_summary(
        run_root / "reports" / "summary.md",
        overall,
        by_bucket,
        metrics,
        notes,
        strict_mode=os.environ.get("STRICT_MODEL_MODE") in {"1", "true", "yes"},
        fallback_counts=fallback_counts,
        benchmark_valid=benchmark_valid,
    )
    if cp.failed:
        atomic_write_json(run_root / "FAILED.json", {"run_id": run_id, "failed_tasks": cp.failed, "completed_tasks": len(cp.completed), "finished_at_utc": utc_now_iso()})
    else:
        atomic_write_json(run_root / "DONE.json", {"run_id": run_id, "completed_tasks": len(cp.completed), "finished_at_utc": utc_now_iso(), "status": "success"})
    cp.status(
        {
            "stage": "done",
            "parallel_enabled": parallel["forecast_parallel_enabled"],
            "effective_cpu_cores": parallel["effective_cpu_cores"],
            "cpu_task_workers": parallel["forecast_cpu_task_workers"],
            "gpu_task_workers": parallel["forecast_gpu_task_workers"],
            "running_tasks": [],
            "queued_tasks": 0,
            "resource_class_counts": task_run["plan"]["resource_class_counts"],
            "recent_completed_task_ids": sorted(task_run.get("completed_ids", []))[-10:],
            "recent_failed_task_ids": [item["task_id"] for item in task_run.get("failed", [])[-10:]],
        },
        len(tasks),
    )
    logger.info("Finished run %s", run_id)
    return run_root
