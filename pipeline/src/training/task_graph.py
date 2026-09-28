from __future__ import annotations

from src.data.schemas import MODEL_FAMILIES, SMOKE_MODELS


BUDGETS = {
    "smoke": {"features": 1, "n_per_source": 20, "n_windows": 1, "num_samples": 2, "models": "smoke"},
    "pilot": {"features": 2, "n_per_source": 100, "n_windows": 2, "num_samples": 10, "models": "all"},
    "full": {"features": 6, "n_per_source": 750, "n_windows": 3, "num_samples": 50, "models": "all"},
    "large": {"features": 6, "n_per_source": 750, "n_windows": 3, "num_samples": 100, "models": "all"},
}


def budget_config(base: dict, budget: str) -> dict:
    b = BUDGETS[budget]
    cfg = dict(base)
    cfg["_budget_settings"] = b
    cfg["features"] = base["features"][: b["features"]]
    cfg["representative_sampling"] = dict(base["representative_sampling"])
    cfg["representative_sampling"]["n_per_source_dataset"] = b["n_per_source"]
    cfg.setdefault("mlforecast_auto", {})["num_samples"] = b["num_samples"]
    cfg.setdefault("neuralforecast_auto", {})["num_samples"] = b["num_samples"]
    cfg.setdefault("transformer_auto", {})["num_samples"] = b["num_samples"]
    return cfg


def selected_models(budget: str, enabled: dict[str, bool], filter_models: list[str] | None = None) -> dict[str, list[str]]:
    families = SMOKE_MODELS if BUDGETS[budget]["models"] == "smoke" else MODEL_FAMILIES
    out = {}
    for family, models in families.items():
        if not enabled.get(family, False):
            continue
        chosen = [m for m in models if not filter_models or m in filter_models]
        if chosen:
            out[family] = chosen
    return out


BUCKET_FT_MODES = ["finetune_bucket_low", "finetune_bucket_medium", "finetune_bucket_high"]


def finetuning_modes(cfg: dict, family: str, requested: list[str] | None = None) -> list[str]:
    raw = cfg.get("finetuning", {})
    modes = []
    if raw.get("no_finetune", True):
        modes.append("no_finetune")
    # Bucket finetuning is meaningless for statistical (no global training, approximation flag only).
    if raw.get("finetune_by_feature_bucket", True) and family != "statistical":
        modes.append("finetune_by_feature_bucket")
    # Per-series finetuning applies to local (statistical) models only.
    if family == "statistical" and raw.get("finetune_by_individual_series", False):
        modes.append("finetune_by_individual_series")
    # Bucket-specific global training: separate model per Low/Medium/High bucket.
    # Only for global model families (ML, neural, transformer) — not statistical.
    if family != "statistical" and raw.get("finetune_by_bucket", False):
        modes += BUCKET_FT_MODES
    return [m for m in modes if not requested or m in requested]


def make_tasks(
    cfg: dict,
    budget: str,
    frequencies: list[str],
    model_filter: list[str] | None,
    decomp_filter: list[str] | None,
    finetune_filter: list[str] | None,
) -> list[dict]:
    models = selected_models(budget, cfg["model_families"], model_filter)
    decomps = [d for d in cfg["decomposition_methods"] if not decomp_filter or d in decomp_filter]
    tasks = []
    for feature in cfg["features"]:
        for freq in frequencies:
            for decomp in decomps:
                for family, names in models.items():
                    for model in names:
                        for ft in finetuning_modes(cfg, family, finetune_filter):
                            task_id = "|".join([feature, freq, decomp, family, model, ft])
                            tasks.append(
                                {
                                    "task_id": task_id,
                                    "feature_name": feature,
                                    "frequency": freq,
                                    "decomposition_method": decomp,
                                    "model_family": family,
                                    "model_name": model,
                                    "finetuning_mode": ft,
                                }
                            )
    return tasks

