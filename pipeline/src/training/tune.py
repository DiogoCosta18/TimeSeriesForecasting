from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.training.strict_mode import StrictModeViolation, is_strict_mode
from src.utils import atomic_write_json
from src.utils import write_dataframe

ALLOWED_TUNING_STATUSES = frozenset({"disabled_by_budget", "disabled_by_config", "real", "placeholder"})


def write_optuna_artifacts(root: Path, model_name: str, num_samples: int, tuning_status: str = "disabled_by_budget") -> None:
    if tuning_status not in ALLOWED_TUNING_STATUSES:
        raise ValueError(f"Unknown tuning_status {tuning_status!r}; expected one of {sorted(ALLOWED_TUNING_STATUSES)}")
    if tuning_status == "placeholder" and is_strict_mode():
        raise StrictModeViolation(
            f"Strict mode: placeholder Optuna artifacts are not allowed for {model_name!r}. "
            "Use real tuning or set tuning_status='disabled_by_budget'."
        )
    safe = model_name.replace("/", "_")
    atomic_write_json(
        root / "optuna" / f"best_params_{safe}.json",
        {"model_name": model_name, "num_samples": num_samples, "tuning_status": tuning_status},
    )
    (root / "optuna" / f"study_{safe}.db").touch(exist_ok=True)
    write_dataframe(
        root / "optuna" / f"trials_{safe}.parquet",
        pd.DataFrame([{"model_name": model_name, "trial": 0, "tuning_status": tuning_status}]),
    )


# Kept for backwards compatibility with any direct imports in existing tests.
def write_placeholder_optuna_artifacts(root: Path, model_name: str, num_samples: int) -> None:
    write_optuna_artifacts(root, model_name, num_samples, tuning_status="disabled_by_budget")
