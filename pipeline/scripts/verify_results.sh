#!/usr/bin/env bash
# Verify that a downloaded run is complete and all results are valid (no fallbacks).
# Usage: bash scripts/verify_results.sh <path-to-run-dir>
# Example: bash scripts/verify_results.sh ./downloads/run_20260526T040000Z
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/verify_results.sh <run-dir>" >&2
  exit 2
fi

RUN_DIR="$1"

python - "$RUN_DIR" <<'PY'
import json, sys, pathlib
import pandas as pd

run = pathlib.Path(sys.argv[1])
errors = []
warnings = []

# 1. Check DONE / FAILED markers
if (run / "FAILED.json").exists():
    payload = json.loads((run / "FAILED.json").read_text())
    errors.append(f"Run FAILED: {payload}")
elif not (run / "DONE.json").exists():
    warnings.append("No DONE.json — run may still be in progress")
else:
    info = json.loads((run / "DONE.json").read_text())
    print(f"[OK] DONE: {info.get('completed_tasks')} tasks, finished {info.get('finished_at_utc')}")

# 2. Required output files
required = [
    "reports/leaderboard_overall.csv",
    "reports/leaderboard_by_decomposition_method.csv",
    "reports/summary.md",
    "final/test_metrics.parquet",
    "final/forecasts.parquet",
    "sampling/feature_bucket_manifest.parquet",
    "checkpoints/completed_tasks.parquet",
]
for rel in required:
    p = run / rel
    if not p.exists():
        errors.append(f"Missing: {rel}")
    else:
        print(f"[OK] {rel}")

# 3. Fallback check — no trained model should have fallen back
metrics_path = run / "final" / "test_metrics.parquet"
if metrics_path.exists():
    metrics = pd.read_parquet(metrics_path)
    total = len(metrics)
    fallback_mask = ~metrics["model_output_source"].str.startswith("trained_", na=False)
    n_fallback = fallback_mask.sum()
    if n_fallback > 0:
        by_family = metrics[fallback_mask].groupby(["model_family", "model_output_source"]).size()
        errors.append(f"{n_fallback}/{total} rows are fallbacks:\n{by_family.to_string()}")
    else:
        print(f"[OK] All {total} metric rows use trained models (no fallbacks)")

    # 4. NaN / infinite check
    numeric_cols = ["rel_naive_unclipped", "rel_naive_clipped", "mae", "smape", "mase", "wape"]
    for col in numeric_cols:
        if col in metrics.columns:
            bad = metrics[col].isna().sum() + (~metrics[col].between(-1e6, 1e6)).sum()
            if bad > 0:
                warnings.append(f"{bad} bad values in column {col}")

    # 5. Task coverage check
    print(f"\n--- Task coverage ---")
    if "decomposition_method" in metrics.columns:
        print(metrics.groupby(["model_family", "decomposition_method"])["unique_id"].count().rename("n_series_evaluations").to_string())

# 6. Checkpoint coverage
cp_path = run / "checkpoints" / "completed_tasks.parquet"
if cp_path.exists():
    cp = pd.read_parquet(cp_path)
    n_done = (cp["status"].isin(["done", "already_done", "skipped"])).sum()
    n_failed = (cp["status"] == "failed").sum()
    print(f"\n[OK] Checkpoints: {n_done} done, {n_failed} failed")
    if n_failed > 0:
        failed_tasks = cp[cp["status"] == "failed"][["task_id", "error"]].head(10)
        warnings.append(f"Failed tasks:\n{failed_tasks.to_string()}")

# Summary
print("\n=== VERIFICATION SUMMARY ===")
if errors:
    print(f"ERRORS ({len(errors)}):")
    for e in errors:
        print(f"  [ERROR] {e}")
if warnings:
    print(f"WARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"  [WARN]  {w}")
if not errors and not warnings:
    print("ALL CHECKS PASSED — results are valid and ready for presentation.")
elif not errors:
    print("NO ERRORS — results are usable. Review warnings above.")
else:
    print("ERRORS FOUND — results need investigation before use.")
    sys.exit(1)
PY
