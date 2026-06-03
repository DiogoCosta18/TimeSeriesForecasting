"""Rebuild paper parquets from the complete vast.ai run results.

Finetuning modes used in analysis:
  no_finetune         — baseline, global model on all 750 series
  finetune_bucket_low    — global model trained ONLY on Low-bucket series (~250)
  finetune_bucket_medium — global model trained ONLY on Medium-bucket series (~250)
  finetune_bucket_high   — global model trained ONLY on High-bucket series (~250)

NOTE: 'finetune_by_feature_bucket' (constant multiplier, no retraining) is
excluded — its median improvement is +0.00007, indistinguishable from noise.
"""
from pathlib import Path
import numpy as np
import pandas as pd

RUN_ROOT = Path("../../forecasting_pipeline/outputs/run_20260525T112121")
OUT = Path("data")
OUT.mkdir(exist_ok=True)

BUCKET_FT_MODES = {"finetune_bucket_low", "finetune_bucket_medium", "finetune_bucket_high"}
KEEP_MODES      = BUCKET_FT_MODES | {"no_finetune"}

print(f"Using run: {RUN_ROOT.name}")

# ── 1. Master metrics ─────────────────────────────────────────────────────────
print("\nLoading master metrics...")
raw = pd.read_parquet(RUN_ROOT / "cv" / "fold_metrics.parquet")
raw["feature_bucket"] = raw["feature_bucket"].str.capitalize()
raw["frequency"]      = raw["frequency"].str.lower()
raw = raw.drop_duplicates(subset=["task_id", "unique_id", "window"])

# Drop the no-op constant-multiplier mode
excluded = (~raw["finetuning_mode"].isin(KEEP_MODES)).sum()
master = raw[raw["finetuning_mode"].isin(KEEP_MODES)].copy()
print(f"Dropped {excluded:,} rows (finetune_by_feature_bucket no-op mode)")
print(f"Master table: {len(master):,} rows")
print(master["finetuning_mode"].value_counts().to_string())

# ── 2. POCID from fold_forecasts ──────────────────────────────────────────────
print("\nComputing POCID from no_finetune fold_forecasts...")

def compute_pocid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["task_id", "unique_id", "window", "ds"])
    rows = []
    for (tid, uid, win), grp in df.groupby(["task_id", "unique_id", "window"], sort=False):
        if len(grp) < 2:
            continue
        dy    = grp["y"].diff().iloc[1:]
        dyhat = grp["yhat"].diff().iloc[1:]
        rows.append({"task_id": tid, "unique_id": uid, "window": win,
                     "pocid": float((np.sign(dy) == np.sign(dyhat)).mean())})
    return pd.DataFrame(rows)

ff = RUN_ROOT / "cv" / "fold_forecasts.parquet"
if ff.exists():
    print(f"  Loading {ff.name}...")
    forecasts = pd.read_parquet(ff, columns=["task_id","unique_id","window","ds","y","yhat"])
    noft_ids = set(master.loc[master["finetuning_mode"]=="no_finetune","task_id"])
    forecasts_noft = forecasts[forecasts["task_id"].isin(noft_ids)]
    print(f"  {len(forecasts_noft):,} rows — computing POCID...")
    pocid_noft = compute_pocid(forecasts_noft)
    del forecasts, forecasts_noft

    pocid_noft["base_task_id"] = pocid_noft["task_id"].str.rsplit("|", n=1).str[0]
    pocid_base = pocid_noft.groupby(["base_task_id","unique_id","window"])["pocid"].mean().reset_index()

    master["base_task_id"] = master["task_id"].str.rsplit("|", n=1).str[0]
    master = master.merge(pocid_base, on=["base_task_id","unique_id","window"], how="left")
    master.drop(columns="base_task_id", inplace=True)
    print(f"  POCID coverage: {100*master['pocid'].notna().mean():.1f}%")
else:
    print("  fold_forecasts.parquet not found — POCID will be NaN")
    master["pocid"] = np.nan

# ── 3. Features & bucket manifest ─────────────────────────────────────────────
feat_raw = pd.read_parquet(RUN_ROOT / "features" / "feature_table_raw.parquet")
feat_raw["frequency"] = feat_raw["frequency"].str.lower()

bkt = None
bkt_path = RUN_ROOT / "sampling" / "feature_bucket_manifest.parquet"
if bkt_path.exists():
    bkt = pd.read_parquet(bkt_path)
    bkt["frequency"] = bkt["frequency"].str.lower()
    if "feature_tercile_bucket" in bkt.columns:
        bkt["feature_tercile_bucket"] = bkt["feature_tercile_bucket"].str.capitalize()
    print(f"\nBucket manifest: {bkt.shape}")
print(f"Feature table:   {feat_raw.shape}")

# ── 4. STL delta ──────────────────────────────────────────────────────────────
# Computed within each finetuning_mode separately (no_finetune + bucket modes).
STL_MERGE_KEYS = ["feature_name","frequency","model_family","model_name",
                  "finetuning_mode","unique_id","source_dataset","window","feature_bucket"]

baseline_stl = master[master["decomposition_method"]=="without_stl"][
    STL_MERGE_KEYS + ["rel_naive_clipped","pocid"]
].rename(columns={"rel_naive_clipped":"rn_base","pocid":"pocid_base"})

stl_variants = master[master["decomposition_method"]!="without_stl"].copy()
stl_delta = stl_variants.merge(baseline_stl, on=STL_MERGE_KEYS, how="inner")
stl_delta["rn_delta"]    = stl_delta["rn_base"] - stl_delta["rel_naive_clipped"]
stl_delta["pocid_delta"] = stl_delta["pocid"]   - stl_delta["pocid_base"]

print(f"\nSTL delta rows: {len(stl_delta):,}")
print(stl_delta[stl_delta["finetuning_mode"]=="no_finetune"]
      .groupby("decomposition_method")["rn_delta"]
      .agg(mean="mean",median="median",win_rate=lambda x:(x>0).mean()).round(4).to_string())

# ── 5. Bucket finetuning delta ────────────────────────────────────────────────
# Compare each bucket-specific mode against no_finetune for the SAME
# series/window/bucket. Bucket-specific rows only exist for their own bucket,
# so the merge naturally restricts to same-bucket comparisons.
FT_MERGE_KEYS = ["feature_name","frequency","decomposition_method",
                 "model_family","model_name","unique_id","source_dataset",
                 "window","feature_bucket"]

no_ft = master[master["finetuning_mode"]=="no_finetune"][
    FT_MERGE_KEYS + ["rel_naive_clipped","pocid"]
].rename(columns={"rel_naive_clipped":"rn_no_ft","pocid":"pocid_no_ft"})

with_ft = master[master["finetuning_mode"].isin(BUCKET_FT_MODES)].copy()
bucket_ft_delta = with_ft.merge(no_ft, on=FT_MERGE_KEYS, how="inner")
bucket_ft_delta["rn_delta_ft"]    = bucket_ft_delta["rn_no_ft"] - bucket_ft_delta["rel_naive_clipped"]
bucket_ft_delta["pocid_delta_ft"] = bucket_ft_delta["pocid"]    - bucket_ft_delta["pocid_no_ft"]
bucket_ft_delta["trained_on_bucket"] = (bucket_ft_delta["finetuning_mode"]
                                         .str.replace("finetune_bucket_","")
                                         .str.capitalize())

print(f"\nBucket finetuning delta rows: {len(bucket_ft_delta):,}")
print(bucket_ft_delta.groupby(["trained_on_bucket","model_family"])["rn_delta_ft"].agg(
    mean="mean", median="median", win_rate=lambda x:(x>0).mean()).round(4).to_string())

# ── 6. Save ────────────────────────────────────────────────────────────────────
master.to_parquet(OUT / "master_metrics.parquet", index=False)
stl_delta.to_parquet(OUT / "stl_delta.parquet", index=False)
bucket_ft_delta.to_parquet(OUT / "bucket_ft_delta.parquet", index=False)
feat_raw.to_parquet(OUT / "features.parquet", index=False)
if bkt is not None:
    bkt.to_parquet(OUT / "bucket_manifest.parquet", index=False)

# Remove stale finetune_delta.parquet (replaced by bucket_ft_delta)
stale = OUT / "finetune_delta.parquet"
if stale.exists():
    stale.unlink()
    print("Removed stale finetune_delta.parquet")

print("\nSaved:")
for f in sorted(OUT.glob("*.parquet")):
    print(f"  {f.name:<38} {f.stat().st_size/1e6:.1f} MB")
print("\nDone.")
