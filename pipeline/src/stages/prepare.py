"""Stage R1, prepare (protocol Sections 3.2-3.6): the hashed manifest bundle.

    python -m src.stages.prepare --config configs/rerun_v2.yaml --run DIR \
        --data-dir FROZEN_COPY --code-commit SHA [--jobs N]

For each frequency: load the verified frozen copy; measure every series; compute the six
features (on the history before the first test window) for the series that pass the
length rule; build the eligibility table; draw the six feature samples; assign buckets;
draw the tuning set; cut the windows of every eligible series. The results are written
once to DIR/prepare/ with bundle.json, which records the checksum of every file, the
configuration, the frozen-data manifest, the code commit and the environment lock.
Nothing downstream re-samples, re-buckets or recomputes cutoffs.

The bundle hash covers the file checksums, the configuration, the data manifest and
the code commit, not the creation time: the same inputs give the same hash.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.eligibility import eligibility_length, eligibility_table, length_table, require_eligible
from src.data.frozen import DEFAULT_MANIFEST, load_manifest, sha256_file
from src.data.load_m_datasets import frozen_data_provenance, load_dataset_pair
from src.data.sample_series import feature_sample, tercile_buckets, tuning_set
from src.data.schemas import FEATURE_NAMES, selected_features
from src.data.validation import make_cutoffs
from src.features.compute_all import QUALITY_FLAG_NAMES, compute_feature_table
from src.numeric_platform import platform_info
from src.utils import atomic_write_json

BUNDLE_FORMAT = "rerun-prepare-bundle/1"
LOCKFILE = Path(__file__).resolve().parents[2] / "environment" / "requirements-lock-linux-cu121.txt"
BUCKET_COLUMNS = ["feature_name", "frequency", "source_dataset", "unique_id", "feature_value",
                  "feature_tercile_bucket", "tercile_cut_1", "tercile_cut_2"]


class PrepareError(RuntimeError):
    """The prepare stage cannot produce a bundle that satisfies the protocol."""


def _sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _ineligibility_counts(table: pd.DataFrame) -> dict:
    out = {}
    for source, g in table.groupby("source_dataset", sort=True):
        not_ok = {}
        for name in FEATURE_NAMES:
            col = QUALITY_FLAG_NAMES[name]
            flags = g.loc[g["length_eligible"] & g[col].ne("ok"), col].value_counts()
            if len(flags):
                not_ok[name] = {str(k): int(v) for k, v in flags.items()}
        out[source] = {
            "pool": int(len(g)),
            "length_eligible": int(g["length_eligible"].sum()),
            "eligible": int(g["eligible"].sum()),
            "feature_not_ok": not_ok,
        }
    return out


def _prepare_frequency(frequency, fcfg, config, data_dir, manifest, cache_dir, n_jobs, logger) -> dict:
    m, h = int(fcfg["season_length"]), int(fcfg["horizon"])
    n_windows = int(config["validation"]["n_windows"])
    s0 = int(config["random_seed"])
    L = eligibility_length(m, h)

    data = load_dataset_pair({"frequency": frequency, **fcfg}, data_dir, manifest)
    lengths = length_table(data, h, n_windows)
    computed = set(lengths.loc[lengths["history_length"] >= L, "unique_id"])
    raw, flags = compute_feature_table(
        data[data["unique_id"].isin(computed)], m, h, n_windows,
        n_jobs=n_jobs, cache_dir=cache_dir, frequency=frequency, logger=logger,
    )
    eligibility = eligibility_table(lengths, flags, m, h).assign(frequency=frequency)
    features = raw.merge(eligibility[["unique_id", "eligible"]], on="unique_id", how="left", validate="one_to_one")
    features = features.assign(frequency=frequency)
    eligible = features[features["eligible"]]

    samples, buckets, summaries = [], [], []
    for feature in selected_features(config):
        sample = feature_sample(eligible, feature, frequency, m, s0,
                                int(config["sampling"]["n_per_source"]), int(config["sampling"]["n_strata"]))
        assigned, summary = tercile_buckets(sample, float(config["buckets"]["min_share"]))
        samples.append(sample)
        buckets.append(assigned[BUCKET_COLUMNS])
        summaries.append(summary)
    samples, buckets = pd.concat(samples, ignore_index=True), pd.concat(buckets, ignore_index=True)

    tuning = tuning_set(eligible[["unique_id", "source_dataset"]], frequency, m, s0, int(config["tuning_set"]["n_per_source"]))
    tuning = tuning.merge(lengths[["unique_id", "history_length"]], on="unique_id", how="left", validate="one_to_one")
    tuning["tuning_train_end_t"] = tuning["history_length"] - h      # validation = the last h points
    tuning["tuning_validation_end_t"] = tuning["history_length"]     # ... before the first cutoff (D13)
    tuning = tuning.drop(columns="history_length")

    eligible_ids = set(eligible["unique_id"])
    cutoffs = make_cutoffs(data[data["unique_id"].isin(eligible_ids)], h, n_windows, h, min_history=L).assign(frequency=frequency)

    # Runtime checks of U4 / G8 / G7: only eligible series downstream; features end at the first cutoff.
    for name, ids in [("samples", samples["unique_id"]), ("tuning set", tuning["unique_id"]), ("cutoffs", cutoffs["unique_id"])]:
        require_eligible(eligibility, ids)
    if set(cutoffs["unique_id"]) != eligible_ids:
        raise PrepareError(f"{frequency}: cutoffs do not cover exactly the eligible series")
    first = cutoffs[cutoffs["window"] == 0].set_index("unique_id")["train_end_idx"]
    feature_end = eligible.set_index("unique_id")["history_end_t"]
    if not feature_end.sort_index().equals(first.sort_index().astype(feature_end.dtype)):
        raise PrepareError(f"{frequency}: feature histories do not end at the first cutoff")
    if (tuning["tuning_validation_end_t"].to_numpy() != first.loc[tuning["unique_id"]].to_numpy()).any():
        raise PrepareError(f"{frequency}: tuning validation blocks do not end at the first cutoff")

    logger.info("prepare %s: %s", frequency, _ineligibility_counts(eligibility))
    return {
        "eligibility": eligibility, "features": features, "samples": samples, "buckets": buckets,
        "bucket_summary": pd.DataFrame(summaries), "tuning_set": tuning, "cutoffs": cutoffs,
        "counts": _ineligibility_counts(eligibility),
    }


def run_prepare(
    config: dict,
    run_dir: Path,
    data_dir: Path,
    code_commit: str,
    n_jobs: int = 1,
    manifest_path: Path = DEFAULT_MANIFEST,
    logger: logging.Logger | None = None,
) -> dict:
    """Run stage R1 into run_dir/prepare and return the bundle record (bundle.json)."""
    logger = logger or logging.getLogger(__name__)
    if not re.fullmatch(r"[0-9a-f]{40}", code_commit):
        raise PrepareError("code_commit must be a full 40-character git commit hash")
    out = Path(run_dir) / "prepare"
    if (out / "bundle.json").exists():
        raise PrepareError(f"{out} already holds a bundle; a bundle is never overwritten")
    state = {
        "code_commit": code_commit,
        "config_sha256": _sha256_json(config),
        "data_manifest_sha256": sha256_file(manifest_path),
    }
    state_path = out / "prepare_state.json"
    if state_path.exists() and json.loads(state_path.read_text(encoding="utf-8")) != state:
        raise PrepareError(f"{out} holds an interrupted prepare with other inputs")
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(state_path, state)

    manifest = load_manifest(manifest_path)
    parts = {}
    for frequency in sorted(config["frequencies"]):
        parts[frequency] = _prepare_frequency(
            frequency, config["frequencies"][frequency], config, data_dir, manifest,
            out / "cache" / frequency, n_jobs, logger,
        )

    files = {}
    for name in ["eligibility", "features", "samples", "buckets", "bucket_summary", "tuning_set", "cutoffs"]:
        df = pd.concat([parts[f][name] for f in sorted(parts)], ignore_index=True)
        path = out / f"{name}.parquet"
        df.to_parquet(path, index=False)
        files[name] = {"file": path.name, "sha256": sha256_file(path), "rows": int(len(df))}

    bundle = {
        "format": BUNDLE_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_commit": code_commit,
        "environment_lock_sha256": sha256_file(LOCKFILE) if LOCKFILE.exists() else None,
        "config": config,
        "config_sha256": state["config_sha256"],
        "data": frozen_data_provenance(manifest_path),
        "platform": platform_info(),
        "files": files,
        "counts": {f: parts[f]["counts"] for f in sorted(parts)},
        "bucket_summary": [r for f in sorted(parts) for r in parts[f]["bucket_summary"].to_dict("records")],
    }
    bundle["bundle_sha256"] = _sha256_json({
        "files": {k: v["sha256"] for k, v in sorted(files.items())},
        "config_sha256": state["config_sha256"],
        "data_manifest_sha256": state["data_manifest_sha256"],
        "code_commit": code_commit,
    })
    atomic_write_json(out / "bundle.json", bundle)
    return bundle

