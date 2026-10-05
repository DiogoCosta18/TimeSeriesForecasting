"""configs/rerun_v2.yaml encodes the protocol's decisions exactly (Section 2)."""
from __future__ import annotations

from pathlib import Path

from src.data.eligibility import eligibility_length
from src.data.sample_series import MIN_BUCKET_SHARE
from src.utils import read_yaml

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "rerun_v2.yaml"


def test_config_matches_the_protocol():
    cfg = read_yaml(CONFIG)
    assert isinstance(cfg["random_seed"], int)
    assert cfg["seed_check_seeds"] == [cfg["random_seed"] + 1, cfg["random_seed"] + 2]  # D17: two extra seeds
    assert cfg["frequencies"] == {
        "monthly": {"season_length": 12, "horizon": 18, "m3_group": "Monthly", "m4_group": "Monthly"},
        "quarterly": {"season_length": 4, "horizon": 8, "m3_group": "Quarterly", "m4_group": "Quarterly"},
    }                                                                        # D2
    assert cfg["validation"] == {"n_windows": 3}                             # D8
    assert cfg["sampling"] == {"n_per_source": 600, "n_strata": 30}          # D6
    assert cfg["buckets"] == {"min_share": 0.20} and MIN_BUCKET_SHARE == 0.20  # D7
    assert cfg["tuning_set"] == {"n_per_source": 600}                        # D13
    assert cfg["tuning"]["num_samples"] == 20                                # D12
    assert set(cfg["tuning"]["early_stopping"]) == {"patience", "check_steps"}  # tuning only (Section 4.3)
    assert [eligibility_length(f["season_length"], f["horizon"]) for f in cfg["frequencies"].values()] == [54, 20]  # D3


def test_pilot_config_differs_from_the_full_run_only_where_section_8_1_says():
    full = read_yaml(CONFIG)
    pilot = read_yaml(CONFIG.with_name("pilot_v2.yaml"))
    changed = {("seed_check_seeds",), ("features",), ("sampling", "n_per_source"), ("sampling", "n_strata"),
               ("tuning_set", "n_per_source"), ("tuning", "num_samples")}

    def flat(d, prefix=()):
        out = {}
        for k, v in d.items():
            out.update(flat(v, prefix + (k,)) if isinstance(v, dict) else {prefix + (k,): v})
        return out

    f, p = flat(full), flat(pilot)
    assert {k for k in f.keys() | p.keys() if f.get(k) != p.get(k)} == changed
    assert pilot["seed_check_seeds"] == [] and pilot["features"] == ["feature_evolving_seasonality", "feature_nonlinearity"]
    assert (pilot["sampling"]["n_per_source"], pilot["tuning_set"]["n_per_source"], pilot["tuning"]["num_samples"]) == (100, 100, 3)
    assert pilot["sampling"]["n_per_source"] % pilot["sampling"]["n_strata"] == 0
