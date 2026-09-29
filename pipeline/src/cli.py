from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .utils import read_yaml, truthy


@dataclass
class CliArgs:
    config: Path
    monthly_config: Path
    quarterly_config: Path
    vast_config: Path | None
    run_root: Path
    budget: str
    resume: bool
    features: list[str] | None
    frequencies: list[str] | None
    models: list[str] | None
    decomposition_methods: list[str] | None
    finetuning_modes: list[str] | None
    skip_existing: bool
    debug: bool
    dry_run: bool
    strict_model_mode: bool | None = None
    data_dir: Path | None = None


def _csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args(argv: list[str] | None = None) -> CliArgs:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--monthly-config", required=True)
    p.add_argument("--quarterly-config", required=True)
    p.add_argument("--vast-config")
    p.add_argument("--run-root", required=True)
    p.add_argument("--budget", choices=["smoke", "pilot", "full", "large"], default="smoke")
    p.add_argument("--resume", default="true")
    p.add_argument("--features")
    p.add_argument("--frequencies")
    p.add_argument("--models")
    p.add_argument("--decomposition-methods")
    p.add_argument("--finetuning-modes")
    p.add_argument("--skip-existing", default="true")
    p.add_argument("--debug", default="false")
    p.add_argument("--dry-run", default="false")
    p.add_argument("--strict-model-mode", default=None)
    p.add_argument("--data-dir", help="folder holding the frozen M3/M4 copy (default: $RERUN_DATA_DIR)")
    ns = p.parse_args(argv)
    return CliArgs(
        config=Path(ns.config),
        monthly_config=Path(ns.monthly_config),
        quarterly_config=Path(ns.quarterly_config),
        vast_config=Path(ns.vast_config) if ns.vast_config else None,
        run_root=Path(ns.run_root),
        budget=ns.budget,
        resume=truthy(ns.resume),
        features=_csv(ns.features),
        frequencies=_csv(ns.frequencies),
        models=_csv(ns.models),
        decomposition_methods=_csv(ns.decomposition_methods),
        finetuning_modes=_csv(ns.finetuning_modes),
        skip_existing=truthy(ns.skip_existing),
        debug=truthy(ns.debug),
        dry_run=truthy(ns.dry_run),
        strict_model_mode=truthy(ns.strict_model_mode) if ns.strict_model_mode is not None else None,
        data_dir=Path(ns.data_dir) if ns.data_dir else None,
    )


def load_configs(args: CliArgs) -> dict:
    cfg = read_yaml(args.config)
    cfg["monthly"] = read_yaml(args.monthly_config)
    cfg["quarterly"] = read_yaml(args.quarterly_config)
    cfg["vast"] = read_yaml(args.vast_config) if args.vast_config else {}
    return cfg

