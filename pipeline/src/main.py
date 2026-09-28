from __future__ import annotations

from .cli import load_configs, parse_args
from .training.run_experiment import run_experiment


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_configs(args)
    run_experiment(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

