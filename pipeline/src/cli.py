"""Command line of the rerun (protocol Section 6.3).

    python -m src.cli freeze-data ...                       # the frozen copy (src.data.freeze_data)
    python -m src.cli prepare  --config C --run DIR --data-dir D [--jobs N]
    python -m src.cli tune     --config C --run DIR --data-dir D --shard FAMILY-FREQUENCY
    python -m src.cli freeze   --config C --run DIR        # configs_frozen.json + hash
    python -m src.cli evaluate --config C --run DIR --data-dir D --shard FAMILY-FREQUENCY [--only TASK_ID ...]
    python -m src.cli merge    --run DIR                   # refuses mismatched hashes
    python -m src.cli gates    --run DIR                   # G1-G13; blocks the analysis on failure
    python -m src.cli analyse  --run DIR --out OUT

Every stage reads the code commit from git and refuses a working tree with uncommitted
changes; the other hashes come from the files themselves.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from src.utils import read_yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("freeze-data", add_help=False)
    for name in ("prepare", "tune", "freeze", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, type=Path)
        p.add_argument("--run", required=True, type=Path)
        if name != "freeze":
            p.add_argument("--data-dir", required=True, type=Path)
        if name in ("tune", "evaluate"):
            p.add_argument("--shard", required=True)
        if name == "evaluate":
            p.add_argument("--only", nargs="*", default=None, help="rerun only these task ids (D16)")
        if name == "prepare":
            p.add_argument("--jobs", default="1")
    for name in ("merge", "gates"):
        sub.add_parser(name).add_argument("--run", required=True, type=Path)
    p = sub.add_parser("analyse")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)

    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "freeze-data":
        from src.data.freeze_data import main as freeze_data_main

        return freeze_data_main(argv[1:])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.stage == "prepare":
        from src.stages.io import code_commit
        from src.stages.prepare import run_prepare
        from src.utils import effective_cpu_count

        jobs = effective_cpu_count() if args.jobs == "auto" else int(args.jobs)
        result = run_prepare(read_yaml(args.config), args.run, args.data_dir, code_commit(), n_jobs=jobs)
        print(json.dumps({"bundle_sha256": result["bundle_sha256"], "counts": result["counts"]}, indent=2))
    elif args.stage == "tune":
        from src.stages.tune import run_tune

        print(run_tune(args.run, read_yaml(args.config), args.data_dir, args.shard))
    elif args.stage == "freeze":
        from src.stages.tune import run_freeze

        print({"configs_sha256": run_freeze(args.run, read_yaml(args.config))})
    elif args.stage == "evaluate":
        from src.stages.evaluate import run_evaluate

        only = set(args.only) if args.only else None
        print(run_evaluate(args.run, read_yaml(args.config), args.data_dir, args.shard, only=only))
    elif args.stage == "merge":
        from src.stages.merge import run_merge

        print(run_merge(args.run))
    elif args.stage == "gates":
        from src.validation.gates import run_gates

        report = run_gates(args.run)
        print(json.dumps({k: v["passed"] for k, v in report["gates"].items()}, indent=2))
        return 0 if report["passed"] else 1
    elif args.stage == "analyse":
        from src.analysis.run import run_analysis

        print(run_analysis(args.run, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
