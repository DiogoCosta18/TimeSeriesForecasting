"""What the pilot must show (protocol Section 8.2), from a finished pilot run.

    python scripts/pilot_report.py --run PILOT_RUN --full-config configs/rerun_v2.yaml \
        --full-prepare FULL_RUN/prepare --out PILOT_RUN/pilot

Writes report.json and report.md with: stage and shard wall times; time per task and per
tuning trial by family, frequency, strategy and scope; failures by family; the gate results;
the I3 check (every output of Table 6 produced); the GPU memory measurement if
scripts/gpu_memory_probe.py has run; and a projection of the full run's wall-clock time on the
four machines of Section 9.3 with its assumptions:
- evaluation tasks are scaled by the number of full-run tasks of the same kind (family,
  model, frequency, strategy, scope), counted on the full run's prepare bundle;
- ML training time grows with the pool (rows of the training frame); neural and transformer
  training runs a fixed number of steps, so their time is given as a range, from unchanged
  (low) to growing with the pool (high), and, with --neural-pool-factor, as the measured
  growth (the "measured" bound; scripts/pool_scaling measurement of the pilot);
- statistical time grows with the number of series; tuning time with the number of trials,
  and for ML also with the tuning set;
- machines A-D as in Section 9.3; a machine's GPU shards run in sequence; on machine D the ML
  chain and the two statistical shards run side by side, each evaluation shard with the given
  number of worker processes (evaluate --workers), assumed to divide its time (ideal scaling).
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.analysis.run import OUTPUTS  # noqa: E402
from src.forecast.registry import FAMILY  # noqa: E402
from src.stages.io import read_json  # noqa: E402
from src.stages.tasks import evaluation_tasks, tuning_tasks  # noqa: E402
from src.utils import read_yaml  # noqa: E402

MACHINES = {"A": ["neural-monthly"], "B": ["neural-quarterly", "transformer-quarterly"], "C": ["transformer-monthly"],
            "D": ["ml-monthly", "ml-quarterly", "statistical-monthly", "statistical-quarterly"]}
LOG_LINE = re.compile(r"^=== (\S+) (\S+) (.*)$")


def _ts(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")


def stage_times(run: Path) -> dict[str, dict]:
    """Start, end and exit of the last attempt of every stage (logs written by run_stage.sh)."""
    out = {}
    for log in sorted((run / "logs").glob("*.log")):
        start = end = status = None
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = LOG_LINE.match(line)
            if not m:
                continue
            if m.group(2) == "exit":  # "=== TIME exit N" (run_stage.sh); otherwise "=== TIME HOST COMMAND"
                end, status = _ts(m.group(1)), int(m.group(3))
            else:
                start, end, status = _ts(m.group(1)), None, None
        if start is not None:
            out[log.stem] = {"start": start.isoformat(), "end": end.isoformat() if end else None, "exit": status,
                             "hours": (end - start).total_seconds() / 3600 if end else None}
    return out


def task_times(run: Path, stages: dict) -> pd.DataFrame:
    """Time of every evaluation task: its recorded compute seconds, or (records written before
    they were kept, as in the first pilot) the time since the previous completion in its shard."""
    rows = []
    for shard_dir in sorted((run / "evaluate").glob("*")):
        records = sorted((read_json(p) for p in (shard_dir / "tasks").glob("*.done.json")), key=lambda r: r["completed_at_utc"])
        previous = datetime.fromisoformat(stages[f"evaluate-{shard_dir.name}"]["start"])
        for r in records:
            done = _ts(r["completed_at_utc"])
            seconds = r.get("compute_seconds", (done - previous).total_seconds())
            rows.append({"shard": shard_dir.name, "task_id": r["task_id"], "seconds": seconds,
                         "rows": r["rows"], "failed_rows": r["failed_rows"], "failed_attempts": len(r.get("failed_attempts", []))})
            previous = done
    df = pd.DataFrame(rows)
    parts = df["task_id"].str.split("|", expand=True)
    stat = df["task_id"].str.startswith("stat|")
    df["kind"] = parts[0]
    df["model"] = parts[1].where(stat, parts[4])
    df["frequency"] = parts[2]
    df["strategy"] = parts[3]
    df["scope"] = parts[5].where(~stat, "cohort")
    df["seed"] = parts[6].where(~stat, None)
    df["family"] = df["model"].map(FAMILY)
    df["scope_kind"] = df["scope"].where(df["scope"] == "cohort", "tercile")
    return df


def trial_times(run: Path) -> pd.DataFrame:
    rows = []
    for db in sorted((run / "tune").glob("*/studies.sqlite")):
        con = sqlite3.connect(db)
        q = """SELECT s.study_name, t.number, v.value_json FROM trials t JOIN studies s ON t.study_id = s.study_id
               LEFT JOIN trial_user_attributes v ON v.trial_id = t.trial_id AND v.key = 'duration_seconds'"""
        for study, number, value in con.execute(q):
            model, frequency, target = study.split("__")
            rows.append({"shard": db.parent.name, "model": model, "frequency": frequency, "target": target, "trial": number,
                         "seconds": float(json.loads(value)) if value else float("nan")})
        con.close()
    return pd.DataFrame(rows)


def full_task_counts(full_prepare: Path, full_config: dict) -> pd.DataFrame:
    tasks = evaluation_tasks(pd.read_parquet(full_prepare / "bucket_summary.parquet"), pd.read_parquet(full_prepare / "samples.parquet"),
                             int(full_config["random_seed"]), list(full_config["seed_check_seeds"]))
    df = pd.DataFrame([{"family": t.family, "model": t.model, "frequency": t.frequency, "strategy": t.strategy,
                        "scope_kind": "cohort" if t.scope == "cohort" else "tercile", "kind": t.kind} for t in tasks])
    return df.groupby(["kind", "family", "model", "frequency", "strategy", "scope_kind"]).size().rename("full_tasks").reset_index()


def project(tasks: pd.DataFrame, trials: pd.DataFrame, pilot_cfg: dict, full_cfg: dict, full_prepare: Path,
            pilot_prepare: Path, stat_workers: int = 1, ml_workers: int = 1, neural_factor: float | None = None) -> dict:
    pool = full_cfg["sampling"]["n_per_source"] / pilot_cfg["sampling"]["n_per_source"]
    tuning_pool = full_cfg["tuning_set"]["n_per_source"] / pilot_cfg["tuning_set"]["n_per_source"]
    trial_factor = full_cfg["tuning"]["num_samples"] / pilot_cfg["tuning"]["num_samples"]
    full_union = pd.read_parquet(full_prepare / "samples.parquet").groupby("frequency")["unique_id"].nunique()
    pilot_union = pd.read_parquet(pilot_prepare / "samples.parquet").groupby("frequency")["unique_id"].nunique()

    glob = tasks[tasks["kind"] == "eval"]
    per_kind = glob.groupby(["family", "model", "frequency", "strategy", "scope_kind"])["seconds"].mean().rename("pilot_seconds").reset_index()
    counts = full_task_counts(full_prepare, full_cfg)
    merged = counts[counts["kind"] == "global"].merge(per_kind, on=["family", "model", "frequency", "strategy", "scope_kind"], how="left")
    if merged["pilot_seconds"].isna().any():
        raise SystemExit(f"no pilot time for {merged[merged['pilot_seconds'].isna()].head().to_dict('records')}")
    ml = merged["family"] == "ml"
    merged["low_hours"] = merged["full_tasks"] * merged["pilot_seconds"] * ml.map({True: pool, False: 1.0}) / 3600
    merged["high_hours"] = merged["full_tasks"] * merged["pilot_seconds"] * pool / 3600
    bounds = ["low_hours", "high_hours"]
    if neural_factor is not None:
        merged["measured_hours"] = merged["full_tasks"] * merged["pilot_seconds"] * ml.map({True: pool, False: neural_factor}) / 3600
        bounds.append("measured_hours")
    merged["shard"] = merged["family"] + "-" + merged["frequency"]
    evaluation = merged.groupby("shard")[bounds].sum()

    stat = tasks[tasks["kind"] == "stat"].groupby("frequency")["seconds"].sum() / 3600
    for frequency, hours in stat.items():
        factor = full_union[frequency] / pilot_union[frequency]
        evaluation.loc[f"statistical-{frequency}"] = [hours * factor] * len(bounds)

    study = trials.groupby(["shard", "model", "frequency", "target"])["seconds"].sum().reset_index()
    study["factor"] = trial_factor * study["model"].map(lambda m: tuning_pool if FAMILY[m] == "ml" else 1.0)
    study["high_factor"] = trial_factor * tuning_pool
    tuning = pd.DataFrame({"low_hours": (study["seconds"] * study["factor"]).groupby(study["shard"]).sum() / 3600,
                           "high_hours": (study["seconds"] * study["high_factor"]).groupby(study["shard"]).sum() / 3600})
    tuning.loc[tuning.index.str.startswith("ml"), "high_hours"] = tuning.loc[tuning.index.str.startswith("ml"), "low_hours"]
    if neural_factor is not None:
        study["measured_factor"] = trial_factor * study["model"].map(lambda m: tuning_pool if FAMILY[m] == "ml" else neural_factor)
        tuning["measured_hours"] = (study["seconds"] * study["measured_factor"]).groupby(study["shard"]).sum() / 3600

    one_process = evaluation.copy()
    evaluation.loc[evaluation.index.str.startswith("statistical")] /= stat_workers
    evaluation.loc[evaluation.index.str.startswith("ml")] /= ml_workers
    machines = {}
    for name, shards in MACHINES.items():
        res = {}
        for bound in bounds:
            if name == "D":
                ml_chain = sum(tuning[bound].get(s, 0) + evaluation[bound].get(s, 0) for s in shards if s.startswith("ml"))
                stat_chain = max(evaluation[bound].get(s, 0) for s in shards if s.startswith("statistical"))
                res[bound] = max(ml_chain, stat_chain)
            else:
                res[bound] = sum(tuning[bound].get(s, 0) + evaluation[bound].get(s, 0) for s in shards)
        machines[name] = res
    return {
        "factors": {"pool": pool, "tuning_set": tuning_pool, "trials": trial_factor,
                    "statistical_series": {f: float(full_union[f] / pilot_union[f]) for f in full_union.index}},
        "tuning_hours": tuning.round(2).to_dict("index"),
        "workers": {"statistical": stat_workers, "ml": ml_workers},
        "evaluation_hours": evaluation.round(2).to_dict("index"),
        "evaluation_hours_one_process": one_process.round(2).to_dict("index"),
        "machines_hours": {k: {b: round(v, 2) for b, v in r.items()} for k, r in machines.items()},
        "wall_clock_hours": {b: round(max(r[b] for r in machines.values()), 2) for b in bounds},
        "criterion_hours": 72,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--full-config", required=True, type=Path)
    ap.add_argument("--full-prepare", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stat-workers", type=int, default=1)
    ap.add_argument("--ml-workers", type=int, default=1)
    ap.add_argument("--neural-pool-factor", type=float, default=None, help="measured growth of neural fit time with the pool")
    args = ap.parse_args()
    run = args.run
    pilot_cfg = read_json(run / "prepare" / "bundle.json")["config"]
    full_cfg = read_yaml(args.full_config)

    stages = stage_times(run)
    tasks = task_times(run, stages)
    trials = trial_times(run)
    merge = read_json(run / "merged" / "merge.json")
    gates = read_json(run / "merged" / "gates.json")
    analysis = read_json(run / "analysis" / "analysis.json") if (run / "analysis" / "analysis.json").exists() else None
    probe = run / "pilot" / "gpu_memory.json"

    rows = pd.read_parquet(run / "merged" / "rows.parquet", columns=["family", "status"])
    failed = pd.read_parquet(run / "merged" / "failed_rows.parquet", columns=["family", "model", "failure"])
    failures = {fam: {"rows": int((rows["family"] == fam).sum()), "failed_rows": int((failed["family"] == fam).sum()),
                      "tasks_retried": int(tasks.loc[tasks["family"] == fam, "failed_attempts"].gt(0).sum())}
                for fam in sorted(rows["family"].unique())}
    expected = {p for paths in OUTPUTS.values() for p in paths}
    if not pilot_cfg.get("seed_check_seeds"):
        expected.discard("figures/seed_spread.png")  # no seed check, no seed figure (src/analysis/run.py)
    report = {
        "run": str(run), "code_commit": merge["provenance"]["code_commit"],
        "stages": stages,
        "task_seconds": tasks.groupby(["family", "frequency", "strategy", "scope_kind"])["seconds"].describe()[["count", "mean", "50%", "max"]]
                             .round(1).reset_index().to_dict("records"),
        "trial_seconds": trials.groupby(["model", "frequency"])["seconds"].describe()[["count", "mean", "50%", "max"]]
                               .round(1).reset_index().to_dict("records"),
        "failures": failures, "listed_failures": merge["grid"]["listed_failures"], "d16_reruns": list(merge["d16_reruns"]),
        "failure_reasons": failed.groupby(["model", failed["failure"].str.split(":").str[0]]).size().rename("rows").reset_index()
                                 .to_dict("records") if len(failed) else [],
        "gates": {k: v["passed"] for k, v in gates["gates"].items()}, "gates_passed": gates["passed"],
        "degenerate": gates["gates"]["G12"]["degenerate"],
        "i3": None if analysis is None else {"outputs_expected": len(expected), "missing": sorted(expected - set(analysis["files"]))},
        "gpu_memory": read_json(probe)["summary"] if probe.exists() else None,
        "projection": project(tasks, trials, pilot_cfg, full_cfg, args.full_prepare, run / "prepare",
                              args.stat_workers, args.ml_workers, args.neural_pool_factor),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    p = report["projection"]
    lines = [f"# Pilot report ({run.name}, commit {report['code_commit'][:7]})", "",
             f"Gates: {'all passed' if report['gates_passed'] else 'FAILED: ' + ', '.join(k for k, v in report['gates'].items() if not v)}; "
             f"degenerate: {[d['feature'] + '|' + d['frequency'] for d in report['degenerate']]}.",
             f"I3: {report['i3']}.", "", "## Stage wall times (hours)", ""]
    lines += [f"- {k}: {v['hours']:.2f} (exit {v['exit']})" for k, v in stages.items() if v["hours"] is not None]
    lines += ["", "## Failures by family", ""] + [f"- {k}: {v}" for k, v in failures.items()]
    lines += ["", "## Projection of the full run (hours; low-high)", ""]
    show = lambda v: f"{v['low_hours']:.1f} - {v['high_hours']:.1f}" + (f"; measured {v['measured_hours']:.1f}" if "measured_hours" in v else "")
    lines += [f"- machine {k}: {show(v)}" for k, v in p["machines_hours"].items()]
    lines += [f"- wall clock: {show(p['wall_clock_hours'])} (criterion <= 72)"]
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
