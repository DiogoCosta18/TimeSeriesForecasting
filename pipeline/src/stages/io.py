"""Run-directory primitives: provenance, atomic outputs, completion records, heartbeat.

Protocol D16, D19, D20, Section 9.4; gates G2-G4, G13; test I4. A task's output becomes
visible only when complete (written under a temporary name, then renamed), followed by
a completion record with the output's SHA-256 and the provenance it was produced with.
A rerun skips a task whose record and output agree with the current provenance and
refuses one produced with any other provenance; nothing is mixed.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.frozen import DEFAULT_MANIFEST, sha256_file
from src.forecast.engine import Provenance

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKFILE = REPO_ROOT / "environment" / "requirements-lock-linux-cu121.txt"


class StageError(RuntimeError):
    """A stage cannot proceed without breaking the protocol."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def code_commit(repo: Path = REPO_ROOT) -> str:
    """HEAD of the repository; refuses a working tree with uncommitted changes (G4)."""
    if os.environ.get("RERUN_CODE_COMMIT"):  # set by tests and by machines without git metadata
        commit = os.environ["RERUN_CODE_COMMIT"]
    else:
        def git(*args):
            return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()

        if git("status", "--porcelain"):
            raise StageError("the working tree has uncommitted changes; results must come from a committed state")
        commit = git("rev-parse", "HEAD")
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise StageError(f"not a full commit hash: {commit!r}")
    return commit


def is_ancestor(old: str, new: str, repo: Path = REPO_ROOT) -> bool:
    """Whether commit ``old`` is an ancestor of (or equal to) commit ``new``."""
    return subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", old, new], capture_output=True).returncode == 0


def environment_lock_sha256() -> str:
    return sha256_file(LOCKFILE)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_parquet(path: Path, df: pd.DataFrame) -> str:
    """Write atomically; return the file's SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return sha256_file(path)


def sha256_json(obj) -> str:
    """Content hash of a JSON-able object (the same canonical form as the prepare bundle's)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_bundle(run_dir: Path, config: dict, manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    """The prepare bundle, after checking that the stage runs with the bundle's configuration,
    data manifest and unchanged prepare files (G3, G4); a mismatch is refused, never mixed."""
    prepare = Path(run_dir) / "prepare"
    bundle = read_json(prepare / "bundle.json")
    if sha256_json(config) != bundle["config_sha256"]:
        raise StageError("the configuration differs from the one the prepare bundle was built with")
    if sha256_file(manifest_path) != bundle["data"]["manifest_sha256"]:
        raise StageError("the data manifest differs from the one the prepare bundle was built with")
    for entry in bundle["files"].values():
        if sha256_file(prepare / entry["file"]) != entry["sha256"]:
            raise StageError(f"prepare/{entry['file']} changed since the bundle was written")
    return bundle


def stage_provenance(bundle: dict, manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    """Provenance of tuning outputs: code, environment, data and bundle (D19)."""
    return {"code_commit": code_commit(), "environment_lock_sha256": environment_lock_sha256(),
            "data_manifest_sha256": sha256_file(manifest_path), "bundle_sha256": bundle["bundle_sha256"]}


def evaluation_provenance(run_dir: Path, bundle: dict, manifest_path: Path = DEFAULT_MANIFEST) -> Provenance:
    """D19 for evaluation rows, read from the files themselves (never typed in)."""
    configs = read_json(Path(run_dir) / "configs_frozen.json")
    return Provenance(
        code_commit=code_commit(),
        environment_lock_sha256=environment_lock_sha256(),
        data_manifest_sha256=sha256_file(manifest_path),
        bundle_sha256=bundle["bundle_sha256"],
        configs_sha256=configs["configs_sha256"],
    )


def safe_name(task_id: str) -> str:
    return task_id.replace("|", "__")


class TaskStore:
    """Outputs and completion records of the tasks of one stage and shard."""

    def __init__(self, root: Path, provenance: dict):
        self.root = Path(root)
        self.provenance = dict(provenance)

    def output(self, task_id: str, suffix: str = ".parquet") -> Path:
        return self.root / "tasks" / f"{safe_name(task_id)}{suffix}"

    def record_path(self, task_id: str) -> Path:
        return self.root / "tasks" / f"{safe_name(task_id)}.done.json"

    def is_done(self, task_id: str, suffix: str = ".parquet") -> bool:
        record_path = self.record_path(task_id)
        if not record_path.exists():
            return False
        record = read_json(record_path)
        if record["provenance"] != self.provenance:
            raise StageError(f"{task_id} was produced with other provenance {record['provenance']}; "
                             "use a new run directory instead of mixing results")
        output = self.output(task_id, suffix)
        if not output.exists() or sha256_file(output) != record["sha256"]:
            raise StageError(f"{task_id}: output missing or changed since it was recorded")
        return True

    def complete(self, task_id: str, sha256: str, extra: dict | None = None, suffix: str = ".parquet") -> None:
        write_json(self.record_path(task_id), {"task_id": task_id, "sha256": sha256, "file": self.output(task_id, suffix).name,
                                               "provenance": self.provenance, "completed_at_utc": utc_now(), **(extra or {})})

    def failure_path(self, task_id: str) -> Path:
        return self.root / "failures" / f"{safe_name(task_id)}.json"

    def failure(self, task_id: str) -> dict | None:
        path = self.failure_path(task_id)
        return read_json(path) if path.exists() else None

    def record_failure(self, task_id: str, attempts: list[str]) -> None:
        """List a failure; an earlier listing of the same task is kept in ``history``."""
        previous = self.failure(task_id)
        history = [] if previous is None else previous["history"] + [{k: previous[k] for k in ("attempts", "provenance", "failed_at_utc")}]
        write_json(self.failure_path(task_id), {"task_id": task_id, "attempts": attempts, "provenance": self.provenance,
                                                "failed_at_utc": utc_now(), "history": history})

    def clear_failure(self, task_id: str) -> None:
        if self.failure_path(task_id).exists():
            self.failure_path(task_id).unlink()


class Heartbeat:
    """Writes heartbeat.json every ``interval`` seconds while a stage runs (Section 9.4)."""

    def __init__(self, path: Path, stage: str, shard: str | None, interval: float = 60.0):
        self.path, self.stage, self.shard, self.interval = Path(path), stage, shard, interval
        self.state = {"task": None, "done": 0, "total": 0, "failed": 0}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _write(self) -> None:
        write_json(self.path, {"stage": self.stage, "shard": self.shard, "host": socket.gethostname(),
                               "pid": os.getpid(), "time_utc": utc_now(), **self.state})

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._write()

    def update(self, **state) -> None:
        self.state.update(state)
        self._write()

    def __enter__(self):
        self._write()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)
        self.state["finished"] = True
        self._write()
        return False
