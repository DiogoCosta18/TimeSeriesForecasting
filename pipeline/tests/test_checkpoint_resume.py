from pathlib import Path

from src.training.checkpointing import CheckpointManager


def test_checkpoint_resume_skips_completed_and_writes_status(tmp_path: Path):
    cp = CheckpointManager(tmp_path, "run_test")
    task = {"task_id": "a|b", "feature_name": "a"}
    cp.mark_done(task)
    assert cp.is_done("a|b")
    cp.status({"stage": "test", **task}, total_tasks=1)
    assert (tmp_path / "STATUS.json").exists()


def test_failed_task_recorded_without_raising(tmp_path: Path):
    cp = CheckpointManager(tmp_path, "run_test")
    cp.mark_failed({"task_id": "x"}, RuntimeError("boom"))
    assert len(cp.failed) == 1

