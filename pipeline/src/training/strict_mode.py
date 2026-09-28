from __future__ import annotations

import os


class StrictModeViolation(RuntimeError):
    """Raised when strict model mode is enabled and a scientific guardrail is tripped."""


def is_strict_mode() -> bool:
    return os.environ.get("STRICT_MODEL_MODE", "0").strip().lower() in {"1", "true", "yes"}
