from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(run_root: Path, debug: bool = False) -> logging.Logger:
    logger = logging.getLogger("forecasting_pipeline")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for path, level in [(run_root / "logs" / "train.log", logging.DEBUG), (run_root / "logs" / "errors.log", logging.ERROR)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.addHandler(sh)
    return logger

