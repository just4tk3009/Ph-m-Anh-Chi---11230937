"""Consistent logging to console and to ``logs/pipeline.log``."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from .config import REPO_ROOT

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        log_dir = REPO_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                                "%Y-%m-%d %H:%M:%S")
        root = logging.getLogger("fdic_ews")
        root.setLevel(logging.INFO)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        fh = logging.FileHandler(Path(log_dir) / "pipeline.log", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(sh)
        root.addHandler(fh)
        _CONFIGURED = True
    return logging.getLogger(f"fdic_ews.{name}")
