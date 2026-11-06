"""Configuration loading and path resolution.

All modules call :func:`load_config` so that a single YAML file
(``config/config.yaml``) fully determines a pipeline run.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class Paths:
    """Resolved folder layout of the Medallion lakehouse and the report outputs."""

    data: Path
    reports: Path

    @property
    def bronze(self) -> Path:
        return self.data / "bronze"

    @property
    def silver(self) -> Path:
        return self.data / "silver"

    @property
    def gold(self) -> Path:
        return self.data / "gold"

    @property
    def tables(self) -> Path:
        return self.reports / "tables"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    def ensure(self) -> "Paths":
        for p in (self.bronze, self.silver, self.gold, self.tables, self.figures):
            p.mkdir(parents=True, exist_ok=True)
        return self


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_config(path: str | Path | None = None, data_dir: str | None = None,
                reports_dir: str | None = None) -> dict[str, Any]:
    """Load the YAML config and attach a resolved :class:`Paths` under ``cfg['_paths']``.

    ``data_dir`` / ``reports_dir`` override the YAML values (used by the
    synthetic smoke test so it never touches the real data folder).
    """
    cfg_path = _resolve(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    paths = Paths(
        data=_resolve(data_dir or cfg["paths"]["data_dir"]),
        reports=_resolve(reports_dir or cfg["paths"]["reports_dir"]),
    ).ensure()
    cfg["_paths"] = paths
    cfg["_config_file"] = str(cfg_path)
    return cfg


def to_date(value: Any) -> date:
    """Accept a ``date`` (YAML parses ISO dates natively) or an ISO string."""
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))
