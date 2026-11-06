"""Small I/O helpers: hashing, JSON manifests, quarter calendars, folder sizes."""
from __future__ import annotations

import gzip
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)  # atomic rename: no half-written manifests


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_gzip_json(path: Path, payload: bytes) -> None:
    """Store the raw API bytes unchanged (gzip-compressed) = immutable Bronze record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=6) as fh:
        fh.write(payload)
    tmp.replace(path)


def read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rb") as fh:
        return json.loads(fh.read())


def quarter_ends(start: str, end: str) -> list[date]:
    """'2001Q1', '2001Q4' -> [2001-03-31, 2001-06-30, 2001-09-30, 2001-12-31]."""
    def parse(q: str) -> tuple[int, int]:
        y, qq = q.upper().split("Q")
        return int(y), int(qq)

    ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
    (y, q), (y1, q1) = parse(start), parse(end)
    out: list[date] = []
    while (y, q) <= (y1, q1):
        m, d = ends[q]
        out.append(date(y, m, d))
        q += 1
        if q == 5:
            y, q = y + 1, 1
    return out


def folder_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def md_table(rows: Iterable[dict[str, Any]], floatfmt: str = "{:,.3f}") -> str:
    """Render a list of dicts as a GitHub-markdown table (for reports/*.md)."""
    rows = list(rows)
    if not rows:
        return "_(no rows)_\n"
    cols = list(rows[0].keys())

    def fmt(v: Any) -> str:
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            return floatfmt.format(v)
        if isinstance(v, int):
            return f"{v:,}"
        return "" if v is None else str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"
