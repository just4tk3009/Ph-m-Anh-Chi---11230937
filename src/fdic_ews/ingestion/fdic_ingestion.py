"""Module 1 / Part 1: reproducible ingestion of FDIC data into the Bronze layer.

Bronze layout (immutable, append-only, raw API bytes gzip-compressed)::

    data/bronze/
      financials/REPDTE=20231231/page_0000.json.gz   # one folder per quarter
      institutions/page_0000.json.gz
      failures/page_0000.json.gz
      _manifest.jsonl                                # one line per page written

The manifest stores URL parameters, row counts, the API-reported total, a
SHA-256 of the raw bytes and a timestamp, so every Silver row can be traced
back to the exact request that produced it (lineage), and a rerun can skip
quarters that are already complete (idempotent / resumable ingestion).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..utils.config import Paths
from ..utils.io_utils import quarter_ends, utc_now, write_gzip_json
from ..utils.logging_utils import get_logger
from .fdic_client import FDICClient

log = get_logger("ingestion")


def _manifest_path(paths: Paths) -> Path:
    return paths.bronze / "_manifest.jsonl"


def _append_manifest(paths: Paths, entry: dict[str, Any]) -> None:
    with open(_manifest_path(paths), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def _completed_partitions(paths: Paths) -> set[str]:
    """Partitions whose download finished (marker written after the last page)."""
    mp = _manifest_path(paths)
    if not mp.exists():
        return set()
    done = set()
    for line in mp.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e.get("event") == "partition_complete":
            done.add(e["partition"])
    return done


def _download_partition(client: FDICClient, paths: Paths, endpoint: str, partition: str,
                        out_dir: Path, filters: str | None, fields: list[str] | None,
                        sort_by: str) -> int:
    t0 = time.perf_counter()
    n_total = 0
    for i, page in enumerate(client.iter_pages(endpoint, filters, fields, sort_by)):
        f = out_dir / f"page_{i:04d}.json.gz"
        write_gzip_json(f, page.raw)
        n_total += page.n_rows
        _append_manifest(paths, {
            "event": "page", "partition": partition, "endpoint": endpoint,
            "file": str(f.relative_to(paths.bronze)), "params": page.params,
            "n_rows": page.n_rows, "api_total": page.total, "bytes": len(page.raw),
            "sha256": page.sha256, "fetched_at": page.fetched_at,
        })
    _append_manifest(paths, {"event": "partition_complete", "partition": partition,
                             "endpoint": endpoint, "n_rows": n_total,
                             "seconds": round(time.perf_counter() - t0, 2),
                             "completed_at": utc_now()})
    return n_total


def ingest_financials(cfg: dict[str, Any], start: str | None = None, end: str | None = None,
                      force: bool = False) -> dict[str, int]:
    """Download quarterly Call Report financials, one Bronze partition per quarter."""
    paths: Paths = cfg["_paths"]
    ing = cfg["ingestion"]
    client = FDICClient.from_config(cfg)
    done = set() if force else _completed_partitions(paths)
    counts: dict[str, int] = {}
    quarters = quarter_ends(start or ing["start_quarter"], end or ing["end_quarter"])
    log.info("Financials: %d quarters requested (%s .. %s)", len(quarters), quarters[0], quarters[-1])
    for q in quarters:
        rep = q.strftime("%Y%m%d")
        partition = f"financials/REPDTE={rep}"
        if partition in done:
            log.info("skip %s (already complete in manifest)", partition)
            continue
        out_dir = paths.bronze / partition
        if force and out_dir.exists():
            for old in out_dir.glob("*.json.gz"):
                old.unlink()
        counts[rep] = _download_partition(client, paths, "financials", partition, out_dir,
                                          filters=f"REPDTE:{rep}",
                                          fields=ing["financial_fields"], sort_by="CERT")
    return counts


def ingest_reference(cfg: dict[str, Any], force: bool = False) -> dict[str, int]:
    """Download the institutions table and the failed-bank list (small, full refresh)."""
    paths: Paths = cfg["_paths"]
    ing = cfg["ingestion"]
    client = FDICClient.from_config(cfg)
    done = set() if force else _completed_partitions(paths)
    out: dict[str, int] = {}
    for endpoint, fields, sort_by in (("institutions", ing["institution_fields"], "CERT"),
                                      ("failures", ing["failure_fields"], "FAILDATE")):
        if endpoint in done:
            log.info("skip %s (already complete in manifest)", endpoint)
            continue
        out_dir = paths.bronze / endpoint
        for old in out_dir.glob("*.json.gz"):
            old.unlink()
        out[endpoint] = _download_partition(client, paths, endpoint, endpoint, out_dir,
                                            filters=None, fields=fields, sort_by=sort_by)
    return out


def probe(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Check that every configured field exists in the live API (run before a full download)."""
    client = FDICClient.from_config(cfg)
    ing = cfg["ingestion"]
    report = {}
    for endpoint, wanted, flt in (("financials", ing["financial_fields"], "REPDTE:20231231"),
                                  ("institutions", ing["institution_fields"], None),
                                  ("failures", ing["failure_fields"], None)):
        available = set(client.probe_fields(endpoint, flt))
        missing = [f for f in wanted if f not in available]
        report[endpoint] = {"missing": missing, "n_available": len(available)}
        if missing:
            log.warning("%s: configured fields NOT returned by the API: %s", endpoint, missing)
        else:
            log.info("%s: all %d configured fields available", endpoint, len(wanted))
    return report
