"""Facts required on page 1 of the report (exam section 4.1), computed, not typed:
download method and date, row counts, development and test periods with failures."""
from __future__ import annotations

import json
from typing import Any

import polars as pl

from ..quality.temporal_split import temporal_split
from ..storage.gold import read_gold
from ..utils.config import Paths
from ..utils.io_utils import md_table, read_json, utc_now, write_json


def run_facts(cfg: dict[str, Any]) -> dict[str, Any]:
    paths: Paths = cfg["_paths"]
    events = [json.loads(l) for l in (paths.bronze / "_manifest.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    done = [e for e in events if e.get("event") == "partition_complete"]
    dates = sorted(e["completed_at"] for e in done)
    silver = read_json(paths.silver / "_silver_build_stats.json")
    gold = read_gold(cfg, lazy=False)
    split = temporal_split(gold, cfg)
    st = {r["split"]: r for r in split.stats}
    dev = pl.concat([split.train, split.valid])
    tgt = [c for c in gold.columns if c.startswith("target_")][0]
    facts = {
        "generated_at": utc_now(),
        "download_method": "FDIC BankFind Suite public API (api.fdic.gov/banks), "
                           "`python pipeline.py ingest` (own resumable client, see README)",
        "download_first_completed": dates[0] if dates else None,
        "download_last_completed": dates[-1] if dates else None,
        "financials_quarters": silver["financials"]["n_quarters"],
        "financials_period": f'{silver["financials"]["min_repdte"]} .. '
                             f'{silver["financials"]["max_repdte"]}',
        "financials_rows": silver["financials"]["rows_silver"],
        "financials_banks": silver["financials"]["n_banks"],
        "failures_rows_raw": silver["failures"]["rows_bronze"],
        "failures_used_RESTYPE_FAILURE": silver["failures"]["rows_silver"],
        "institutions_rows": silver["institutions"]["rows_silver"],
        "development_period": f'{st["train"]["from"]} .. {st["valid"]["to"]}',
        "development_rows": dev.height,
        "development_failed_banks": int(dev.filter(pl.col(tgt) == 1)["cert"].n_unique()),
        "development_positive_rows": int(dev[tgt].sum()),
        "test_period": f'{st["test"]["from"]} .. {st["test"]["to"]}',
        "test_rows": st["test"]["rows"],
        "test_failed_banks": st["test"]["failed_banks"],
        "test_positive_rows": st["test"]["positives"],
        "label": f"{tgt}: fails within (q, q+4 quarters]; mergers are 0",
    }
    write_json(paths.tables / "report_page1_facts.json", facts)
    (paths.tables / "report_page1_facts.md").write_text(
        "# Page-1 facts (computed)\n\n" + md_table([{"item": k, "value": str(v)}
                                                   for k, v in facts.items()]),
        encoding="utf-8")
    return facts
