"""Command-line entry point for the BFT65B FDIC early-warning pipeline (Modules 1-2).

Examples
--------
python pipeline.py probe                                # check field names against the live API
python pipeline.py ingest --start 2023Q1 --end 2023Q4   # trial download (a few quarters)
python pipeline.py ingest                               # full range from config.yaml (~1 h)
python pipeline.py silver                               # Bronze -> Silver (Parquet)
python pipeline.py gold                                 # Silver -> Gold (Delta, labelled panel)
python pipeline.py benchmark                            # CSV vs Parquet vs Delta proof
python pipeline.py audit                                # data-quality audit tables/figures
python pipeline.py leakage                              # anti-leakage verification
python pipeline.py features                             # Module 3: CAMELS features (Polars lazy)
python pipeline.py bench3                               # Module 3: Pandas vs Polars vs PySpark
python pipeline.py select                               # Module 4: feature-selection funnel
python pipeline.py freedman                             # Module 4: Freedman simulation
python pipeline.py facts                                # page-1 facts: rows, periods, failures
python pipeline.py all                                  # silver -> gold -> benchmark -> audit -> leakage -> features -> bench3

Add ``--data-dir data_synthetic --reports-dir reports_synthetic`` to run on the
synthetic smoke-test data from ``tools/make_synthetic_bronze.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fdic_ews.utils.config import load_config  # noqa: E402
from fdic_ews.utils.logging_utils import get_logger  # noqa: E402

log = get_logger("cli")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["probe", "ingest", "silver", "gold", "benchmark",
                                      "audit", "leakage", "features", "bench3", "select", "freedman", "facts", "all"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--reports-dir", default=None)
    ap.add_argument("--start", default=None, help="first quarter, e.g. 2001Q1")
    ap.add_argument("--end", default=None, help="last quarter, e.g. 2025Q2")
    ap.add_argument("--force", action="store_true", help="re-download completed partitions")
    ap.add_argument("--skip-reference", action="store_true",
                    help="ingest: do not (re)download institutions and failures")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.data_dir, a.reports_dir)

    t0 = time.perf_counter()
    if a.stage == "probe":
        from fdic_ews.ingestion.fdic_ingestion import probe
        print(json.dumps(probe(cfg), indent=2))
    elif a.stage == "ingest":
        from fdic_ews.ingestion.fdic_ingestion import ingest_financials, ingest_reference
        if not a.skip_reference:
            ingest_reference(cfg, force=a.force)
        counts = ingest_financials(cfg, a.start, a.end, force=a.force)
        log.info("downloaded %d quarters, %d rows", len(counts), sum(counts.values()))
    stages = (["silver", "gold", "benchmark", "audit", "leakage", "features", "bench3",
               "select", "freedman", "facts"]
              if a.stage == "all" else [a.stage])
    for st in stages:
        if st == "silver":
            from fdic_ews.storage.silver import build_silver
            build_silver(cfg)
        elif st == "gold":
            from fdic_ews.storage.gold import build_gold
            build_gold(cfg)
        elif st == "benchmark":
            from fdic_ews.storage.storage_benchmark import run_storage_benchmark
            print(run_storage_benchmark(cfg))
        elif st == "audit":
            from fdic_ews.quality.audit import run_audit
            print(json.dumps(run_audit(cfg), indent=2, default=str))
        elif st == "features":
            from fdic_ews.features.camels import run_features
            print(json.dumps(run_features(cfg), indent=2, default=str))
        elif st == "bench3":
            from fdic_ews.features.processing_benchmark import run_processing_benchmark
            print(run_processing_benchmark(cfg))
        elif st == "select":
            from fdic_ews.selection.funnel import run_selection
            print(json.dumps(run_selection(cfg), indent=2, default=str))
        elif st == "freedman":
            from fdic_ews.selection.freedman import run_freedman
            print(run_freedman(cfg))
        elif st == "facts":
            from fdic_ews.reporting.facts import run_facts
            print(json.dumps(run_facts(cfg), indent=2, default=str))
        elif st == "leakage":
            from fdic_ews.quality.leakage_checks import run_leakage_checks
            print(json.dumps(run_leakage_checks(cfg), indent=2, default=str))
    log.info("stage %s finished in %.1fs", a.stage, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
