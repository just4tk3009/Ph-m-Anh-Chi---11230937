"""SYNTHETIC smoke-test data. NEVER use its numbers in the report.

Writes a small Bronze layer in exactly the FDIC API envelope format
(``{"meta": {"total": n}, "data": [{"data": {...}}]}``) so the whole pipeline
(silver -> gold -> benchmark -> audit -> leakage checks) can be tested
without network access, e.g. in CI or before the real download finishes.

Usage:  python tools/make_synthetic_bronze.py --data-dir data_synthetic
"""
from __future__ import annotations

import argparse
import gzip
import json
from datetime import date
from pathlib import Path

import numpy as np

QEND = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}


def quarters(y0: int, y1: int) -> list[date]:
    return [date(y, *QEND[q]) for y in range(y0, y1 + 1) for q in range(1, 5)]


def dump(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": {"total": len(rows)}, "data": [{"data": r, "score": 0} for r in rows]}
    with gzip.open(path, "wb") as fh:
        fh.write(json.dumps(payload).encode())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_synthetic")
    ap.add_argument("--banks", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    bronze = Path(a.data_dir) / "bronze"
    qs = quarters(2001, 2024)
    certs = np.arange(10001, 10001 + a.banks)
    fail_q = {}
    for c in rng.choice(certs, size=max(4, a.banks // 15), replace=False):
        qi = int(rng.integers(20, len(qs) - 2))
        fail_q[int(c)] = qi
    manifest = []
    for qi, q in enumerate(qs):
        rows = []
        for c in certs:
            c = int(c)
            if c in fail_q and qi >= fail_q[c]:
                continue
            stress = 0.0
            if c in fail_q and fail_q[c] - qi <= 6:
                stress = (7 - (fail_q[c] - qi)) / 7
            asset = float(np.exp(rng.normal(12, 1.4)))
            loans = asset * rng.uniform(0.45, 0.8)
            eq = asset * max(0.002, rng.normal(0.10, 0.02) - 0.08 * stress)
            npl = loans * max(0.0, rng.lognormal(-4.2, 0.9) + 0.08 * stress)
            r = {"CERT": c, "REPDTE": q.strftime("%Y%m%d"), "ASSET": round(asset),
                 "DEP": round(asset * rng.uniform(0.7, 0.9)), "DEPDOM": round(asset * 0.8),
                 "EQ": round(eq), "LNLSGR": round(loans), "LNLSNET": round(loans * 0.985),
                 "LNATRES": round(loans * 0.015), "NCLNLS": round(npl),
                 "P3ASSET": round(loans * rng.uniform(0, 0.02)),
                 "P9ASSET": round(loans * rng.uniform(0, 0.01)), "NAASSET": round(npl * 0.7),
                 "ORE": round(asset * rng.uniform(0, 0.01) * (1 + 5 * stress)),
                 "NTLNLS": round(loans * 0.002), "ELNATR": round(loans * 0.003),
                 "NETINC": round(asset * (0.003 - 0.02 * stress) * q.month / 12),
                 "INTINC": round(asset * 0.012 * q.month / 3),
                 "EINTEXP": round(asset * 0.004 * q.month / 3),
                 "NONII": round(asset * 0.002), "NONIX": round(asset * 0.007),
                 "SC": round(asset * rng.uniform(0.05, 0.35)),
                 "CHBAL": round(asset * rng.uniform(0.02, 0.1)),
                 "BRO": round(asset * rng.uniform(0, 0.05) * (1 + 3 * stress)),
                 "RBC1AAJ": round(100 * eq / asset, 3), "RBCRWAJ": round(150 * eq / asset, 3),
                 "ROA": round(100 * (0.012 - 0.04 * stress), 3), "ROE": 9.5,
                 "NIMY": round(rng.normal(3.6, 0.5), 3)}
            if rng.random() < 0.03:
                r["BRO"] = None                       # random missing
            if q.year < 2005:
                r["P3ASSET"] = None                   # structural missing (form change)
            if stress > 0 and rng.random() < 0.3:
                r["ORE"] = None                       # informative missing
            if rng.random() < 0.001:
                r["ASSET"] = r["ASSET"] * 1000        # unit error outlier
            rows.append(r)
        part = f"financials/REPDTE={q.strftime('%Y%m%d')}"
        dump(bronze / part / "page_0000.json.gz", rows)
        if qi == 30:                                 # duplicate page (re-fetch) -> duplicates
            dump(bronze / part / "page_0001.json.gz", rows[:25])
        manifest.append({"event": "partition_complete", "partition": part, "n_rows": len(rows),
                         "completed_at": "2026-11-06T00:00:00Z"})
    inst = [{"CERT": int(c), "NAME": f"Synthetic Bank {c}", "CITY": "Testville",
             "STALP": str(rng.choice(["GA", "FL", "IL", "CA", "TX"])), "BKCLASS": "NM",
             "REGAGNT": "FDIC", "ESTYMD": "01/01/1990", "ACTIVE": 0 if int(c) in fail_q else 1,
             "ENDEFYMD": "12/31/9999"} for c in certs]
    dump(bronze / "institutions" / "page_0000.json.gz", inst)
    fails = [{"CERT": c, "NAME": f"Synthetic Bank {c}", "CITYST": "TESTVILLE, GA",
              "FAILDATE": f"{qs[qi].month - 1}/15/{qs[qi].year}", "FAILYR": qs[qi].year,
              "RESTYPE": "FAILURE", "RESTYPE1": "PA", "SAVR": "DIF", "QBFASSET": 1000.0,
              "QBFDEP": 900.0, "COST": 100.0} for c, qi in fail_q.items()]
    dump(bronze / "failures" / "page_0000.json.gz", fails)
    for p in ("institutions", "failures"):
        manifest.append({"event": "partition_complete", "partition": p,
                         "completed_at": "2026-11-06T00:00:00Z"})
    (bronze / "_manifest.jsonl").write_text("\n".join(json.dumps(m) for m in manifest) + "\n")
    (Path(a.data_dir) / "SYNTHETIC_DATA_DO_NOT_REPORT.txt").write_text(
        "This folder contains synthetic data for smoke tests only.\n")
    print(f"synthetic bronze written to {bronze} ({len(qs)} quarters, {a.banks} banks, "
          f"{len(fail_q)} failures)")


if __name__ == "__main__":
    main()
