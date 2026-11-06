# A Leakage-Free Big-Data Pipeline for an FDIC Bank-Failure Early-Warning System

**Course:** BFT65B Big Data Analytics in Finance, National Economics University (NEU)
**Student:** Phạm Anh Chi (11230937) · **Instructor:** Dr. Nguyen Tuan Long
**Report:** `report/BFT65B_final_report.pdf` (link to this repository on page 1)

The repository builds a reproducible pipeline on public FDIC data:
FDIC BankFind API → **Bronze** (raw JSON) → **Silver** (typed, deduplicated Parquet)
→ **Gold** (labelled bank-quarter panel, Delta Lake) → data-quality audit →
CAMELS features (Polars, PySpark demo) → leakage-safe feature selection →
imbalanced early-warning model.

> Status: Modules 1-3 implemented (ingestion, storage + benchmark, data quality,
> anti-leakage preprocessing, CAMELS features, Pandas/Polars/PySpark benchmark).
> Modules 4-6 follow `docs/ROADMAP.md`. Measured results are in `reports/`.

## Repository layout

```
config/config.yaml          single source of truth: fields, dates, split, thresholds
pipeline.py                 CLI: probe | ingest | silver | gold | benchmark | audit | leakage | all
src/fdic_ews/
  ingestion/fdic_client.py  HTTP client: retries, backoff, pagination, completeness check
  ingestion/fdic_ingestion.py  Bronze writer + manifest (lineage, resumable)
  storage/silver.py         schema enforcement, cast-failure counts, de-duplication
  storage/gold.py           point-in-time failure label, censoring flag, LEAKY_COLUMNS
  storage/storage_benchmark.py  CSV vs Parquet (snappy/zstd) vs Delta, subprocess-isolated
  quality/temporal_split.py purged temporal train/valid/test split
  quality/preprocessing.py  sklearn transformers fitted on train only
  quality/audit.py          9 data-quality checks -> tables + figures
  quality/leakage_checks.py empirical proof that preprocessing never sees test data
  features/camels.py        50 CAMELS + dynamic features, Polars Lazy, look-ahead check
  features/engines.py       one workload in Pandas / Polars eager / Polars lazy / PySpark
  features/processing_benchmark.py  time + peak memory (incl. Spark JVM) + result equality
tools/make_synthetic_bronze.py  SYNTHETIC data for smoke tests only (never reported)
tests/                      pytest suite (runs offline on synthetic data)
notebooks/colab_modules_1_2.ipynb  run everything on Google Colab
reports/tables, reports/figures   generated evidence used in the report
docs/ROADMAP.md             plan for all six parts and the report
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python pipeline.py probe                              # 1. are all configured fields served by the API?
python pipeline.py ingest --start 2023Q1 --end 2023Q4 # 2. trial download (4 quarters)
python pipeline.py ingest                             # 3. full range in config.yaml (~1 h, resumable)
python pipeline.py all                                # 4. silver -> gold -> benchmark -> audit -> leakage -> features -> bench3
pytest -q                                             # offline tests on synthetic data
```

`ingest` is idempotent: completed quarters are recorded in `data/bronze/_manifest.jsonl`
and skipped on the next run, so an interrupted download simply resumes. Use `--force`
to re-download.

## Medallion layers

| Layer | Path | Content | Format |
|---|---|---|---|
| Bronze | `data/bronze/financials/REPDTE=YYYYMMDD/page_*.json.gz` | raw API responses, unchanged | gzip JSON + manifest (SHA-256, params, row counts) |
| Silver | `data/silver/{financials,institutions,failures}.parquet` | typed, lower-case names, duplicates resolved, sorted by (repdte, cert) | Parquet, zstd, row-group statistics |
| Gold | `data/gold/bank_quarter_panel/` | one row per bank-quarter + `target_fail_4q`, `available_date`, `label_observable` | Delta Lake, partitioned by year |

Silver deliberately does **not** impute or clip: those steps learn parameters and are
fitted later on the training window only (`quality/preprocessing.py`).

## Leakage controls implemented so far

1. Label `target_fail_4q = 1` iff the bank fails in (repdte, repdte + 4 quarters].
2. Purged temporal split: rows whose label window crosses a split boundary are dropped.
3. `available_date = repdte + 60 days` (Call Reports are not public on the report date).
4. Right-censoring flag for labels not yet observable at download time.
5. `LEAKY_COLUMNS` (failure date, `active`, institution end date, …) can never be features.
6. All learned preprocessing (near-constant filter, IQR fences, medians, scales) is inside a
   scikit-learn `ColumnTransformer` fitted on train; `pipeline.py leakage` proves it numerically.

## Module 3: features

`data/gold/features.parquet` holds 50 features per bank-quarter (dictionary with formulas and
source fields: `reports/tables/feature_dictionary.csv`). Year-to-date income items are
de-cumulated to quarterly flows (null across reporting gaps) and annualised. Rolling statistics
use a time-based window `(t-12 months, t]`, so they never look forward and never silently
stretch across a missing quarter. `features_lookahead_check.csv` shows that recomputing all
features on data truncated at the train cut-off changes nothing (and a unit test proves the
check catches a deliberately leaky feature).

## Data source

Federal Deposit Insurance Corporation, BankFind Suite API, <https://api.fdic.gov/banks/docs/>
(endpoints `financials`, `institutions`, `failures`). Amounts are in thousands of USD;
income-statement items are year-to-date.

## References and acknowledgements

- FDIC (2026). *BankFind Suite API documentation*. https://api.fdic.gov/banks/docs/
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley (purging, embargo).
- Kuhn, M., & Johnson, K. (2019). *Feature Engineering and Selection*. CRC Press.
- Breunig, M. et al. (2000). LOF: identifying density-based local outliers. *SIGMOD*.
- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley (IQR fences).
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR* 12.
- Polars, PyArrow, delta-rs (deltalake) open-source documentation.
- **AI assistance:** parts of this code were drafted with Claude (Anthropic), then reviewed,
  run and modified by the author. All reported numbers come from the author's own runs.
