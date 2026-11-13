# A Leakage-Free Big-Data Pipeline for an FDIC Bank-Failure Early-Warning System

**Course:** BFT65B Big Data Analytics in Finance, National Economics University (NEU)
**Student:** Phạm Anh Chi (11230937) · **Instructor:** Dr. Nguyen Tuan Long
**Report:** `report/BFT65B_final_report.pdf` (link to this repository on page 1)

The repository builds a reproducible pipeline on public FDIC data:
FDIC BankFind API → **Bronze** (raw JSON) → **Silver** (typed, deduplicated Parquet)
→ **Gold** (labelled bank-quarter panel, Delta Lake) → data-quality audit →
CAMELS features (Polars, PySpark demo) → leakage-safe feature selection →
imbalanced early-warning model.

> Scope (exam section 2.2): Financials for all banks 1984Q1-2026Q2 plus the complete
> Failures and Institutions tables. Development period 1984Q1-2007Q4, gap 2008, test
> period 2009Q1-2025Q2 (see `config/config.yaml`, `reports/tables/report_page1_facts.md`).
>
> Status: Parts 1-5 implemented (acquisition, storage + benchmark, data quality and exits,
> Polars/Pandas/PySpark processing with execution plans, leak-free features and selection,
> models with time-aware validation, Spark MLlib demo, cost-based threshold, single test).

## Repository layout

```
config/config.yaml          single source of truth: fields, dates, split, thresholds
pipeline.py                 CLI: probe | ingest | silver | gold | benchmark | audit | leakage |
                            features | bench3 | select | freedman | model | mllib | facts | all
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
  features/processing_benchmark.py  time + peak memory (incl. Spark JVM) + result equality,
                            Polars naive/optimised plans and Spark formatted plan
  selection/funnel.py       variance -> F-test/MI + collinearity -> L1 stability -> permutation
  selection/freedman.py     simulation: feature selection before vs inside CV
  modeling/train_eval.py    baseline vs LR / RF / HGB (+ imbalance strategies), purged
                            expanding-window CV on development, Platt calibration, cost-
                            minimising threshold, single logged test evaluation
  modeling/costs.py         CFN from FDIC failure COST, CFP assumption, cost curve
  modeling/metrics.py       PR-AUC, Recall@top-k% per quarter, confusion at a threshold
  modeling/mllib_demo.py    same logistic regression in Spark MLlib and scikit-learn
  quality/sanity.py         27 logic checks on the real outputs (keys, label, split, features,
                            single test look, report numbers vs data)
  reporting/facts.py        page-1 facts (download date, rows, periods, failures)
tools/make_synthetic_bronze.py  SYNTHETIC data for smoke tests only (never reported)
tests/                      pytest suite (runs offline on synthetic data)
notebooks/BFT65B_full_run_2026-11-07.ipynb  the Colab run behind reports/ (with outputs)
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
python pipeline.py all                                # 4. silver -> gold -> benchmark -> audit -> leakage ->
                                                      #    features -> bench3 -> select -> freedman -> model -> mllib -> facts
python pipeline.py check                              # 5. logic checks on the real outputs -> sanity_checks.md
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

## Exits (failure vs. other exits)

`gold.classify_exits` labels every bank's exit as `failure` (FDIC failures list,
RESTYPE = FAILURE), `merger_or_other_exit` (inactive, not failed; CHANGEC1 code kept),
`active`, or `stopped_unexplained`. Open-bank assistance transactions are excluded from the
label. Only failures create positive labels; mergers are 0. Tables: `gold_exit_types.csv`,
`gold_exit_changec1_codes.csv`, `gold_failure_report_gap.csv`.

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
check catches a deliberately leaky feature). The four-engine benchmark (`features/engines.py`) deliberately
uses row-based windows, the only window type with identical semantics in all four APIs; it
proves the engines agree on the same workload, not that they reproduce the production features,
whose gap-aware windows are tested separately (`test_rolling_window_is_gap_aware_and_ignores_nulls`).

## Module 5: models, threshold and the single test

`python pipeline.py model` compares a baseline (logistic regression on four classic ratios)
with logistic regression, random forest and gradient boosting (no re-weighting, class
weights, undersampling) by expanding-window, purged cross-validation inside 1984-2007
(`model_cv_summary.csv`). The chosen model is fitted on 1984-1992, Platt-calibrated on
1993-2007, and its alarm threshold minimises the expected cost on 1993-2007, with the cost of
a missed failure estimated from the FDIC `COST` field of development-period failures
(`model_costs.json`, sensitivity in `model_cost_sensitivity.csv`). The choice is frozen in
`model_choice.json` before the test window 2009Q1-2025Q2 is scored; every test evaluation is
appended to `data/gold/_test_evaluations.jsonl`. Results: `reports/module5_models.md`.
`python pipeline.py predict [--quarter 2026Q2]` scores one quarter (default: the latest) with
the frozen model in `data/models/ews_model.joblib`, without refitting.
`python pipeline.py mllib` fits the same logistic regression in Spark MLlib and
scikit-learn on development data and reports runtime and agreement.

## Data source

Federal Deposit Insurance Corporation, BankFind Suite API, <https://api.fdic.gov/banks/docs/>
(endpoints `financials`, `institutions`, `failures`). Amounts are in thousands of USD;
income-statement items are year-to-date.

## References and acknowledgements

Sources used in the code and in the report (APA 7; the same list as the report's reference list).

- Anthropic. (2026). *Claude* [Large language model]. https://claude.ai
- Armbrust, M., Das, T., Sun, L., Yavuz, B., Zhu, S., Murthy, M., Torres, J., van Hovell, H., Ionescu, A., Łuszczak, A., Świtakowski, M., Szafrański, M., Li, X., Ueshin, T., Mokhtar, M., Boncz, P., Ghodsi, A., Paranjpye, S., Senster, P., … Zaharia, M. (2020). Delta Lake: High-performance ACID table storage over cloud object stores. *Proceedings of the VLDB Endowment, 13*(12), 3411-3424. https://vldb.org/pvldb/vol13/p3411-armbrust.pdf
- Board of Governors of the Federal Reserve System. (2023). *Review of the Federal Reserve's supervision and regulation of Silicon Valley Bank*. https://www.federalreserve.gov/publications/files/svb-review-20230428.pdf
- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5-32. https://doi.org/10.1023/A:1010933404324
- Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). LOF: Identifying density-based local outliers. In *Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data* (pp. 93-104). ACM. https://doi.org/10.1145/342009.335388
- Cole, R. A., & White, L. J. (2012). Déjà vu all over again: The causes of U.S. commercial bank failures this time around. *Journal of Financial Services Research, 42*(1-2), 5-29. https://doi.org/10.1007/s10693-011-0116-9
- Elkan, C. (2001). The foundations of cost-sensitive learning. In *Proceedings of the Seventeenth International Joint Conference on Artificial Intelligence* (pp. 973-978). Morgan Kaufmann. https://mlanthology.org/ijcai/2001/elkan2001ijcai-foundations
- Federal Deposit Insurance Corporation. (n.d.). *BankFind Suite API documentation*. Retrieved November 7, 2026, from https://api.fdic.gov/banks/docs/
- Freedman, D. A. (1983). A note on screening regression equations. *The American Statistician, 37*(2), 152-155. https://doi.org/10.1080/00031305.1983.10482729
- Kuhn, M., & Johnson, K. (2019). *Feature engineering and selection: A practical approach for predictive models*. CRC Press. https://doi.org/10.1201/9781315108230
- López de Prado, M. (2018). *Advances in financial machine learning*. Wiley. https://www.oreilly.com/library/view/-/9781119482086/
- Martin, D. (1977). Early warning of bank failure: A logit regression approach. *Journal of Banking & Finance, 1*(3), 249-276. https://doi.org/10.1016/0378-4266(77)90022-X
- Meinshausen, N., & Bühlmann, P. (2010). Stability selection. *Journal of the Royal Statistical Society Series B: Statistical Methodology, 72*(4), 417-473. https://doi.org/10.1111/j.1467-9868.2010.00740.x
- National Assembly of Vietnam. (2024). *Law on Credit Institutions* (Law No. 32/2024/QH15). https://english.luatvietnam.vn/tai-chinh/law-on-credit-institution-no-32-2024-qh15-296639-d1.html
- Office of the Comptroller of the Currency, Board of Governors of the Federal Reserve System, & Federal Deposit Insurance Corporation. (2019). Regulatory capital rule: Capital simplification for qualifying community banking organizations. *Federal Register, 84*, 61776-61804. https://www.federalregister.gov/documents/2019/11/13/2019-23472/regulatory-capital-rule-capital-simplification-for-qualifying-community-banking-organizations
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*(85), 2825-2830. https://jmlr.org/papers/v12/pedregosa11a.html
- Platt, J. C. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In A. J. Smola, P. Bartlett, B. Schölkopf, & D. Schuurmans (Eds.), *Advances in large margin classifiers* (pp. 61-74). MIT Press.
- Polars Developers. (2025). *Polars* (Version 1.35.2) [Computer software]. https://pypi.org/project/polars/1.35.2/
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), Article e0118432. https://doi.org/10.1371/journal.pone.0118432
- State Bank of Vietnam. (2016). *Circular No. 41/2016/TT-NHNN prescribing capital adequacy ratios for banks and foreign bank branches*. https://thuvienphapluat.vn/van-ban/Tien-te-Ngan-hang/Thong-tu-41-2016-TT-NHNN-ty-le-an-toan-ngan-hang-chi-nhanh-ngan-hang-nuoc-ngoai-336513.aspx
- The Investor. (2022, October). Central bank put Saigon Commercial Bank under special control. https://theinvestor.vn/central-bank-put-saigon-commercial-bank-under-special-control-d2163.html
- Tukey, J. W. (1977). *Exploratory data analysis*. Addison-Wesley.
- Vietstock. (2015, October). Over $500mn in liquidity reserved for three lenders acquired by Vietnam cbank. https://en.vietstock.vn/2015/10/over-$500mn-in-liquidity-reserved-for-three-lenders-acquired-by-vietnam-cbank-37-218944.htm
- VnEconomy. (2024, October 18). Two weak banks transferred to Vietcombank, MB Bank. https://en.vneconomy.vn/two-weak-banks-transfered-to-vietcombank-mb-bank.htm
- VnEconomy. (2025, January 18). GPBank and DongA Bank transferred to VPBank and HDBank. https://en.vneconomy.vn/gpbank-and-donga-bank-transferred-to-vpbank-and-hdbank.htm
- Zaharia, M., Xin, R. S., Wendell, P., Das, T., Armbrust, M., Dave, A., Meng, X., Rosen, J., Venkataraman, S., Franklin, M. J., Ghodsi, A., Gonzalez, J., Shenker, S., & Stoica, I. (2016). Apache Spark: A unified engine for big data processing. *Communications of the ACM, 59*(11), 56-65. https://doi.org/10.1145/2934664

Software: Polars, pandas, PyArrow, delta-rs (deltalake), PySpark, scikit-learn and matplotlib, at the pinned versions in `requirements.txt`.

**AI assistance:** parts of this code and of the report text were drafted with Claude (Anthropic), then reviewed,
run and modified by the author. All reported numbers come from the author's own runs.
