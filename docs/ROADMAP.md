# Project roadmap (6 Nov - 26 Nov 2026, deadline 23:00 on 26/11)

Rule for the whole project: **every number in the report comes from a file in
`reports/tables/` produced by a script in this repository.** Numbers in the early
outline (for example "$42.8M saved", "2.4% failure spike", "35% cost reduction",
"8x faster", ">75% smaller", "150 -> 20 features", "99.2% accuracy") were
placeholders, not results, and must not appear unless your own run produces them.

## Timeline

| Dates | Part | Work | Evidence produced |
|---|---|---|---|
| 6-8 Nov | 1 | `probe`, trial ingest of 4 quarters, then full ingest 2001Q1-2025Q2 | `bronze/_manifest.jsonl`, rows per quarter |
| 8-9 Nov | 1 | `silver`, `gold`, `benchmark` | `storage_benchmark.md/.csv/.png`, `gold_label_counts_by_year.csv` |
| 9-11 Nov | 2 | `audit`, `leakage`; read every table and decide | `module2_data_quality.md`, `module2_leakage_checks.md` |
| 11-15 Nov (done 6 Nov) | 3 | CAMELS ratios in Polars Lazy; de-cumulate YTD income; QoQ, 4q rolling mean/std; PySpark window demo; Pandas vs Polars vs PySpark benchmark | `processing_benchmark.csv`, feature dictionary |
| 15-18 Nov | 4 | Selection funnel fitted inside CV folds; Freedman simulation (select-before-split vs after) | funnel table, simulation table |
| 18-21 Nov | 5 | Purged walk-forward CV; LR / RF / GBM; class weights vs undersampling vs SMOTE (train fold only); PR-AUC, Recall@top-k; cost-based threshold | metric tables, PR curves, cost curve |
| 21-22 Nov | 6 | Serialize the fitted pipeline (`joblib`), `predict.py` that scores a new quarter, README final | `models/`, end-to-end rerun log |
| 22-25 Nov | - | Write the report (<= 20 pages + <= 10 appendix), figures, citations incl. AI use | PDF |
| 25-26 Nov | - | Fresh clone -> run -> check numbers match report; make repo public; final commit before 23:00 | |

## Decisions that must be made from measured data

1. **Split dates.** Bank failures cluster in 2008-2012; after 2015 there are only a
   handful per year. Open `reports/tables/gold_label_counts_by_year.csv` before fixing
   `split:` in `config.yaml`. A test window with ~15 positive banks gives very noisy
   PR-AUC. Options: (a) train 2001-2012 / valid 2013-2015 / test 2016-2024 (default);
   (b) rolling-origin evaluation (train up to year Y, test Y+1, repeat) and report the
   mean and spread. Report positives per split in the paper either way.
2. **Label horizon.** 4 quarters by default (`gold.label_horizon_quarters`). A 2-quarter
   horizon is easier but less useful for supervisors; justify the choice.
3. **Missing-value strategy.** Use `dq_missingness.csv`: structural gaps (item not on the
   form in some years) are handled by feature choice, informative gaps keep a missing flag.
4. **Outliers.** Compare `train_iqr_flag_pct` with `train_gauss_flag_pct`, and the failure
   rate among flagged rows. If flagged rows fail more often, clip, do not delete.

## Technical corrections to keep in mind for Modules 3-5

* Call Report income items (NETINC, INTINC, EINTEXP, NONII, NONIX, ELNATR, NTLNLS) are
  **year-to-date**. Quarterly flow = YTD(q) - YTD(q-1) within the same year; annualise
  before dividing by assets.
* A 4-quarter rolling window in PySpark is `rowsBetween(-3, 0)` (current + 3 previous),
  not `(-4, 0)`, which is 5 rows. Row-based windows are wrong when a bank skips a quarter;
  use `rangeBetween` on an integer quarter index (year*4 + quarter) or Polars
  `rolling(index_column=..., period="12mo")`.
* "Non-interest expense / total assets" is an efficiency proxy, not a cost-to-income ratio
  (that is non-interest expense / (net interest income + non-interest income)).
* `t* = CFP / (CFP + CFN)` is the Bayes-optimal threshold only for **calibrated**
  probabilities. Class weights and SMOTE distort probabilities, so either calibrate on the
  validation window or choose the threshold by minimising expected cost on validation,
  then report the cost on test. CFN/CFP values are assumptions: run a sensitivity table.
* SMOTE, scaling, imputation and feature selection go inside the CV fold
  (`imblearn.pipeline.Pipeline`), never before the split.
* Banks that disappear through mergers are labelled 0; mention this as a limitation.
* The `institutions` table is a snapshot taken at download time; static attributes such
  as charter class may have changed. `active` and `endefymd` are excluded as leaky.

## Report structure (suggested page budget)

| Pages | Section | Main evidence |
|---|---|---|
| 1 | Title, student info, GitHub URL | |
| 1 | Executive summary, problem, contributions | key measured KPIs only |
| 3 | Part 1 Ingestion and storage | architecture diagram, manifest stats, storage benchmark |
| 3 | Part 2 Data quality | duplicates, missingness table + heat-map, IQR vs 3 sd, LOF, leakage proof |
| 3 | Part 3 Processing and features | CAMELS mapping table, processing benchmark |
| 2 | Part 4 Feature selection | funnel, Freedman simulation |
| 4 | Part 5 Model | split table, PR curves, metrics, cost curve and threshold |
| 1 | Part 6 Reproducibility and governance | pipeline diagram, rerun instructions |
| 1-2 | Limitations, conclusion | |
| 1 | References (incl. FDIC API, libraries, AI tool use) | |
