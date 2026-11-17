# Project log (6 Nov - 26 Nov 2026, deadline 23:00 on 26/11)

Rule for the whole project: every number in the report comes from a file in
`reports/` produced by a script in this repository.

The instructor's suggested plan was week 1 for Parts 1-2, week 2 for Parts 3-4 and
week 3 for Parts 5-6, the report and a final check of the repository. The table below
records when each part was actually done, with the commit that holds the evidence.

## What was done, and when

| Dates | Part | Work | Evidence |
|---|---|---|---|
| 6 Nov | 1 | `probe`, trial ingest of a few quarters, Bronze/Silver/Gold layers, storage benchmark | `bronze/_manifest.jsonl`, `storage_benchmark.md` |
| 6 Nov | 2 | `audit`, `leakage`; failures vs. assistance transactions; exit types | `module2_data_quality.md`, `module2_leakage_checks.md` |
| 6 Nov | 3 | CAMELS features in Polars Lazy (YTD income to quarterly flows, gap-aware rolling windows); pandas vs Polars vs PySpark benchmark | `processing_benchmark.md`, `feature_dictionary.csv`, `features_lookahead_check.csv` |
| 6 Nov | - | Scope aligned with the exam brief: Financials 1984Q1-2026Q2 for all banks, full Failures and Institutions, exam split | `config/config.yaml` |
| 7 Nov | 1-3 | Full download (01:23-01:37 UTC) and full run on Colab; two data-quality fixes; pinned `requirements.txt` | `reports/`, `requirements.txt` |
| 7 Nov | 4 | Four-stage selection funnel inside the training data; Freedman simulation | `module4_feature_selection.md`, `freedman_simulation.md` |
| 7 Nov | 5 | Time-aware CV with purging; baseline vs LR / RF / gradient boosting; class weights vs undersampling; Platt calibration; cost-based threshold; Spark MLlib demo; single logged test evaluation | `module5_models.md`, `model_*.csv` |
| 7 Nov | 6 | `pipeline.py predict` scores the latest quarter with the frozen model; Colab notebook with outputs | `predict_latest_quarter.md`, `notebooks/` |
| 10 Nov | - | `pipeline.py check`: logic checks on the real outputs; Gold asserts join cardinality | `pipeline.py check` |
| 12-13 Nov | - | Report written (20 pages + appendix), sources cited in the report and in the README | report (PDF/DOCX) |
| before 26 Nov | - | Fresh clone, rerun of the checks, repository kept public until grades are released | |

## Decisions taken from measured data

1. **Split dates.** Train 1984-1992, validation 1993-2007, gap 2008, test 2009Q1-2025Q2,
   chosen after reading `gold_label_counts_by_year.csv`: the test window has 501 failed
   banks (the exam requires at least 50).
2. **Label horizon.** 4 quarters, as the exam requires: y = 1 if the bank fails in (q, q + 4].
3. **Missing values.** Structural gaps (item not collected in some years) are handled by the
   variance filter; informative gaps keep a missing-value flag (`dq_missingness.csv`).
4. **Outliers.** Rows outside the Tukey fences fail far more often, so values are clipped at
   limits learned on train, never deleted (`dq_outliers_iqr_vs_gauss.csv`).

## Technical notes

* Call Report income items (NETINC, INTINC, EINTEXP, NONII, NONIX, ELNATR, NTLNLS) are
  year-to-date. Quarterly flow = YTD(q) - YTD(q-1) within the same year, only when the
  previous report is exactly one quarter earlier; annualised before dividing by assets.
* Production rolling features use a time window (t - 12 months, t] on a quarter index, so
  they do not stretch across missing quarters. The four-engine benchmark uses the row
  window `rowsBetween(-3, 0)` (current + 3 previous) because it behaves the same in all engines.
* `t* = CFP / (CFP + CFN)` is optimal only for calibrated probabilities, so the threshold
  is chosen by minimising expected cost on the validation window and reported once on test,
  with a sensitivity table for the cost assumptions.
* Scaling, imputation, clipping and feature selection are fitted inside the training data
  only (scikit-learn `Pipeline`), never before the split.
* Banks that disappear through mergers are labelled 0 (exam rule); this is a limitation.
* The `institutions` table is a snapshot taken at download time; `active`, `endefymd`
  and `changec1` are blocked as features.
