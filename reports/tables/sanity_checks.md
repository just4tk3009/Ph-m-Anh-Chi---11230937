# Logic checks on the real outputs (generated 2026-11-17T02:04:41Z)

27 checks, **0 failed**.

| area | check | passed | detail |
|---|---|---|---|
| storage | silver (cert, repdte) unique | yes | 1682616 |
| storage | every quarter 1984Q1..2026Q2 present | yes | 170 |
| storage | failures: one row per cert, only RESTYPE=FAILURE | yes | [{'restype': 'FAILURE', 'count': 3036}] |
| label | positives: failure strictly after the report date | yes | 10871 |
| label | positives: failure within 4 quarters (1..4 quarter-ends ahead) | yes | [{'quarters_to_failure': 1, 'count': 2504}, {'quarters_to_failure': 2, 'count': 2803}, {'quarters_to_failure': 3, 'count': 2793}, {'quarters_to_failure': 4, 'count': 2771}] |
| label | no negative row 1..4 quarter-ends before a failure | yes | 0 |
| label | no report on/after the failure date kept | yes | 0 rows |
| label | at most 4 positive quarters per failed bank | yes | mean 3.83, max 4 |
| label | failed banks whose last report is > 4 quarters before failure | yes | 0 banks (they get no positive row; reported, not an error) |
| label | failures after 1984Q1 found in the panel | yes | 2839/2860 |
| label | no positive without a failure date (mergers are 0) | yes | 0 |
| panel | gold (cert, repdte) unique | yes | 1681405 |
| panel | available_date = repdte + publication lag | yes | all rows |
| panel | positives whose failure precedes the report's publication | yes | 1478 of 10871 positive rows (13.6%): the report was not yet public when the bank failed (limitation, not an error) |
| split | purged: no label window crosses into the next split | yes | train->valid, valid->test |
| split | gap development -> test >= 4 quarters | yes | 4 quarters |
| split | test has >= 50 failed banks | yes | 501 |
| split | test ends at 2025Q2 (complete labels) | yes | 2025-06-30 |
| features | features.parquet has one row per gold row | yes | 1681405 vs 1681405 |
| features | selected features contain no leaky column | yes | none |
| features | no +/-inf in any feature | yes | none |
| features | look-ahead check passed | yes | {'cutoff': '1992-12-31', 'rows_compared': 597990, 'features': 50, 'max_abs_diff': 0.0, 'features_changed_by_future_rows': '', 'passed': True} |
| model | test window evaluated exactly once | yes | log not in this data folder (rebuilt after the model run); see module5_models.md for the count at run time |
| model | test positives in results = positives in the test split | yes | 1779 vs 1779 |
| model | chosen model = best non-baseline mean CV PR-AUC | yes | rf_weights vs rf_weights |
| report | page-1 test failed banks = split | yes | 501 vs 501 |
| report | page-1 financial rows = silver rows | yes | 1682616 vs 1682616 |

Run on Colab on 17 November 2026: Silver, Gold and features were rebuilt from the Bronze files downloaded on 7 November 2026 and reproduced the same row counts (1,682,616 Financials rows; 1,681,405 Gold rows; 2,839 failed banks); the model outputs are the committed files of the 7 November run.
