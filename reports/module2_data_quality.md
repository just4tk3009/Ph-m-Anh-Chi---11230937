# Module 2 - Data-quality audit (measured, generated 2026-11-07T12:21:32Z)

All numbers below are computed by `python pipeline.py audit` on the data in `/content/data`. Fences, variance filters and LOF are fitted on the TRAIN window only.

## 1. Duplicates and lineage

| table | rows_bronze | exact_duplicates_removed | conflicting_key_duplicates | rows_silver | cast_failures |
|---|---|---|---|---|---|
| financials | 1,682,616 | 0 | 0 | 1,682,616 | 0 |
| institutions | 27,835 | 0 | 0 | 27,835 | 0 |
| failures | 4,117 | 0 | 0 | 3,036 | 0 |
| gold (re-check) |  |  | 0 | 1,681,405 |  |

Cast failures by column (value present in Bronze, null after the dtype cast):

none

## 2. Panel coverage

| quarters | min_banks_per_quarter | max_banks_per_quarter | banks_with_internal_gaps | missing_quarters_inside_series |
|---|---|---|---|---|
| 170 | 4,313 | 18,119 | 32 | 92 |

## 3. Temporal split (purged) and class balance

| split | from | to | rows_in_window | rows_purged | rows | banks | positives | failed_banks | base_rate_pct | max_label_window_end |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 1984-03-31 | 1992-12-31 | 597,990 | 57,007 | 540,983 | 21,139 | 7,977 | 2,167 | 1.475 | 1992-12-31 |
| valid | 1992-12-31 | 2007-12-31 | 630,440 | 0 | 630,440 | 16,369 | 369 | 132 | 0.059 | 2008-12-31 |
| test | 2009-03-31 | 2025-06-30 | 401,443 | 0 | 401,443 | 8,472 | 1,779 | 501 | 0.443 | 2026-06-30 |

## 4. Missingness and evidence for its mechanism

_MNAR cannot be proven from observed data; the label below is a data-driven indication only._

| column | missing_pct | min_year_pct | max_year_pct | size_quintile_gap_pp | missing_pct_failing | missing_pct_nonfailing | evidence_based_mechanism |
|---|---|---|---|---|---|---|---|
| p3asset | 57.732 | 0.000 | 100.000 | 0.008 | 79.082 | 57.593 | structural: item not reported in some years (form change) |
| ntlnls | 4.702 | 0.196 | 18.485 | 8.061 | 21.571 | 4.592 | informative: differs failing vs non-failing -> keep missing flag (MNAR-like) |
| p9asset | 4.572 | 0.000 | 19.632 | 7.842 | 21.571 | 4.461 | informative: differs failing vs non-failing -> keep missing flag (MNAR-like) |
| naasset | 4.572 | 0.000 | 19.632 | 7.842 | 21.571 | 4.461 | informative: differs failing vs non-failing -> keep missing flag (MNAR-like) |
| elnatr | 2.562 | 0.185 | 18.485 | 4.583 | 2.668 | 2.561 | no association found (consistent with MCAR) |
| roe | 1.051 | 0.196 | 3.128 | 2.132 | 21.534 | 0.918 | informative: differs failing vs non-failing -> keep missing flag (MNAR-like) |
| bro | 0.996 | 0.175 | 17.726 | 1.866 | 0.534 | 0.999 | no association found (consistent with MCAR) |
| bank_age_years | 0.654 | 0.183 | 1.281 | 2.128 | 0.000 | 0.658 | no association found (consistent with MCAR) |
| eintexp | 0.383 | 0.177 | 1.852 | 1.034 | 0.000 | 0.385 | no association found (consistent with MCAR) |
| intinc | 0.279 | 0.175 | 0.918 | 0.690 | 0.000 | 0.281 | no association found (consistent with MCAR) |
| netinc | 0.279 | 0.175 | 0.918 | 0.690 | 0.000 | 0.281 | no association found (consistent with MCAR) |
| nonii | 0.279 | 0.175 | 0.918 | 0.690 | 0.000 | 0.281 | no association found (consistent with MCAR) |
| nonix | 0.279 | 0.175 | 0.918 | 0.690 | 0.000 | 0.281 | no association found (consistent with MCAR) |
| lnatres | 0.247 | 0.160 | 0.392 | 0.579 | 0.000 | 0.248 | no association found (consistent with MCAR) |
| ore | 0.247 | 0.160 | 0.392 | 0.579 | 0.000 | 0.248 | no association found (consistent with MCAR) |
| eq | 0.247 | 0.160 | 0.392 | 0.578 | 0.000 | 0.248 | no association found (consistent with MCAR) |
| lncon | 0.234 | 0.000 | 0.392 | 0.564 | 0.000 | 0.236 | no association found (consistent with MCAR) |
| nclnls | 0.149 | 0.000 | 1.807 | 0.471 | 0.000 | 0.150 | no association found (consistent with MCAR) |
| chbal | 0.000 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | negligible (<0.1%) |
| dep | 0.000 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | negligible (<0.1%) |
| depdom | 0.000 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | negligible (<0.1%) |
| lnci | 0.000 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | negligible (<0.1%) |
| lnlsgr | 0.000 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | negligible (<0.1%) |
| lnlsnet | 0.000 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | negligible (<0.1%) |
| lnre | 0.000 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | negligible (<0.1%) |
| sc | 0.000 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | negligible (<0.1%) |

## 4b. Structural zeros (exact-zero share per year)

_A column that is ~100% zero in some years and rarely zero in others is not measured as 0: the item was not collected. Features treat it as missing._

| column | zero_pct_min_year | zero_pct_max_year | years_with_zero_pct_ge_95 | first_year | last_year | structural_zero |
|---|---|---|---|---|---|---|
| rbcrwaj | 0.196 | 100.000 | 6 | 1,984 | 1,989 | yes |

## 5. Zero / near-zero variance (decided on train)

Columns dropped: ['p3asset']


| column | train_n_unique | train_mode_share | train_std | full_n_unique | full_mode_share | full_std | drop_decision_train_only |
|---|---|---|---|---|---|---|---|
| p3asset | 0 | 1.000 | 0.000 | 37,035 | 0.088 | 189,305.562 | yes |
| bro | 16,619 | 0.920 | 78,938.944 | 98,172 | 0.794 | 1,232,752.333 | no |
| rbcrwaj | 122,013 | 0.774 | 376.078 | 1,211,000 | 0.278 | 287.079 | no |
| ore | 21,947 | 0.297 | 27,398.928 | 37,405 | 0.400 | 30,847.691 | no |
| naasset | 21,236 | 0.216 | 74,260.356 | 57,601 | 0.201 | 189,552.078 | no |
| elnatr | 14,124 | 0.172 | 20,485.386 | 35,849 | 0.207 | 111,459.977 | no |
| p9asset | 12,451 | 0.153 | 7,308.304 | 26,836 | 0.318 | 184,408.203 | no |
| rbc1aaj | 460,566 | 0.140 | 790.024 | 1,584,927 | 0.047 | 449.053 | no |
| ntlnls | 12,398 | 0.107 | 15,324.353 | 33,252 | 0.129 | 94,062.023 | no |
| lnci | 67,161 | 0.103 | 516,689.835 | 178,741 | 0.076 | 2,906,050.315 | no |
| nclnls | 31,676 | 0.055 | 74,568.558 | 66,730 | 0.089 | 343,502.438 | no |
| lnatres | 22,357 | 0.036 | 42,932.076 | 63,507 | 0.023 | 256,260.243 | no |
| lnre | 132,884 | 0.010 | 570,595.994 | 391,522 | 0.018 | 5,420,852.208 | no |
| lncon | 65,328 | 0.010 | 204,110.667 | 130,989 | 0.021 | 2,604,289.169 | no |
| sc | 104,937 | 0.009 | 284,853.844 | 269,738 | 0.017 | 6,286,669.791 | no |

## 6. Accounting identity and domain-rule violations

| rule | violations | pct_rows | banks_affected |
|---|---|---|---|
| total assets <= 0 | 0 | 0.000 | 0 |
| deposits > total assets | 5,473 | 0.326 | 1,500 |
| equity > total assets | 1 | 0.000 | 1 |
| negative equity (insolvent book value) | 14,579 | 0.867 | 2,819 |
| net loans > total assets | 2 | 0.000 | 2 |
| net loans > gross loans | 9 | 0.001 | 7 |
| noncurrent loans > gross loans | 37 | 0.002 | 16 |
| allowance < 0 | 9 | 0.001 | 7 |
| securities > total assets | 15 | 0.001 | 3 |
| |ROA| > 100% | 240 | 0.014 | 55 |
| leverage ratio outside [-100, 100]% | 2,921 | 0.174 | 1,425 |

## 7. Univariate outliers: Tukey IQR vs mean ± 3 sd (fences fitted on train)

| ratio | train_skew | train_kurtosis | train_iqr_flag_pct | train_gauss_flag_pct | test_iqr_flag_pct | fail_rate_pct_if_iqr_flagged | fail_rate_pct_if_not_flagged |
|---|---|---|---|---|---|---|---|
| equity_to_assets | -2.733 | 484.461 | 3.367 | 1.303 | 4.699 | 8.904 | 1.216 |
| npl_to_loans | 7.391 | 175.732 | 2.848 | 1.500 | 1.190 | 19.635 | 0.942 |
| alll_to_loans | 25.541 | 3,315.483 | 3.110 | 1.038 | 1.592 | 13.540 | 1.087 |
| loans_to_deposits | 209.401 | 60,336.492 | 0.467 | 0.033 | 0.347 | 0.554 | 1.479 |
| securities_to_assets | 0.517 | 1.649 | 0.001 | 0.461 | 0.000 | 0.000 | 1.475 |
| cash_to_assets | 3.510 | 22.676 | 2.358 | 1.858 | 6.153 | 0.768 | 1.492 |
| brokered_to_deposits | 13.442 | 239.025 | 7.834 | 1.205 | 40.789 | 6.245 | 1.069 |
| ore_to_assets | 10.817 | 204.233 | 5.650 | 1.355 | 2.930 | 13.533 | 0.753 |
| log_assets | 0.912 | 1.922 | 0.220 | 1.310 | 1.951 | 1.513 | 1.474 |

## 8. Multivariate outliers: Local Outlier Factor (novelty mode)

| split | rows | lof_flag_pct | fail_rate_pct_flagged | fail_rate_pct_not_flagged |
|---|---|---|---|---|
| train_sample | 60,000 | 0.900 | 3.519 | 1.519 |
| valid | 630,440 | 2.699 | 0.165 | 0.056 |
| test | 401,443 | 8.133 | 0.159 | 0.468 |
