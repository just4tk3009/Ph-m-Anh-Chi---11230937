# Module 2 - Data-quality audit (measured, generated 2026-11-06T07:49:32Z)

All numbers below are computed by `python pipeline.py audit` on the data in `/content/data`. Fences, variance filters and LOF are fitted on the TRAIN window only.

## 1. Duplicates and lineage

| table | rows_bronze | exact_duplicates_removed | conflicting_key_duplicates | rows_silver | cast_failures |
|---|---|---|---|---|---|
| financials | 693,162 | 0 | 0 | 693,162 | 0 |
| institutions | 27,835 | 0 | 0 | 27,835 | 0 |
| failures | 4,117 | 0 | 0 | 3,036 | 0 |
| gold (re-check) |  |  | 0 | 693,162 |  |

## 2. Panel coverage

| quarters | min_banks_per_quarter | max_banks_per_quarter | banks_with_internal_gaps | missing_quarters_inside_series |
|---|---|---|---|---|
| 98 | 4,494 | 9,940 | 1 | 9 |

## 3. Temporal split (purged) and class balance

| split | from | to | rows_in_window | rows_purged | rows | banks | positives | base_rate_pct | max_label_window_end |
|---|---|---|---|---|---|---|---|---|---|
| train | 2001-03-31 | 2012-12-31 | 415,567 | 29,157 | 386,410 | 11,140 | 1,880 | 0.487 | 2012-12-31 |
| valid | 2012-12-31 | 2015-12-31 | 80,315 | 25,521 | 54,794 | 7,105 | 125 | 0.228 | 2015-12-31 |
| test | 2015-12-31 | 2024-12-31 | 188,250 | 0 | 188,250 | 6,277 | 102 | 0.054 | 2025-12-31 |

## 4. Missingness and evidence for its mechanism

_MNAR cannot be proven from observed data; the label below is a data-driven indication only._

| column | missing_pct | min_year_pct | max_year_pct | size_quintile_gap_pp | missing_pct_failing | missing_pct_nonfailing | evidence_based_mechanism |
|---|---|---|---|---|---|---|---|
| bank_age_years | 0.997 | 0.948 | 1.229 | 3.526 | 0.000 | 1.000 | no association found (consistent with MCAR) |
| roe | 0.279 | 0.196 | 0.729 | 0.811 | 1.547 | 0.275 | no association found (consistent with MCAR) |
| bro | 0.251 | 0.196 | 0.377 | 0.793 | 0.000 | 0.252 | no association found (consistent with MCAR) |
| eintexp | 0.251 | 0.196 | 0.377 | 0.793 | 0.000 | 0.252 | no association found (consistent with MCAR) |
| elnatr | 0.251 | 0.196 | 0.377 | 0.793 | 0.000 | 0.252 | no association found (consistent with MCAR) |
| eq | 0.251 | 0.196 | 0.377 | 0.793 | 0.000 | 0.252 | no association found (consistent with MCAR) |
| intinc | 0.251 | 0.196 | 0.377 | 0.793 | 0.000 | 0.252 | no association found (consistent with MCAR) |
| lnatres | 0.251 | 0.196 | 0.377 | 0.793 | 0.000 | 0.252 | no association found (consistent with MCAR) |
| netinc | 0.251 | 0.196 | 0.377 | 0.793 | 0.000 | 0.252 | no association found (consistent with MCAR) |
| nonii | 0.251 | 0.196 | 0.377 | 0.793 | 0.000 | 0.252 | no association found (consistent with MCAR) |
| nonix | 0.251 | 0.196 | 0.377 | 0.793 | 0.000 | 0.252 | no association found (consistent with MCAR) |
| ntlnls | 0.251 | 0.196 | 0.377 | 0.793 | 0.000 | 0.252 | no association found (consistent with MCAR) |
| ore | 0.251 | 0.196 | 0.377 | 0.793 | 0.000 | 0.252 | no association found (consistent with MCAR) |

## 5. Zero / near-zero variance (decided on train)

Columns dropped: none


| column | train_n_unique | train_mode_share | train_std | full_n_unique | full_mode_share | full_std | drop_decision_train_only |
|---|---|---|---|---|---|---|---|
| bro | 49,876 | 0.680 | 966,999.857 | 84,256 | 0.644 | 1,824,738.133 | no |
| ore | 18,086 | 0.454 | 37,675.651 | 24,950 | 0.451 | 36,512.223 | no |
| p9asset | 11,893 | 0.390 | 251,174.852 | 17,962 | 0.449 | 279,106.366 | no |
| naasset | 30,999 | 0.202 | 282,162.206 | 44,802 | 0.184 | 275,204.806 | no |
| elnatr | 18,084 | 0.185 | 171,870.888 | 26,377 | 0.219 | 163,732.289 | no |
| ntlnls | 17,649 | 0.144 | 151,497.617 | 24,655 | 0.139 | 137,588.275 | no |
| nclnls | 32,959 | 0.115 | 506,377.857 | 48,131 | 0.115 | 524,978.975 | no |
| p3asset | 26,437 | 0.076 | 181,026.573 | 36,193 | 0.087 | 188,056.243 | no |
| sc | 128,872 | 0.028 | 3,920,078.703 | 207,139 | 0.028 | 9,166,974.625 | no |
| lnatres | 29,792 | 0.020 | 302,277.148 | 48,377 | 0.020 | 377,060.540 | no |
| lnlsnet | 218,485 | 0.017 | 11,051,640.961 | 349,889 | 0.017 | 18,056,330.616 | no |
| lnlsgr | 219,422 | 0.017 | 11,307,923.236 | 351,833 | 0.017 | 18,387,281.432 | no |
| eintexp | 33,709 | 0.011 | 268,373.082 | 44,564 | 0.011 | 336,789.607 | no |
| depdom | 236,002 | 0.010 | 10,683,624.594 | 384,869 | 0.010 | 25,514,068.071 | no |
| dep | 236,152 | 0.010 | 14,723,848.727 | 385,114 | 0.010 | 30,355,099.260 | no |

## 6. Accounting identity and domain-rule violations

| rule | violations | pct_rows | banks_affected |
|---|---|---|---|
| total assets <= 0 | 0 | 0.000 | 0 |
| deposits > total assets | 143 | 0.021 | 70 |
| equity > total assets | 1 | 0.000 | 1 |
| negative equity (insolvent book value) | 393 | 0.057 | 184 |
| net loans > total assets | 0 | 0.000 | 0 |
| net loans > gross loans | 4 | 0.001 | 2 |
| noncurrent loans > gross loans | 0 | 0.000 | 0 |
| allowance < 0 | 4 | 0.001 | 2 |
| securities > total assets | 0 | 0.000 | 0 |
| |ROA| > 100% | 156 | 0.023 | 23 |
| leverage ratio outside [-100, 100]% | 1,275 | 0.184 | 547 |

## 7. Univariate outliers: Tukey IQR vs mean ± 3 sd (fences fitted on train)

| ratio | train_skew | train_kurtosis | train_iqr_flag_pct | train_gauss_flag_pct | test_iqr_flag_pct | fail_rate_pct_if_iqr_flagged | fail_rate_pct_if_not_flagged |
|---|---|---|---|---|---|---|---|
| equity_to_assets | 4.986 | 60.922 | 3.836 | 2.073 | 3.046 | 0.182 | 0.499 |
| npl_to_loans | 9.454 | 207.756 | 3.602 | 1.585 | 1.066 | 10.599 | 0.109 |
| alll_to_loans | 27.068 | 1,413.022 | 3.017 | 0.751 | 1.584 | 7.514 | 0.268 |
| loans_to_deposits | 158.926 | 34,840.376 | 0.718 | 0.039 | 0.179 | 0.072 | 0.490 |
| securities_to_assets | 1.102 | 1.682 | 0.192 | 1.203 | 0.081 | 0.000 | 0.487 |
| cash_to_assets | 5.302 | 42.568 | 4.210 | 1.779 | 12.617 | 0.775 | 0.474 |
| brokered_to_deposits | 273.209 | 101,859.176 | 14.424 | 0.639 | 15.457 | 1.771 | 0.270 |
| ore_to_assets | 8.034 | 109.265 | 8.219 | 1.748 | 3.242 | 4.200 | 0.154 |
| log_assets | 0.948 | 3.001 | 0.497 | 1.441 | 1.194 | 0.208 | 0.488 |

## 8. Multivariate outliers: Local Outlier Factor (novelty mode)

| split | rows | lof_flag_pct | fail_rate_pct_flagged | fail_rate_pct_not_flagged |
|---|---|---|---|---|
| train_sample | 60,000 | 0.917 | 2.364 | 0.422 |
| valid | 54,794 | 0.964 | 1.705 | 0.214 |
| test | 188,250 | 1.448 | 0.330 | 0.050 |
