# Module 2 - Anti-leakage verification (generated 2026-11-07T12:23:19Z)

| n_numeric_features_in | n_categorical_features_in | n_numeric_kept_after_near_constant | dropped_near_constant | all_medians_equal_train_median | share_medians_equal_full_sample_pct | median_abs_rel_shift_if_fitted_naively_pct | params_invariant_to_test_data | transform_is_row_wise_on_test | output_features_after_preprocessing |
|---|---|---|---|---|---|---|---|---|---|
| 31 | 4 | 30 | {'p3asset': 'all-null in train'} | True | 3.3333333333333335 | 59.71195078506399 | True | True | 101 |

## Learned imputation medians vs window medians

| index | learned_median | train_median | full_sample_median | test_median | equals_train | equals_full |
|---|---|---|---|---|---|---|
| asset | 46,379.000 | 46,379.000 | 85,281.000 | 203,179.000 | yes | no |
| bro | 0.000 | 0.000 | 0.000 | 0.000 | yes | yes |
| chbal | 2,548.000 | 2,548.000 | 4,283.000 | 13,899.000 | yes | no |
| dep | 41,180.000 | 41,180.000 | 72,378.000 | 169,610.000 | yes | no |
| depdom | 41,147.000 | 41,147.000 | 72,282.000 | 169,513.000 | yes | no |
| eintexp | 1,461.000 | 1,461.000 | 1,164.000 | 704.000 | yes | no |
| elnatr | 57.000 | 57.000 | 56.000 | 88.000 | yes | no |
| eq | 3,499.000 | 3,499.000 | 7,913.000 | 21,545.000 | yes | no |
| intinc | 2,473.000 | 2,473.000 | 3,159.000 | 4,618.000 | yes | no |
| lnatres | 264.000 | 264.000 | 595.000 | 1,738.000 | yes | no |
| lnci | 2,770.000 | 2,770.000 | 5,216.000 | 12,440.000 | yes | no |
| lncon | 3,541.000 | 3,541.000 | 3,629.000 | 3,301.000 | yes | no |
| lnlsgr | 24,696.000 | 24,696.000 | 49,632.000 | 125,962.000 | yes | no |
| lnlsnet | 24,334.000 | 24,334.000 | 48,940.000 | 124,037.000 | yes | no |
| lnre | 11,752.000 | 11,752.000 | 30,259.000 | 91,905.000 | yes | no |
| naasset | 170.000 | 170.000 | 245.000 | 847.000 | yes | no |
| nclnls | 448.000 | 448.000 | 463.000 | 1,026.000 | yes | no |
| netinc | 189.000 | 189.000 | 391.000 | 873.000 | yes | no |
| nimy | 4.197 | 4.197 | 4.042 | 3.613 | yes | no |
| nonii | 157.000 | 157.000 | 267.000 | 566.000 | yes | no |
| nonix | 781.000 | 781.000 | 1,401.000 | 3,190.000 | yes | no |
| ntlnls | 30.000 | 30.000 | 21.000 | 26.000 | yes | no |
| ore | 94.000 | 94.000 | 51.000 | 102.000 | yes | no |
| p9asset | 106.000 | 106.000 | 41.000 | 0.000 | yes | no |
| rbc1aaj | 7.698 | 7.698 | 9.093 | 10.196 | yes | no |
| rbcrwaj | 0.000 | 0.000 | 13.166 | 15.378 | yes | no |
| roa | 0.893 | 0.893 | 0.972 | 0.870 | yes | no |
| roe | 11.240 | 11.240 | 10.300 | 8.080 | yes | no |
| sc | 11,440.000 | 11,440.000 | 18,087.000 | 36,061.000 | yes | no |
| bank_age_years | 63.244 | 63.244 | 73.848 | 91.313 | yes | no |

## Relative parameter shift if fitted on all periods (naive = leaky)

| column | impute_median_rel_diff_pct | clip_lower_rel_diff_pct | clip_upper_rel_diff_pct | scale_center_rel_diff_pct | scale_iqr_rel_diff_pct |
|---|---|---|---|---|---|
| asset | 83.878 | 126.591 | 117.660 | 83.878 | 121.154 |
| bro | nan | nan | nan | nan | 0.000 |
| chbal | 68.093 | 121.280 | 112.764 | 68.093 | 116.127 |
| dep | 75.760 | 112.705 | 105.000 | 75.760 | 108.014 |
| depdom | 75.668 | 112.583 | 104.888 | 75.668 | 107.898 |
| eintexp | 20.329 | 14.076 | 15.466 | 20.329 | 14.578 |
| elnatr | 1.754 | 16.256 | 14.166 | 1.754 | 26.404 |
| eq | 126.150 | 219.286 | 198.854 | 126.150 | 206.473 |
| intinc | 27.740 | 36.633 | 34.939 | 27.740 | 35.657 |
| lnatres | 125.379 | 182.945 | 175.583 | 125.379 | 178.292 |
| lnci | 88.303 | 93.804 | 93.974 | 88.303 | 93.903 |
| lncon | 2.485 | 2.853 | 2.773 | 2.485 | 2.693 |
| lnlsgr | 100.972 | 140.226 | 132.275 | 100.972 | 135.443 |
| lnlsnet | 101.118 | 139.867 | 132.008 | 101.118 | 135.140 |
| lnre | 157.480 | 180.251 | 175.993 | 157.480 | 177.744 |
| naasset | 44.118 | 72.337 | 72.091 | 44.118 | 105.383 |
| nclnls | 3.348 | 23.362 | 20.988 | 3.348 | 22.638 |
| netinc | 106.878 | 131.756 | 131.846 | 106.878 | 131.733 |
| nimy | 3.696 | 54.169 | 8.356 | 3.696 | 13.021 |
| nonii | 70.064 | 108.797 | 103.882 | 70.064 | 105.913 |
| nonix | 79.385 | 115.379 | 108.780 | 79.385 | 111.521 |
| ntlnls | 30.000 | 7.006 | 7.006 | 30.000 | 11.864 |
| ore | 45.745 | 10.112 | 10.112 | 45.745 | 10.158 |
| p9asset | 61.321 | 26.898 | 28.424 | 61.321 | 14.643 |
| rbc1aaj | 18.127 | 31.172 | 7.619 | 18.127 | 0.112 |
| rbcrwaj | nan | nan | nan | nan | 1,705.936 |
| roa | 8.894 | 18.720 | 5.125 | 8.894 | 10.134 |
| roe | 8.363 | 13.618 | 9.330 | 8.363 | 8.776 |
| sc | 58.103 | 84.564 | 80.120 | 58.103 | 81.892 |
| bank_age_years | 16.766 | 14.219 | 15.389 | 16.766 | 14.485 |
