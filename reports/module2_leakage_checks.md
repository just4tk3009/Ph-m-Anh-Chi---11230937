# Module 2 - Anti-leakage verification (generated 2026-11-06T07:31:41Z)

| n_numeric_features_in | n_categorical_features_in | n_numeric_kept_after_near_constant | dropped_near_constant | all_medians_equal_train_median | share_medians_equal_full_sample_pct | median_abs_rel_shift_if_fitted_naively_pct | params_invariant_to_test_data | transform_is_row_wise_on_test | output_features_after_preprocessing |
|---|---|---|---|---|---|---|---|---|---|
| 28 | 4 | 28 | {} | True | 3.571428571428571 | 24.636226793778224 | True | True | 97 |

## Learned imputation medians vs window medians

| index | learned_median | train_median | full_sample_median | test_median | equals_train | equals_full |
|---|---|---|---|---|---|---|
| asset | 119,524.000 | 119,524.000 | 153,586.000 | 256,789.000 | yes | no |
| bro | 0.000 | 0.000 | 0.000 | 0.000 | yes | yes |
| chbal | 5,183.000 | 5,183.000 | 8,129.000 | 17,832.000 | yes | no |
| dep | 97,863.000 | 97,863.000 | 126,634.000 | 215,270.000 | yes | no |
| depdom | 97,739.000 | 97,739.000 | 126,533.000 | 215,108.000 | yes | no |
| eintexp | 1,237.000 | 1,237.000 | 977.000 | 711.000 | yes | no |
| elnatr | 90.000 | 90.000 | 75.000 | 58.000 | yes | no |
| eq | 12,533.000 | 12,533.000 | 16,225.000 | 27,218.500 | yes | no |
| intinc | 3,726.000 | 3,726.000 | 4,182.000 | 5,601.500 | yes | no |
| lnatres | 935.000 | 935.000 | 1,219.000 | 1,997.000 | yes | no |
| lnlsgr | 76,178.500 | 76,178.500 | 96,795.000 | 160,487.000 | yes | no |
| lnlsnet | 75,115.000 | 75,115.000 | 95,423.000 | 158,311.000 | yes | no |
| naasset | 353.000 | 353.000 | 457.000 | 610.000 | yes | no |
| nclnls | 551.000 | 551.000 | 645.000 | 767.000 | yes | no |
| netinc | 514.000 | 514.000 | 704.000 | 1,355.000 | yes | no |
| nimy | 3.937 | 3.937 | 3.776 | 3.534 | yes | no |
| nonii | 379.000 | 379.000 | 467.000 | 719.000 | yes | no |
| nonix | 1,993.000 | 1,993.000 | 2,484.000 | 3,828.000 | yes | no |
| ntlnls | 31.000 | 31.000 | 22.000 | 6.000 | yes | no |
| ore | 32.000 | 32.000 | 30.000 | 0.000 | yes | no |
| p3asset | 728.000 | 728.000 | 689.000 | 612.000 | yes | no |
| p9asset | 20.000 | 20.000 | 8.000 | 0.000 | yes | no |
| rbc1aaj | 9.467 | 9.467 | 9.901 | 10.494 | yes | no |
| rbcrwaj | 14.927 | 14.927 | 15.079 | 14.807 | yes | no |
| roa | 0.917 | 0.917 | 0.930 | 0.988 | yes | no |
| roe | 8.780 | 8.780 | 8.820 | 9.300 | yes | no |
| sc | 21,271.000 | 21,271.000 | 27,129.500 | 44,554.000 | yes | no |
| bank_age_years | 77.582 | 77.582 | 84.999 | 97.878 | yes | no |

## Relative parameter shift if fitted on all periods (naive = leaky)

| column | impute_median_rel_diff_pct | clip_lower_rel_diff_pct | clip_upper_rel_diff_pct | scale_center_rel_diff_pct | scale_iqr_rel_diff_pct |
|---|---|---|---|---|---|
| asset | 28.498 | 42.375 | 39.823 | 28.498 | 40.825 |
| bro | nan | 80.939 | 80.939 | nan | 80.902 |
| chbal | 56.840 | 83.473 | 78.144 | 56.840 | 80.259 |
| dep | 29.399 | 44.800 | 41.839 | 29.399 | 42.999 |
| depdom | 29.460 | 44.726 | 41.777 | 29.460 | 42.932 |
| eintexp | 21.019 | 10.821 | 12.482 | 21.019 | 11.885 |
| elnatr | 16.667 | 4.313 | 5.648 | 16.667 | 4.852 |
| eq | 29.458 | 46.621 | 42.773 | 29.458 | 44.160 |
| intinc | 12.238 | 17.553 | 16.712 | 12.238 | 16.974 |
| lnatres | 30.374 | 40.657 | 38.570 | 30.374 | 39.275 |
| lnlsgr | 27.063 | 39.047 | 37.109 | 27.063 | 37.881 |
| lnlsnet | 27.036 | 39.150 | 37.195 | 27.036 | 37.973 |
| naasset | 29.462 | 23.543 | 23.935 | 29.462 | 23.768 |
| nclnls | 17.060 | 18.410 | 18.292 | 17.060 | 18.342 |
| netinc | 36.965 | 44.205 | 44.071 | 36.965 | 43.856 |
| nimy | 4.091 | 70.185 | 2.803 | 4.091 | 1.523 |
| nonii | 23.219 | 33.504 | 32.531 | 23.219 | 32.993 |
| nonix | 24.636 | 34.714 | 33.021 | 24.636 | 33.622 |
| ntlnls | 29.032 | 13.964 | 13.964 | 29.032 | 14.027 |
| ore | 6.250 | 8.966 | 8.966 | 6.250 | 9.007 |
| p3asset | 5.357 | 1.631 | 2.322 | 5.357 | 2.036 |
| p9asset | 60.000 | 17.012 | 17.012 | 60.000 | 17.012 |
| rbc1aaj | 4.591 | 26.155 | 0.763 | 4.591 | 3.679 |
| rbcrwaj | 1.018 | 6.930 | 1.990 | 1.018 | 3.013 |
| roa | 1.416 | 8.984 | 3.612 | 1.416 | 5.503 |
| roe | 0.456 | 10.866 | 5.685 | 0.456 | 7.582 |
| sc | 27.542 | 39.898 | 38.028 | 27.542 | 38.783 |
| bank_age_years | 9.560 | 3.263 | 1.837 | 9.560 | 0.399 |
