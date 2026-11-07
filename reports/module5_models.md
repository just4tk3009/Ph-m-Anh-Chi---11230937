# Module 5 - Models, costs and the single test evaluation (generated 2026-11-07T13:36:31Z)

Choices below use the development period only (1984-2007). The test window (2009Q1-2025Q2) was evaluated after `model_choice.json` was written. Test evaluations recorded so far: **1** (`gold/_test_evaluations.jsonl`).

Features (Module 4, 6): texas_ratio_sd4, roa_ann_sd4, texas_ratio, equity_to_assets, texas_ratio_qoq, leverage_ratio

## 1. Time-aware CV on development (mean over folds)

| model | strategy | pr_auc_mean | roc_auc_mean | recall_top1pct_mean | recall_top5pct_mean | recall_top10pct_mean | folds | pr_auc_sd | fit_seconds_mean |
|---|---|---|---|---|---|---|---|---|---|
| rf_weights | class_weights | 0.305 | 0.947 | 0.506 | 0.797 | 0.869 | 4 | 0.178 | 120.577 |
| hgb_weights | class_weights | 0.302 | 0.938 | 0.501 | 0.783 | 0.848 | 4 | 0.178 | 4.325 |
| hgb_none | none | 0.296 | 0.942 | 0.493 | 0.787 | 0.852 | 4 | 0.176 | 3.905 |
| hgb_undersample | undersample | 0.281 | 0.943 | 0.488 | 0.778 | 0.858 | 4 | 0.158 | 1.495 |
| lr_weights | class_weights | 0.268 | 0.933 | 0.492 | 0.786 | 0.864 | 4 | 0.152 | 1.813 |
| baseline_lr_4ratios | none | 0.194 | 0.930 | 0.458 | 0.765 | 0.846 | 4 | 0.116 | 0.673 |


Chosen model: **rf_weights** (best mean CV pr_auc).

## 2. Costs estimated from FDIC data

| failures_in_development | failures_with_cost | cost_median_thousand_usd | cost_mean_thousand_usd | cost_p25_thousand_usd | cost_p75_thousand_usd | loss_rate_median_pct_of_assets | avoidable_loss_share_alpha | positive_rows_per_failed_bank | cfn_per_missed_bank_thousand_usd | cfn_per_positive_row_thousand_usd | cfp_per_false_alarm_thousand_usd | cfn_over_cfp | bayes_threshold_calibrated | dollars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2,342 | 2,312 | 8,693.239 | 63,695.858 | 2,976.239 | 31,552.894 | 20.996 | 0.250 | 3.769 | 2,173.310 | 576.577 | 50.000 | 11.532 | 0.080 | nominal (not inflation-adjusted) |

## 3. Final fit on train, calibration and threshold on validation

| model | pr_auc | roc_auc | recall_top1pct | recall_top5pct | recall_top10pct | brier | tp | fp | fn | cost_thousand_usd | threshold_cost_min | threshold_bayes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_lr_4ratios | 0.090 | 0.912 | 0.659 | 0.810 | 0.829 | 0.001 | 166 | 925 | 203 | 163,295.157 | 0.018 | 0.080 |
| lr_weights | 0.156 | 0.903 | 0.678 | 0.816 | 0.835 | 0.001 | 181 | 1,091 | 188 | 162,946.500 | 0.037 | 0.080 |
| rf_weights | 0.167 | 0.933 | 0.661 | 0.821 | 0.859 | 0.001 | 158 | 846 | 211 | 163,957.774 | 0.055 | 0.080 |
| hgb_none | 0.170 | 0.925 | 0.645 | 0.816 | 0.856 | 0.001 | 151 | 845 | 218 | 167,943.814 | 0.062 | 0.080 |
| hgb_weights | 0.166 | 0.920 | 0.648 | 0.816 | 0.840 | 0.001 | 158 | 864 | 211 | 164,857.774 | 0.058 | 0.080 |
| hgb_undersample | 0.170 | 0.929 | 0.634 | 0.808 | 0.840 | 0.001 | 151 | 810 | 218 | 166,193.814 | 0.062 | 0.080 |

## 4. Test 2009Q1-2025Q2 (single evaluation)

| chosen | model | pr_auc | roc_auc | recall_top1pct | recall_top5pct | recall_top10pct | brier | threshold | tp | fp | fn | cost_thousand_usd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no | baseline_lr_4ratios | 0.617 | 0.985 | 0.549 | 0.911 | 0.956 | 0.004 | 0.018 | 1,315 | 1,643 | 464 | 349,681.788 |
| no | lr_weights | 0.637 | 0.987 | 0.569 | 0.920 | 0.961 | 0.003 | 0.037 | 1,407 | 1,924 | 372 | 310,686.692 |
| yes | rf_weights | 0.661 | 0.985 | 0.583 | 0.918 | 0.961 | 0.003 | 0.055 | 1,305 | 1,228 | 474 | 334,697.559 |
| no | hgb_none | 0.637 | 0.987 | 0.570 | 0.914 | 0.957 | 0.003 | 0.062 | 1,286 | 1,325 | 493 | 350,502.525 |
| no | hgb_weights | 0.646 | 0.986 | 0.580 | 0.917 | 0.957 | 0.003 | 0.058 | 1,306 | 1,386 | 473 | 342,020.982 |
| no | hgb_undersample | 0.614 | 0.987 | 0.567 | 0.916 | 0.958 | 0.003 | 0.062 | 1,278 | 1,330 | 501 | 355,365.142 |


Reference costs on test: {"cost_no_model_all_missed": 1025730.7, "cost_flag_everything": 19983200.0}

## 5. Failed banks detected in the test period

| model | failed_banks | detected_at_threshold | detection_rate_threshold | detected_top5pct | detection_rate_top5pct | median_lead_quarters_if_detected |
|---|---|---|---|---|---|---|
| rf_weights | 501 | 457 | 0.912 | 485 | 0.968 | 3.000 |
| baseline_lr_4ratios | 501 | 456 | 0.910 | 483 | 0.964 | 3.000 |

## 6. Cost sensitivity (threshold chosen on validation, applied to test)

| alpha | cfp | cfn_row | cfn_over_cfp | threshold_valid | bayes_threshold | valid_alerts | valid_recall | valid_cost | test_recall | test_alerts | test_cost | test_cost_no_model |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.100 | 10.000 | 230.631 | 23.063 | 0.029 | 0.042 | 1,531 | 0.509 | 55,174.184 | 0.824 | 3,900 | 96,527.457 | 410,292.285 |
| 0.100 | 25.000 | 230.631 | 9.225 | 0.070 | 0.098 | 848 | 0.388 | 69,747.572 | 0.680 | 2,115 | 153,853.955 | 410,292.285 |
| 0.100 | 50.000 | 230.631 | 4.613 | 0.198 | 0.178 | 216 | 0.206 | 74,574.840 | 0.402 | 880 | 253,641.226 | 410,292.285 |
| 0.100 | 100.000 | 230.631 | 2.306 | 0.232 | 0.302 | 177 | 0.184 | 80,319.886 | 0.341 | 706 | 280,529.989 | 410,292.285 |
| 0.100 | 250.000 | 230.631 | 0.923 | 0.534 | 0.520 | 1 | 0.003 | 84,872.153 | 0.008 | 15 | 406,832.822 | 410,292.285 |
| 0.250 | 10.000 | 576.577 | 57.658 | 0.011 | 0.017 | 3,086 | 0.612 | 111,050.529 | 0.907 | 7,016 | 149,155.226 | 1,025,730.712 |
| 0.250 | 25.000 | 576.577 | 23.063 | 0.029 | 0.042 | 1,531 | 0.509 | 137,935.460 | 0.824 | 3,900 | 241,318.641 | 1,025,730.712 |
| 0.250 | 50.000 | 576.577 | 11.532 | 0.055 | 0.080 | 1,004 | 0.428 | 163,957.774 | 0.734 | 2,533 | 334,697.559 | 1,025,730.712 |
| 0.250 | 100.000 | 576.577 | 5.766 | 0.198 | 0.148 | 216 | 0.206 | 182,937.099 | 0.402 | 880 | 629,978.065 | 1,025,730.712 |
| 0.250 | 250.000 | 576.577 | 2.306 | 0.232 | 0.302 | 177 | 0.184 | 200,799.716 | 0.341 | 706 | 701,324.972 | 1,025,730.712 |
| 0.500 | 10.000 | 1,153.154 | 115.315 | 0.007 | 0.009 | 4,741 | 0.667 | 186,787.974 | 0.936 | 9,430 | 209,109.585 | 2,051,461.425 |
| 0.500 | 25.000 | 1,153.154 | 46.126 | 0.011 | 0.021 | 3,086 | 0.612 | 236,401.059 | 0.907 | 7,016 | 325,320.453 | 2,051,461.425 |
| 0.500 | 50.000 | 1,153.154 | 23.063 | 0.029 | 0.042 | 1,531 | 0.509 | 275,870.921 | 0.824 | 3,900 | 482,637.283 | 2,051,461.425 |
| 0.500 | 100.000 | 1,153.154 | 11.532 | 0.055 | 0.080 | 1,004 | 0.428 | 327,915.548 | 0.734 | 2,533 | 669,395.118 | 2,051,461.425 |
| 0.500 | 250.000 | 1,153.154 | 4.613 | 0.198 | 0.178 | 216 | 0.206 | 372,874.198 | 0.402 | 880 | 1,268,206.130 | 2,051,461.425 |
