# Module 4 - Feature-selection funnel (generated 2026-11-07T03:43:30Z)

All stages fitted on the train window; stage 4 and the scores use validation; the test window is not read.

| stage | n_features | note | valid_pr_auc | valid_roc_auc | fit_seconds |
|---|---|---|---|---|---|
| 0 all candidates | 50 | all engineered features | 0.123 | 0.913 | 25.970 |
| 1 variance filter | 49 | dropped: ['pd30_89_to_assets'] | 0.123 | 0.913 | 23.686 |
| 2 F-test + MI + |rho|<=0.85 | 36 | 5 failed univariate, 8 pruned as collinear | 0.127 | 0.914 | 13.078 |
| 3 L1 stability selection | 21 | best C=0.001 | 0.129 | 0.916 | 6.471 |
| 4 permutation importance > 0 | 17 | HGB, PR-AUC drop on validation | 0.125 | 0.915 | 6.872 |


Selected (17): texas_ratio_sd4, equity_to_assets, texas_ratio, nonaccrual_to_assets, roa_ann_sd4, liquid_assets_to_assets_ma4, roa_ann_ma4, roe_ann, equity_to_assets_sd4, alll_to_loans, ore_to_assets, efficiency_ratio, npl_ratio_sd4, log_assets, equity_to_assets_qoq, texas_ratio_qoq, pd90_to_assets


## MDI vs permutation: noise controls

51 columns ranked. 
| feature | mdi_rank | perm_rank | perm_mean |
|---|---|---|---|
| noise_continuous | 48 | 39 | -0.001 |
| noise_binary | 50 | 33 | -0.000 |
