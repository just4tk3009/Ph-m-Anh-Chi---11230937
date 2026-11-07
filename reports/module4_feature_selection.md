# Module 4 - Feature-selection funnel (generated 2026-11-07T13:19:39Z)

All stages fitted on the train window; stage 4 and the scores use validation; the test window is not read.

| stage | n_features | note | valid_pr_auc | valid_roc_auc | fit_seconds |
|---|---|---|---|---|---|
| 0 all candidates | 50 | all engineered features | 0.158 | 0.913 | 37.197 |
| 1 variance filter | 49 | dropped: ['pd30_89_to_assets'] | 0.158 | 0.913 | 23.336 |
| 2 F-test + MI + |rho|<=0.85 | 38 | 4 failed univariate, 7 pruned as collinear | 0.159 | 0.914 | 13.430 |
| 3 L1 stability selection | 12 | best C=0.0003 | 0.132 | 0.901 | 5.477 |
| 4 permutation importance > 0 | 6 | HGB, PR-AUC drop on validation | 0.153 | 0.901 | 2.982 |


Selected (6): texas_ratio_sd4, roa_ann_sd4, texas_ratio, equity_to_assets, texas_ratio_qoq, leverage_ratio


## MDI vs permutation: noise controls

51 columns ranked. 
| feature | mdi_rank | perm_rank | perm_mean |
|---|---|---|---|
| noise_continuous | 47 | 40 | -0.000 |
| noise_binary | 50 | 38 | -0.000 |
