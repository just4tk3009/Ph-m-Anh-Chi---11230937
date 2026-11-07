# Spark MLlib demo (generated 2026-11-07T13:38:01Z)

Unregularised, class-weighted logistic regression; median imputation + standardisation fitted on train (1984-1992); scored on validation (1993-2007). Features: texas_ratio_sd4, roa_ann_sd4, texas_ratio, equity_to_assets, texas_ratio_qoq, leverage_ratio.

| engine | fit_seconds | predict_seconds | startup_seconds | valid_pr_auc | valid_roc_auc |
|---|---|---|---|---|---|
| scikit-learn | 8.677 | 0.138 | 0.000 | 0.049 | 0.884 |
| spark-mllib | 60.140 | 13.238 | 0.387 | 0.049 | 0.884 |


Agreement of the two engines' validation probabilities:

| max_abs_prob_diff | mean_abs_prob_diff | spearman_rank_corr |
|---|---|---|
| 0.000 | 0.000 | 1.000 |
