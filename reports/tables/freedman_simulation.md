# Freedman's paradox: selection before vs inside CV (generated 2026-11-07T03:44:43Z)

Pure noise (true AUC = 0.5). Leaky = top-k features chosen on all rows, then 5-fold CV; correct = selection refitted inside each training fold.

| N | P | k_selected | reps | r_max_mean | r_max_theory_sqrt_2lnP_over_N | auc_leaky_mean | auc_leaky_p05 | auc_leaky_p95 | auc_correct_mean | auc_correct_p05 | auc_correct_p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 200 | 10 | 200 | 0.413 | 0.460 | 0.880 | 0.794 | 0.960 | 0.509 | 0.343 | 0.730 |
| 200 | 200 | 10 | 200 | 0.206 | 0.230 | 0.724 | 0.676 | 0.768 | 0.498 | 0.405 | 0.596 |
| 1,000 | 200 | 10 | 200 | 0.093 | 0.103 | 0.605 | 0.582 | 0.625 | 0.499 | 0.454 | 0.538 |
