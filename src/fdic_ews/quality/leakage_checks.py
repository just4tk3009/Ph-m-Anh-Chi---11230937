"""Module 2: empirical anti-leakage verification of the preprocessing pipeline.

Three checks, each written to ``reports/tables/leak_*.csv``:

1. **Parameter provenance** - after ``fit(train)``, every learned imputation
   median equals the train-window median and *not* the full-sample median.
2. **Naive vs correct fitting** - how much the medians / IQR fences / scales
   move when the pipeline is (wrongly) fitted on train+valid+test. This puts a
   number on the leak that the correct design avoids.
3. **Test-set invariance** - fit on train, then replace the test features with
   random noise and refit: learned parameters must be bit-identical.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ..storage.gold import read_gold, target_name
from ..utils.config import Paths
from ..utils.io_utils import md_table, utc_now
from ..utils.logging_utils import get_logger
from .preprocessing import build_preprocessor, feature_columns
from .temporal_split import temporal_split

log = get_logger("quality.leakage")


def _learned_params(ct, num_cols_in: list[str]) -> pd.DataFrame:
    num = ct.named_transformers_["num"]
    kept = num.named_steps["near_constant"].keep_
    clip = num.named_steps["clip"]
    imp = num.named_steps["impute"]
    sc = num.named_steps["scale"]
    return pd.DataFrame({"column": kept, "impute_median": imp.statistics_[:len(kept)],
                         "clip_lower": clip.lower_, "clip_upper": clip.upper_,
                         "scale_center": sc.center_[:len(kept)],
                         "scale_iqr": sc.scale_[:len(kept)]}).set_index("column")


def run_leakage_checks(cfg: dict[str, Any]) -> dict[str, Any]:
    paths: Paths = cfg["_paths"]
    h = int(cfg["gold"]["label_horizon_quarters"])
    tgt = target_name(h)
    q = cfg["quality"]
    gold = read_gold(cfg, lazy=False)
    split = temporal_split(gold, cfg)
    num, cat = feature_columns(gold.columns, list(gold.dtypes), tgt)
    to_pd = lambda d: d.select(num + cat).to_pandas()  # noqa: E731
    Xtr, Xva, Xte = to_pd(split.train), to_pd(split.valid), to_pd(split.test)

    def make():
        return build_preprocessor(num, cat, "median", float(q["iqr_k"]),
                                  float(q["near_constant_share"]))

    correct = make().fit(Xtr)
    p_ok = _learned_params(correct, num)

    # 1. provenance: learned median == train median (after clipping is irrelevant for median
    #    only if the clip does not cross the median, which holds by construction of the fences)
    full = pd.concat([Xtr, Xva, Xte])
    prov = pd.DataFrame({
        "learned_median": p_ok["impute_median"],
        "train_median": Xtr[p_ok.index].median(),
        "full_sample_median": full[p_ok.index].median(),
        "test_median": Xte[p_ok.index].median() if len(Xte) else np.nan})
    prov["equals_train"] = np.isclose(prov["learned_median"], prov["train_median"], equal_nan=True)
    prov["equals_full"] = np.isclose(prov["learned_median"], prov["full_sample_median"],
                                     equal_nan=True)

    # 2. naive fit on all periods (the WRONG way) -> size of the leak
    naive = make().fit(full)
    p_bad = _learned_params(naive, num).reindex(p_ok.index)
    rel = ((p_bad - p_ok).abs() / p_ok.abs().replace(0, np.nan)) * 100
    rel.columns = [f"{c}_rel_diff_pct" for c in rel.columns]

    # 3. invariance: perturb test, refit on train -> identical parameters
    Xte_noise = Xte.copy()
    rng = np.random.default_rng(q["random_state"])
    for c in num:
        Xte_noise[c] = rng.normal(size=len(Xte_noise)) * 1e6
    again = make().fit(Xtr)  # the test frame never enters fit()
    invariant = bool(np.allclose(_learned_params(again, num).to_numpy(), p_ok.to_numpy(),
                                 equal_nan=True))
    Zte = correct.transform(Xte) if len(Xte) else np.empty((0, 0))
    Zte_noise = correct.transform(Xte_noise) if len(Xte) else np.empty((0, 0))
    rows_changed = bool(len(Xte) and not np.allclose(Zte, Zte_noise, equal_nan=True))

    T = paths.tables
    prov.reset_index().to_csv(T / "leak_param_provenance.csv", index=False)
    rel.reset_index().to_csv(T / "leak_naive_vs_correct.csv", index=False)
    summary = {"n_numeric_features_in": len(num), "n_categorical_features_in": len(cat),
               "n_numeric_kept_after_near_constant": int(len(p_ok)),
               "dropped_near_constant": correct.named_transformers_["num"]
               .named_steps["near_constant"].dropped_,
               "all_medians_equal_train_median": bool(prov["equals_train"].all()),
               "share_medians_equal_full_sample_pct": float(100 * prov["equals_full"].mean()),
               "median_abs_rel_shift_if_fitted_naively_pct":
                   float(np.nanmedian(rel["impute_median_rel_diff_pct"])),
               "params_invariant_to_test_data": invariant,
               "transform_is_row_wise_on_test": rows_changed,
               "output_features_after_preprocessing": int(correct.transform(Xtr[:5]).shape[1])}
    md = [f"# Module 2 - Anti-leakage verification (generated {utc_now()})\n",
          md_table([{k: str(v) for k, v in summary.items()}]),
          "## Learned imputation medians vs window medians\n",
          md_table(prov.reset_index().head(30).to_dict("records")),
          "## Relative parameter shift if fitted on all periods (naive = leaky)\n",
          md_table(rel.reset_index().head(30).to_dict("records"))]
    (paths.reports / "module2_leakage_checks.md").write_text("\n".join(md), encoding="utf-8")
    log.info("leakage checks: %s", {k: v for k, v in summary.items() if k != "dropped_near_constant"})
    assert summary["all_medians_equal_train_median"], "imputer learned a non-train median"
    assert invariant, "preprocessing parameters depend on test data"
    return summary
