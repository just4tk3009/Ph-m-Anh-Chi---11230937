"""Module 4 / Part 4: leakage-safe, multi-stage feature-selection funnel.

Every stage is fitted on the TRAIN window only (the test window is never read
here; the validation window is used only by stage 4 and for evaluation):

Stage 1  unsupervised   NearConstantDropper (mode share) + VarianceThreshold(0)
Stage 2  supervised     ANOVA F-test (p < alpha) AND mutual information > 0, then
                        collinearity pruning: of any pair with |Spearman rho| > 0.85
                        keep the one with the higher mutual information
Stage 3  embedded       L1-penalised logistic regression, *stability selection*
                        over expanding-window folds inside the train window
                        (C chosen by mean inner-fold PR-AUC); keep features
                        selected in >= ``min_fold_share`` of folds
Stage 4  validation     permutation importance (drop in PR-AUC) of a gradient-
                        boosted model on the validation window; keep importance > 0

After each stage the same reference model (L2 logistic, balanced weights, inside
a Pipeline with train-fitted preprocessing) is refitted on train and scored on
validation, so the report can show what each stage costs or gains.

Audit: Random-forest MDI vs permutation importance with two control columns
(continuous noise, binary noise) to show MDI's bias towards high-cardinality
features (Strobl et al., 2007).
"""
from __future__ import annotations

import json
import time
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from ..features.camels import feature_names
from ..quality.preprocessing import IQRClipper, NearConstantDropper, assert_no_leaky_columns
from ..quality.temporal_split import TemporalSplit, temporal_split
from ..storage.gold import target_name
from ..utils.config import Paths, to_date
from ..utils.io_utils import md_table, utc_now, write_json
from ..utils.logging_utils import get_logger

log = get_logger("selection.funnel")


def _prep(iqr_k: float) -> Pipeline:
    from sklearn.preprocessing import RobustScaler
    return Pipeline([("clip", IQRClipper(iqr_k)), ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scale", RobustScaler())])


def _sample_negatives(X: pd.DataFrame, y: np.ndarray, rate: float, seed: int):
    """Keep every positive and a random share of negatives (speed only, not a
    rebalancing choice: models below still use class weights)."""
    if rate >= 1:
        return X, y
    rng = np.random.default_rng(seed)
    keep = (y == 1) | (rng.random(len(y)) < rate)
    return X[keep], y[keep]


def _ref_scores(cols: list[str], Xtr, ytr, Xva, yva, iqr_k: float) -> dict[str, float]:
    model = Pipeline([("prep", _prep(iqr_k)),
                      ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])
    t0 = time.perf_counter()
    model.fit(Xtr[cols], ytr)
    p = model.predict_proba(Xva[cols])[:, 1]
    return {"valid_pr_auc": float(average_precision_score(yva, p)),
            "valid_roc_auc": float(roc_auc_score(yva, p)),
            "fit_seconds": time.perf_counter() - t0}


def _inner_folds(train: pl.DataFrame, years: list[int], h: int):
    """Expanding windows inside the train split, purged by the label horizon."""
    lw_end = pl.col("repdte").dt.offset_by(f"{3 * h}mo").dt.month_end()
    for y in years:
        cut = date(y, 12, 31)
        tr = train.filter(lw_end <= cut)
        va = train.filter((pl.col("repdte") > cut) & (pl.col("repdte") <= date(y + 1, 12, 31)))
        if tr.height and va.height:
            yield y, tr, va


def stage3_stability(split: TemporalSplit, cols: list[str], tgt: str, sc: dict[str, Any],
                     h: int, iqr_k: float, seed: int) -> tuple[list[str], pd.DataFrame, float]:
    years = [int(y) for y in sc["inner_fold_years"]]
    grid = [float(c) for c in sc["l1_C_grid"]]
    folds = list(_inner_folds(split.train, years, h))
    sel_counts = pd.DataFrame(0, index=cols, columns=[str(c) for c in grid])
    scores = {c: [] for c in grid}
    for y, tr, va in folds:
        Xtr, ytr = tr.select(cols).to_pandas(), tr[tgt].to_numpy()
        Xtr, ytr = _sample_negatives(Xtr, ytr, float(sc["neg_sample_rate"]), seed + y)
        Xva, yva = va.select(cols).to_pandas(), va[tgt].to_numpy()
        prep = _prep(iqr_k).fit(Xtr)
        Ztr, Zva = prep.transform(Xtr), prep.transform(Xva)
        for C in grid:
            clf = LogisticRegression(penalty="l1", C=C, solver="liblinear",
                                     class_weight="balanced", max_iter=2000).fit(Ztr, ytr)
            nz = np.abs(clf.coef_[0]) > 1e-8
            sel_counts.loc[np.array(cols)[nz], str(C)] += 1
            if yva.sum() > 0:
                scores[C].append(average_precision_score(yva, clf.predict_proba(Zva)[:, 1]))
        log.info("stage 3 fold <=%d: %d train rows, %d valid rows (%d positives)",
                 y, len(ytr), len(yva), int(yva.sum()))
    mean_scores = {C: (float(np.mean(v)) if v else -1.0) for C, v in scores.items()}
    best_C = max(mean_scores, key=mean_scores.get)
    share = sel_counts[str(best_C)] / max(len(folds), 1)
    kept = [c for c in cols if share[c] >= float(sc["min_fold_share"])]
    table = pd.DataFrame({"feature": cols, "selection_share_at_best_C": share.values})
    for C in grid:
        table[f"share_C={C}"] = (sel_counts[str(C)] / max(len(folds), 1)).values
    log.info("stage 3: inner-fold PR-AUC by C %s -> best C=%s, %d kept", mean_scores, best_C,
             len(kept))
    return kept, table, best_C


def mdi_vs_permutation(Xtr, ytr, Xva, yva, cols: list[str], seed: int,
                       neg_rate: float, valid_neg_rate: float = 1.0) -> pd.DataFrame:
    """The audit compares *rankings* of importances, so it is scored on all validation
    positives + a share of validation negatives (speed: 51 columns x 5 repeats of a
    200-tree forest on 630k rows took ~1 h on Colab)."""
    rng = np.random.default_rng(seed)
    Xtr, ytr = _sample_negatives(Xtr, ytr, neg_rate, seed)
    Xva, yva = _sample_negatives(Xva, yva, valid_neg_rate, seed + 7)
    Xtr = Xtr[cols].copy()
    Xva = Xva[cols].copy()
    for X in (Xtr, Xva):
        X["noise_continuous"] = rng.standard_normal(len(X))
        X["noise_binary"] = rng.integers(0, 2, len(X)).astype(float)
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr)
    Ztr = pd.DataFrame(imp.transform(Xtr), columns=Xtr.columns)
    Zva = pd.DataFrame(imp.transform(Xva), columns=Xva.columns)
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, max_features="sqrt",
                                class_weight="balanced_subsample", n_jobs=-1,
                                random_state=seed).fit(Ztr, ytr)
    perm = permutation_importance(rf, Zva, yva, scoring="average_precision", n_repeats=5,
                                  random_state=seed, n_jobs=1)
    out = pd.DataFrame({"feature": Ztr.columns, "mdi": rf.feature_importances_,
                        "perm_mean": perm.importances_mean, "perm_std": perm.importances_std})
    out["mdi_rank"] = out["mdi"].rank(ascending=False).astype(int)
    out["perm_rank"] = out["perm_mean"].rank(ascending=False).astype(int)
    return out.sort_values("mdi_rank")


def run_selection(cfg: dict[str, Any]) -> dict[str, Any]:
    paths: Paths = cfg["_paths"]
    h = int(cfg["gold"]["label_horizon_quarters"])
    tgt = target_name(h)
    q, sc = cfg["quality"], cfg["selection"]
    seed = int(q["random_state"])
    iqr_k = float(q["iqr_k"])
    feats_all = feature_names()
    assert_no_leaky_columns(feats_all, tgt)

    data = pl.read_parquet(paths.gold / "features.parquet")
    split = temporal_split(data, cfg)
    Xtr = split.train.select(feats_all).to_pandas()
    ytr = split.train[tgt].to_numpy()
    Xva = split.valid.select(feats_all).to_pandas()
    yva = split.valid[tgt].to_numpy()
    log.info("train %d rows (%d pos), valid %d rows (%d pos); test not loaded into selection",
             len(ytr), int(ytr.sum()), len(yva), int(yva.sum()))
    funnel: list[dict[str, Any]] = []

    def record(stage: str, cols: list[str], note: str) -> None:
        funnel.append({"stage": stage, "n_features": len(cols), "note": note,
                       **_ref_scores(cols, Xtr, ytr, Xva, yva, iqr_k)})
        log.info("%s: %d features, valid PR-AUC %.4f", stage, len(cols),
                 funnel[-1]["valid_pr_auc"])

    record("0 all candidates", feats_all, "all engineered features")

    # ---- stage 1: unsupervised
    nc = NearConstantDropper(float(q["near_constant_share"])).fit(Xtr)
    s1 = list(nc.keep_)
    vt = VarianceThreshold(0.0).fit(SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(Xtr[s1]))
    s1 = [c for c, keep in zip(s1, vt.get_support()) if keep]
    record("1 variance filter", s1, f"dropped: {sorted(set(feats_all) - set(s1)) or 'none'}")

    # ---- stage 2: supervised filter + collinearity pruning (train only)
    prep2 = _prep(iqr_k).fit(Xtr[s1])
    Ztr = pd.DataFrame(prep2.transform(Xtr[s1]), columns=s1)
    F, pval = f_classif(Ztr, ytr)
    Zs, ys = _sample_negatives(Ztr, ytr, float(sc["mi_neg_sample_rate"]), seed)
    mi = mutual_info_classif(Zs, ys, random_state=seed)
    uni = pd.DataFrame({"feature": s1, "F": F, "p_value": pval, "mutual_info": mi})
    alpha = float(sc["f_test_alpha"])
    passed = uni[(uni["p_value"] < alpha) & (uni["mutual_info"] > 0)].sort_values(
        "mutual_info", ascending=False)
    rho = Zs[passed["feature"]].rank().corr()  # Spearman on the (sampled) train fold
    kept2, pruned = [], []
    for f in passed["feature"]:  # greedy, in descending MI
        clash = [k for k in kept2 if abs(rho.loc[f, k]) > float(sc["max_abs_spearman"])]
        if clash:
            pruned.append({"feature": f, "pruned_because_of": clash[0],
                           "abs_spearman": float(abs(rho.loc[f, clash[0]]))})
        else:
            kept2.append(f)
    uni["passed_univariate"] = uni["feature"].isin(passed["feature"])
    uni["kept_after_collinearity"] = uni["feature"].isin(kept2)
    uni.sort_values("mutual_info", ascending=False).to_csv(
        paths.tables / "selection_stage2_univariate.csv", index=False)
    pd.DataFrame(pruned).to_csv(paths.tables / "selection_stage2_collinearity_pruned.csv",
                                index=False)
    record("2 F-test + MI + |rho|<=%.2f" % float(sc["max_abs_spearman"]), kept2,
           f"{len(s1) - len(passed)} failed univariate, {len(pruned)} pruned as collinear")

    # ---- stage 3: L1 stability selection inside the train window
    kept3, stab, best_C = stage3_stability(split, kept2, tgt, sc, h, iqr_k, seed)
    stab.to_csv(paths.tables / "selection_stage3_stability.csv", index=False)
    record("3 L1 stability selection", kept3, f"best C={best_C}")

    # ---- stage 4: permutation importance on validation
    Xs, ys = _sample_negatives(Xtr[kept3], ytr, float(sc["neg_sample_rate"]), seed)
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         class_weight="balanced", random_state=seed)
    hgb.fit(Xs, ys)  # HGB handles missing values natively; trees need no scaling
    perm = permutation_importance(hgb, Xva[kept3], yva, scoring="average_precision",
                                  n_repeats=int(sc["perm_repeats"]), random_state=seed,
                                  n_jobs=1)
    pi = pd.DataFrame({"feature": kept3, "perm_mean": perm.importances_mean,
                       "perm_std": perm.importances_std}).sort_values("perm_mean",
                                                                      ascending=False)
    pi.to_csv(paths.tables / "selection_stage4_permutation.csv", index=False)
    kept4 = pi.loc[pi["perm_mean"] > 0, "feature"].tolist()
    if len(kept4) < int(sc["min_features"]):
        kept4 = pi["feature"].head(int(sc["min_features"])).tolist()
    record("4 permutation importance > 0", kept4, "HGB, PR-AUC drop on validation")

    # ---- audit: MDI vs permutation with noise controls
    audit = mdi_vs_permutation(Xtr, ytr, Xva, yva, s1, seed, float(sc["neg_sample_rate"]),
                               float(sc.get("audit_valid_neg_rate", 1.0)))
    audit.to_csv(paths.tables / "selection_mdi_vs_permutation.csv", index=False)

    fdf = pd.DataFrame(funnel)
    fdf.to_csv(paths.tables / "selection_funnel.csv", index=False)
    result = {"built_at": utc_now(), "target": tgt, "selected_features": kept4,
              "stage_sizes": {r["stage"]: r["n_features"] for r in funnel}, "best_C": best_C}
    write_json(paths.gold / "selected_features.json", result)
    write_json(paths.tables / "selected_features.json", result)
    noise = audit[audit["feature"].str.startswith("noise_")][["feature", "mdi_rank", "perm_rank",
                                                              "perm_mean"]]
    md = [f"# Module 4 - Feature-selection funnel (generated {utc_now()})\n",
          "All stages fitted on the train window; stage 4 and the scores use validation; "
          "the test window is not read.\n",
          md_table(fdf.to_dict("records")),
          f"\nSelected ({len(kept4)}): {', '.join(kept4)}\n",
          "\n## MDI vs permutation: noise controls\n",
          f"{len(audit)} columns ranked. ", md_table(noise.to_dict("records"))]
    (paths.reports / "module4_feature_selection.md").write_text("\n".join(md), encoding="utf-8")
    _plot(fdf, paths)
    log.info("selection done: %s", json.dumps(result["stage_sizes"]))
    return result


def _plot(fdf: pd.DataFrame, paths: Paths) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax1 = plt.subplots(figsize=(8, 3.4))
    x = range(len(fdf))
    ax1.bar(x, fdf["n_features"], color="#9ecae1")
    ax1.set_ylabel("Features kept")
    ax1.set_xticks(list(x), [s.split(" ", 1)[1] if " " in s else s for s in fdf["stage"]],
                   rotation=12, fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(list(x), fdf["valid_pr_auc"], color="#08519c", marker="o")
    ax2.set_ylabel("Validation PR-AUC (reference model)")
    for ax in (ax1, ax2):
        ax.spines[["top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(paths.figures / "selection_funnel.png", dpi=200)
    plt.close(fig)
