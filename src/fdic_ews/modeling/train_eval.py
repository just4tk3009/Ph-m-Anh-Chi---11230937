"""Module 5 / Part 5: baseline vs models, time-aware validation, costs, single test.

Protocol (every choice is made on the DEVELOPMENT period 1984-2007):

1. **Candidates.** A simple baseline (logistic regression on four classic CAMELS
   ratios, no re-weighting) against logistic regression, random forest and
   histogram gradient boosting on the features chosen in Module 4. For
   gradient boosting three imbalance strategies are compared: none, class
   weights, random undersampling of the training fold. (SMOTE is not used:
   interpolating between bank-quarters of different banks and eras creates
   synthetic balance sheets that never existed.)
2. **Time-aware CV.** Expanding windows inside development (``model.cv_folds``);
   a training row is used only if its 4-quarter label window ended before the
   validation window starts (purging). All preprocessing sits inside the
   sklearn Pipeline, so it is refitted on each training fold.
3. **Model choice** = best mean CV PR-AUC (Recall@top-k reported next to it).
4. **Final fit** of every candidate on the train window (1984-1992, purged);
   Platt calibration and the cost-minimising alarm threshold are fitted on the
   validation window (1993-2007). Costs: see ``costs.py``.
5. **Test (2009Q1-2025Q2) is read once**, after all choices are frozen and
   written to ``model_choice.json``. Every evaluation is appended to
   ``gold/_test_evaluations.jsonl`` so the number of test looks is auditable.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import date, timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from ..quality.preprocessing import IQRClipper, assert_no_leaky_columns
from ..quality.temporal_split import temporal_split
from ..storage.gold import target_name
from ..utils.config import Paths, to_date
from ..utils.io_utils import md_table, utc_now, write_json
from ..utils.logging_utils import get_logger
from .costs import best_threshold, cost_curve, estimate_costs
from .metrics import confusion_at, score_metrics, top_k_flags

log = get_logger("modeling")


# --------------------------------------------------------------------------- candidates
def _linear_prep(iqr_k: float) -> list:
    return [("clip", IQRClipper(iqr_k)),
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", RobustScaler())]


def candidates(cfg: dict[str, Any], selected: list[str]) -> dict[str, dict[str, Any]]:
    """name -> {features, strategy, make: () -> unfitted Pipeline, family}."""
    mc, q = cfg["model"], cfg["quality"]
    seed, iqr_k = int(q["random_state"]), float(q["iqr_k"])
    rf, hgb = mc["rf"], mc["hgb"]

    def lr(weighted: bool) -> Callable[[], Pipeline]:
        return lambda: Pipeline(_linear_prep(iqr_k) + [("clf", LogisticRegression(
            class_weight="balanced" if weighted else None, C=1.0, max_iter=3000))])

    def forest() -> Pipeline:
        return Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                         ("clf", RandomForestClassifier(
                             n_estimators=int(rf["n_estimators"]),
                             min_samples_leaf=int(rf["min_samples_leaf"]),
                             max_features=rf["max_features"], class_weight="balanced_subsample",
                             n_jobs=-1, random_state=seed))])

    def boost(weighted: bool) -> Callable[[], Pipeline]:
        # HGB handles missing values natively: no imputation, no scaling needed.
        return lambda: Pipeline([("clf", HistGradientBoostingClassifier(
            max_iter=int(hgb["max_iter"]), learning_rate=float(hgb["learning_rate"]),
            max_leaf_nodes=int(hgb["max_leaf_nodes"]),
            l2_regularization=float(hgb["l2_regularization"]),
            min_samples_leaf=int(hgb["min_samples_leaf"]),
            class_weight="balanced" if weighted else None,
            early_stopping=False, random_state=seed))])

    base = [f for f in mc["baseline_features"]]
    return {
        "baseline_lr_4ratios": {"features": base, "strategy": "none", "make": lr(False),
                                "family": "baseline"},
        "lr_weights": {"features": selected, "strategy": "class_weights", "make": lr(True),
                       "family": "logistic"},
        "rf_weights": {"features": selected, "strategy": "class_weights", "make": forest,
                       "family": "random_forest"},
        "hgb_none": {"features": selected, "strategy": "none", "make": boost(False),
                     "family": "gradient_boosting"},
        "hgb_weights": {"features": selected, "strategy": "class_weights", "make": boost(True),
                        "family": "gradient_boosting"},
        "hgb_undersample": {"features": selected, "strategy": "undersample", "make": boost(False),
                            "family": "gradient_boosting"},
    }


def _sample(X: pd.DataFrame, y: np.ndarray, rate: float, seed: int):
    """All positives + a random share of negatives (train fold only)."""
    if rate >= 1:
        return X, y
    keep = (y == 1) | (np.random.default_rng(seed).random(len(y)) < rate)
    return X[keep], y[keep]


def fit_candidate(spec: dict[str, Any], X: pd.DataFrame, y: np.ndarray, mc: dict[str, Any],
                  seed: int) -> tuple[Pipeline, float]:
    base_rate = float(mc["train_neg_sample_rate"])
    Xs, ys = _sample(X[spec["features"]], y, base_rate, seed)
    if spec["strategy"] == "undersample":
        rel = min(1.0, float(mc["undersample_neg_rate"]) / base_rate)
        Xs, ys = _sample(Xs, ys, rel, seed + 1)
    t0 = time.perf_counter()
    model = spec["make"]().fit(Xs, ys)
    return model, time.perf_counter() - t0


# --------------------------------------------------------------------------- calibration
class PlattCalibrator:
    """Logistic regression on the logit of the raw score (fitted on validation only)."""

    def fit(self, p: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(self._z(p), y)
        return self

    @staticmethod
    def _z(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self.lr_.predict_proba(self._z(p))[:, 1]


# --------------------------------------------------------------------------- data
def _load(cfg: dict[str, Any], selected: list[str]) -> tuple[pl.DataFrame, Any, str]:
    paths: Paths = cfg["_paths"]
    h = int(cfg["gold"]["label_horizon_quarters"])
    tgt = target_name(h)
    data = pl.read_parquet(paths.gold / "features.parquet")
    split = temporal_split(data, cfg)
    return data, split, tgt


def _dev_rows(data: pl.DataFrame, cfg: dict[str, Any], h: int) -> pl.DataFrame:
    """Development rows whose label is known before the test period starts."""
    s = cfg["split"]
    label_by = to_date(s["test_start"]) - timedelta(days=1)
    lw_end = pl.col("repdte").dt.offset_by(f"{3 * h}mo").dt.month_end()
    return data.filter(pl.col("repdte").is_between(to_date(s["train_start"]), to_date(s["valid_end"]))
                       & (lw_end <= label_by))


def cv_folds(dev: pl.DataFrame, cfg: dict[str, Any], h: int):
    lw_end = pl.col("repdte").dt.offset_by(f"{3 * h}mo").dt.month_end()
    for known_by, v0, v1 in cfg["model"]["cv_folds"]:
        kb, a, b = to_date(known_by), to_date(v0), to_date(v1)
        assert kb < a, "training labels must be known before the validation window starts"
        tr = dev.filter(lw_end <= kb)
        va = dev.filter(pl.col("repdte").is_between(a, b))
        yield f"train_labels<= {kb} | valid {a}..{b}", tr, va


# --------------------------------------------------------------------------- main
def run_models(cfg: dict[str, Any]) -> dict[str, Any]:
    paths: Paths = cfg["_paths"]
    mc, q = cfg["model"], cfg["quality"]
    seed = int(q["random_state"])
    h = int(cfg["gold"]["label_horizon_quarters"])
    ks = [float(k) for k in mc["recall_top_k_pct"]]
    sel_file = paths.gold / "selected_features.json"
    if not sel_file.exists():
        raise FileNotFoundError("run `python pipeline.py select` first (Module 4)")
    selected = json.loads(sel_file.read_text(encoding="utf-8"))["selected_features"]
    tgt = target_name(h)
    cands = candidates(cfg, selected)
    for spec in cands.values():
        assert_no_leaky_columns(spec["features"], tgt)
    T = paths.tables

    data, split, tgt = _load(cfg, selected)
    dev = _dev_rows(data, cfg, h)
    log.info("development rows %d (%d positives); test rows are not touched until step 5",
             dev.height, int(dev[tgt].sum()))

    # ---- 1-3: time-aware CV on development, choose the model
    cv_rows = []
    for fold_i, (fold, tr, va) in enumerate(cv_folds(dev, cfg, h)):
        Xtr, ytr = tr.to_pandas(), tr[tgt].to_numpy()
        Xva, yva = va.to_pandas(), va[tgt].to_numpy()
        if ytr.sum() == 0 or yva.sum() == 0:
            log.warning("CV fold %d skipped: %d train / %d valid positives", fold_i + 1,
                        int(ytr.sum()), int(yva.sum()))
            continue
        for name, spec in cands.items():
            model, secs = fit_candidate(spec, Xtr, ytr, mc, seed + fold_i)
            p = model.predict_proba(Xva[spec["features"]])[:, 1]
            m = score_metrics(yva, p, va["repdte"].to_numpy(), ks)
            cv_rows.append({"fold": fold_i + 1, "window": fold, "model": name,
                            "strategy": spec["strategy"], "train_rows": len(ytr),
                            "train_positives": int(ytr.sum()), "fit_seconds": secs, **m})
            log.info("CV fold %d %-20s PR-AUC %.4f  recall@top5%% %.3f  (%.1fs)", fold_i + 1,
                     name, m["pr_auc"], m.get("recall_top5pct", float("nan")), secs)
    cv = pl.DataFrame(cv_rows)
    cv.write_csv(T / "model_cv_folds.csv")
    metric = mc.get("selection_metric", "pr_auc")
    agg_cols = ["pr_auc", "roc_auc"] + [f"recall_top{k:g}pct" for k in ks]
    cv_summary = (cv.group_by("model", "strategy")
                  .agg([pl.col(c).fill_nan(None).mean().alias(f"{c}_mean") for c in agg_cols]
                       + [pl.len().alias("folds"),
                          pl.col(metric).fill_nan(None).std().alias(f"{metric}_sd"),
                          pl.col("fit_seconds").mean().alias("fit_seconds_mean")])
                  .sort(f"{metric}_mean", descending=True))
    cv_summary.write_csv(T / "model_cv_summary.csv")
    chosen = cv_summary.filter(~pl.col("model").str.starts_with("baseline"))["model"][0]
    log.info("model chosen on development CV: %s", chosen)

    # ---- 4: final fit on train, calibration + threshold on validation
    Xtr, ytr = split.train.to_pandas(), split.train[tgt].to_numpy()
    Xva, yva = split.valid.to_pandas(), split.valid[tgt].to_numpy()
    pos_per_bank = float(dev.filter(pl.col(tgt) == 1).group_by("cert").len()["len"].mean() or 1)
    c = mc["costs"]
    cost_stats, cost_by_year = estimate_costs(
        paths, to_date(cfg["split"]["train_start"]),
        to_date(cfg["split"]["test_start"]) - timedelta(days=1), pos_per_bank,
        float(c["avoidable_loss_share"]), float(c["cfp_thousand_usd"]))
    cfn, cfp = cost_stats["cfn_per_positive_row_thousand_usd"], float(c["cfp_thousand_usd"])
    write_json(T / "model_costs.json", cost_stats)
    cost_by_year.write_csv(T / "model_costs_by_fail_year.csv")

    finals: dict[str, dict[str, Any]] = {}
    valid_rows = []
    for name, spec in cands.items():
        model, secs = fit_candidate(spec, Xtr, ytr, mc, seed)
        raw = model.predict_proba(Xva[spec["features"]])[:, 1]
        cal = PlattCalibrator().fit(raw, yva)
        pva = cal.transform(raw)
        t_best, cc = best_threshold(yva, pva, cfn, cfp)
        finals[name] = {"model": model, "cal": cal, "threshold": t_best, "fit_seconds": secs}
        cm = confusion_at(yva, pva, t_best)
        valid_rows.append({"model": name, **score_metrics(yva, pva, split.valid["repdte"].to_numpy(), ks),
                           "threshold_cost_min": t_best,
                           "threshold_bayes": cost_stats["bayes_threshold_calibrated"],
                           **cm, "cost_thousand_usd": cm["fn"] * cfn + cm["fp"] * cfp})
        if name == chosen:
            cc.write_csv(T / "model_cost_curve_valid.csv")
            chosen_curve = cc
            chosen_valid_p = pva
    valid_tbl = pl.DataFrame(valid_rows)
    valid_tbl.write_csv(T / "model_valid_final.csv")

    # cost sensitivity (thresholds chosen on validation only)
    sens = []
    for a in c["sensitivity_avoidable_loss_share"]:
        for fp_cost in c["sensitivity_cfp_thousand_usd"]:
            cfn_s = float(a) * cost_stats["cost_median_thousand_usd"] / max(pos_per_bank, 1.0)
            t_s, _ = best_threshold(yva, chosen_valid_p, cfn_s, float(fp_cost))
            cm = confusion_at(yva, chosen_valid_p, t_s)
            sens.append({"alpha": float(a), "cfp": float(fp_cost), "cfn_row": cfn_s,
                         "cfn_over_cfp": cfn_s / float(fp_cost), "threshold_valid": t_s,
                         "bayes_threshold": float(fp_cost) / (float(fp_cost) + cfn_s),
                         "valid_alerts": cm["tp"] + cm["fp"], "valid_recall": cm["tp"] / max(1, yva.sum()),
                         "valid_cost": cm["fn"] * cfn_s + cm["fp"] * float(fp_cost)})

    choice = {"frozen_at": utc_now(), "chosen_model": chosen, "selection_metric": metric,
              "features": cands[chosen]["features"], "threshold": finals[chosen]["threshold"],
              "cfn_row": cfn, "cfp": cfp, "config_sha1": _cfg_hash(cfg)}
    write_json(T / "model_choice.json", choice)

    # ---- 5: the single test evaluation
    test = split.test
    Xte, yte, dte = test.to_pandas(), test[tgt].to_numpy(), test["repdte"].to_numpy()
    _log_test_look(paths, choice)
    test_rows, curves = [], {}
    for name, spec in cands.items():
        f = finals[name]
        pte = f["cal"].transform(f["model"].predict_proba(Xte[spec["features"]])[:, 1])
        cm = confusion_at(yte, pte, f["threshold"])
        test_rows.append({"model": name, "chosen": name == chosen, "strategy": spec["strategy"],
                          **score_metrics(yte, pte, dte, ks), "threshold": f["threshold"], **cm,
                          "cost_thousand_usd": cm["fn"] * cfn + cm["fp"] * cfp})
        curves[name] = pte
    test_tbl = pl.DataFrame(test_rows)
    test_tbl.write_csv(T / "model_test_results.csv")
    P = int(yte.sum())
    ref_costs = {"cost_no_model_all_missed": P * cfn, "cost_flag_everything": (len(yte) - P) * cfp}

    for row in sens:
        cm = confusion_at(yte, curves[chosen], row["threshold_valid"])
        row["test_recall"] = cm["tp"] / max(1, P)
        row["test_alerts"] = cm["tp"] + cm["fp"]
        row["test_cost"] = cm["fn"] * row["cfn_row"] + cm["fp"] * row["cfp"]
        row["test_cost_no_model"] = P * row["cfn_row"]
    sens_tbl = pl.DataFrame(sens)
    sens_tbl.write_csv(T / "model_cost_sensitivity.csv")

    banks = bank_level_detection(test, tgt, curves, {n: finals[n]["threshold"] for n in finals},
                                 [chosen, "baseline_lr_4ratios"], ks)
    banks["by_year"].write_csv(T / "model_test_detection_by_fail_year.csv")
    banks["summary"].write_csv(T / "model_test_detection_summary.csv")

    _save_model(paths, cands[chosen], finals[chosen], choice)
    _figures(paths, yte, curves, chosen, chosen_curve, yva, chosen_valid_p, cv)
    _write_md(paths, cv_summary, valid_tbl, test_tbl, cost_stats, sens_tbl, banks, choice,
              ref_costs, selected)
    return {"chosen": chosen, "test": test_tbl.filter(pl.col("chosen")).to_dicts()[0],
            "costs": cost_stats}


# --------------------------------------------------------------------------- helpers
def bank_level_detection(test: pl.DataFrame, tgt: str, curves: dict[str, np.ndarray],
                         thresholds: dict[str, float], models: list[str],
                         ks: list[float]) -> dict[str, pl.DataFrame]:
    """Per failed bank: was it flagged in at least one of its positive quarters, and how
    many quarters before failure was the first alert (lead time)?"""
    base = test.select("cert", "repdte", "faildate", tgt) if "faildate" in test.columns \
        else test.select("cert", "repdte", tgt).with_columns(faildate=pl.lit(None, pl.Date))
    summ, by_year = [], []
    k_mid = 5.0 if 5.0 in ks else ks[0]
    for m in models:
        df = base.with_columns(
            p=pl.Series(curves[m]),
            alert=pl.Series(curves[m] >= thresholds[m]),
            topk=pl.Series(top_k_flags(test["repdte"].to_numpy(), curves[m], k_mid)))
        pos = df.filter(pl.col(tgt) == 1).with_columns(
            q_before=((pl.col("faildate").dt.year() * 4 + pl.col("faildate").dt.quarter())
                      - (pl.col("repdte").dt.year() * 4 + pl.col("repdte").dt.quarter())))
        g = pos.group_by("cert").agg(
            fail_year=pl.col("faildate").dt.year().first(),
            detected=pl.col("alert").any(), detected_topk=pl.col("topk").any(),
            lead_q=pl.when(pl.col("alert")).then(pl.col("q_before")).max())
        summ.append({"model": m, "failed_banks": g.height,
                     "detected_at_threshold": int(g["detected"].sum()),
                     "detection_rate_threshold": float(g["detected"].mean()),
                     f"detected_top{k_mid:g}pct": int(g["detected_topk"].sum()),
                     f"detection_rate_top{k_mid:g}pct": float(g["detected_topk"].mean()),
                     "median_lead_quarters_if_detected": float(g["lead_q"].median())
                     if g["lead_q"].drop_nulls().len() else None})
        by_year.append(g.group_by("fail_year").agg(
            failed_banks=pl.len(), detected=pl.col("detected").sum(),
            detected_topk=pl.col("detected_topk").sum()).with_columns(model=pl.lit(m))
            .sort("fail_year"))
    return {"summary": pl.DataFrame(summ), "by_year": pl.concat(by_year)}


def _cfg_hash(cfg: dict[str, Any]) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha1(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()


def _log_test_look(paths: Paths, choice: dict[str, Any]) -> None:
    f = paths.gold / "_test_evaluations.jsonl"
    with open(f, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"evaluated_at": utc_now(), **{k: choice[k] for k in
                             ("chosen_model", "threshold", "config_sha1")}}) + "\n")


def read_test_looks(paths: Paths) -> list[dict[str, Any]]:
    f = paths.gold / "_test_evaluations.jsonl"
    if not f.exists():
        return []
    return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]


def _save_model(paths: Paths, spec: dict[str, Any], final: dict[str, Any],
                choice: dict[str, Any]) -> None:
    import joblib
    d = paths.data / "models"
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final["model"], "calibrator": final["cal"],
                 "threshold": final["threshold"], "features": spec["features"],
                 "choice": choice}, d / "ews_model.joblib")


def _figures(paths, yte, curves, chosen, cost_curve_valid, yva, pva, cv) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
    except ImportError:
        return
    F = paths.figures
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, p in curves.items():
        if yte.sum() == 0:
            break
        pr, rc, _ = precision_recall_curve(yte, p)
        lw = 2.2 if name == chosen else 1.0
        ax.plot(rc, pr, lw=lw, label=name + (" (chosen)" if name == chosen else ""))
    ax.axhline(yte.mean(), color="grey", ls="--", lw=1, label="base rate")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Test 2009Q1-2025Q2: precision-recall (fitted on 1984-1992)", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(F / "model_pr_curves_test.png", dpi=200)
    plt.close(fig)

    cc = cost_curve_valid.filter(pl.col("threshold") <= 1)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(cc["threshold"], cc["cost_thousand_usd"] / 1000)
    best = cc.sort("cost_thousand_usd").row(0, named=True)
    ax.axvline(best["threshold"], color="C3", ls="--", lw=1,
               label=f"cost-minimising t = {best['threshold']:.3f}")
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("Alarm threshold on calibrated probability")
    ax.set_ylabel("Cost on validation ($ million)")
    ax.set_title(f"{chosen}: expected cost vs threshold (validation 1993-2007)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(F / "model_cost_curve_valid.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    piv = cv.select("fold", "model", "pr_auc").pivot(on="model", index="fold").sort("fold")
    for m in [c for c in piv.columns if c != "fold"]:
        ax.plot(piv["fold"], piv[m], marker="o", lw=2.2 if m == chosen else 1.0, label=m)
    ax.set_xticks(piv["fold"].to_list())
    ax.set_xlabel("CV fold (expanding window inside 1984-2007)")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Time-aware cross-validation on the development period", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(F / "model_cv_pr_auc.png", dpi=200)
    plt.close(fig)


def _write_md(paths, cv_summary, valid_tbl, test_tbl, cost_stats, sens_tbl, banks, choice,
              ref_costs, selected) -> None:
    looks = read_test_looks(paths)
    keep = ["model", "pr_auc", "roc_auc", "recall_top1pct", "recall_top5pct", "recall_top10pct",
            "brier", "threshold", "tp", "fp", "fn", "cost_thousand_usd"]
    md = [f"# Module 5 - Models, costs and the single test evaluation (generated {utc_now()})\n",
          "Choices below use the development period only (1984-2007). The test window "
          "(2009Q1-2025Q2) was evaluated after `model_choice.json` was written. "
          f"Test evaluations recorded so far: **{len(looks)}** (`gold/_test_evaluations.jsonl`).\n",
          f"Features (Module 4, {len(selected)}): {', '.join(selected)}\n",
          "## 1. Time-aware CV on development (mean over folds)\n", md_table(cv_summary.to_dicts()),
          f"\nChosen model: **{choice['chosen_model']}** (best mean CV {choice['selection_metric']}).\n",
          "## 2. Costs estimated from FDIC data\n", md_table([cost_stats]),
          "## 3. Final fit on train, calibration and threshold on validation\n",
          md_table(valid_tbl.select([c for c in keep if c in valid_tbl.columns]
                                    + ["threshold_cost_min", "threshold_bayes"]).to_dicts()),
          "## 4. Test 2009Q1-2025Q2 (single evaluation)\n",
          md_table(test_tbl.select(["chosen"] + [c for c in keep if c in test_tbl.columns]).to_dicts()),
          f"\nReference costs on test: {json.dumps({k: round(v, 1) for k, v in ref_costs.items()})}\n",
          "## 5. Failed banks detected in the test period\n", md_table(banks["summary"].to_dicts()),
          "## 6. Cost sensitivity (threshold chosen on validation, applied to test)\n",
          md_table(sens_tbl.to_dicts())]
    (paths.reports / "module5_models.md").write_text("\n".join(md), encoding="utf-8")
