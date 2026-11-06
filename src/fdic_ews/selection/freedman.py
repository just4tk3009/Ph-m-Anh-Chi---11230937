"""Module 4 / Part 4: Freedman's paradox, measured by simulation.

Freedman (1983): when many candidate predictors are screened on the same data
that is later used to evaluate the model, pure noise looks predictive.

Simulation (repeated ``reps`` times for each (N, P)):
* X ~ N(0, 1) of shape N x P, y ~ Bernoulli(0.5), independent of X -> the true
  achievable AUC is 0.5;
* **leaky**: pick the k columns most correlated with y using ALL rows, then
  estimate AUC with 5-fold CV of a logistic regression on those k columns;
* **correct**: the same selection (``SelectKBest``) sits inside the CV pipeline,
  so it is refitted on each training fold only.
Also reported: the observed max |corr(X_j, y)| versus the extreme-value
approximation sqrt(2 ln P / N) for the largest of P near-normal correlations.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from ..utils.config import Paths
from ..utils.io_utils import md_table, utc_now
from ..utils.logging_utils import get_logger

log = get_logger("selection.freedman")


def simulate_once(n: int, p: int, k: int, rng: np.random.Generator) -> dict[str, float]:
    X = rng.standard_normal((n, p))
    y = rng.integers(0, 2, n)
    if y.min() == y.max():
        y[0] = 1 - y[0]
    yc = (y - y.mean()) / y.std()
    Xc = (X - X.mean(0)) / X.std(0)
    corr = np.abs(Xc.T @ yc) / n
    cv = StratifiedKFold(5, shuffle=True, random_state=int(rng.integers(1_000_000)))
    top = np.argsort(corr)[-k:]
    clf = LogisticRegression(max_iter=1000)
    leaky = cross_val_score(clf, X[:, top], y, cv=cv, scoring="roc_auc").mean()
    correct = cross_val_score(make_pipeline(SelectKBest(f_classif, k=k),
                                            LogisticRegression(max_iter=1000)),
                              X, y, cv=cv, scoring="roc_auc").mean()
    return {"r_max": float(corr.max()), "auc_leaky": float(leaky),
            "auc_correct": float(correct)}


def run_freedman(cfg: dict[str, Any]) -> pl.DataFrame:
    paths: Paths = cfg["_paths"]
    fc = cfg.get("freedman", {})
    grid = fc.get("grid", [[50, 200], [200, 200], [1000, 200]])
    k = int(fc.get("k", 10))
    reps = int(fc.get("reps", 200))
    rng = np.random.default_rng(int(cfg["quality"]["random_state"]))
    raw, rows = [], []
    for n, p in grid:
        sims = [simulate_once(int(n), int(p), k, rng) for _ in range(reps)]
        df = pl.DataFrame(sims).with_columns(N=pl.lit(n), P=pl.lit(p))
        raw.append(df)
        rows.append({"N": n, "P": p, "k_selected": k, "reps": reps,
                     "r_max_mean": float(df["r_max"].mean()),
                     "r_max_theory_sqrt_2lnP_over_N": math.sqrt(2 * math.log(p) / n),
                     "auc_leaky_mean": float(df["auc_leaky"].mean()),
                     "auc_leaky_p05": float(df["auc_leaky"].quantile(0.05)),
                     "auc_leaky_p95": float(df["auc_leaky"].quantile(0.95)),
                     "auc_correct_mean": float(df["auc_correct"].mean()),
                     "auc_correct_p05": float(df["auc_correct"].quantile(0.05)),
                     "auc_correct_p95": float(df["auc_correct"].quantile(0.95))})
        log.info("Freedman N=%s P=%s: %s", n, p, rows[-1])
    summ = pl.DataFrame(rows)
    pl.concat(raw).write_csv(paths.tables / "freedman_simulation_raw.csv")
    summ.write_csv(paths.tables / "freedman_simulation.csv")
    (paths.tables / "freedman_simulation.md").write_text(
        f"# Freedman's paradox: selection before vs inside CV (generated {utc_now()})\n\n"
        "Pure noise (true AUC = 0.5). Leaky = top-k features chosen on all rows, then "
        "5-fold CV; correct = selection refitted inside each training fold.\n\n"
        + md_table(summ.to_dicts()), encoding="utf-8")
    _plot(pl.concat(raw), paths)
    return summ


def _plot(raw: pl.DataFrame, paths: Paths) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    groups = raw.select("N", "P").unique().sort("N").rows()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    data, labels = [], []
    for n, p in groups:
        g = raw.filter((pl.col("N") == n) & (pl.col("P") == p))
        data += [g["auc_leaky"].to_list(), g["auc_correct"].to_list()]
        labels += [f"leaky\nN={n}", f"correct\nN={n}"]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=8)
    ax.axhline(0.5, color="grey", lw=1, ls="--")
    ax.set_ylabel("Cross-validated AUC on pure noise")
    ax.set_title("Selecting features before the split manufactures signal", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(paths.figures / "freedman_simulation.png", dpi=200)
    plt.close(fig)
