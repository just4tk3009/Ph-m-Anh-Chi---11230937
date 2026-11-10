"""Module 5: metrics, costs, purged CV folds and the end-to-end model stage (synthetic)."""
import json
from datetime import timedelta

import numpy as np
import polars as pl

from fdic_ews.features.camels import run_features
from fdic_ews.modeling.costs import best_threshold, cost_curve
from fdic_ews.modeling.metrics import score_metrics, top_k_flags
from fdic_ews.modeling.train_eval import _dev_rows, cv_folds, run_models, read_test_looks
from fdic_ews.utils.config import to_date


def test_top_k_flags_per_quarter():
    q = np.array(["a"] * 10 + ["b"] * 20)
    s = np.r_[np.arange(10), np.arange(20)].astype(float)
    f = top_k_flags(q, s, 10)
    assert f[:10].sum() == 1 and f[9]          # ceil(10% of 10) = 1, the highest score
    assert f[10:].sum() == 2 and f[-1] and f[-2]


def test_cost_curve_and_threshold():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    cc = cost_curve(y, p, cfn=10, cfp=1)
    assert cc.filter(pl.col("threshold") == 0)["cost_thousand_usd"][0] == 3  # flag all: 3 FP
    t, _ = best_threshold(y, p, cfn=10, cfp=1)
    assert 0.3 < t <= 0.8                                  # perfect separation, cost 0
    m = score_metrics(y, p, np.array(["q"] * 5), [20])
    assert m["pr_auc"] == 1.0 and m["recall_top20pct"] == 0.5


def _prepare(cfg):
    paths = cfg["_paths"]
    if not (paths.gold / "features.parquet").exists():
        run_features(cfg)
    feats = ["equity_to_assets", "npl_ratio", "roa_ann", "texas_ratio", "ore_to_assets"]
    (paths.gold / "selected_features.json").write_text(
        json.dumps({"selected_features": feats}), encoding="utf-8")
    return pl.read_parquet(paths.gold / "features.parquet")


def test_cv_folds_are_purged_and_inside_development(synthetic_cfg):
    data = _prepare(synthetic_cfg)
    h = 4
    dev = _dev_rows(data, synthetic_cfg, h)
    assert dev["repdte"].max() <= to_date(synthetic_cfg["split"]["valid_end"])
    for _, tr, va in cv_folds(dev, synthetic_cfg, h):
        lw_end = tr.select(pl.col("repdte").dt.offset_by("12mo").dt.month_end()).to_series()
        assert lw_end.max() < va["repdte"].min()          # label windows end before validation


def test_run_models_end_to_end(tmp_path):
    # the shared 150-bank fixture has no failures in the 1984-1992 train window:
    # build a larger synthetic panel for the end-to-end run
    import subprocess
    import sys
    from pathlib import Path

    from fdic_ews.storage.gold import build_gold
    from fdic_ews.storage.silver import build_silver
    from fdic_ews.utils.config import load_config
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "tools" / "make_synthetic_bronze.py"),
                    "--data-dir", str(tmp_path / "data"), "--banks", "300"], check=True)
    base = load_config(None, str(tmp_path / "data"), str(tmp_path / "reports"))
    build_silver(base)
    build_gold(base)
    _prepare(base)
    cfg = dict(base)
    cfg["model"] = {**cfg["model"], "rf": {**cfg["model"]["rf"], "n_estimators": 20},
                    "hgb": {**cfg["model"]["hgb"], "max_iter": 30}}
    before = len(read_test_looks(cfg["_paths"]))
    res = run_models(cfg)
    T = cfg["_paths"].tables
    choice = json.loads((T / "model_choice.json").read_text(encoding="utf-8"))
    assert choice["chosen_model"] == res["chosen"] and not res["chosen"].startswith("baseline")
    assert len(read_test_looks(cfg["_paths"])) == before + 1   # every test look is logged
    tt = pl.read_csv(T / "model_test_results.csv")
    assert tt.height == 6 and tt["chosen"].sum() == 1
    assert (cfg["_paths"].data / "models" / "ews_model.joblib").exists()
    costs = json.loads((T / "model_costs.json").read_text(encoding="utf-8"))
    assert costs["failures_with_cost"] > 0 and costs["cfn_per_positive_row_thousand_usd"] > 0
    from fdic_ews.modeling.predict import run_predict
    s = run_predict(cfg)
    assert s["banks_scored"] > 0 and s["model"] == res["chosen"]
    from fdic_ews.quality.sanity import run_checks
    out = run_checks(cfg)
    # the synthetic panel is too small for the two scope checks; every logic check must pass
    assert set(out["failed_checks"]) <= {"every quarter 1984Q1..2026Q2 present",
                                          "test has >= 50 failed banks"}, out
