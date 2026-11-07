"""Module 5: Spark MLlib demo - the same logistic regression in MLlib and scikit-learn.

Both engines fit an *unregularised*, class-weighted logistic regression with
median imputation (fitted on train) and standardisation, on the train window
(1984-1992), and score the validation window (1993-2007). Development data
only: the test window is not read. Without a penalty the optimum does not
depend on the feature scale, so the two engines should give (almost) the same
probabilities; the remaining gap measures numerical differences (L-BFGS
tolerances, exact vs. approximate medians), not a modelling choice.
"""
from __future__ import annotations

import json
import shutil
import time
from typing import Any

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..quality.temporal_split import temporal_split
from ..storage.gold import target_name
from ..utils.config import Paths
from ..utils.io_utils import md_table, utc_now
from ..utils.logging_utils import get_logger

log = get_logger("modeling.mllib")


def _weights(y: np.ndarray) -> tuple[float, float]:
    """'balanced' weights: n / (2 * n_class)."""
    n, n1 = len(y), int(y.sum())
    return n / (2 * (n - n1)), n / (2 * n1)


def run_mllib(cfg: dict[str, Any]) -> dict[str, Any]:
    paths: Paths = cfg["_paths"]
    h = int(cfg["gold"]["label_horizon_quarters"])
    tgt = target_name(h)
    sel = json.loads((paths.gold / "selected_features.json").read_text(encoding="utf-8"))
    feats = sel["selected_features"]
    data = pl.read_parquet(paths.gold / "features.parquet")
    split = temporal_split(data, cfg)
    tr = split.train.select(feats + [tgt]).with_columns(pl.col(feats).cast(pl.Float64))
    va = split.valid.select(feats + [tgt]).with_columns(pl.col(feats).cast(pl.Float64))
    ytr, yva = tr[tgt].to_numpy(), va[tgt].to_numpy()
    w0, w1 = _weights(ytr)
    rows: list[dict[str, Any]] = []

    # ---- scikit-learn
    t0 = time.perf_counter()
    sk = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler()),
                   ("clf", LogisticRegression(penalty=None, class_weight="balanced",
                                              max_iter=5000, tol=1e-8))])
    sk.fit(tr.select(feats).to_pandas(), ytr)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    p_sk = sk.predict_proba(va.select(feats).to_pandas())[:, 1]
    rows.append({"engine": "scikit-learn", "fit_seconds": fit_s,
                 "predict_seconds": time.perf_counter() - t0, "startup_seconds": 0.0,
                 "valid_pr_auc": float(average_precision_score(yva, p_sk)),
                 "valid_roc_auc": float(roc_auc_score(yva, p_sk))})

    # ---- Spark MLlib
    try:
        from pyspark.ml import Pipeline as SparkPipeline
        from pyspark.ml.classification import LogisticRegression as SparkLR
        from pyspark.ml.feature import Imputer, StandardScaler as SparkScaler, VectorAssembler
        from pyspark.sql import functions as F

        from ..features.engines import spark_session
    except ImportError as e:  # pragma: no cover - depends on the environment
        log.warning("PySpark not available (%s): MLlib demo skipped", e)
        return _write(paths, rows, None, feats, skipped=str(e))

    tmp = paths.data / "_mllib_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    tr.with_columns(w=pl.when(pl.col(tgt) == 1).then(w1).otherwise(w0)) \
      .write_parquet(tmp / "train.parquet")
    va.with_row_index("row_id").write_parquet(tmp / "valid.parquet")
    pc = cfg.get("processing_benchmark", {})
    t0 = time.perf_counter()
    spark = spark_session(pc.get("spark_driver_memory", "4g"),
                          int(pc.get("spark_shuffle_partitions", 8)))
    startup = time.perf_counter() - t0
    try:
        str_ = spark.read.parquet(str(tmp / "train.parquet"))
        sva = spark.read.parquet(str(tmp / "valid.parquet"))
        imp_cols = [f"{c}__imp" for c in feats]
        pipe = SparkPipeline(stages=[
            Imputer(inputCols=feats, outputCols=imp_cols, strategy="median", relativeError=0.0),
            VectorAssembler(inputCols=imp_cols, outputCol="x_raw"),
            SparkScaler(inputCol="x_raw", outputCol="x", withMean=True, withStd=True),
            SparkLR(featuresCol="x", labelCol=tgt, weightCol="w", regParam=0.0,
                    maxIter=500, tol=1e-10, standardization=False)])
        t0 = time.perf_counter()
        fitted = pipe.fit(str_)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        from pyspark.ml.functions import vector_to_array
        pred = (fitted.transform(sva)
                .select("row_id", vector_to_array("probability")[1].alias("p"))
                .orderBy("row_id").toPandas())
        pred_s = time.perf_counter() - t0
        plan = fitted.transform(sva)._jdf.queryExecution().simpleString()
        (paths.tables / "plan_spark_mllib_scoring.txt").write_text(plan, encoding="utf-8")
    finally:
        spark.stop()
        shutil.rmtree(tmp, ignore_errors=True)
    p_sp = pred["p"].to_numpy()
    rows.append({"engine": "spark-mllib", "fit_seconds": fit_s, "predict_seconds": pred_s,
                 "startup_seconds": startup,
                 "valid_pr_auc": float(average_precision_score(yva, p_sp)),
                 "valid_roc_auc": float(roc_auc_score(yva, p_sp))})
    agree = {"max_abs_prob_diff": float(np.max(np.abs(p_sk - p_sp))),
             "mean_abs_prob_diff": float(np.mean(np.abs(p_sk - p_sp))),
             "spearman_rank_corr": float(pl.DataFrame({"a": p_sk, "b": p_sp})
                                         .select(pl.corr("a", "b", method="spearman")).item())}
    log.info("MLlib vs sklearn: %s | %s", rows, agree)
    return _write(paths, rows, agree, feats)


def _write(paths: Paths, rows, agree, feats, skipped: str | None = None) -> dict[str, Any]:
    tbl = pl.DataFrame(rows)
    tbl.write_csv(paths.tables / "model_mllib_vs_sklearn.csv")
    md = [f"# Spark MLlib demo (generated {utc_now()})\n",
          "Unregularised, class-weighted logistic regression; median imputation + "
          "standardisation fitted on train (1984-1992); scored on validation (1993-2007). "
          f"Features: {', '.join(feats)}.\n", md_table(tbl.to_dicts())]
    if agree:
        md += ["\nAgreement of the two engines' validation probabilities:\n", md_table([agree])]
    if skipped:
        md.append(f"\nMLlib part skipped: {skipped}\n")
    (paths.tables / "model_mllib_vs_sklearn.md").write_text("\n".join(md), encoding="utf-8")
    return {"engines": rows, "agreement": agree}
