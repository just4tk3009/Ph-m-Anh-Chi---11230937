"""Module 2: leakage-safe preprocessing as scikit-learn transformers.

Every step that *learns* something from data is a transformer with ``fit`` /
``transform``, and all of them live in one ``Pipeline``/``ColumnTransformer``.
Calling ``fit`` on the train window and only ``transform`` on validation/test
guarantees that medians, IQR fences, scales and the list of dropped columns
are computed from the past only (Kuhn & Johnson, 2019; scikit-learn docs,
"Common pitfalls: data leakage").

Steps (numeric branch)::

    NearConstantDropper -> IQRClipper -> SimpleImputer(median, +missing flags) -> RobustScaler
    (KNN variant)       -> IQRClipper -> RobustScaler -> KNNImputer

Why these choices (to be backed by the audit tables in the report):
* financial ratios are heavy-tailed, so fences use quartiles (Tukey k=3),
  not mean +/- 3 sd, and scaling uses median/IQR (``RobustScaler``);
* outliers are *clipped* (winsorised), not deleted: the audit shows extreme
  values are over-represented among failing banks, so they carry signal;
* missingness can itself be informative (MNAR), so ``add_indicator=True``.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.utils.validation import check_is_fitted

from ..storage.gold import KEY_COLS, LEAKY_COLUMNS, META_COLS


class LeakageError(ValueError):
    """Raised when a column that encodes future information is used as a feature."""


def assert_no_leaky_columns(columns: Iterable[str], target: str | None = None) -> None:
    cols = set(columns)
    forbidden = LEAKY_COLUMNS | ({target} if target else set()) \
        | {c for c in cols if c.startswith("target_")}
    bad = sorted(cols & forbidden)
    if bad:
        raise LeakageError(f"future-information columns used as features: {bad}")


def feature_columns(df_columns: Sequence[str], dtypes: Sequence[object],
                    target: str) -> tuple[list[str], list[str]]:
    """Split candidate columns into numeric / categorical, excluding keys, meta and leaky ones."""
    excluded = set(KEY_COLS) | LEAKY_COLUMNS | META_COLS | {target, "estymd"}
    num, cat = [], []
    for c, t in zip(df_columns, dtypes):
        if c in excluded or c.startswith("target_") or c.startswith("_"):
            continue
        name = str(t)
        if name.startswith(("Float", "Int", "UInt")) or name in ("float64", "int64", "Int64"):
            num.append(c)
        elif name in ("String", "Utf8", "Categorical", "object"):
            cat.append(c)
    assert_no_leaky_columns(num + cat, target)
    return num, cat


class NearConstantDropper(BaseEstimator, TransformerMixin):
    """Drop columns that are (near-)constant *in the training window*.

    A column is dropped if it has <= 1 distinct non-null value or if its most
    frequent value covers >= ``max_mode_share`` of non-null training rows.
    Fully-null training columns are dropped as well.
    """

    def __init__(self, max_mode_share: float = 0.995):
        self.max_mode_share = max_mode_share

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        keep, self.dropped_ = [], {}
        for c in X.columns:
            s = X[c].dropna()
            if s.empty:
                self.dropped_[c] = "all-null in train"
                continue
            vc = s.value_counts(normalize=True)
            if len(vc) <= 1:
                self.dropped_[c] = "zero variance in train"
            elif vc.iloc[0] >= self.max_mode_share:
                self.dropped_[c] = f"mode share {vc.iloc[0]:.4f} >= {self.max_mode_share}"
            else:
                keep.append(c)
        self.keep_ = keep
        return self

    def transform(self, X):
        check_is_fitted(self, "keep_")
        return pd.DataFrame(X, columns=self.feature_names_in_)[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.keep_, dtype=object)


class IQRClipper(BaseEstimator, TransformerMixin):
    """Winsorise each column to Tukey fences [Q1 - k*IQR, Q3 + k*IQR] learned on train.

    NaNs pass through untouched (imputation is a separate, later step).
    """

    def __init__(self, k: float = 3.0):
        self.k = k

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = (q1 - self.k * iqr).to_numpy(dtype=float)
        self.upper_ = (q3 + self.k * iqr).to_numpy(dtype=float)
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_")
        arr = pd.DataFrame(X, columns=self.feature_names_in_).to_numpy(dtype=float)
        return np.clip(arr, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)


def build_preprocessor(num_cols: list[str], cat_cols: list[str], imputer: str = "median",
                       iqr_k: float = 3.0, max_mode_share: float = 0.995,
                       knn_neighbors: int = 5) -> ColumnTransformer:
    """Return an *unfitted* ColumnTransformer. Fit it on the train window only."""
    assert_no_leaky_columns(num_cols + cat_cols)
    if imputer == "median":
        num_steps = [("near_constant", NearConstantDropper(max_mode_share)),
                     ("clip", IQRClipper(iqr_k)),
                     ("impute", SimpleImputer(strategy="median", add_indicator=True,
                                              keep_empty_features=False)),
                     ("scale", RobustScaler())]
    elif imputer == "knn":
        # KNN distances need comparable scales, so scale before imputing.
        num_steps = [("near_constant", NearConstantDropper(max_mode_share)),
                     ("clip", IQRClipper(iqr_k)),
                     ("scale", RobustScaler()),
                     ("impute", KNNImputer(n_neighbors=knn_neighbors, add_indicator=True))]
    else:
        raise ValueError(f"unknown imputer {imputer!r}")
    transformers = [("num", Pipeline(num_steps), num_cols)]
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.005,
                                     sparse_output=False)),
        ]), cat_cols))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
