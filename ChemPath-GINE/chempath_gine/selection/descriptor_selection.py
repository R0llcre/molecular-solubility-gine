from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class FeatureSelectionSpec:
    method: str = "none"
    top_k: int | None = None
    max_missing: float = 0.2
    variance_threshold: float = 1e-12
    corr_threshold: float = 0.98


class FeaturePreprocessor:
    """Train-only feature cleaner, selector, imputer, and scaler.

    The class deliberately fits every statistic on the training matrix only.
    Validation and test matrices are transformed with the stored train values.
    """

    def __init__(self, spec: FeatureSelectionSpec | None = None, scale: bool = True):
        self.spec = spec or FeatureSelectionSpec()
        self.scale = bool(scale)
        self.columns_: list[str] = []
        self.selected_columns_: list[str] = []
        self.keep_idx_: np.ndarray | None = None
        self.fill_values_: np.ndarray | None = None
        self.scaler_: StandardScaler | None = None

    def fit(self, x: pd.DataFrame | np.ndarray, y: np.ndarray) -> "FeaturePreprocessor":
        x_df = _as_frame(x)
        self.columns_ = list(x_df.columns)
        arr = x_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if arr.shape[0] != y_arr.shape[0]:
            raise ValueError("x and y have different row counts")

        missing_rate = np.mean(~np.isfinite(arr), axis=0)
        keep = missing_rate <= self.spec.max_missing
        arr = arr[:, keep]
        columns = [c for c, k in zip(self.columns_, keep) if k]
        fill = np.nanmean(np.where(np.isfinite(arr), arr, np.nan), axis=0)
        fill = np.where(np.isfinite(fill), fill, 0.0)
        arr = _fill(arr, fill)

        variance = np.var(arr, axis=0)
        keep = variance > self.spec.variance_threshold
        arr = arr[:, keep]
        columns = [c for c, k in zip(columns, keep) if k]
        fill = fill[keep]

        keep = _correlation_keep_mask(arr, threshold=self.spec.corr_threshold)
        arr = arr[:, keep]
        columns = [c for c, k in zip(columns, keep) if k]
        fill = fill[keep]

        keep = _score_keep_mask(arr, y_arr, columns, self.spec.method, self.spec.top_k)
        arr = arr[:, keep]
        columns = [c for c, k in zip(columns, keep) if k]
        fill = fill[keep]

        original_idx = np.array([self.columns_.index(c) for c in columns], dtype=np.int64)
        self.keep_idx_ = original_idx
        self.fill_values_ = fill.astype(np.float64)
        self.selected_columns_ = columns
        if self.scale:
            self.scaler_ = StandardScaler().fit(arr)
        else:
            self.scaler_ = None
        return self

    def transform(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.keep_idx_ is None or self.fill_values_ is None:
            raise RuntimeError("FeaturePreprocessor is not fitted")
        x_df = _as_frame(x, columns=self.columns_ if not isinstance(x, pd.DataFrame) else None)
        arr = x_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        arr = arr[:, self.keep_idx_]
        arr = _fill(arr, self.fill_values_)
        if self.scaler_ is not None:
            arr = self.scaler_.transform(arr)
        return arr.astype(np.float32)

    def fit_transform(self, x: pd.DataFrame | np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(x, y).transform(x)


def _as_frame(x: pd.DataFrame | np.ndarray, columns: list[str] | None = None) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    arr = np.asarray(x)
    if columns is None:
        columns = [f"f{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=columns)


def _fill(arr: np.ndarray, fill: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).copy()
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = np.take(fill, np.nonzero(bad)[1])
    return out


def _correlation_keep_mask(arr: np.ndarray, threshold: float) -> np.ndarray:
    n_features = arr.shape[1]
    if n_features <= 1 or threshold <= 0 or threshold >= 1:
        return np.ones(n_features, dtype=bool)
    # Dense correlation over thousands of fingerprint bits is expensive and
    # rarely useful; tree/linear regularization handles those redundancies.
    if n_features > 2000:
        return np.ones(n_features, dtype=bool)
    corr = np.corrcoef(arr, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    keep = np.ones(n_features, dtype=bool)
    for i in range(n_features):
        if not keep[i]:
            continue
        dup = np.where((corr[i] > threshold) & keep)[0]
        dup = dup[dup > i]
        keep[dup] = False
    return keep


def _score_keep_mask(
    arr: np.ndarray,
    y: np.ndarray,
    columns: list[str],
    method: str,
    top_k: int | None,
) -> np.ndarray:
    n_features = arr.shape[1]
    if n_features == 0:
        return np.zeros(0, dtype=bool)
    if top_k is None or top_k <= 0 or top_k >= n_features:
        return np.ones(n_features, dtype=bool)

    method = method.lower()
    if method in {"none", ""}:
        scores = np.ones(n_features, dtype=np.float64)
    elif method in {"pearson", "corr", "abs_pearson"}:
        scores = []
        for j in range(n_features):
            xj = arr[:, j]
            if np.std(xj) < 1e-12 or np.std(y) < 1e-12:
                scores.append(0.0)
            else:
                val = np.corrcoef(xj, y)[0, 1]
                scores.append(abs(float(val)) if np.isfinite(val) else 0.0)
        scores = np.asarray(scores, dtype=np.float64)
    elif method in {"mutual_info", "mi"}:
        scores = mutual_info_regression(arr, y, random_state=0)
    elif method in {"elasticnet", "enet"}:
        model = ElasticNetCV(
            l1_ratio=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9],
            alphas=np.logspace(-4, 2, 20),
            cv=min(5, max(2, len(y) // 5)),
            max_iter=20000,
            random_state=0,
        )
        scaled = StandardScaler().fit_transform(arr)
        model.fit(scaled, y)
        scores = np.abs(model.coef_)
    else:
        raise ValueError(
            f"Unknown selection method {method!r}; use none, pearson, mutual_info, or elasticnet"
        )

    order = np.argsort(scores)[::-1]
    keep = np.zeros(n_features, dtype=bool)
    keep[order[:top_k]] = True
    if not keep.any():
        keep[np.arange(min(top_k, n_features))] = True
    return keep
