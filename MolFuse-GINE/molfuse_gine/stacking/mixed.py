from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from molfuse_gine.stacking.oof import build_meta_model
from molfuse_gine.training.metrics import regression_metrics


class SimpleAverageRegressor:
    def __init__(self):
        self.coef_: np.ndarray | None = None
        self.intercept_ = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray):
        del y
        self.coef_ = np.full(x.shape[1], 1.0 / max(1, x.shape[1]), dtype=np.float64)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("SimpleAverageRegressor must be fit before predict")
        return x @ self.coef_


def build_mixed_meta_model(name: str, seed: int):
    key = name.lower()
    if key in {"average", "mean", "simple_average"}:
        return SimpleAverageRegressor()
    if key in {"nonnegative", "nnls", "positive_linear"}:
        return LinearRegression(positive=True)
    return build_meta_model(key, seed)


@dataclass(frozen=True)
class MixedStackResult:
    summary: dict
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    coefficients: pd.DataFrame
    correlation: pd.DataFrame


def load_expert_prediction_dir(path: Path, split: str) -> pd.DataFrame:
    file_map = {
        "train": "expert_oof_train.csv",
        "val": "expert_val_predictions.csv",
        "test": "expert_test_predictions.csv",
    }
    if split not in file_map:
        raise ValueError(f"Unknown split={split!r}")
    file = path / file_map[split]
    if not file.exists():
        raise FileNotFoundError(f"Missing {split} prediction file: {file}")
    df = pd.read_csv(file)
    required = {"source_index", "y_true"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"{file} is missing columns: {sorted(missing)}")
    return df


def prediction_columns(df: pd.DataFrame) -> list[str]:
    ignored = {"source_index", "y_true", "absolute_error", "stack_absolute_error", "stack_pred"}
    ignored_suffixes = ("_true",)
    cols = [
        col
        for col in df.columns
        if col not in ignored and not any(col.endswith(suffix) for suffix in ignored_suffixes)
    ]
    if not cols:
        raise ValueError("Expert prediction table contains no prediction columns")
    return cols


def _safe_expert_prefix(path: Path) -> str:
    parts = path.parts[-3:] if len(path.parts) >= 3 else path.parts
    prefix = "__".join(parts)
    for char in (",", ":", " ", "/", "\\"):
        prefix = prefix.replace(char, "_")
    return prefix


def merge_expert_predictions(paths: list[Path], split: str) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    expected_index: pd.Index | None = None
    expected_y: pd.Series | None = None
    for path in paths:
        df = load_expert_prediction_dir(path, split)
        if df["source_index"].duplicated().any():
            dup = df.loc[df["source_index"].duplicated(), "source_index"].head(5).tolist()
            raise ValueError(f"{path} {split} has duplicate source_index values: {dup}")
        df = df.sort_values("source_index").reset_index(drop=True)
        current_index = pd.Index(df["source_index"])
        current_y = pd.to_numeric(df["y_true"], errors="raise").reset_index(drop=True)
        if expected_index is None:
            expected_index = current_index
            expected_y = current_y
        else:
            if not current_index.equals(expected_index):
                raise ValueError(f"{path} {split} source_index set/order does not match other experts")
            assert expected_y is not None
            if not np.allclose(current_y.to_numpy(dtype=float), expected_y.to_numpy(dtype=float), atol=1e-8):
                raise ValueError(f"{path} {split} y_true does not match other experts")
        cols = prediction_columns(df)
        prefix = _safe_expert_prefix(path)
        rename = {col: f"{prefix}__{col}" for col in cols}
        if len(set(rename.values())) != len(rename):
            raise ValueError(f"{path} {split} produced duplicate expert column names")
        if merged is not None:
            overlap = set(rename.values()) & set(merged.columns)
            if overlap:
                raise ValueError(f"{path} {split} expert columns collide after prefixing: {sorted(overlap)[:5]}")
        slim = df[["source_index", "y_true", *cols]].rename(columns=rename)
        if merged is None:
            merged = slim
        else:
            merged = merged.merge(
                slim.drop(columns=["y_true"]),
                on="source_index",
                how="inner",
                validate="one_to_one",
            )
    if merged is None:
        raise ValueError("No expert prediction dirs were provided")
    if expected_index is not None and len(merged) != len(expected_index):
        raise ValueError(f"Merged {split} expert table lost rows: {len(merged)} != {len(expected_index)}")
    return merged


def run_mixed_prediction_stack(
    expert_dirs: list[Path],
    out_dir: Path,
    meta_model_name: str = "ridge",
    seed: int = 1,
    stack_name: str | None = None,
) -> MixedStackResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    train = merge_expert_predictions(expert_dirs, "train")
    val = merge_expert_predictions(expert_dirs, "val")
    test = merge_expert_predictions(expert_dirs, "test")
    split_sets = {
        "train": set(train["source_index"]),
        "val": set(val["source_index"]),
        "test": set(test["source_index"]),
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_sets[left] & split_sets[right]
        if overlap:
            sample = sorted(overlap)[:5]
            raise ValueError(f"{left}/{right} source_index overlap in mixed stack: {sample}")
    expert_cols = [col for col in train.columns if col not in {"source_index", "y_true"}]
    if not expert_cols:
        raise ValueError("Mixed stack needs at least one expert prediction column")

    model = build_mixed_meta_model(meta_model_name, seed)
    x_train = train[expert_cols].to_numpy(dtype=np.float64)
    y_train = train["y_true"].to_numpy(dtype=np.float64)
    model.fit(x_train, y_train)

    for df in (train, val, test):
        pred = model.predict(df[expert_cols].to_numpy(dtype=np.float64)).reshape(-1)
        df["stack_pred"] = pred
        df["stack_absolute_error"] = np.abs(df["y_true"].to_numpy(dtype=np.float64) - pred)

    coef = getattr(model, "coef_", np.full(len(expert_cols), np.nan))
    coef_df = pd.DataFrame({"expert": expert_cols, "coefficient": np.asarray(coef).reshape(-1)})
    corr_df = train[expert_cols].corr()

    summary = {
        "meta_model": meta_model_name,
        "stack_name": stack_name or out_dir.parent.name,
        "seed": int(seed),
        "n_expert_columns": len(expert_cols),
        "expert_dirs": [str(path) for path in expert_dirs],
        **{
            f"train_{key}": value
            for key, value in regression_metrics(train["stack_pred"], train["y_true"]).items()
        },
        **{f"val_{key}": value for key, value in regression_metrics(val["stack_pred"], val["y_true"]).items()},
        **{
            f"test_{key}": value
            for key, value in regression_metrics(test["stack_pred"], test["y_true"]).items()
        },
    }

    train.to_csv(out_dir / "expert_oof_train.csv", index=False)
    val.to_csv(out_dir / "expert_val_predictions.csv", index=False)
    test.to_csv(out_dir / "expert_test_predictions.csv", index=False)
    train.to_csv(out_dir / "merged_expert_oof_train.csv", index=False)
    val.to_csv(out_dir / "merged_expert_val_predictions.csv", index=False)
    test.to_csv(out_dir / "merged_stack_test_predictions.csv", index=False)
    test.to_csv(out_dir / "merged_expert_test_predictions.csv", index=False)
    coef_df.to_csv(out_dir / "meta_coefficients.csv", index=False)
    corr_df.to_csv(out_dir / "expert_correlation.csv")
    with (out_dir / "mixed_stack_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return MixedStackResult(
        summary=summary,
        train=train,
        val=val,
        test=test,
        coefficients=coef_df,
        correlation=corr_df,
    )
