from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from molfuse_gine.stacking.mixed import build_mixed_meta_model, merge_expert_predictions
from molfuse_gine.training.metrics import regression_metrics


@dataclass(frozen=True)
class ExpertPruningSpec:
    top_k: int = 6
    max_experts: int = 10
    corr_threshold: float = 0.95
    oof_rmse_slack: float = 0.15
    include_patterns: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class PrunedStackResult:
    summary: dict
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    coefficients: pd.DataFrame
    selection: pd.DataFrame
    leave_one_out: pd.DataFrame


def _compile(patterns: tuple[str, ...]) -> list[re.Pattern]:
    return [re.compile(pattern) for pattern in patterns if pattern]


def _matches(value: str, patterns: list[re.Pattern]) -> bool:
    return any(pattern.search(value) for pattern in patterns)


def _prediction_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in {"source_index", "y_true"}]


def score_experts(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = train["y_true"].to_numpy(dtype=np.float64)
    for col in _prediction_columns(train):
        pred = train[col].to_numpy(dtype=np.float64)
        metrics = regression_metrics(pred, y)
        rows.append(
            {
                "expert": col,
                "oof_rmse": metrics["rmse"],
                "oof_mae": metrics["mae"],
                "oof_r2": metrics["r2"],
            }
        )
    if not rows:
        raise ValueError("No expert columns available for pruning")
    return pd.DataFrame(rows).sort_values(["oof_rmse", "expert"]).reset_index(drop=True)


def select_experts(train: pd.DataFrame, spec: ExpertPruningSpec) -> pd.DataFrame:
    scores = score_experts(train)
    corr = train[_prediction_columns(train)].corr().abs().fillna(0.0)
    include = _compile(spec.include_patterns)
    exclude = _compile(spec.exclude_patterns)
    best_rmse = float(scores["oof_rmse"].min())
    selected: list[str] = []
    reasons: dict[str, str] = {}

    for expert in scores["expert"]:
        if exclude and _matches(expert, exclude):
            reasons[expert] = "excluded_by_pattern"
            continue
        if include and _matches(expert, include):
            selected.append(expert)
            reasons[expert] = "include_pattern"

    for _, row in scores.head(max(0, spec.top_k)).iterrows():
        expert = str(row["expert"])
        if expert not in selected and not (exclude and _matches(expert, exclude)):
            selected.append(expert)
            reasons[expert] = "top_oof"

    for _, row in scores.iterrows():
        expert = str(row["expert"])
        if expert in selected or (exclude and _matches(expert, exclude)):
            continue
        if len(selected) >= spec.max_experts:
            break
        rmse = float(row["oof_rmse"])
        if rmse > best_rmse + float(spec.oof_rmse_slack):
            reasons[expert] = "outside_oof_slack"
            continue
        max_corr = max((float(corr.loc[expert, prev]) for prev in selected), default=0.0)
        if max_corr < float(spec.corr_threshold):
            selected.append(expert)
            reasons[expert] = "diverse_low_corr"
        else:
            reasons[expert] = "high_corr"

    if not selected:
        selected = [str(scores.iloc[0]["expert"])]
        reasons[selected[0]] = "fallback_best_oof"

    selected_set = set(selected[: spec.max_experts])
    rows = []
    for _, row in scores.iterrows():
        expert = str(row["expert"])
        max_corr = max((float(corr.loc[expert, prev]) for prev in selected_set if prev != expert), default=0.0)
        rows.append(
            {
                **row.to_dict(),
                "selected": expert in selected_set,
                "selection_reason": reasons.get(expert, "not_selected"),
                "max_abs_corr_to_selected": max_corr,
            }
        )
    return pd.DataFrame(rows).sort_values(["selected", "oof_rmse"], ascending=[False, True])


def _fit_predict(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, cols: list[str], meta_model: str, seed: int):
    model = build_mixed_meta_model(meta_model, seed)
    y = train["y_true"].to_numpy(dtype=np.float64)
    model.fit(train[cols].to_numpy(dtype=np.float64), y)
    return (
        model,
        model.predict(train[cols].to_numpy(dtype=np.float64)).reshape(-1),
        model.predict(val[cols].to_numpy(dtype=np.float64)).reshape(-1),
        model.predict(test[cols].to_numpy(dtype=np.float64)).reshape(-1),
    )


def leave_one_expert_out(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    meta_model: str,
    seed: int,
    base_metrics: dict,
) -> pd.DataFrame:
    rows = []
    if len(cols) <= 1:
        return pd.DataFrame(
            columns=[
                "dropped_expert",
                "n_experts",
                "val_rmse",
                "test_rmse",
                "delta_val_rmse",
                "delta_test_rmse_posthoc",
            ]
        )
    for dropped in cols:
        kept = [col for col in cols if col != dropped]
        _, _, val_pred, test_pred = _fit_predict(train, val, test, kept, meta_model, seed)
        val_metrics = regression_metrics(val_pred, val["y_true"])
        test_metrics = regression_metrics(test_pred, test["y_true"])
        rows.append(
            {
                "dropped_expert": dropped,
                "n_experts": len(kept),
                "val_rmse": val_metrics["rmse"],
                "test_rmse": test_metrics["rmse"],
                "delta_val_rmse": val_metrics["rmse"] - base_metrics["val_rmse"],
                "delta_test_rmse_posthoc": test_metrics["rmse"] - base_metrics["test_rmse"],
            }
        )
    return pd.DataFrame(rows).sort_values("delta_val_rmse", ascending=False)


def run_pruned_prediction_stack(
    expert_dirs: list[Path],
    out_dir: Path,
    pruning: ExpertPruningSpec | None = None,
    meta_model_name: str = "ridge",
    seed: int = 1,
    stack_name: str | None = None,
    requested_expert_dirs: list[Path] | None = None,
    missing_expert_dirs: list[Path] | None = None,
) -> PrunedStackResult:
    pruning = pruning or ExpertPruningSpec()
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
            raise ValueError(f"{left}/{right} source_index overlap in pruned stack: {sample}")
    n_candidate_experts = len(_prediction_columns(train))
    selection = select_experts(train, pruning)
    cols = selection.loc[selection["selected"], "expert"].astype(str).tolist()
    _, train_pred, val_pred, test_pred = _fit_predict(train, val, test, cols, meta_model_name, seed)

    train = train[["source_index", "y_true", *cols]].copy()
    val = val[["source_index", "y_true", *cols]].copy()
    test = test[["source_index", "y_true", *cols]].copy()
    for df, pred in ((train, train_pred), (val, val_pred), (test, test_pred)):
        df["stack_pred"] = pred
        df["stack_absolute_error"] = np.abs(df["y_true"].to_numpy(dtype=np.float64) - pred)

    train_metrics = regression_metrics(train["stack_pred"], train["y_true"])
    val_metrics = regression_metrics(val["stack_pred"], val["y_true"])
    test_metrics = regression_metrics(test["stack_pred"], test["y_true"])
    model = build_mixed_meta_model(meta_model_name, seed)
    model.fit(train[cols].to_numpy(dtype=np.float64), train["y_true"].to_numpy(dtype=np.float64))
    coef = getattr(model, "coef_", np.full(len(cols), np.nan))
    coefficients = pd.DataFrame({"expert": cols, "coefficient": np.asarray(coef).reshape(-1)})
    intercept = float(getattr(model, "intercept_", np.nan))
    coefficients = pd.concat(
        [
            pd.DataFrame({"expert": ["__intercept__"], "coefficient": [intercept]}),
            coefficients,
        ],
        ignore_index=True,
    )
    coefficients_ranked = coefficients.assign(abs_coefficient=coefficients["coefficient"].abs()).sort_values(
        "abs_coefficient", ascending=False
    )
    loo = leave_one_expert_out(
        train,
        val,
        test,
        cols,
        meta_model_name,
        seed,
        {"val_rmse": val_metrics["rmse"], "test_rmse": test_metrics["rmse"]},
    )
    summary = {
        "meta_model": meta_model_name,
        "stack_name": stack_name or out_dir.parent.name,
        "seed": int(seed),
        "n_candidate_experts": n_candidate_experts,
        "n_selected_experts": len(cols),
        "n_requested_expert_dirs": len(requested_expert_dirs or expert_dirs),
        "n_existing_expert_dirs": len(expert_dirs),
        "n_missing_expert_dirs": len(missing_expert_dirs or []),
        "expert_dirs": [str(path) for path in expert_dirs],
        "requested_expert_dirs": [str(path) for path in (requested_expert_dirs or expert_dirs)],
        "missing_expert_dirs": [str(path) for path in (missing_expert_dirs or [])],
        "selected_experts": cols,
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"val_{key}": value for key, value in val_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }

    train.to_csv(out_dir / "expert_oof_train.csv", index=False)
    val.to_csv(out_dir / "expert_val_predictions.csv", index=False)
    test.to_csv(out_dir / "expert_test_predictions.csv", index=False)
    test.to_csv(out_dir / "merged_stack_test_predictions.csv", index=False)
    selection.to_csv(out_dir / "expert_selection_report.csv", index=False)
    train[cols].corr().to_csv(out_dir / "expert_correlation.csv")
    coefficients.to_csv(out_dir / "meta_coefficients.csv", index=False)
    coefficients_ranked.to_csv(out_dir / "meta_coefficients_ranked.csv", index=False)
    loo.to_csv(out_dir / "leave_one_expert_out.csv", index=False)
    with (out_dir / "pruned_stack_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return PrunedStackResult(
        summary=summary,
        train=train,
        val=val,
        test=test,
        coefficients=coefficients,
        selection=selection,
        leave_one_out=loo,
    )
