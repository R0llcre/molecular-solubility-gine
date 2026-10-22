from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, ElasticNetCV, RidgeCV
from sklearn.model_selection import KFold, train_test_split

from molfuse_gine.baselines.tabular import (
    build_model,
    build_preprocessor,
    fit_predict,
    make_tabular_features,
    _make_raw_features,
    parse_feature_spec,
)
from molfuse_gine.training.metrics import regression_metrics


@dataclass(frozen=True)
class ExpertSpec:
    feature_set: str
    model: str

    @property
    def name(self) -> str:
        return f"{self.feature_set}__{self.model}"


def parse_expert_specs(value: str) -> list[ExpertSpec]:
    specs: list[ExpertSpec] = []
    for item in [part.strip() for part in value.split(",") if part.strip()]:
        if ":" in item:
            feature_set, model = item.split(":", 1)
        else:
            feature_set, model = item, "xgb"
        specs.append(ExpertSpec(feature_set.strip(), model.strip()))
    if not specs:
        raise ValueError("At least one expert spec is required")
    return specs


def build_meta_model(name: str, seed: int):
    key = name.lower()
    if key == "ridge":
        return RidgeCV(alphas=np.logspace(-4, 4, 41))
    if key == "elasticnet":
        return ElasticNetCV(
            alphas=np.logspace(-4, 2, 25),
            l1_ratio=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9],
            cv=5,
            max_iter=20000,
            random_state=seed,
            precompute=False,
        )
    if key == "bayesian_ridge":
        return BayesianRidge()
    raise ValueError(f"Unknown meta model {name!r}; use ridge, elasticnet, or bayesian_ridge")


def run_tabular_oof_stack(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    smiles_col: str,
    target_col: str,
    expert_specs: list[ExpertSpec],
    out_dir: Path,
    seed: int,
    folds: int = 5,
    meta_model_name: str = "ridge",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    y_train = pd.to_numeric(train_df[target_col], errors="coerce").to_numpy(dtype=np.float64)
    y_val = pd.to_numeric(val_df[target_col], errors="coerce").to_numpy(dtype=np.float64)
    y_test = pd.to_numeric(test_df[target_col], errors="coerce").to_numpy(dtype=np.float64)

    if not np.isfinite(y_train).all() or not np.isfinite(y_val).all() or not np.isfinite(y_test).all():
        raise ValueError("OOF stacking requires finite train/val/test targets")

    n_splits = min(max(2, int(folds)), len(train_df))
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    expert_train = pd.DataFrame({"source_index": train_df.get("source_index", train_df.index), "y_true": y_train})
    expert_val = pd.DataFrame({"source_index": val_df.get("source_index", val_df.index), "y_true": y_val})
    expert_test = pd.DataFrame({"source_index": test_df.get("source_index", test_df.index), "y_true": y_test})

    selected_rows = []
    for expert in expert_specs:
        spec = parse_feature_spec(expert.feature_set)
        oof = np.zeros(len(train_df), dtype=np.float64)
        val_fold_preds = []
        test_fold_preds = []
        feature_counts = []
        vocab_sizes = []

        for fold_idx, (fit_idx, hold_idx) in enumerate(kfold.split(np.arange(len(train_df))), 1):
            if len(fit_idx) >= 4:
                inner_fit_idx, inner_eval_idx = train_test_split(
                    fit_idx,
                    test_size=max(1, int(round(len(fit_idx) * 0.15))),
                    random_state=seed + fold_idx,
                    shuffle=True,
                )
            else:
                inner_fit_idx = fit_idx
                inner_eval_idx = fit_idx

            fit_df = train_df.iloc[inner_fit_idx].reset_index(drop=True)
            eval_df = train_df.iloc[inner_eval_idx].reset_index(drop=True)
            hold_df = train_df.iloc[hold_idx].reset_index(drop=True)
            x_fit_raw, x_eval_raw, x_hold_raw, inner_vocab = make_tabular_features(
                fit_df, eval_df, hold_df, smiles_col=smiles_col, spec=spec
            )
            x_val_raw_inner = _make_raw_features(
                val_df, smiles_col, spec, vocab=inner_vocab
            ).reset_index(drop=True)
            x_test_raw_inner = _make_raw_features(
                test_df, smiles_col, spec, vocab=inner_vocab
            ).reset_index(drop=True)
            y_fit = y_train[inner_fit_idx]
            y_eval = y_train[inner_eval_idx]
            pre = build_preprocessor(spec)
            x_fit = pre.fit_transform(x_fit_raw, y_fit)
            x_eval = pre.transform(x_eval_raw)
            x_hold = pre.transform(x_hold_raw)
            x_val = pre.transform(x_val_raw_inner)
            x_test = pre.transform(x_test_raw_inner)
            model = build_model(expert.model, seed + fold_idx)
            oof[hold_idx] = fit_predict(
                model, expert.model, x_fit, y_fit, x_eval, y_eval, x_hold
            )
            val_fold_preds.append(model.predict(x_val).reshape(-1))
            test_fold_preds.append(model.predict(x_test).reshape(-1))
            feature_counts.append(int(x_fit.shape[1]))
            vocab_sizes.append(len(inner_vocab))
            selected_rows.append(
                {
                    "expert": expert.name,
                    "fold": fold_idx,
                    "n_features": int(x_fit.shape[1]),
                    "selected_features": "|".join(pre.selected_columns_[:200]),
                    "brics_vocab_size": len(inner_vocab),
                }
            )

        expert_train[expert.name] = oof
        expert_val[expert.name] = np.mean(np.vstack(val_fold_preds), axis=0)
        expert_test[expert.name] = np.mean(np.vstack(test_fold_preds), axis=0)
        selected_rows.append(
            {
                "expert": expert.name,
                "fold": "mean",
                "n_features": float(np.mean(feature_counts)),
                "selected_features": "",
                "brics_vocab_size": float(np.mean(vocab_sizes)) if vocab_sizes else 0.0,
            }
        )

    expert_cols = [expert.name for expert in expert_specs]
    meta = build_meta_model(meta_model_name, seed)
    meta.fit(expert_train[expert_cols].to_numpy(dtype=np.float64), y_train)
    train_pred = meta.predict(expert_train[expert_cols].to_numpy(dtype=np.float64)).reshape(-1)
    val_pred = meta.predict(expert_val[expert_cols].to_numpy(dtype=np.float64)).reshape(-1)
    test_pred = meta.predict(expert_test[expert_cols].to_numpy(dtype=np.float64)).reshape(-1)
    expert_train["stack_pred"] = train_pred
    expert_val["stack_pred"] = val_pred
    expert_test["stack_pred"] = test_pred
    expert_train["stack_absolute_error"] = np.abs(y_train - train_pred)
    expert_val["stack_absolute_error"] = np.abs(y_val - val_pred)
    expert_test["stack_absolute_error"] = np.abs(y_test - test_pred)

    expert_train.to_csv(out_dir / "expert_oof_train.csv", index=False)
    expert_val.to_csv(out_dir / "expert_val_predictions.csv", index=False)
    expert_test.to_csv(out_dir / "expert_test_predictions.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(out_dir / "expert_selected_features.csv", index=False)

    coef = getattr(meta, "coef_", np.full(len(expert_cols), np.nan))
    coef_df = pd.DataFrame({"expert": expert_cols, "coefficient": np.asarray(coef).reshape(-1)})
    coef_df.to_csv(out_dir / "meta_coefficients.csv", index=False)

    val_metrics = regression_metrics(val_pred, y_val)
    test_metrics = regression_metrics(test_pred, y_test)
    summary = {
        "seed": int(seed),
        "folds": int(n_splits),
        "meta_model": meta_model_name,
        "experts": ",".join(expert_cols),
        **{f"val_{key}": value for key, value in val_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    with (out_dir / "stack_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
