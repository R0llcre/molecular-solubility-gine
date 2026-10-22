from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chempath_gine.data import make_splits
from chempath_gine.training.metrics import regression_metrics


def _parse_list(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def _xy(df: pd.DataFrame, descriptor_cols: list[str], target_col: str):
    x = df[descriptor_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(x).all(axis=1) & np.isfinite(y)
    return x[mask], y[mask], df.loc[mask].copy()


def _build_model(name: str, seed: int):
    if name == "ridge":
        return RidgeCV(alphas=np.logspace(-4, 4, 25))
    if name == "elasticnet":
        return ElasticNetCV(
            alphas=np.logspace(-4, 2, 20),
            l1_ratio=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9],
            cv=5,
            max_iter=20000,
            random_state=seed,
        )
    if name == "svr":
        return SVR(C=10.0, epsilon=0.05, gamma="scale")
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=800,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=1,
        )
    if name == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError("xgboost is required for --models xgb") from exc
        return XGBRegressor(
            n_estimators=1500,
            learning_rate=0.02,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            early_stopping_rounds=80,
        )
    raise ValueError(f"Unknown descriptor baseline {name!r}")


def _fit_predict(model, name: str, x_train, y_train, x_val, y_val, x_test):
    if name == "xgb":
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    else:
        model.fit(x_train, y_train)
    return model.predict(x_test).reshape(-1)


def _write_predictions(path: Path, test_df: pd.DataFrame, pred: np.ndarray, target_col: str):
    out = test_df.copy()
    out["y_true"] = out[target_col].astype(float)
    out["y_pred"] = pred.astype(float)
    out["absolute_error"] = np.abs(out["y_true"].to_numpy() - out["y_pred"].to_numpy())
    out.to_csv(path, index=False)


def _write_aggregate(summary_path: Path, aggregate_path: Path) -> None:
    df = pd.read_csv(summary_path)
    metrics = ["rmse", "mae", "r2", "pearson", "spearman"]
    rows = []
    for model, group in df.groupby("model", sort=False):
        row = {"model": model, "n": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(aggregate_path, index=False)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", default="rf,xgb")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--out", default="baseline_runs")
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    descriptor_cols = list(cfg["data"].get("descriptor_cols") or [])
    if not descriptor_cols:
        raise ValueError("Descriptor baselines require data.descriptor_cols")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "descriptor_baseline_summary.csv"
    aggregate_path = out_dir / "descriptor_baseline_aggregate.csv"
    if summary_path.exists() and aggregate_path.exists():
        print(f"resume existing {summary_path}")
        print(f"resume existing {aggregate_path}")
        return
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed_text in _parse_list(args.seeds):
        seed = int(seed_text)
        split_paths = make_splits(
            csv_path=cfg["data"]["csv"],
            out_dir=out_dir / "splits" / f"seed{seed}",
            smiles_col=cfg["data"]["smiles_col"],
            split=cfg["data"]["split"],
            train_ratio=float(cfg["data"]["train_ratio"]),
            val_ratio=float(cfg["data"]["val_ratio"]),
            seed=seed,
        )
        train_df = pd.read_csv(split_paths["train"])
        val_df = pd.read_csv(split_paths["val"])
        test_df = pd.read_csv(split_paths["test"])
        x_train, y_train, train_df = _xy(train_df, descriptor_cols, cfg["data"]["target_col"])
        x_val, y_val, _ = _xy(val_df, descriptor_cols, cfg["data"]["target_col"])
        x_test, y_test, test_df = _xy(test_df, descriptor_cols, cfg["data"]["target_col"])

        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

        for model_name in _parse_list(args.models):
            model = _build_model(model_name, seed)
            pred = _fit_predict(model, model_name, x_train, y_train, x_val, y_val, x_test)
            metrics = regression_metrics(pred, y_test)
            pred_path = pred_dir / f"{model_name}_seed{seed}.csv"
            _write_predictions(pred_path, test_df, pred, cfg["data"]["target_col"])
            rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "split": cfg["data"]["split"],
                    "n_train": len(y_train),
                    "n_val": len(y_val),
                    "n_test": len(y_test),
                    "prediction_path": str(pred_path),
                    **metrics,
                }
            )

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _write_aggregate(summary_path, aggregate_path)
    print(f"wrote {summary_path}")
    print(f"wrote {aggregate_path}")


if __name__ == "__main__":
    main()
