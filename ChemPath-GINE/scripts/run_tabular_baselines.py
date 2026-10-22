from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import RDLogger
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chempath_gine.baselines.tabular import (
    build_model,
    build_preprocessor,
    fit_predict,
    make_tabular_features,
    parse_feature_spec,
)
from chempath_gine.data import make_splits
from chempath_gine.training.metrics import regression_metrics


def _parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _target(df: pd.DataFrame, target_col: str) -> np.ndarray:
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError(f"Non-finite target values found in {target_col}")
    return y


def _write_predictions(path: Path, test_df: pd.DataFrame, pred: np.ndarray, target_col: str) -> None:
    out = test_df.copy()
    out["y_true"] = pd.to_numeric(out[target_col], errors="coerce").astype(float)
    out["y_pred"] = pred.astype(float)
    out["absolute_error"] = np.abs(out["y_true"].to_numpy() - out["y_pred"].to_numpy())
    out.to_csv(path, index=False)


def _write_aggregate(summary_path: Path, aggregate_path: Path) -> None:
    df = pd.read_csv(summary_path)
    rows = []
    metrics = ["rmse", "mae", "r2", "pearson", "spearman", "n_features"]
    for (feature_set, model), group in df.groupby(["feature_set", "model"], sort=False):
        row = {"feature_set": feature_set, "model": model, "n": len(group)}
        for metric in metrics:
            if metric in group:
                row[f"{metric}_mean"] = group[metric].mean()
                row[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(aggregate_path, index=False)


def _write_rows(summary_path: Path, aggregate_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_aggregate(summary_path, aggregate_path)


def main(argv=None):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    RDLogger.DisableLog("rdApp.*")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--feature-sets",
        default=None,
    )
    parser.add_argument("--models", default=None)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--out", default="tabular_runs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tabular_cfg = cfg.get("tabular", {})
    feature_sets = args.feature_sets or ",".join(
        tabular_cfg.get(
            "feature_sets",
            [
                "desc16",
                "morgan2",
                "desc16_morgan2",
                "desc16_morgan2_maccs",
                "rdkit2d_selected32_morgan2_maccs",
            ],
        )
    )
    models = args.models or ",".join(tabular_cfg.get("models", ["ridge", "elasticnet", "svr", "rf", "xgb"]))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "tabular_summary.csv"
    aggregate_path = out_dir / "tabular_aggregate.csv"
    rows = []
    done = set()
    if summary_path.exists() and not args.overwrite:
        existing = pd.read_csv(summary_path)
        rows = existing.to_dict("records")
        for row in rows:
            done.add((str(row["feature_set"]), str(row["model"]), int(row["seed"])))
        print(f"resume {len(rows)} existing rows from {summary_path}")

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

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
        y_train = _target(train_df, cfg["data"]["target_col"])
        y_val = _target(val_df, cfg["data"]["target_col"])
        y_test = _target(test_df, cfg["data"]["target_col"])

        for feature_set in _parse_list(feature_sets):
            spec = parse_feature_spec(feature_set)
            x_train_raw, x_val_raw, x_test_raw, vocab = make_tabular_features(
                train_df,
                val_df,
                test_df,
                smiles_col=cfg["data"]["smiles_col"],
                spec=spec,
            )
            pre = build_preprocessor(spec)
            x_train = pre.fit_transform(x_train_raw, y_train)
            x_val = pre.transform(x_val_raw)
            x_test = pre.transform(x_test_raw)

            for model_name in _parse_list(models):
                key = (feature_set, model_name, seed)
                if key in done:
                    print(f"resume existing feature_set={feature_set} model={model_name} seed={seed}")
                    continue
                model = build_model(model_name, seed)
                pred = fit_predict(model, model_name, x_train, y_train, x_val, y_val, x_test)
                metrics = regression_metrics(pred, y_test)
                pred_path = pred_dir / f"{feature_set}_{model_name}_seed{seed}.csv"
                _write_predictions(pred_path, test_df, pred, cfg["data"]["target_col"])
                row = (
                    {
                        "feature_set": feature_set,
                        "model": model_name,
                        "seed": seed,
                        "split": cfg["data"]["split"],
                        "n_train": len(y_train),
                        "n_val": len(y_val),
                        "n_test": len(y_test),
                        "n_features_raw": int(x_train_raw.shape[1]),
                        "n_features": int(x_train.shape[1]),
                        "selected_features": "|".join(pre.selected_columns_[:200]),
                        "brics_vocab_size": len(vocab),
                        "prediction_path": str(pred_path),
                        **metrics,
                    }
                )
                rows.append(row)
                done.add(key)
                _write_rows(summary_path, aggregate_path, rows)

    if not rows:
        raise RuntimeError("No tabular runs were executed")
    _write_rows(summary_path, aggregate_path, rows)
    print(f"wrote {summary_path}")
    print(f"wrote {aggregate_path}")


if __name__ == "__main__":
    main()
