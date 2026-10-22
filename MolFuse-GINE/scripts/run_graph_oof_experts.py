from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from molfuse_gine.data import make_splits
from molfuse_gine.training.predict import run_prediction
from molfuse_gine.training.train import main as train_main
from molfuse_gine.training.metrics import regression_metrics


def _parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _prediction_series(path: Path, pred_col: str = "y_pred") -> pd.Series:
    df = pd.read_csv(path)
    if "source_index" not in df.columns:
        raise KeyError(f"{path} must contain source_index")
    return df.set_index("source_index")[pred_col].astype(float)


def _make_inner_train_val(
    fit_df: pd.DataFrame,
    val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(fit_df) < 4:
        raise ValueError("Graph OOF fold is too small to create an inner validation split")
    train_idx, val_idx = train_test_split(
        np.arange(len(fit_df)),
        test_size=max(1, int(round(len(fit_df) * val_ratio))),
        random_state=seed,
        shuffle=True,
    )
    return fit_df.iloc[train_idx].reset_index(drop=True), fit_df.iloc[val_idx].reset_index(drop=True)


def _write_explicit_config(
    base: dict,
    cfg_path: Path,
    run_name: str,
    output_dir: Path,
    variant: str,
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    seed: int,
) -> Path:
    cfg = copy.deepcopy(base)
    cfg["run"]["seed"] = int(seed)
    cfg["run"]["name"] = run_name
    cfg["run"]["output_dir"] = str(output_dir)
    cfg["model"]["variant"] = variant
    cfg["data"]["train_csv"] = str(train_csv)
    cfg["data"]["val_csv"] = str(val_csv)
    cfg["data"]["test_csv"] = str(test_csv)
    cfg["data"]["cache_dir"] = str((output_dir / "cache" / run_name).resolve())
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path


def _average_prediction_series(series_list: list[pd.Series]) -> pd.Series:
    df = pd.concat(series_list, axis=1)
    return df.mean(axis=1)


def _completed_expert(expert_dir: Path) -> bool:
    required = [
        expert_dir / "expert_oof_train.csv",
        expert_dir / "expert_val_predictions.csv",
        expert_dir / "expert_test_predictions.csv",
        expert_dir / "graph_oof_summary.json",
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def _load_completed_row(expert_dir: Path) -> dict:
    with (expert_dir / "graph_oof_summary.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _cleanup_fold_artifacts(expert_dir: Path, run_dir: Path) -> None:
    # OOF scripts only need the fold predictions after each fold. Training
    # checkpoints and PyG dataset caches are large and can fill small VMs.
    shutil.rmtree(run_dir, ignore_errors=True)
    shutil.rmtree(expert_dir / "train_runs" / "cache", ignore_errors=True)
    for cache_dir in (expert_dir / "predictions").glob("*_predict_cache"):
        shutil.rmtree(cache_dir, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variants", default="gine_desc")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", default="graph_oof_runs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "graph_oof_summary.csv"
    if summary_path.exists() and not args.overwrite:
        print(f"resume existing {summary_path}")
        return

    rows = []
    for seed_text in _parse_list(args.seeds):
        seed = int(seed_text)
        split_paths = make_splits(
            csv_path=base["data"]["csv"],
            out_dir=out_dir / "splits" / f"seed{seed}",
            smiles_col=base["data"]["smiles_col"],
            split=base["data"]["split"],
            train_ratio=float(base["data"]["train_ratio"]),
            val_ratio=float(base["data"]["val_ratio"]),
            seed=seed,
        )
        outer_train = pd.read_csv(split_paths["train"])
        outer_val = pd.read_csv(split_paths["val"])
        outer_test = pd.read_csv(split_paths["test"])
        y_train = pd.to_numeric(outer_train[base["data"]["target_col"]], errors="coerce")
        y_val = pd.to_numeric(outer_val[base["data"]["target_col"]], errors="coerce")
        y_test = pd.to_numeric(outer_test[base["data"]["target_col"]], errors="coerce")

        for variant in _parse_list(args.variants):
            expert_name = f"{variant}"
            expert_dir = out_dir / "runs" / f"seed{seed}_{variant}"
            if _completed_expert(expert_dir) and not args.overwrite:
                print(f"resume completed graph expert {expert_dir}")
                rows.append(_load_completed_row(expert_dir))
                continue

            cfg_dir = expert_dir / "configs"
            data_dir = expert_dir / "fold_data"
            pred_dir = expert_dir / "predictions"
            pred_dir.mkdir(parents=True, exist_ok=True)
            oof = pd.Series(index=outer_train["source_index"], dtype=float, name=expert_name)
            val_fold_preds: list[pd.Series] = []
            test_fold_preds: list[pd.Series] = []

            kfold = KFold(n_splits=min(max(2, args.folds), len(outer_train)), shuffle=True, random_state=seed)
            for fold_idx, (fit_idx, hold_idx) in enumerate(kfold.split(outer_train), 1):
                fit_df = outer_train.iloc[fit_idx].reset_index(drop=True)
                hold_df = outer_train.iloc[hold_idx].reset_index(drop=True)
                inner_train, inner_val = _make_inner_train_val(
                    fit_df, float(base["data"].get("val_ratio", 0.15)), seed + fold_idx
                )
                fold_data = data_dir / f"fold{fold_idx}"
                fold_data.mkdir(parents=True, exist_ok=True)
                train_csv = fold_data / "train.csv"
                val_csv = fold_data / "val.csv"
                hold_csv = fold_data / "holdout.csv"
                outer_val_csv = fold_data / "outer_val.csv"
                outer_test_csv = fold_data / "outer_test.csv"
                inner_train.to_csv(train_csv, index=False)
                inner_val.to_csv(val_csv, index=False)
                hold_df.to_csv(hold_csv, index=False)
                outer_val.to_csv(outer_val_csv, index=False)
                outer_test.to_csv(outer_test_csv, index=False)
                cfg_path = _write_explicit_config(
                    base,
                    cfg_dir / f"fold{fold_idx}.yaml",
                    f"graph_oof_seed{seed}_{variant}_fold{fold_idx}",
                    expert_dir / "train_runs",
                    variant,
                    train_csv,
                    val_csv,
                    hold_csv,
                    seed + fold_idx,
                )
                summary = train_main(["--config", str(cfg_path)])
                run_dir = Path(summary["run_dir"])
                hold_pred = _prediction_series(run_dir / "test_predictions.csv")
                oof.loc[hold_pred.index] = hold_pred
                val_pred_path = pred_dir / f"fold{fold_idx}_outer_val.csv"
                test_pred_path = pred_dir / f"fold{fold_idx}_outer_test.csv"
                run_prediction(run_dir / "best.pt", outer_val_csv, val_pred_path)
                run_prediction(run_dir / "best.pt", outer_test_csv, test_pred_path)
                val_fold_preds.append(_prediction_series(val_pred_path))
                test_fold_preds.append(_prediction_series(test_pred_path))
                _cleanup_fold_artifacts(expert_dir, run_dir)

            if oof.isna().any():
                raise ValueError(f"OOF predictions missing for variant={variant}, seed={seed}")
            val_pred = _average_prediction_series(val_fold_preds)
            test_pred = _average_prediction_series(test_fold_preds)

            train_out = pd.DataFrame(
                {
                    "source_index": oof.index,
                    "y_true": y_train.to_numpy(dtype=float),
                    expert_name: oof.to_numpy(dtype=float),
                }
            )
            val_out = pd.DataFrame(
                {
                    "source_index": outer_val["source_index"],
                    "y_true": y_val.to_numpy(dtype=float),
                    expert_name: val_pred.loc[outer_val["source_index"]].to_numpy(dtype=float),
                }
            )
            test_out = pd.DataFrame(
                {
                    "source_index": outer_test["source_index"],
                    "y_true": y_test.to_numpy(dtype=float),
                    expert_name: test_pred.loc[outer_test["source_index"]].to_numpy(dtype=float),
                }
            )
            train_out.to_csv(expert_dir / "expert_oof_train.csv", index=False)
            val_out.to_csv(expert_dir / "expert_val_predictions.csv", index=False)
            test_out.to_csv(expert_dir / "expert_test_predictions.csv", index=False)
            val_metrics = regression_metrics(val_out[expert_name], val_out["y_true"])
            test_metrics = regression_metrics(test_out[expert_name], test_out["y_true"])
            row = {
                "variant": variant,
                "seed": seed,
                "run_dir": str(expert_dir),
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
            rows.append(row)
            with (expert_dir / "graph_oof_summary.json").open("w", encoding="utf-8") as f:
                json.dump(row, f, indent=2)
            shutil.rmtree(expert_dir / "train_runs", ignore_errors=True)

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
