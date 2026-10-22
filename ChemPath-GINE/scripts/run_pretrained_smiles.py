from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chempath_gine.baselines.tabular import build_model, fit_predict
from chempath_gine.data import make_splits
from chempath_gine.features.pretrained_smiles_bank import SmilesEmbeddingConfig, embed_smiles_frame
from chempath_gine.selection import FeaturePreprocessor, FeatureSelectionSpec
from chempath_gine.training.metrics import regression_metrics


def _parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _target(df: pd.DataFrame, target_col: str) -> np.ndarray:
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("Non-finite targets found")
    return y


def _write_predictions(path: Path, test_df: pd.DataFrame, pred: np.ndarray, target_col: str) -> None:
    out = test_df.copy()
    out["y_true"] = pd.to_numeric(out[target_col], errors="coerce").astype(float)
    out["y_pred"] = pred.astype(float)
    out["absolute_error"] = np.abs(out["y_true"].to_numpy() - out["y_pred"].to_numpy())
    out.to_csv(path, index=False)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-name", default="DeepChem/ChemBERTa-77M-MLM")
    parser.add_argument("--head-models", default="ridge,elasticnet,xgb")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--random-smiles", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="pretrained_smiles_runs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "pretrained_smiles_summary.csv"
    if summary_path.exists() and not args.overwrite:
        print(f"resume existing {summary_path}")
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
        emb_cfg = SmilesEmbeddingConfig(
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device,
            n_random=args.random_smiles,
            seed=seed,
        )
        x_train_raw = embed_smiles_frame(train_df[cfg["data"]["smiles_col"]], emb_cfg)
        x_val_raw = embed_smiles_frame(val_df[cfg["data"]["smiles_col"]], emb_cfg)
        x_test_raw = embed_smiles_frame(test_df[cfg["data"]["smiles_col"]], emb_cfg)
        y_train = _target(train_df, cfg["data"]["target_col"])
        y_val = _target(val_df, cfg["data"]["target_col"])
        y_test = _target(test_df, cfg["data"]["target_col"])
        pre = FeaturePreprocessor(FeatureSelectionSpec(method="none", top_k=None), scale=True)
        x_train = pre.fit_transform(x_train_raw, y_train)
        x_val = pre.transform(x_val_raw)
        x_test = pre.transform(x_test_raw)

        for head in _parse_list(args.head_models):
            model = build_model(head, seed)
            pred = fit_predict(model, head, x_train, y_train, x_val, y_val, x_test)
            metrics = regression_metrics(pred, y_test)
            pred_path = pred_dir / f"{head}_seed{seed}.csv"
            _write_predictions(pred_path, test_df, pred, cfg["data"]["target_col"])
            rows.append(
                {
                    "model_name": args.model_name,
                    "head_model": head,
                    "seed": seed,
                    "split": cfg["data"]["split"],
                    "random_smiles": args.random_smiles,
                    "n_features": int(x_train.shape[1]),
                    "prediction_path": str(pred_path),
                    **metrics,
                }
            )

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
