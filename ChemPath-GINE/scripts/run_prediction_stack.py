from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chempath_gine.stacking.oof import build_meta_model
from chempath_gine.training.metrics import regression_metrics


def _parse_list(value: str) -> list[Path]:
    return [Path(item.strip()) for item in value.split(",") if item.strip()]


def _load_expert_table(path: Path, split: str) -> pd.DataFrame:
    candidates = {
        "train": ["expert_oof_train.csv"],
        "val": ["expert_val_predictions.csv"],
        "test": ["expert_test_predictions.csv"],
    }[split]
    for name in candidates:
        file = path / name
        if file.exists():
            df = pd.read_csv(file)
            if "source_index" not in df.columns or "y_true" not in df.columns:
                raise KeyError(f"{file} must contain source_index and y_true")
            return df
    raise FileNotFoundError(f"No {split} prediction file found in {path}")


def _merge_tables(paths: list[Path], split: str) -> pd.DataFrame:
    merged = None
    for path in paths:
        df = _load_expert_table(path, split)
        expert_cols = [
            c
            for c in df.columns
            if c not in {"source_index", "y_true", "stack_absolute_error", "absolute_error"}
            and not c.endswith("_true")
        ]
        if "stack_pred" in expert_cols:
            expert_cols.remove("stack_pred")
            expert_cols.append("stack_pred")
        prefix = path.name.replace(",", "_").replace(":", "_")
        rename = {c: f"{prefix}__{c}" for c in expert_cols}
        slim = df[["source_index", "y_true", *expert_cols]].rename(columns=rename)
        if merged is None:
            merged = slim
        else:
            merged = merged.merge(slim.drop(columns=["y_true"]), on="source_index", how="inner")
    if merged is None:
        raise ValueError("No expert paths provided")
    return merged


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-dirs", required=True, help="Comma list of dirs with expert_oof_train/val/test CSVs")
    parser.add_argument("--meta-model", default="ridge")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    paths = _parse_list(args.expert_dirs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = _merge_tables(paths, "train")
    val = _merge_tables(paths, "val")
    test = _merge_tables(paths, "test")
    expert_cols = [c for c in train.columns if c not in {"source_index", "y_true"}]

    meta = build_meta_model(args.meta_model, args.seed)
    meta.fit(train[expert_cols].to_numpy(dtype=np.float64), train["y_true"].to_numpy(dtype=np.float64))
    train_pred = meta.predict(train[expert_cols].to_numpy(dtype=np.float64)).reshape(-1)
    val_pred = meta.predict(val[expert_cols].to_numpy(dtype=np.float64)).reshape(-1)
    test_pred = meta.predict(test[expert_cols].to_numpy(dtype=np.float64)).reshape(-1)

    train["stack_pred"] = train_pred
    val["stack_pred"] = val_pred
    test["stack_pred"] = test_pred
    train["stack_absolute_error"] = np.abs(train["y_true"].to_numpy(dtype=float) - train_pred)
    val["stack_absolute_error"] = np.abs(val["y_true"].to_numpy(dtype=float) - val_pred)
    test["stack_absolute_error"] = np.abs(test["y_true"].to_numpy(dtype=float) - test_pred)
    train.to_csv(out_dir / "merged_expert_oof_train.csv", index=False)
    val.to_csv(out_dir / "merged_expert_val_predictions.csv", index=False)
    test.to_csv(out_dir / "merged_stack_test_predictions.csv", index=False)
    train.to_csv(out_dir / "expert_oof_train.csv", index=False)
    val.to_csv(out_dir / "expert_val_predictions.csv", index=False)
    test.to_csv(out_dir / "expert_test_predictions.csv", index=False)

    coef = getattr(meta, "coef_", np.full(len(expert_cols), np.nan))
    pd.DataFrame({"expert": expert_cols, "coefficient": np.asarray(coef).reshape(-1)}).to_csv(
        out_dir / "meta_coefficients.csv", index=False
    )
    summary = {
        "meta_model": args.meta_model,
        "n_experts": len(expert_cols),
        "experts": ",".join(expert_cols),
        **{f"val_{k}": v for k, v in regression_metrics(val_pred, val["y_true"]).items()},
        **{f"test_{k}": v for k, v in regression_metrics(test_pred, test["y_true"]).items()},
    }
    with (out_dir / "prediction_stack_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / "prediction_stack_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
