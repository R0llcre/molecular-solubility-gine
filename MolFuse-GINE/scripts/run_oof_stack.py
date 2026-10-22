from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from molfuse_gine.data import make_splits
from molfuse_gine.stacking import parse_expert_specs, run_tabular_oof_stack


def _parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _write_aggregate(summary_path: Path, aggregate_path: Path) -> None:
    df = pd.read_csv(summary_path)
    rows = []
    metrics = ["val_rmse", "val_mae", "val_r2", "test_rmse", "test_mae", "test_r2"]
    for experts, group in df.groupby("experts", sort=False):
        row = {"experts": experts, "n": len(group)}
        for metric in metrics:
            if metric in group:
                row[f"{metric}_mean"] = group[metric].mean()
                row[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(aggregate_path, index=False)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--experts",
        default=None,
        help="Comma list of feature_set:model specs.",
    )
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--meta-model", default=None)
    parser.add_argument("--out", default="stack_runs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    stack_cfg = cfg.get("stack", {})
    experts_text = args.experts or ",".join(
        stack_cfg.get(
            "experts",
            [
                "desc16:ridge",
                "desc16_morgan2:xgb",
                "desc16_morgan2_maccs:xgb",
                "rdkit2d_selected32_morgan2_maccs:ridge",
            ],
        )
    )
    folds = int(args.folds if args.folds is not None else stack_cfg.get("folds", 5))
    meta_model = args.meta_model or stack_cfg.get("meta_model", "ridge")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "stack_summary.csv"
    aggregate_path = out_dir / "stack_aggregate.csv"
    if summary_path.exists() and aggregate_path.exists() and not args.overwrite:
        print(f"resume existing {summary_path}")
        print(f"resume existing {aggregate_path}")
        return

    experts = parse_expert_specs(experts_text)
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
        run_dir = out_dir / "runs" / f"seed{seed}_{meta_model}"
        summary = run_tabular_oof_stack(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            smiles_col=cfg["data"]["smiles_col"],
            target_col=cfg["data"]["target_col"],
            expert_specs=experts,
            out_dir=run_dir,
            seed=seed,
            folds=folds,
            meta_model_name=meta_model,
        )
        rows.append({"split": cfg["data"]["split"], "run_dir": str(run_dir), **summary})

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_aggregate(summary_path, aggregate_path)
    print(f"wrote {summary_path}")
    print(f"wrote {aggregate_path}")


if __name__ == "__main__":
    main()
