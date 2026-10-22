from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chempath_gine.distill import attach_teacher_predictions
from chempath_gine.training.train import main as train_main


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--split-dir", required=True, help="Directory containing train.csv, val.csv, test.csv")
    parser.add_argument("--teacher-run", required=True, help="OOF stack run directory for one seed")
    parser.add_argument("--out", default="distill_runs")
    parser.add_argument("--teacher-col", default="teacher_pred")
    parser.add_argument("--prediction-col", default="stack_pred")
    parser.add_argument("--distill-weight", type=float, default=0.5)
    parser.add_argument("--variant", default="gine_desc")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    with open(args.base_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out)
    data_dir = out_dir / "distill_data"
    split_dir = Path(args.split_dir)
    teacher_run = Path(args.teacher_run)
    paths = {
        "train": attach_teacher_predictions(
            split_dir / "train.csv",
            teacher_run / "expert_oof_train.csv",
            data_dir / "train.csv",
            teacher_col=args.teacher_col,
            prediction_col=args.prediction_col,
        ),
        "val": attach_teacher_predictions(
            split_dir / "val.csv",
            teacher_run / "expert_val_predictions.csv",
            data_dir / "val.csv",
            teacher_col=args.teacher_col,
            prediction_col=args.prediction_col,
        ),
        "test": attach_teacher_predictions(
            split_dir / "test.csv",
            teacher_run / "expert_test_predictions.csv",
            data_dir / "test.csv",
            teacher_col=args.teacher_col,
            prediction_col=args.prediction_col,
        ),
    }

    distill_cfg = copy.deepcopy(cfg)
    distill_cfg["run"]["output_dir"] = str(out_dir / "runs")
    distill_cfg["run"]["name"] = f"distill_{args.variant}"
    if args.seed is not None:
        distill_cfg["run"]["seed"] = int(args.seed)
    distill_cfg["data"]["train_csv"] = str(paths["train"])
    distill_cfg["data"]["val_csv"] = str(paths["val"])
    distill_cfg["data"]["test_csv"] = str(paths["test"])
    distill_cfg["data"]["teacher_col"] = args.teacher_col
    distill_cfg["model"]["variant"] = args.variant
    distill_cfg.setdefault("train", {})["distill_weight"] = float(args.distill_weight)

    cfg_path = out_dir / "distill_config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(distill_cfg, f, sort_keys=False)

    summary = train_main(["--config", str(cfg_path)])
    with (out_dir / "distill_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
