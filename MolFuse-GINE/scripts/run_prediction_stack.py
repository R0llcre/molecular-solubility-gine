from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from molfuse_gine.stacking import run_mixed_prediction_stack
from molfuse_gine.stacking.diagnostics import write_stack_diagnostics


def _parse_list(value: str) -> list[Path]:
    return [Path(item.strip()) for item in value.split(",") if item.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Backward-compatible wrapper around validated mixed stacking."
    )
    parser.add_argument("--expert-dirs", required=True, help="Comma list of expert prediction dirs")
    parser.add_argument("--meta-model", default="ridge")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    summary_path = out_dir / "prediction_stack_summary.csv"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"{summary_path} exists; pass --overwrite to replace it")

    result = run_mixed_prediction_stack(
        expert_dirs=_parse_list(args.expert_dirs),
        out_dir=out_dir,
        meta_model_name=args.meta_model,
        seed=args.seed,
        stack_name="legacy_prediction_stack",
    )
    write_stack_diagnostics(out_dir)
    with (out_dir / "prediction_stack_summary.json").open("w", encoding="utf-8") as f:
        json.dump(result.summary, f, indent=2)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(result.summary))
        writer.writeheader()
        writer.writerow(result.summary)
    print(json.dumps(result.summary, indent=2))
    return result.summary


if __name__ == "__main__":
    main()
