from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from molfuse_gine.stacking.pruned import ExpertPruningSpec, run_pruned_prediction_stack


def _parse_paths(value: str) -> list[Path]:
    return [Path(item.strip()) for item in value.split(",") if item.strip()]


def _parse_patterns(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run OOF mixed stack with train-OOF expert pruning.")
    parser.add_argument("--expert-dirs", required=True, help="Comma-separated expert prediction dirs.")
    parser.add_argument("--meta-model", default="ridge")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stack-name", default=None)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-experts", type=int, default=10)
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument("--oof-rmse-slack", type=float, default=0.15)
    parser.add_argument("--include-patterns", default="")
    parser.add_argument("--exclude-patterns", default="")
    parser.add_argument("--allow-missing-experts", action="store_true")
    parser.add_argument("--min-existing-experts", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    summary_path = out_dir / "pruned_stack_summary.csv"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"{summary_path} exists; pass --overwrite to replace it")

    requested_expert_dirs = _parse_paths(args.expert_dirs)
    expert_dirs = requested_expert_dirs
    missing_expert_dirs = [path for path in requested_expert_dirs if not path.exists()]
    if args.allow_missing_experts:
        expert_dirs = [path for path in expert_dirs if path.exists()]
        if missing_expert_dirs:
            print(
                "missing expert dirs skipped: "
                + ", ".join(str(path) for path in missing_expert_dirs),
                file=sys.stderr,
            )
    elif missing_expert_dirs:
        raise FileNotFoundError(f"Missing expert dirs: {[str(path) for path in missing_expert_dirs]}")
    if len(expert_dirs) < int(args.min_existing_experts):
        raise ValueError(
            f"Only {len(expert_dirs)} expert dirs remain; "
            f"min-existing-experts={args.min_existing_experts}"
        )

    result = run_pruned_prediction_stack(
        expert_dirs=expert_dirs,
        out_dir=out_dir,
        pruning=ExpertPruningSpec(
            top_k=args.top_k,
            max_experts=args.max_experts,
            corr_threshold=args.corr_threshold,
            oof_rmse_slack=args.oof_rmse_slack,
            include_patterns=_parse_patterns(args.include_patterns),
            exclude_patterns=_parse_patterns(args.exclude_patterns),
        ),
        meta_model_name=args.meta_model,
        seed=args.seed,
        stack_name=args.stack_name,
        requested_expert_dirs=requested_expert_dirs,
        missing_expert_dirs=missing_expert_dirs,
    )

    with (out_dir / "missing_experts.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "requested_expert_dirs": [str(path) for path in requested_expert_dirs],
                "existing_expert_dirs": [str(path) for path in expert_dirs],
                "missing_expert_dirs": [str(path) for path in missing_expert_dirs],
            },
            f,
            indent=2,
        )

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(result.summary))
        writer.writeheader()
        writer.writerow(result.summary)

    print(json.dumps(result.summary, indent=2))
    return result.summary


if __name__ == "__main__":
    main()
