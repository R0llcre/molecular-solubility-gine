from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from molfuse_gine.training.train import main as train_main


def _parse_list(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def _write_aggregate(summary_path: Path, aggregate_path: Path) -> None:
    df = pd.read_csv(summary_path)
    metrics = [
        "rmse",
        "mae",
        "r2",
        "pearson",
        "spearman",
        "best_epoch",
        "best_val_loss",
        "diag_attention_0_path_scale_tanh",
        "diag_attention_0_attn_res_scale",
        "diag_attention_0_ffn_res_scale",
        "diag_attention_0_path_bias_abs_mean",
        "diag_attention_0_sparse_pair_ratio",
        "diag_descriptor_gate_mean",
    ]
    rows = []
    for variant, group in df.groupby("variant", sort=False):
        row = {"variant": variant, "n": len(group)}
        for metric in metrics:
            if metric in group:
                row[f"{metric}_mean"] = group[metric].mean()
                row[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(aggregate_path, index=False)


def _flatten_summary(summary: dict) -> dict:
    flat = {k: v for k, v in summary.items() if k != "model_diagnostics"}
    for key, value in (summary.get("model_diagnostics") or {}).items():
        flat[f"diag_{key}"] = value
    return flat


def _existing_summary(out_dir: Path, variant: str, seed_text: str) -> dict | None:
    run_root = out_dir / "runs"
    matches = sorted(
        run_root.glob(f"ablation_seed{seed_text}_{variant}_*/summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in matches:
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            actual = (payload.get("model_diagnostics") or {}).get("variant")
            if actual != variant:
                continue
            return _flatten_summary(payload)
        except Exception:
            continue
    return None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--variants",
        default="gine,gine_attn,gine_spd,molfuse_gine,chempath_desc",
    )
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--out", default="ablation_runs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    split_name = str(base.get("data", {}).get("split", "")).lower()

    out_dir = Path(args.out)
    cfg_dir = out_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant in _parse_list(args.variants):
        for seed_text in _parse_list(args.seeds):
            existing = None if args.overwrite else _existing_summary(out_dir, variant, seed_text)
            if existing is not None:
                print(f"resume existing variant={variant} seed={seed_text}")
                rows.append({"variant": variant, "seed": int(seed_text), **existing})
                continue

            cfg = copy.deepcopy(base)
            cfg["run"]["seed"] = int(seed_text)
            cfg["run"]["output_dir"] = str(out_dir / "runs")
            cfg["run"]["name"] = f"ablation_seed{seed_text}"
            cfg["model"]["variant"] = variant
            if not Path(str(cfg["data"].get("cache_dir", ""))).is_absolute():
                cache_key = "scaffold" if split_name == "scaffold" else f"seed{seed_text}"
                cfg["data"]["cache_dir"] = str((out_dir / "shared_cache" / cache_key).resolve())

            cfg_path = cfg_dir / f"{variant}_seed{seed_text}.yaml"
            with cfg_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

            summary = _flatten_summary(train_main(["--config", str(cfg_path)]))
            rows.append({"variant": variant, "seed": int(seed_text), **summary})

    summary_path = out_dir / "ablation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    aggregate_path = out_dir / "ablation_aggregate.csv"
    _write_aggregate(summary_path, aggregate_path)
    print(f"wrote {summary_path}")
    print(f"wrote {aggregate_path}")


if __name__ == "__main__":
    main()
