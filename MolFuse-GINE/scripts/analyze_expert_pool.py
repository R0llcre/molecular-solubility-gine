from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def collect_seed_rows(root: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(root.glob("*")):
        if not run_dir.is_dir():
            continue
        dataset = run_dir.name.replace("_gine_desc16", "").replace("_scaffold", "")
        split = "scaffold" if run_dir.name.endswith("_scaffold") else "random"

        graph = _read(run_dir / "graph_ablation" / "ablation_summary.csv")
        if not graph.empty:
            for _, row in graph.iterrows():
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "stage": "graph",
                        "model": row.get("variant"),
                        "seed": int(row.get("seed")),
                        "rmse": row.get("rmse"),
                        "mae": row.get("mae"),
                    }
                )

        tabular = _read(run_dir / "tabular" / "tabular_summary.csv")
        if not tabular.empty:
            for _, row in tabular.iterrows():
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "stage": "tabular",
                        "model": f"{row.get('feature_set')}:{row.get('model')}",
                        "seed": int(row.get("seed")),
                        "rmse": row.get("rmse"),
                        "mae": row.get("mae"),
                    }
                )

        tabular_extra = _read(run_dir / "tabular_extra" / "tabular_summary.csv")
        if not tabular_extra.empty:
            for _, row in tabular_extra.iterrows():
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "stage": "tabular_extra",
                        "model": f"{row.get('feature_set')}:{row.get('model')}",
                        "seed": int(row.get("seed")),
                        "rmse": row.get("rmse"),
                        "mae": row.get("mae"),
                    }
                )

        graph_oof = _read(run_dir / "graph_oof" / "graph_oof_summary.csv")
        if not graph_oof.empty:
            for _, row in graph_oof.iterrows():
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "stage": "graph_oof",
                        "model": row.get("variant"),
                        "seed": int(row.get("seed")),
                        "rmse": row.get("test_rmse"),
                        "mae": row.get("test_mae"),
                    }
                )

        stack = _read(run_dir / "tabular_stack" / "stack_summary.csv")
        if not stack.empty:
            for _, row in stack.iterrows():
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "stage": "tabular_stack",
                        "model": f"tabular_stack:{row.get('meta_model', 'ridge')}",
                        "seed": int(row.get("seed")),
                        "rmse": row.get("test_rmse"),
                        "mae": row.get("test_mae"),
                    }
                )

        stack_extra = _read(run_dir / "tabular_stack_extra" / "stack_summary.csv")
        if not stack_extra.empty:
            for _, row in stack_extra.iterrows():
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "stage": "tabular_stack_extra",
                        "model": f"tabular_stack_extra:{row.get('meta_model', 'ridge')}",
                        "seed": int(row.get("seed")),
                        "rmse": row.get("test_rmse"),
                        "mae": row.get("test_mae"),
                    }
                )

        pretrained_oof = _read(run_dir / "pretrained_oof" / "pretrained_oof_summary.csv")
        if not pretrained_oof.empty:
            for _, row in pretrained_oof.iterrows():
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "stage": "pretrained_oof",
                        "model": (
                            f"{row.get('model_name')}:"
                            f"{row.get('feature_set', 'embedding')}:"
                            f"{row.get('head_model')}"
                        ),
                        "seed": int(row.get("seed")),
                        "rmse": row.get("test_rmse"),
                        "mae": row.get("test_mae"),
                    }
                )

        for stack_root in sorted(run_dir.glob("mixed_stack*")):
            runs_dir = stack_root / "runs"
            if not runs_dir.exists():
                continue
            mixed_summary_paths = sorted(
                set(runs_dir.glob("seed*_*/*mixed_stack_summary.csv"))
                | set(runs_dir.glob("seed*_*/*stack_summary.csv"))
            )
            for path in mixed_summary_paths:
                df = _read(path)
                if df.empty:
                    continue
                row = df.iloc[0]
                seed = int(str(path.parent.name).split("_", 1)[0].removeprefix("seed"))
                stack_name = row.get("stack_name", stack_root.name)
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "stage": "mixed_stack",
                        "model": f"mixed:{stack_name}:{row.get('meta_model', path.parent.name)}",
                        "seed": seed,
                        "rmse": row.get("test_rmse"),
                        "mae": row.get("test_mae"),
                    }
                )

        for stack_root in sorted(run_dir.glob("pruned_mixed_stack*")):
            runs_dir = stack_root / "runs"
            if not runs_dir.exists():
                continue
            summary_paths = sorted(
                set(runs_dir.glob("seed*_*/*pruned_stack_summary.csv"))
            )
            for path in summary_paths:
                df = _read(path)
                if df.empty:
                    continue
                row = df.iloc[0]
                seed = int(str(path.parent.name).split("_", 1)[0].removeprefix("seed"))
                stack_name = row.get("stack_name", stack_root.name)
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "stage": "pruned_stack",
                        "model": f"pruned:{stack_name}:{row.get('meta_model', path.parent.name)}",
                        "seed": seed,
                        "rmse": row.get("test_rmse"),
                        "mae": row.get("test_mae"),
                    }
                )

        for path in sorted((run_dir / "distill").glob("seed*/distill_summary.json")):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "stage": "distill",
                    "model": f"{row.get('distill_variant', 'unknown')}:w{row.get('distill_weight', 'unknown')}",
                    "seed": int(str(path.parent.name).split("_", 1)[0].removeprefix("seed")),
                    "rmse": row.get("rmse"),
                    "mae": row.get("mae"),
                }
            )
    return pd.DataFrame(rows)


def write_analysis(seed_rows: pd.DataFrame, out_dir: Path, baseline: str = "gine_desc") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_rows.to_csv(out_dir / "seed_level_results.csv", index=False)
    if seed_rows.empty:
        return

    aggregate_rows = []
    for (dataset, split, stage, model), group in seed_rows.groupby(
        ["dataset", "split", "stage", "model"], dropna=False
    ):
        rmse = pd.to_numeric(group["rmse"], errors="coerce").dropna()
        mae = pd.to_numeric(group["mae"], errors="coerce").dropna()
        aggregate_rows.append(
            {
                "dataset": dataset,
                "split": split,
                "stage": stage,
                "model": model,
                "n": len(rmse),
                "rmse_mean": rmse.mean(),
                "rmse_std": rmse.std(ddof=1) if len(rmse) > 1 else 0.0,
                "mae_mean": mae.mean() if len(mae) else None,
            }
        )
    aggregate = pd.DataFrame(aggregate_rows).sort_values(["dataset", "split", "rmse_mean"])
    aggregate.to_csv(out_dir / "aggregate_results.csv", index=False)

    deltas = []
    for (dataset, split), group in seed_rows.groupby(["dataset", "split"]):
        base = group[group["model"] == baseline][["seed", "rmse"]].rename(columns={"rmse": "baseline_rmse"})
        if base.empty:
            continue
        for (stage, model), candidate in group.groupby(["stage", "model"]):
            merged = candidate[["seed", "rmse"]].merge(base, on="seed", how="inner")
            if merged.empty:
                continue
            merged["delta_rmse"] = pd.to_numeric(merged["rmse"], errors="coerce") - pd.to_numeric(
                merged["baseline_rmse"], errors="coerce"
            )
            deltas.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "stage": stage,
                    "model": model,
                    "baseline": baseline,
                    "n": len(merged),
                    "delta_rmse_mean": merged["delta_rmse"].mean(),
                    "win_rate": float((merged["delta_rmse"] < 0).mean()),
                }
            )
    delta_df = pd.DataFrame(deltas)
    if not delta_df.empty:
        delta_df = delta_df.sort_values(["dataset", "split", "delta_rmse_mean"])
    delta_df.to_csv(out_dir / "paired_deltas_vs_baseline.csv", index=False)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="sota_runs_v2")
    parser.add_argument("--out", default="reports/sota_search_diagnostics")
    parser.add_argument("--baseline", default="gine_desc")
    args = parser.parse_args(argv)
    rows = collect_seed_rows(Path(args.root))
    write_analysis(rows, Path(args.out), baseline=args.baseline)
    print(f"wrote diagnostics to {args.out}")


if __name__ == "__main__":
    main()
