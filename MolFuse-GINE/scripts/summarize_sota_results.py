from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _mean_std(group: pd.DataFrame, metric: str) -> tuple[float | None, float | None]:
    if metric not in group or group.empty:
        return None, None
    values = pd.to_numeric(group[metric], errors="coerce").dropna()
    if values.empty:
        return None, None
    return float(values.mean()), float(values.std(ddof=1) if len(values) > 1 else 0.0)


def _dataset_split_from_name(name: str) -> tuple[str, str]:
    split = "scaffold" if name.endswith("_scaffold") else "random"
    dataset = name.removesuffix("_gine_desc16").removesuffix("_scaffold")
    dataset = dataset.replace("_gine_desc16", "")
    return dataset, split


def _leaderboard_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_dir in sorted(root.glob("*")):
        if not run_dir.is_dir():
            continue
        dataset, split = _dataset_split_from_name(run_dir.name)
        graph = _read_csv(run_dir / "graph_ablation" / "ablation_aggregate.csv")
        if not graph.empty:
            for _, row in graph.iterrows():
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "stage": "graph",
                        "model": row.get("variant"),
                        "n": int(row.get("n", 0)),
                        "rmse_mean": row.get("rmse_mean"),
                        "rmse_std": row.get("rmse_std"),
                        "mae_mean": row.get("mae_mean"),
                        "source": str(run_dir / "graph_ablation" / "ablation_aggregate.csv"),
                    }
                )

        tabular = _read_csv(run_dir / "tabular" / "tabular_summary.csv")
        if not tabular.empty:
            for (feature_set, model), group in tabular.groupby(["feature_set", "model"], dropna=False):
                rmse_mean, rmse_std = _mean_std(group, "rmse")
                mae_mean, _ = _mean_std(group, "mae")
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "stage": "tabular",
                        "model": f"{feature_set}:{model}",
                        "n": len(group),
                        "rmse_mean": rmse_mean,
                        "rmse_std": rmse_std,
                        "mae_mean": mae_mean,
                        "source": str(run_dir / "tabular" / "tabular_summary.csv"),
                    }
                )

        tabular_extra = _read_csv(run_dir / "tabular_extra" / "tabular_summary.csv")
        if not tabular_extra.empty:
            for (feature_set, model), group in tabular_extra.groupby(["feature_set", "model"], dropna=False):
                rmse_mean, rmse_std = _mean_std(group, "rmse")
                mae_mean, _ = _mean_std(group, "mae")
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "stage": "tabular_extra",
                        "model": f"{feature_set}:{model}",
                        "n": len(group),
                        "rmse_mean": rmse_mean,
                        "rmse_std": rmse_std,
                        "mae_mean": mae_mean,
                        "source": str(run_dir / "tabular_extra" / "tabular_summary.csv"),
                    }
                )

        stack = _read_csv(run_dir / "tabular_stack" / "stack_summary.csv")
        if not stack.empty:
            rmse_mean, rmse_std = _mean_std(stack, "test_rmse")
            mae_mean, _ = _mean_std(stack, "test_mae")
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "stage": "stack",
                    "model": "tabular_oof_ridge",
                    "n": len(stack),
                    "rmse_mean": rmse_mean,
                    "rmse_std": rmse_std,
                    "mae_mean": mae_mean,
                    "source": str(run_dir / "tabular_stack" / "stack_summary.csv"),
                }
            )
        stack_extra = _read_csv(run_dir / "tabular_stack_extra" / "stack_summary.csv")
        if not stack_extra.empty:
            rmse_mean, rmse_std = _mean_std(stack_extra, "test_rmse")
            mae_mean, _ = _mean_std(stack_extra, "test_mae")
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "stage": "tabular_stack_extra",
                    "model": "tabular_extra_oof_ridge",
                    "n": len(stack_extra),
                    "rmse_mean": rmse_mean,
                    "rmse_std": rmse_std,
                    "mae_mean": mae_mean,
                    "source": str(run_dir / "tabular_stack_extra" / "stack_summary.csv"),
                }
            )

        pretrained = _read_csv(run_dir / "pretrained_smiles" / "pretrained_smiles_summary.csv")
        if not pretrained.empty:
            for (model_name, head), group in pretrained.groupby(["model_name", "head_model"], dropna=False):
                rmse_mean, rmse_std = _mean_std(group, "rmse")
                mae_mean, _ = _mean_std(group, "mae")
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "stage": "pretrained_smiles",
                        "model": f"{model_name}:{head}",
                        "n": len(group),
                        "rmse_mean": rmse_mean,
                        "rmse_std": rmse_std,
                        "mae_mean": mae_mean,
                        "source": str(run_dir / "pretrained_smiles" / "pretrained_smiles_summary.csv"),
                    }
                )

        pretrained_oof = _read_csv(run_dir / "pretrained_oof" / "pretrained_oof_summary.csv")
        if not pretrained_oof.empty:
            if "feature_set" not in pretrained_oof:
                pretrained_oof["feature_set"] = "embedding"
            for (model_name, feature_set, head), group in pretrained_oof.groupby(
                ["model_name", "feature_set", "head_model"], dropna=False
            ):
                rmse_mean, rmse_std = _mean_std(group, "test_rmse")
                mae_mean, _ = _mean_std(group, "test_mae")
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "stage": "pretrained_oof",
                        "model": f"{model_name}:{feature_set}:{head}",
                        "n": len(group),
                        "rmse_mean": rmse_mean,
                        "rmse_std": rmse_std,
                        "mae_mean": mae_mean,
                        "source": str(run_dir / "pretrained_oof" / "pretrained_oof_summary.csv"),
                    }
                )

        graph_oof = _read_csv(run_dir / "graph_oof" / "graph_oof_summary.csv")
        if not graph_oof.empty:
            for variant, group in graph_oof.groupby("variant", dropna=False):
                rmse_mean, rmse_std = _mean_std(group, "test_rmse")
                mae_mean, _ = _mean_std(group, "test_mae")
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "stage": "graph_oof",
                        "model": variant,
                        "n": len(group),
                        "rmse_mean": rmse_mean,
                        "rmse_std": rmse_std,
                        "mae_mean": mae_mean,
                        "source": str(run_dir / "graph_oof" / "graph_oof_summary.csv"),
                    }
                )

        mixed_by_model: dict[str, list[dict]] = {}
        mixed_source_by_model: dict[str, str] = {}
        for stack_root in sorted(run_dir.glob("mixed_stack*")):
            runs_dir = stack_root / "runs"
            if not runs_dir.exists():
                continue
            mixed_summary_paths = sorted(
                set(runs_dir.glob("seed*_*/*mixed_stack_summary.csv"))
                | set(runs_dir.glob("seed*_*/*stack_summary.csv"))
            )
            for summary_path in mixed_summary_paths:
                df_summary = _read_csv(summary_path)
                if not df_summary.empty:
                    stack_name = str(df_summary.iloc[0].get("stack_name", stack_root.name))
                    meta_model = str(df_summary.iloc[0].get("meta_model", summary_path.parent.name))
                    model = f"{stack_name}:{meta_model}"
                    mixed_by_model.setdefault(model, []).extend(df_summary.to_dict("records"))
                    mixed_source_by_model.setdefault(model, str(stack_root / "runs"))
        # Backward-compatible old script output.
        for summary_path in sorted((run_dir / "mixed_stack" / "runs").glob("seed*_*/*prediction_stack_summary.csv")):
            df_summary = _read_csv(summary_path)
            if not df_summary.empty:
                model = str(df_summary.iloc[0].get("meta_model", summary_path.parent.name))
                mixed_by_model.setdefault(model, []).extend(df_summary.to_dict("records"))
                mixed_source_by_model.setdefault(model, str(run_dir / "mixed_stack" / "runs"))
        for model, mixed_rows in mixed_by_model.items():
            mixed = pd.DataFrame(mixed_rows)
            rmse_mean, rmse_std = _mean_std(mixed, "test_rmse")
            mae_mean, _ = _mean_std(mixed, "test_mae")
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "stage": "mixed_stack",
                    "model": f"mixed_{model}",
                    "n": len(mixed),
                    "rmse_mean": rmse_mean,
                    "rmse_std": rmse_std,
                    "mae_mean": mae_mean,
                    "source": mixed_source_by_model.get(model, str(run_dir / "mixed_stack" / "runs")),
                }
            )

        pruned_by_model: dict[str, list[dict]] = {}
        pruned_source_by_model: dict[str, str] = {}
        for stack_root in sorted(run_dir.glob("pruned_mixed_stack*")):
            runs_dir = stack_root / "runs"
            if not runs_dir.exists():
                continue
            for summary_path in sorted(runs_dir.glob("seed*_*/*pruned_stack_summary.csv")):
                df_summary = _read_csv(summary_path)
                if not df_summary.empty:
                    stack_name = str(df_summary.iloc[0].get("stack_name", stack_root.name))
                    meta_model = str(df_summary.iloc[0].get("meta_model", summary_path.parent.name))
                    model = f"{stack_name}:{meta_model}"
                    pruned_by_model.setdefault(model, []).extend(df_summary.to_dict("records"))
                    pruned_source_by_model.setdefault(model, str(stack_root / "runs"))
        for model, pruned_rows in pruned_by_model.items():
            pruned = pd.DataFrame(pruned_rows)
            rmse_mean, rmse_std = _mean_std(pruned, "test_rmse")
            mae_mean, _ = _mean_std(pruned, "test_mae")
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "stage": "pruned_stack",
                    "model": f"pruned_{model}",
                    "n": len(pruned),
                    "rmse_mean": rmse_mean,
                    "rmse_std": rmse_std,
                    "mae_mean": mae_mean,
                    "source": pruned_source_by_model.get(model, str(run_dir / "pruned_mixed_stack" / "runs")),
                }
            )

        distill_rows = []
        for summary_path in sorted((run_dir / "distill").glob("seed*/distill_summary.json")):
            try:
                distill_rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
            except Exception:
                continue
        if distill_rows:
            distill = pd.DataFrame(distill_rows)
            group_cols = ["distill_variant", "distill_weight"]
            for col in group_cols:
                if col not in distill:
                    distill[col] = "unknown"
            for (variant, weight), group in distill.groupby(group_cols, dropna=False):
                rmse_mean, rmse_std = _mean_std(group, "rmse")
                mae_mean, _ = _mean_std(group, "mae")
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "stage": "distill",
                        "model": f"{variant}:w{weight}",
                        "n": len(group),
                        "rmse_mean": rmse_mean,
                        "rmse_std": rmse_std,
                        "mae_mean": mae_mean,
                        "source": str(run_dir / "distill"),
                    }
                )
    rows.sort(
        key=lambda row: (
            str(row["dataset"]),
            str(row["split"]),
            float(row["rmse_mean"]) if row["rmse_mean"] is not None else float("inf"),
        )
    )
    return rows


def _format_float(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def _write_markdown(rows: list[dict[str, object]], path: Path, root: Path) -> None:
    lines = [
        "# SOTA System Results",
        "",
        f"This report is generated from `{root}/`. Lower RMSE is better.",
        "",
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        lines.append("No results found.")
    else:
        for (dataset, split), group in df.groupby(["dataset", "split"], sort=True):
            top = group.sort_values("rmse_mean", na_position="last").head(12)
            lines.extend(
                [
                    f"## {dataset} {split}",
                    "",
                    "| Rank | Stage | Model | n | RMSE | MAE |",
                    "| ---: | --- | --- | ---: | ---: | ---: |",
                ]
            )
            for rank, (_, row) in enumerate(top.iterrows(), 1):
                rmse = _format_float(row["rmse_mean"])
                std = _format_float(row["rmse_std"])
                rmse_text = f"{rmse} ± {std}" if std else rmse
                lines.append(
                    f"| {rank} | {row['stage']} | `{row['model']}` | {int(row['n'])} | {rmse_text} | {_format_float(row['mae_mean'])} |"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="sota_runs_v2")
    parser.add_argument("--out", default="reports/sota_search")
    args = parser.parse_args(argv)

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = _leaderboard_rows(root)
    csv_path = out / "leaderboard.csv"
    md_path = out / "sota_system_report.md"
    json_path = out / "summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in rows for key in row}) if rows else [
            "dataset",
            "split",
            "stage",
            "model",
            "n",
            "rmse_mean",
            "rmse_std",
            "mae_mean",
            "source",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown(rows, md_path, root)
    payload = {"git_sha": _git_sha(), "n_rows": len(rows), "leaderboard": str(csv_path), "report": str(md_path)}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
