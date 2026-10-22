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

        mixed_rows = []
        for summary_path in sorted((run_dir / "mixed_stack" / "runs").glob("seed*_ridge/prediction_stack_summary.csv")):
            df_summary = _read_csv(summary_path)
            if not df_summary.empty:
                mixed_rows.extend(df_summary.to_dict("records"))
        if mixed_rows:
            mixed = pd.DataFrame(mixed_rows)
            rmse_mean, rmse_std = _mean_std(mixed, "test_rmse")
            mae_mean, _ = _mean_std(mixed, "test_mae")
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "stage": "mixed_stack",
                    "model": "graph_oof_plus_tabular_stack",
                    "n": len(mixed),
                    "rmse_mean": rmse_mean,
                    "rmse_std": rmse_std,
                    "mae_mean": mae_mean,
                    "source": str(run_dir / "mixed_stack" / "runs"),
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
            rmse_mean, rmse_std = _mean_std(distill, "rmse")
            mae_mean, _ = _mean_std(distill, "mae")
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "stage": "distill",
                    "model": "gine_desc_concat_from_mixed_stack",
                    "n": len(distill),
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


def _write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# SOTA System Results",
        "",
        "This report is generated from `sota_runs/`. Lower RMSE is better.",
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
    parser.add_argument("--root", default="sota_runs")
    parser.add_argument("--out", default="reports/sota_system")
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
    _write_markdown(rows, md_path)
    payload = {"git_sha": _git_sha(), "n_rows": len(rows), "leaderboard": str(csv_path), "report": str(md_path)}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
