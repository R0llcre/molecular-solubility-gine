"""Aggregate calibration runs into tables and a reliability diagram.

Example
-------
python scripts/summarize_calibration.py \
  --roots calibration_runs/esol_random,calibration_runs/esol_scaffold \
  --out calibration_runs/summary \
  --fig-dir ../reports/figures/calibration_interpretability_2026-10-17
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
ARCHIVED = {
    "random": ROOT / "ablation_runs/esol_fusion_random_5seed/ablation_aggregate.csv",
    "scaffold": ROOT / "ablation_runs/esol_fusion_scaffold_5seed/ablation_aggregate.csv",
}
MODE_ORDER = ["mse", "nll", "beta_nll", "faithful"]
MODE_LABEL = {
    "mse": "MSE baseline + constant σ (= val RMSE)",
    "nll": "Gaussian NLL head",
    "beta_nll": "β-NLL head (β=0.5)",
    "faithful": "Faithful NLL head (stop-grad variance)",
}
METRICS = [
    "rmse", "mae", "nll", "ece", "coverage_50", "coverage_68", "coverage_90", "coverage_95",
    "width_90", "mean_sigma", "rms_z", "spearman_sigma_abs_err",
]
# Fixed categorical slots (validated: dataviz reference palette, light surface).
SERIES = {
    "mse_const": ("#2a78d6", "o", "-", "MSE + constant σ (val RMSE)"),
    "nll_raw": ("#eb6834", "s", "-", "NLL head, raw σ"),
    "nll_recal": ("#1baf7a", "^", "--", "NLL head, σ × val scale"),
    "faithful_recal": ("#eda100", "D", ":", "Faithful NLL, σ × val scale"),
}


def load(roots: list[Path]) -> tuple[pd.DataFrame, dict]:
    rows, curves = [], {}
    for root in roots:
        for path in sorted((root / "summaries").glob("*.json")):
            d = json.loads(path.read_text())
            if int(d.get("member", 0)) != 0:  # deep-ensemble members are summarised by build_ensembles.py
                continue
            row = {
                "split": d["split"], "mode": d["mode"], "seed": d["seed"],
                "best_epoch": d["best_epoch"], "variance_scale": d["variance_scale"],
                "val_rmse": d["val"]["rmse"], "val_nll": d["val"]["nll"], "run_dir": d["run_dir"],
            }
            for blk, prefix in (("test", "raw"), ("test_recalibrated", "recal")):
                for m in METRICS:
                    row[f"{prefix}_{m}"] = d[blk][m]
            row.update(selective_rmse(Path(d["run_dir"]) / "predictions_test.csv"))
            rows.append(row)
            curves[(d["split"], d["mode"], d["seed"])] = d["coverage_curve_test"]
    df = pd.DataFrame(rows)
    df["mode"] = pd.Categorical(df["mode"], MODE_ORDER, ordered=True)
    return df.sort_values(["split", "mode", "seed"]).reset_index(drop=True), curves


def selective_rmse(pred_path: Path) -> dict:
    """RMSE on the half of the test set with the lowest vs highest predicted sigma."""
    p = pd.read_csv(pred_path)
    if p["sigma"].std() < 1e-12:  # constant sigma: no ranking possible
        return {"rmse_low_sigma_half": float("nan"), "rmse_high_sigma_half": float("nan")}
    p = p.sort_values("sigma", kind="mergesort").reset_index(drop=True)
    half = len(p) // 2
    sq = (p["y_true"] - p["mu"]) ** 2
    return {
        "rmse_low_sigma_half": float(np.sqrt(sq.iloc[:half].mean())),
        "rmse_high_sigma_half": float(np.sqrt(sq.iloc[half:].mean())),
    }


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    num = [c for c in df.columns if c not in {"split", "mode", "seed", "run_dir"}]
    g = df.groupby(["split", "mode"], observed=True)[num]
    agg = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std"))
    agg["n_seeds"] = df.groupby(["split", "mode"], observed=True).size()
    return agg.reset_index()


def paired(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in sorted(df["split"].unique()):
        base = df[(df["split"] == split) & (df["mode"] == "mse")].set_index("seed")
        for mode in MODE_ORDER[1:]:
            other = df[(df["split"] == split) & (df["mode"] == mode)].set_index("seed")
            seeds = sorted(set(base.index) & set(other.index))
            if len(seeds) < 2:
                continue
            for metric in ("raw_rmse", "raw_nll", "raw_ece", "recal_nll", "recal_ece"):
                b, o = base.loc[seeds, metric].values, other.loc[seeds, metric].values
                diff = o - b
                rows.append(
                    {
                        "split": split, "mode": mode, "metric": metric, "n": len(seeds),
                        "mse_mean": float(b.mean()), "mode_mean": float(o.mean()),
                        "diff_mean": float(diff.mean()), "diff_std": float(diff.std(ddof=1)),
                        "n_seeds_mode_lower": int((diff < 0).sum()),
                        "paired_t_p": float(ttest_rel(o, b).pvalue),
                    }
                )
    return pd.DataFrame(rows)


def archived() -> dict:
    out = {}
    for split, path in ARCHIVED.items():
        if path.exists():
            a = pd.read_csv(path).set_index("variant")
            out[split] = {"rmse_mean": float(a.loc["gine_desc", "rmse_mean"]),
                          "rmse_std": float(a.loc["gine_desc", "rmse_std"]), "source": str(path)}
    return out


def reliability_figure(curves: dict, fig_dir: Path) -> list[Path]:
    splits = sorted({k[0] for k in curves})
    fig, axes = plt.subplots(1, len(splits), figsize=(5.2 * len(splits), 4.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, split in zip(axes, splits):
        ax.plot([0, 1], [0, 1], color="#9a9890", lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.text(0.9, 0.96, "ideal", color="#52514e", fontsize=8, ha="right", rotation=45, rotation_mode="anchor")
        series = {
            "mse_const": ("mse", "empirical"),
            "nll_raw": ("nll", "empirical"),
            "nll_recal": ("nll", "empirical_recalibrated"),
            "faithful_recal": ("faithful", "empirical_recalibrated"),
        }
        for key, (mode, field) in series.items():
            seeds = sorted(s for (sp, m, s) in curves if sp == split and m == mode)
            if not seeds:
                continue
            levels = np.array(curves[(split, mode, seeds[0])]["levels"])
            emp = np.array([curves[(split, mode, s)][field] for s in seeds])
            color, marker, ls, label = SERIES[key]
            ax.fill_between(levels, emp.min(0), emp.max(0), color=color, alpha=0.10, lw=0, zorder=2)
            ax.plot(levels, emp.mean(0), color=color, marker=marker, ms=4.5, lw=2, ls=ls,
                    label=label, zorder=3, markeredgecolor="#fcfcfb", markeredgewidth=0.8)
        ax.set_title(f"ESOL {split} split (5 seeds; band = seed min–max)", fontsize=10, color="#0b0b0b")
        ax.set_xlabel("nominal central-interval coverage p", fontsize=9, color="#52514e")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(color="#e6e5e0", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#b5b3aa")
        ax.tick_params(colors="#52514e", labelsize=8)
    axes[0].set_ylabel("empirical coverage on test", fontsize=9, color="#52514e")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Reliability diagram: gine_desc uncertainty on ESOL (below the diagonal = overconfident)",
                 fontsize=11, color="#0b0b0b")
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    paths = [fig_dir / "reliability_diagram.png", fig_dir / "reliability_diagram.svg"]
    fig.savefig(paths[0], dpi=150, facecolor=fig.get_facecolor())
    fig.savefig(paths[1], facecolor=fig.get_facecolor())
    plt.close(fig)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fig-dir", required=True)
    args = parser.parse_args(argv)
    roots = [Path(r).resolve() for r in args.roots.split(",") if r.strip()]
    out = Path(args.out).resolve()
    fig_dir = Path(args.fig_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df, curves = load(roots)
    df.to_csv(out / "seed_level.csv", index=False)
    agg = aggregate(df)
    agg.to_csv(out / "aggregate.csv", index=False)
    pairs = paired(df)
    pairs.to_csv(out / "paired_vs_mse.csv", index=False)
    arch = archived()
    figs = reliability_figure(curves, fig_dir)
    with (out / "archived_reference.json").open("w") as f:
        json.dump(arch, f, indent=2)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    show = ["split", "mode", "n_seeds"] + [
        f"{c}_{s}" for c in ("raw_rmse", "raw_nll", "raw_ece", "raw_coverage_90", "raw_width_90",
                             "variance_scale", "recal_nll", "recal_ece", "recal_coverage_90",
                             "recal_width_90", "raw_spearman_sigma_abs_err", "rmse_low_sigma_half",
                             "rmse_high_sigma_half", "best_epoch", "val_rmse")
        for s in ("mean", "std")
    ]
    print(agg[show].round(4).T.to_string())
    print(pairs.round(4).to_string())
    print(json.dumps(arch, indent=2))
    print("figures:", [str(p) for p in figs])


if __name__ == "__main__":
    main()
