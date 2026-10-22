"""Deep ensembles and conformal intervals on top of run_calibration.py outputs.

For each split root, Gaussian mode and outer seed s, the ensemble combines the
base run (member 0, `<mode>_seed<s>`) and members `<mode>_seed<s>_m<k>`, which
share the split of seed s and differ only in initialisation / batch order.

Ensemble prediction (Lakshminarayanan et al. 2017, mixture moments):
    mu      = mean_m mu_m
    sigma^2 = mean_m sigma_m^2  (aleatoric)  +  var_m mu_m  (epistemic, ddof=0)

Every method (single models and ensembles) is evaluated three ways on test:
raw sigma, sigma times the validation variance scale, and normalized split
conformal intervals calibrated on validation (chempath_gine.uncertainty.calibration).

Ensemble runs are written in the run_calibration layout (summaries/*.json with
run_dir; run_dir/predictions_{val,test}.csv; run_dir/splits/*.csv) so other
analysis scripts can consume them.

Example
-------
python scripts/build_ensembles.py \
  --roots calibration_runs/esol_random,calibration_runs/esol_scaffold \
  --out calibration_runs/ensembles \
  --fig-dir ../reports/figures/calibration_interpretability_2026-10-17
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chempath_gine.uncertainty import (
    calibration_metrics,
    conformal_metrics,
    coverage_curve,
    fit_variance_scale,
)

PRED_COLS = ["source_index", "SMILES", "y_true", "mu", "sigma"]


def _parse_list(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_member(summary_path: Path) -> dict:
    d = json.loads(summary_path.read_text())
    run_dir = Path(d["run_dir"])
    return {
        "summary": d,
        "run_dir": run_dir,
        "val": pd.read_csv(run_dir / "predictions_val.csv")[PRED_COLS].sort_values("source_index").reset_index(drop=True),
        "test": pd.read_csv(run_dir / "predictions_test.csv")[PRED_COLS].sort_values("source_index").reset_index(drop=True),
    }


def _combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    base = frames[0]
    for f in frames[1:]:
        if not np.array_equal(f["source_index"].values, base["source_index"].values):
            raise ValueError("ensemble members do not share the same molecules")
        if np.max(np.abs(f["y_true"].values - base["y_true"].values)) > 1e-6:
            raise ValueError("ensemble members disagree on y_true")
    mus = np.stack([f["mu"].values for f in frames])
    sig = np.stack([f["sigma"].values for f in frames])
    alea = np.mean(sig ** 2, axis=0)
    epi = np.var(mus, axis=0)
    out = base[["source_index", "SMILES", "y_true"]].copy()
    out["mu"] = mus.mean(axis=0)
    out["sigma"] = np.sqrt(alea + epi)
    out["sigma_aleatoric"] = np.sqrt(alea)
    out["sigma_epistemic"] = np.sqrt(epi)
    out["abs_error"] = np.abs(out["y_true"] - out["mu"])
    out["z"] = (out["y_true"] - out["mu"]) / out["sigma"]
    return out


def _selective(test: pd.DataFrame) -> dict:
    if test["sigma"].std() < 1e-12:
        return {"rmse_low_sigma_half": float("nan"), "rmse_high_sigma_half": float("nan")}
    t = test.sort_values("sigma", kind="mergesort").reset_index(drop=True)
    half = len(t) // 2
    sq = (t["y_true"] - t["mu"]) ** 2
    return {"rmse_low_sigma_half": float(np.sqrt(sq.iloc[:half].mean())),
            "rmse_high_sigma_half": float(np.sqrt(sq.iloc[half:].mean()))}


def evaluate(val: pd.DataFrame, test: pd.DataFrame) -> dict:
    yv, mv, sv = val["y_true"].values, val["mu"].values, val["sigma"].values
    yt, mt, st = test["y_true"].values, test["mu"].values, test["sigma"].values
    constant_sigma = float(np.std(sv)) < 1e-12
    scale = 1.0 if constant_sigma else fit_variance_scale(yv, mv, sv)
    levels, emp = coverage_curve(yt, mt, st)
    _, emp_recal = coverage_curve(yt, mt, st * scale)
    return {
        "val": calibration_metrics(yv, mv, sv),
        "test": calibration_metrics(yt, mt, st),
        "variance_scale": scale,
        "test_recalibrated": calibration_metrics(yt, mt, st * scale),
        "conformal": conformal_metrics(yv, mv, sv, yt, mt, st),
        "coverage_curve_test": {"levels": [float(v) for v in levels],
                                "empirical": [float(v) for v in emp],
                                "empirical_recalibrated": [float(v) for v in emp_recal]},
        **_selective(test),
    }


def _flat(split: str, method: str, seed: int, ev: dict, extra: dict | None = None) -> dict:
    row = {"split": split, "method": method, "seed": seed, "variance_scale": ev["variance_scale"],
           "rmse_low_sigma_half": ev["rmse_low_sigma_half"], "rmse_high_sigma_half": ev["rmse_high_sigma_half"]}
    for blk, prefix in (("test", "raw"), ("test_recalibrated", "recal")):
        for m in ("rmse", "nll", "ece", "coverage_90", "width_90", "spearman_sigma_abs_err"):
            row[f"{prefix}_{m}"] = ev[blk][m]
    for m in ("ece", "coverage_90", "width_90", "q_90"):
        row[f"conformal_{m}"] = ev["conformal"][m]
    row["val_rmse"] = ev["val"]["rmse"]
    row.update(extra or {})
    return row


def reliability_figure(curves: dict, fig_dir: Path) -> list[Path]:
    series = [
        ("mse", "raw", "#2a78d6", "o", "-", "MSE single + constant σ"),
        ("nll", "raw", "#eb6834", "s", "-", "NLL single, raw σ"),
        ("nll_ens", "raw", "#1baf7a", "^", "-", "NLL 5-member ensemble, raw σ"),
        ("nll_ens", "conformal", "#eda100", "D", "--", "NLL ensemble + conformal (val)"),
    ]
    splits = sorted({k[0] for k in curves})
    fig, axes = plt.subplots(1, len(splits), figsize=(5.2 * len(splits), 4.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, split in zip(axes, splits):
        ax.plot([0, 1], [0, 1], color="#9a9890", lw=1.2, ls=(0, (4, 3)), zorder=1)
        for method, kind, color, marker, ls, label in series:
            seeds = sorted(s for (sp, m, k, s) in curves if sp == split and m == method and k == kind)
            if not seeds:
                continue
            levels = np.array(curves[(split, method, kind, seeds[0])]["levels"])
            emp = np.array([curves[(split, method, kind, s)]["empirical"] for s in seeds])
            ax.fill_between(levels, emp.min(0), emp.max(0), color=color, alpha=0.10, lw=0, zorder=2)
            ax.plot(levels, emp.mean(0), color=color, marker=marker, ms=4.5, lw=2, ls=ls, label=label,
                    zorder=3, markeredgecolor="#fcfcfb", markeredgewidth=0.8)
        ax.set_title(f"ESOL {split} split (5 repeats; band = min–max)", fontsize=10, color="#0b0b0b")
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
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Deep ensemble and conformal intervals (below the diagonal = overconfident)", fontsize=11)
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    paths = [fig_dir / "reliability_ensemble_conformal.png", fig_dir / "reliability_ensemble_conformal.svg"]
    fig.savefig(paths[0], dpi=150, facecolor=fig.get_facecolor())
    fig.savefig(paths[1], facecolor=fig.get_facecolor())
    plt.close(fig)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", required=True)
    parser.add_argument("--modes", default="nll,faithful")
    parser.add_argument("--single-modes", default="mse,nll,faithful")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--members", default="0,1,2,3,4")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fig-dir", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    fig_dir = Path(args.fig_dir).resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in _parse_list(args.seeds)]
    members = [int(m) for m in _parse_list(args.members)]

    rows, curves = [], {}
    for root in [Path(r).resolve() for r in _parse_list(args.roots)]:
        split = None
        # Single models (member 0) with conformal added.
        for mode in _parse_list(args.single_modes):
            for s in seeds:
                mem = _load_member(root / "summaries" / f"{mode}_seed{s}.json")
                split = mem["summary"]["split"]
                ev = evaluate(mem["val"], mem["test"])
                rows.append(_flat(split, mode, s, ev))
                curves[(split, mode, "raw", s)] = {"levels": ev["coverage_curve_test"]["levels"],
                                                   "empirical": ev["coverage_curve_test"]["empirical"]}
                curves[(split, mode, "conformal", s)] = ev["conformal"]["curve"]
        # Deep ensembles.
        for mode in _parse_list(args.modes):
            for s in seeds:
                mems = []
                for m in members:
                    tag = f"{mode}_seed{s}" if m == 0 else f"{mode}_seed{s}_m{m}"
                    mems.append(_load_member(root / "summaries" / f"{tag}.json"))
                split = mems[0]["summary"]["split"]
                val = _combine([m["val"] for m in mems])
                test = _combine([m["test"] for m in mems])
                ev = evaluate(val, test)
                method = f"{mode}_ens"
                run_dir = out / f"esol_{split}" / "runs" / f"{method}_seed{s}"
                (run_dir / "splits").mkdir(parents=True, exist_ok=True)
                for name in ("train", "val", "test"):
                    shutil.copy(mems[0]["run_dir"] / "splits" / f"{name}.csv", run_dir / "splits" / f"{name}.csv")
                val.to_csv(run_dir / "predictions_val.csv", index=False)
                test.to_csv(run_dir / "predictions_test.csv", index=False)
                member_rmse = [float(np.sqrt(np.mean((m["test"]["y_true"] - m["test"]["mu"]) ** 2))) for m in mems]
                extra = {
                    "n_members": len(mems),
                    "member_rmse_mean": float(np.mean(member_rmse)),
                    "epistemic_var_share": float(np.mean(test["sigma_epistemic"] ** 2) / np.mean(test["sigma"] ** 2)),
                    "mean_sigma_aleatoric": float(test["sigma_aleatoric"].mean()),
                    "mean_sigma_epistemic": float(test["sigma_epistemic"].mean()),
                }
                summary = {"mode": method, "split": split, "seed": s, "run_dir": str(run_dir),
                           "members": [str(m["run_dir"]) for m in mems], **extra, **ev}
                sdir = out / f"esol_{split}" / "summaries"
                sdir.mkdir(parents=True, exist_ok=True)
                (sdir / f"{method}_seed{s}.json").write_text(json.dumps(summary, indent=2))
                rows.append(_flat(split, method, s, ev, extra))
                curves[(split, method, "raw", s)] = {"levels": ev["coverage_curve_test"]["levels"],
                                                     "empirical": ev["coverage_curve_test"]["empirical"]}
                curves[(split, method, "conformal", s)] = ev["conformal"]["curve"]

    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "seed_level.csv", index=False)
    num = [c for c in df.columns if c not in {"split", "method", "seed"}]
    g = df.groupby(["split", "method"], sort=False)[num]
    agg = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std")).reset_index()
    agg.to_csv(out / "aggregate.csv", index=False)

    pairs = []
    for split in df["split"].unique():
        refs = {name: df[(df["split"] == split) & (df["method"] == name)].set_index("seed")
                for name in ("mse", "nll", "faithful")}
        for method in df["method"].unique():
            other = df[(df["split"] == split) & (df["method"] == method)].set_index("seed")
            for ref_name, ref in refs.items():
                if method == ref_name or ref.empty or not set(seeds) <= set(ref.index):
                    continue
                for metric in ("raw_rmse", "raw_nll", "recal_nll", "conformal_width_90", "conformal_ece"):
                    b, o = ref.loc[seeds, metric].values, other.loc[seeds, metric].values
                    d = o - b
                    pairs.append({"split": split, "method": method, "reference": ref_name, "metric": metric,
                                  "ref_mean": float(b.mean()), "method_mean": float(o.mean()),
                                  "diff_mean": float(d.mean()), "diff_std": float(d.std(ddof=1)),
                                  "n_seeds_method_lower": int((d < 0).sum()),
                                  "paired_t_p": float(ttest_rel(o, b).pvalue)})
    pd.DataFrame(pairs).to_csv(out / "paired.csv", index=False)
    figs = reliability_figure(curves, fig_dir)

    pd.set_option("display.width", 250)
    show = ["split", "method"] + [f"{c}_mean" for c in (
        "raw_rmse", "raw_nll", "raw_ece", "raw_coverage_90", "raw_width_90", "recal_nll", "recal_ece",
        "recal_coverage_90", "recal_width_90", "conformal_ece", "conformal_coverage_90", "conformal_width_90",
        "raw_spearman_sigma_abs_err", "rmse_low_sigma_half", "rmse_high_sigma_half", "epistemic_var_share",
        "member_rmse_mean")]
    print(agg[[c for c in show if c in agg.columns]].round(3).T.to_string())
    print("figures:", [str(p) for p in figs])


if __name__ == "__main__":
    main()
