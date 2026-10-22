"""Does predicted uncertainty grow for test molecules unlike the training set?

For every seed of one calibration (or ensemble) output root this script reads only

    <runs-root>/summaries/<mode>_seed<k>.json        -> run_dir, variance_scale
    <run_dir>/predictions_test.csv                    -> source_index, SMILES, y_true, mu, sigma
    <run_dir>/splits/train.csv                        -> SMILES

and computes, per test molecule, the max Tanimoto similarity (Morgan radius 2, 2048 bits, on
RDKit-canonical SMILES) to that run's own training set. It then reports, per seed and as
mean +/- sd over seeds:

* Spearman(similarity, sigma) and Spearman(similarity, |error|)  (also Spearman(sigma, |error|),
  Spearman(mu, sigma), Spearman(similarity, y_true) and the partial Spearman(similarity, sigma | mu),
  because a heteroscedastic head tends to tie sigma to the predicted level mu);
* similarity quartile bins (edges fixed per seed on that seed's test set): mean similarity,
  mean sigma, RMSE, 90% coverage of mu +/- 1.645 sigma, both for the raw sigma and for
  sigma * variance_scale (the validation-fitted multiplier stored in the summary JSON).

Bins are labelled Q1 (least similar to train) ... Q4 (most similar).

Example
-------
python scripts/sigma_vs_similarity.py \
  --runs-root calibration_runs/esol_scaffold --mode nll --seeds 1,2,3,4,5 \
  --out calibration_runs/similarity/nll_scaffold --tag nll_scaffold \
  --compare-with calibration_runs/similarity/nll_random

``--compare-with`` takes other output dirs of this script and overlays them in the figure only.
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
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from matplotlib.lines import Line2D
from scipy.stats import rankdata, spearmanr, t as t_dist

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIG_DIR = ROOT.parent / "reports/figures/calibration_interpretability_2026-10-17"
FP_RADIUS, FP_BITS = 2, 2048
Z90 = 1.6448536269514722  # two-sided 90% normal quantile
N_BINS = 4

# Fixed categorical slots, in order (light surface).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
BG, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#b5b3aa"

BIN_METRICS = [
    "n", "sim_lo", "sim_hi", "mean_sim", "mean_sigma", "mean_sigma_recal", "rms_sigma",
    "rms_sigma_recal", "rmse", "mae", "cov90", "cov90_recal",
]
CORR_METRICS = [
    "spearman_sim_sigma", "spearman_sim_abs_err", "spearman_sigma_abs_err",
    "spearman_sim_sigma_partial_mu", "spearman_mu_sigma", "spearman_sim_y_true", "mean_max_sim",
    "median_max_sim", "frac_sim_eq_1", "n_exact_smiles_in_train", "variance_scale",
    "q1_over_q4_mean_sigma", "q1_over_q4_rmse", "q1_minus_q4_cov90", "q1_minus_q4_cov90_recal",
]


# ----------------------------------------------------------------------------- chemistry
def canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


def fingerprints(canon: list[str]):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in canon]


def max_similarity(test_canon: list[str], train_canon: list[str]) -> tuple[np.ndarray, list[str]]:
    train_fps = fingerprints(train_canon)
    sims, nearest = [], []
    for fp in fingerprints(test_canon):
        row = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
        j = int(row.argmax())
        sims.append(float(row[j]))
        nearest.append(train_canon[j])
    return np.asarray(sims), nearest


# ----------------------------------------------------------------------------- per seed
def load_seed(runs_root: Path, mode: str, seed: int) -> tuple[pd.DataFrame, dict]:
    summary_path = runs_root / "summaries" / f"{mode}_seed{seed}.json"
    summary = json.loads(summary_path.read_text())
    run_dir = Path(summary["run_dir"])
    if not run_dir.is_absolute():
        run_dir = (runs_root / run_dir).resolve()
    scale = float(summary.get("variance_scale", 1.0))

    pred = pd.read_csv(run_dir / "predictions_test.csv")
    train = pd.read_csv(run_dir / "splits" / "train.csv")

    test_canon = [canonical(s) for s in pred["SMILES"]]
    train_canon = [c for c in (canonical(s) for s in train["SMILES"]) if c is not None]
    n_bad = sum(c is None for c in test_canon)
    if n_bad:
        raise ValueError(f"{n_bad} unparsable test SMILES in {run_dir}")
    sims, nearest = max_similarity(test_canon, train_canon)

    df = pd.DataFrame(
        {
            "seed": seed,
            "source_index": pred["source_index"] if "source_index" in pred else np.arange(len(pred)),
            "SMILES": pred["SMILES"],
            "canonical_smiles": test_canon,
            "y_true": pred["y_true"],
            "mu": pred["mu"],
            "sigma": pred["sigma"],
            "sigma_recal": pred["sigma"] * scale,
            "abs_error": (pred["y_true"] - pred["mu"]).abs(),
            "max_sim": sims,
            "nearest_train_smiles": nearest,
            "exact_in_train": [c in set(train_canon) for c in test_canon],
        }
    )
    # Deep-ensemble outputs (build_ensembles.py) also carry the variance decomposition.
    for col in ("sigma_aleatoric", "sigma_epistemic"):
        if col in pred:
            df[col] = pred[col].values
    edges = np.quantile(df["max_sim"], [0.25, 0.5, 0.75])
    # Q1 = sim <= e1, Q2 = e1 < sim <= e2, ... (ties go to the lower bin)
    df["sim_bin"] = np.searchsorted(edges, df["max_sim"].values, side="left") + 1
    meta = {
        "seed": seed, "run_dir": str(run_dir), "variance_scale": scale, "n_test": len(df),
        "n_train": len(train_canon), "quartile_edges": [float(e) for e in edges],
    }
    # Sanity check: the stored recalibrated block should equal sigma * variance_scale.
    if "test" in summary and "test_recalibrated" in summary:
        raw, rec = summary["test"].get("mean_sigma"), summary["test_recalibrated"].get("mean_sigma")
        if raw and rec:
            meta["stored_recal_mean_sigma_ratio"] = float(rec / raw)
    return df, meta


def bin_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b in range(1, N_BINS + 1):
        g = df[df["sim_bin"] == b]
        err = g["y_true"] - g["mu"]
        rows.append(
            {
                "seed": int(df["seed"].iloc[0]), "bin": f"Q{b}", "n": len(g),
                "sim_lo": g["max_sim"].min(), "sim_hi": g["max_sim"].max(),
                "mean_sim": g["max_sim"].mean(),
                "mean_sigma": g["sigma"].mean(), "mean_sigma_recal": g["sigma_recal"].mean(),
                "rms_sigma": np.sqrt((g["sigma"] ** 2).mean()),
                "rms_sigma_recal": np.sqrt((g["sigma_recal"] ** 2).mean()),
                "rmse": np.sqrt((err ** 2).mean()), "mae": err.abs().mean(),
                "cov90": (err.abs() <= Z90 * g["sigma"]).mean(),
                "cov90_recal": (err.abs() <= Z90 * g["sigma_recal"]).mean(),
            }
        )
    return pd.DataFrame(rows)


def partial_spearman(x, y, z) -> tuple[float, float]:
    """Spearman(x, y | z): Pearson correlation of rank residuals after regressing out rank(z)."""
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    design = np.column_stack([np.ones_like(rz), rz])
    res = [r - design @ np.linalg.lstsq(design, r, rcond=None)[0] for r in (rx, ry)]
    r = float(np.corrcoef(res[0], res[1])[0, 1])
    dof = len(rx) - 3
    t = r * np.sqrt(dof / max(1e-12, 1 - r * r))
    return r, float(2 * t_dist.sf(abs(t), dof))


def corr_row(df: pd.DataFrame, bins: pd.DataFrame, meta: dict) -> dict:
    def rho(a, b):
        r = spearmanr(a, b)
        return float(r.statistic), float(r.pvalue)

    r_ss, p_ss = rho(df["max_sim"], df["sigma"])
    r_se, p_se = rho(df["max_sim"], df["abs_error"])
    r_sige, p_sige = rho(df["sigma"], df["abs_error"])
    r_part, p_part = partial_spearman(df["max_sim"], df["sigma"], df["mu"])
    q = bins.set_index("bin")
    return {
        "seed": meta["seed"], "n_test": meta["n_test"], "n_train": meta["n_train"],
        "spearman_sim_sigma": r_ss, "p_sim_sigma": p_ss,
        "spearman_sim_abs_err": r_se, "p_sim_abs_err": p_se,
        "spearman_sigma_abs_err": r_sige, "p_sigma_abs_err": p_sige,
        "spearman_sim_sigma_partial_mu": r_part, "p_sim_sigma_partial_mu": p_part,
        "spearman_mu_sigma": rho(df["mu"], df["sigma"])[0],
        "spearman_sim_y_true": rho(df["max_sim"], df["y_true"])[0],
        "mean_max_sim": df["max_sim"].mean(), "median_max_sim": df["max_sim"].median(),
        "frac_sim_eq_1": float((df["max_sim"] >= 1 - 1e-12).mean()),
        "n_exact_smiles_in_train": int(df["exact_in_train"].sum()),
        "variance_scale": meta["variance_scale"],
        "q1_over_q4_mean_sigma": q.loc["Q1", "mean_sigma"] / q.loc["Q4", "mean_sigma"],
        "q1_over_q4_rmse": q.loc["Q1", "rmse"] / q.loc["Q4", "rmse"],
        "q1_minus_q4_cov90": q.loc["Q1", "cov90"] - q.loc["Q4", "cov90"],
        "q1_minus_q4_cov90_recal": q.loc["Q1", "cov90_recal"] - q.loc["Q4", "cov90_recal"],
        "q_edges": ";".join(f"{e:.4f}" for e in meta["quartile_edges"]),
        **({"spearman_sim_sigma_epistemic": rho(df["max_sim"], df["sigma_epistemic"])[0],
            "spearman_sim_sigma_aleatoric": rho(df["max_sim"], df["sigma_aleatoric"])[0],
            "q1_over_q4_mean_sigma_epistemic": (
                df.loc[df["sim_bin"] == 1, "sigma_epistemic"].mean()
                / df.loc[df["sim_bin"] == 4, "sigma_epistemic"].mean())}
           if "sigma_epistemic" in df else {}),
    }


def mean_sd(frame: pd.DataFrame, cols: list[str], by: str | None = None) -> pd.DataFrame:
    if by is None:
        return pd.DataFrame(
            {"metric": cols, "mean": [frame[c].mean() for c in cols],
             "std": [frame[c].std(ddof=1) for c in cols], "n_seeds": len(frame)}
        )
    g = frame.groupby(by, sort=True)[cols]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std"))
    out["n_seeds"] = frame.groupby(by).size()
    return out.reset_index()


# ----------------------------------------------------------------------------- figure
def figure(datasets: list[dict], fig_paths: list[Path], mode: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), sharey=True)
    ax_sig, ax_rmse = axes
    handles = []

    def fmt(v):
        return f"{v:+.2f}".replace("-", "\u2212")

    for i, ds in enumerate(datasets):
        color = COLORS[i % len(COLORS)]
        b = ds["bins"].groupby("bin", sort=True)
        x = b["mean_sim"].mean().values
        corr = ds["corr"]
        rs = f"{fmt(corr['spearman_sim_sigma'].mean())} ± {corr['spearman_sim_sigma'].std(ddof=1):.2f}"
        re = f"{fmt(corr['spearman_sim_abs_err'].mean())} ± {corr['spearman_sim_abs_err'].std(ddof=1):.2f}"
        common = dict(color=color, lw=2, ms=5.5, capsize=2.5, elinewidth=1, capthick=1, zorder=3)
        styles = {
            "raw": dict(marker="o", ls="-", markeredgecolor=BG, markeredgewidth=0.8),
            "recal": dict(marker="o", ls=(0, (4, 2)), markerfacecolor=BG, markeredgecolor=color,
                          markeredgewidth=1.4),
            "rmse": dict(marker="s", ls="-", markeredgecolor=BG, markeredgewidth=0.8),
        }
        for ax, col, key in ((ax_sig, "mean_sigma", "raw"), (ax_sig, "mean_sigma_recal", "recal"),
                             (ax_rmse, "rmse", "rmse")):
            ax.errorbar(x, b[col].mean().values, yerr=b[col].std(ddof=1).values, **styles[key], **common)
        labels = {
            "raw": f"{ds['label']}: mean σ, raw   (Spearman sim–σ {rs})",
            "recal": f"{ds['label']}: mean σ × val variance scale",
            "rmse": f"{ds['label']}: RMSE   (Spearman sim–|err| {re})",
        }
        handles += [Line2D([], [], color=color, lw=2, ms=5.5, label=labels[k], **styles[k]) for k in styles]
    ax_sig.set_title("Predicted uncertainty (mean σ per bin)", fontsize=10, color=INK)
    ax_rmse.set_title("Observed error (RMSE per bin)", fontsize=10, color=INK)
    ax_sig.set_ylabel("log10 solubility units", fontsize=9, color=MUTED)
    for ax in axes:
        ax.set_xlabel("max Tanimoto similarity to train set\n(one point per test quartile; x = bin mean)",
                      fontsize=9, color=MUTED)
        ax.grid(color=GRID, lw=0.6, zorder=0)
        ax.set_facecolor(BG)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8)
    ax_sig.set_ylim(bottom=0)
    n_seeds = max(len(ds["corr"]) for ds in datasets)
    fig.suptitle(
        f"Does predicted σ grow for test molecules unlike the training set? "
        f"({mode} head, {n_seeds} seeds, error bars = sd; Morgan r={FP_RADIUS}, {FP_BITS} bits)",
        fontsize=10.5, color=INK,
    )
    fig.legend(handles=handles, loc="lower center", ncol=2 if len(datasets) > 1 else 3,
               frameon=False, fontsize=8, handlelength=3.4, bbox_to_anchor=(0.5, -0.005))
    fig.patch.set_facecolor(BG)
    n_rows = int(np.ceil(len(handles) / (2 if len(datasets) > 1 else 3)))
    fig.tight_layout(rect=(0, 0.036 * n_rows + 0.015, 1, 0.95))
    fig.savefig(fig_paths[0], dpi=150, facecolor=BG)
    fig.savefig(fig_paths[1], facecolor=BG)
    plt.close(fig)


def load_dataset(out_dir: Path) -> dict:
    meta = json.loads((out_dir / "summary.json").read_text())
    return {
        "label": meta["label"],
        "bins": pd.read_csv(out_dir / "bins_per_seed.csv"),
        "corr": pd.read_csv(out_dir / "correlations_per_seed.csv"),
    }


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", required=True, type=Path)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--label", default=None,
                    help="legend label (default: '<split> split' from the summaries, else the tag)")
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    ap.add_argument("--compare-with", default="",
                    help="comma-separated output dirs of earlier runs of this script to overlay in the figure")
    args = ap.parse_args(argv)

    runs_root = args.runs_root if args.runs_root.is_absolute() else (Path.cwd() / args.runs_root)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    mols, bins, corrs, metas = [], [], [], []
    splits = set()
    for seed in seeds:
        df, meta = load_seed(runs_root, args.mode, seed)
        s = json.loads((runs_root / "summaries" / f"{args.mode}_seed{seed}.json").read_text()).get("split")
        splits.add(s)
        bt = bin_table(df)
        mols.append(df)
        bins.append(bt)
        corrs.append(corr_row(df, bt, meta))
        metas.append(meta)
        print(f"seed {seed}: n_test={meta['n_test']} n_train={meta['n_train']} "
              f"rho(sim,sigma)={corrs[-1]['spearman_sim_sigma']:+.3f} "
              f"rho(sim,|err|)={corrs[-1]['spearman_sim_abs_err']:+.3f} "
              f"edges={corrs[-1]['q_edges']}")

    mol_df = pd.concat(mols, ignore_index=True)
    bin_df = pd.concat(bins, ignore_index=True)
    corr_df = pd.DataFrame(corrs)
    bin_summary = mean_sd(bin_df, BIN_METRICS, by="bin")
    corr_summary = mean_sd(corr_df, CORR_METRICS)

    label = args.label or (f"{splits.pop()} split" if len(splits) == 1 and None not in splits else args.tag)

    mol_df.to_csv(out / "per_molecule.csv", index=False)
    bin_df.to_csv(out / "bins_per_seed.csv", index=False)
    bin_summary.to_csv(out / "bins_summary.csv", index=False)
    corr_df.to_csv(out / "correlations_per_seed.csv", index=False)
    corr_summary.to_csv(out / "correlations_summary.csv", index=False)

    datasets = [{"label": label, "bins": bin_df, "corr": corr_df}]
    for d in (p for p in args.compare_with.split(",") if p.strip()):
        datasets.append(load_dataset(Path(d)))
    datasets.sort(key=lambda ds: ds["label"])  # stable colour slots across figures
    fig_paths = [args.fig_dir / f"sigma_vs_similarity_{args.tag}.png",
                 args.fig_dir / f"sigma_vs_similarity_{args.tag}.svg"]
    figure(datasets, fig_paths, args.mode)

    summary = {
        "tag": args.tag, "label": label, "runs_root": str(runs_root), "mode": args.mode, "seeds": seeds,
        "fingerprint": {"type": "Morgan", "radius": FP_RADIUS, "bits": FP_BITS, "input": "RDKit canonical SMILES"},
        "binning": "per-seed quartiles of test max-Tanimoto; Q1 = least similar; ties to lower bin",
        "coverage": f"|y - mu| <= {Z90:.4f} * sigma",
        "recal": "sigma_recal = sigma * variance_scale from the summary JSON",
        "correlations_mean_sd": {r["metric"]: [r["mean"], r["std"]] for _, r in corr_summary.iterrows()},
        "bins_mean_sd": {
            r["bin"]: {m: [r[f"{m}_mean"], r[f"{m}_std"]] for m in BIN_METRICS} for _, r in bin_summary.iterrows()
        },
        "per_seed_meta": metas,
        "compared_with": [p for p in args.compare_with.split(",") if p.strip()],
        "figure": [str(p) for p in fig_paths],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    pd.set_option("display.width", 200)
    print("\nCorrelations (mean ± sd over seeds)")
    print(corr_summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nQuartile bins (mean over seeds)")
    show = ["bin"] + [f"{m}_mean" for m in ("n", "mean_sim", "mean_sigma", "mean_sigma_recal", "rmse",
                                              "cov90", "cov90_recal")]
    print(bin_summary[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {out} and {fig_paths[0]}")


if __name__ == "__main__":
    main()
