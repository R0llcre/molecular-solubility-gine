"""Atom/bond attributions for trained GaussianGINE (gine_desc + NLL head) checkpoints.

For every seed's checkpoint under --runs-root, computes on the test split:
  * integrated gradients of the predicted mean over atom/bond embeddings and
    descriptors (chempath_gine.uncertainty.attribution), and
  * GraphPool readout-attention weights.

Then for one showcase checkpoint (the seed with the median test RMSE) it picks
the most accurate and the largest-error test molecules and draws RDKit
highlight maps (SVG + PNG). A dataset-level table aggregates atom-level IG by
chemical atom environment across all seeds, so the chemistry reading is not
based on a handful of hand-picked molecules.

Example
-------
python scripts/run_interpretability.py \
  --runs-root calibration_runs/esol_random --mode nll \
  --fig-dir ../reports/figures/calibration_interpretability_2026-10-17 \
  --out calibration_runs/interpretability/esol_random
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import cm, colors as mcolors
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chempath_gine.data.dataset import collate_molecules
from chempath_gine.training.train import _ns, build_datasets, build_model, load_checkpoint
from chempath_gine.uncertainty import GaussianGINE
from chempath_gine.uncertainty.attribution import (
    integrated_gradients,
    readout_attention,
    symmetrize_attributions,
)

SIGNED_CMAP = matplotlib.colormaps["RdBu"]  # red = lowers predicted logS, blue = raises it
ATTN_CMAP = matplotlib.colormaps["Oranges"]


_NITRO = [Chem.MolFromSmarts(x) for x in ("[#6][N+](=O)[O-]", "[#6]N(=O)=O")]


def nitro_atoms(mol: Chem.Mol) -> set[int]:
    """Atom indices of C-NO2 groups (N and both O); excludes N-oxides and nitrates."""
    out: set[int] = set()
    for patt in _NITRO:
        for match in mol.GetSubstructMatches(patt):
            out.update(match[1:])
    return out


def atom_group(atom: Chem.Atom, nitro: set[int] | None = None) -> str:
    nitro = nitro if nitro is not None else nitro_atoms(atom.GetOwningMol())
    sym = atom.GetSymbol()
    n_h = atom.GetTotalNumHs()
    idx = atom.GetIdx()
    if sym == "O":
        if idx in nitro:
            return "O (nitro)"
        if any(n.GetSymbol() == "N" and n.GetFormalCharge() > 0 for n in atom.GetNeighbors()):
            return "O (N-oxide / nitrate)"
        if n_h > 0:
            return "O-H (hydroxyl / acid OH)"
        double_partners = [
            b.GetOtherAtom(atom).GetSymbol()
            for b in atom.GetBonds()
            if b.GetBondType() == Chem.BondType.DOUBLE
        ]
        if "C" in double_partners:
            return "O= (carbonyl C=O)"
        if double_partners:
            return "O= (S=O / P=O)"
        if atom.GetIsAromatic():
            return "O (aromatic ring)"
        return "O (ether / ester)"
    if sym == "N":
        if idx in nitro:
            return "N (nitro)"
        if atom.GetFormalCharge() > 0 and any(n.GetSymbol() == "O" for n in atom.GetNeighbors()):
            return "N (N-oxide / nitrate)"
        if atom.GetIsAromatic():
            return "N (aromatic)"
        if n_h > 0:
            return "N-H (amine / amide)"
        return "N (tertiary / nitrile / other)"
    if sym == "C":
        if atom.GetIsAromatic():
            return "C (aromatic)"
        if any(b.GetBondType() == Chem.BondType.DOUBLE for b in atom.GetBonds()):
            return "C (sp2, non-aromatic)"
        return "C (aliphatic sp3/sp)"
    if sym in {"F", "Cl", "Br", "I"}:
        return f"{sym} (halogen)"
    if sym in {"S", "P"}:
        return f"{sym}"
    return "other"


def _read_complete(path: Path, retries: int = 20) -> bytes:
    """Read a file fully; retry when an iCloud-evicted (dataless) file comes back short."""
    import time

    expected = path.stat().st_size
    for attempt in range(retries):
        try:
            data = path.read_bytes()
        except OSError:
            data = b""
        if len(data) == expected and expected > 0:
            return data
        time.sleep(1.0 + attempt)
    raise OSError(f"could not read {path} completely ({expected} bytes expected)")


def load_gaussian(run_dir: Path):
    import io

    ckpt = load_checkpoint(io.BytesIO(_read_complete(run_dir / "best.pt")), map_location="cpu")
    cfg_raw = ckpt["config"]
    cfg = _ns(cfg_raw)
    train_ds, val_ds, test_ds = build_datasets(cfg, run_dir)
    from chempath_gine.extensions import apply_model_extensions

    base = apply_model_extensions(build_model(cfg, train_ds), cfg)
    model = GaussianGINE(base)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    y_scale = float(ckpt["y_scaler"]["scale"][0])
    y_mean = float(ckpt["y_scaler"]["mean"][0])
    return model, test_ds, y_mean, y_scale, ckpt


def attribute_split(model, test_ds, y_mean, y_scale, steps: int, seed: int):
    mol_rows, atom_rows, desc_rows, details = [], [], [], {}
    for k in range(len(test_ds)):
        data = test_ds[k]
        smiles = data.smiles
        mol = Chem.MolFromSmiles(smiles)
        if mol.GetNumAtoms() != data.num_nodes:
            raise ValueError(f"atom count mismatch for {smiles}")
        with torch.no_grad():
            mu_s, lv_s = model(collate_molecules([data]))
        mu = float(mu_s.item() * y_scale + y_mean)
        sigma = float(np.exp(0.5 * lv_s.item()) * y_scale)
        y_true = float(data.y.item() * y_scale + y_mean)
        ig = integrated_gradients(model, data, n_bonds=mol.GetNumBonds(), steps=steps, y_scale=y_scale)
        ig["atom_raw"], ig["bond_raw"] = ig["atom"], ig["bond"]
        ig["atom"], ig["bond"] = symmetrize_attributions(mol, ig["atom_raw"], ig["bond_raw"])
        attn = readout_attention(model, data)
        src_idx = int(test_ds.df.iloc[int(data.row_idx.item())]["source_index"])
        graph_sum = float(ig["atom"].sum() + ig["bond"].sum() + ig["sentinel_edge"])
        mol_rows.append(
            {
                "seed": seed,
                "source_index": src_idx,
                "SMILES": smiles,
                "heavy_atoms": mol.GetNumAtoms(),
                "y_true": y_true,
                "mu": mu,
                "sigma": sigma,
                "abs_error": abs(y_true - mu),
                "ig_delta": ig["delta"],
                "ig_sum": ig["attr_sum"],
                "ig_completeness_gap": ig["completeness_gap"],
                "ig_graph_sum": graph_sum,
                "ig_desc_sum": float(ig["desc"].sum()),
                "mu_baseline": float(ig["mu_baseline_scaled"] * y_scale + y_mean),
            }
        )
        nitro = nitro_atoms(mol)
        for atom in mol.GetAtoms():
            i = atom.GetIdx()
            atom_rows.append(
                {
                    "seed": seed,
                    "source_index": src_idx,
                    "atom_idx": i,
                    "symbol": atom.GetSymbol(),
                    "group": atom_group(atom, nitro),
                    "ig": float(ig["atom"][i]),
                    "ig_raw": float(ig["atom_raw"][i]),
                    "attention": float(attn[i]),
                }
            )
        for name, val in zip(test_ds.descriptor_cols, ig["desc"]):
            desc_rows.append({"seed": seed, "source_index": src_idx, "descriptor": name, "ig": float(val)})
        details[src_idx] = {
            "mol": mol, "ig": ig, "attn": attn, "row": mol_rows[-1],
            "desc_names": list(test_ds.descriptor_cols),
        }
    return pd.DataFrame(mol_rows), pd.DataFrame(atom_rows), pd.DataFrame(desc_rows), details


def _signed_colors(values: np.ndarray, vmax: float):
    out = {}
    for i, v in enumerate(values):
        rgba = SIGNED_CMAP(0.5 + 0.5 * float(np.clip(v / vmax, -1, 1)))
        out[i] = tuple(float(c) for c in rgba[:3])
    return out


def _draw(mol, atom_vals, bond_vals, legend, signed: bool, w=460, h=360, fmt="svg"):
    drawer = rdMolDraw2D.MolDraw2DSVG(w, h) if fmt == "svg" else rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = drawer.drawOptions()
    opts.legendFontSize = 15
    opts.highlightBondWidthMultiplier = 14
    opts.fillHighlights = True
    opts.padding = 0.08
    if signed:
        vmax = max(float(np.max(np.abs(atom_vals))), float(np.max(np.abs(bond_vals))) if len(bond_vals) else 0.0)
        vmax = vmax if vmax > 0 else 1.0
        atom_cols = _signed_colors(atom_vals, vmax)
        bond_cols = _signed_colors(bond_vals, vmax) if len(bond_vals) else {}
        radii = {i: 0.25 + 0.30 * abs(float(v)) / vmax for i, v in enumerate(atom_vals)}
    else:
        vmax = float(np.max(atom_vals)) if len(atom_vals) else 1.0
        atom_cols = {
            i: tuple(float(c) for c in ATTN_CMAP(0.1 + 0.85 * float(v) / vmax)[:3])
            for i, v in enumerate(atom_vals)
        }
        bond_cols = {}
        radii = {i: 0.25 + 0.30 * float(v) / vmax for i, v in enumerate(atom_vals)}
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        legend=legend,
        highlightAtoms=list(atom_cols.keys()),
        highlightAtomColors=atom_cols,
        highlightAtomRadii=radii,
        highlightBonds=list(bond_cols.keys()),
        highlightBondColors=bond_cols,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText(), vmax


def draw_molecule_panels(entry: dict, tag: str, fig_dir: Path) -> dict:
    mol, ig, attn, row = entry["mol"], entry["ig"], entry["attn"], entry["row"]
    legend = (
        f"{tag}: logS true {row['y_true']:.2f} | pred {row['mu']:.2f} +/- {row['sigma']:.2f}"
    )
    paths = {}
    for kind, vals, bvals, signed in (
        ("ig", ig["atom"], ig["bond"], True),
        ("attention", attn, np.zeros(0), False),
    ):
        svg, vmax = _draw(mol, vals, bvals, legend, signed, fmt="svg")
        png, _ = _draw(mol, vals, bvals, legend, signed, fmt="png")
        svg_path = fig_dir / f"{tag}_{kind}.svg"
        png_path = fig_dir / f"{tag}_{kind}.png"
        svg_path.write_text(svg, encoding="utf-8")
        png_path.write_bytes(png)
        paths[kind] = {"svg": str(svg_path), "png": str(png_path), "vmax": vmax}
    return paths


def overview_figure(selected: list[tuple[str, dict]], fig_dir: Path, title: str) -> Path:
    import io

    from PIL import Image

    n = len(selected)
    fig, axes = plt.subplots(n, 2, figsize=(9.2, 3.25 * n))
    axes = np.atleast_2d(axes)
    fig.subplots_adjust(top=1 - 0.85 / (3.25 * n), bottom=0.9 / (3.25 * n), hspace=0.32, wspace=0.04)
    for r, (tag, entry) in enumerate(selected):
        for c, kind in enumerate(("ig", "attention")):
            vals = entry["ig"]["atom"] if kind == "ig" else entry["attn"]
            bvals = entry["ig"]["bond"] if kind == "ig" else np.zeros(0)
            png, vmax = _draw(entry["mol"], vals, bvals, "", kind == "ig", w=520, h=380, fmt="png")
            axes[r, c].imshow(Image.open(io.BytesIO(png)))
            axes[r, c].axis("off")
            row = entry["row"]
            if kind == "ig":
                axes[r, c].set_title(
                    f"{tag}  IG (max |a|={vmax:.2f} logS)\ntrue {row['y_true']:.2f}, "
                    f"pred {row['mu']:.2f} ± {row['sigma']:.2f}, |err| {row['abs_error']:.2f}",
                    fontsize=9,
                )
            else:
                axes[r, c].set_title(f"{tag}  readout attention (max={vmax:.2f})", fontsize=9)
    sm = cm.ScalarMappable(norm=mcolors.Normalize(-1, 1), cmap=SIGNED_CMAP)
    cax = fig.add_axes([0.14, 0.35 / (3.25 * n), 0.32, 0.12 / (3.25 * n)])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("IG / max|IG| per molecule  (blue: raises predicted logS = more soluble; red: lowers it)", fontsize=8)
    fig.suptitle(title, fontsize=11, y=1 - 0.15 / (3.25 * n))
    out = fig_dir / "overview_ig_attention.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--mode", default="nll")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--n-best", type=int, default=4)
    parser.add_argument("--n-worst", type=int, default=4)
    parser.add_argument("--min-heavy-best", type=int, default=8)
    parser.add_argument("--fig-dir", default=None, help="draw showcase figures here")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)

    runs_root = Path(args.runs_root).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        with (runs_root / "summaries" / f"{args.mode}_seed{seed}.json").open() as f:
            summaries[seed] = json.load(f)

    all_mol, all_atom, all_desc, details_by_seed = [], [], [], {}
    for seed, summ in summaries.items():
        model, test_ds, y_mean, y_scale, _ = load_gaussian(Path(summ["run_dir"]))
        mol_df, atom_df, desc_df, details = attribute_split(model, test_ds, y_mean, y_scale, args.steps, seed)
        # The attribution forward must reproduce the saved test predictions.
        saved = pd.read_csv(Path(summ["run_dir"]) / "predictions_test.csv")
        merged = mol_df.merge(saved[["source_index", "mu", "sigma"]], on="source_index", suffixes=("", "_saved"))
        dev = float(np.max(np.abs(merged["mu"] - merged["mu_saved"])))
        print(f"seed={seed} n_test={len(mol_df)} max|mu-mu_saved|={dev:.2e} "
              f"max|IG gap|={mol_df['ig_completeness_gap'].abs().max():.4f}", flush=True)
        if dev > 1e-4 or len(merged) != len(mol_df):
            raise RuntimeError(f"prediction mismatch for seed {seed}: {dev}")
        all_mol.append(mol_df)
        all_atom.append(atom_df)
        all_desc.append(desc_df)
        details_by_seed[seed] = details

    mol_df = pd.concat(all_mol, ignore_index=True)
    atom_df = pd.concat(all_atom, ignore_index=True)
    desc_df = pd.concat(all_desc, ignore_index=True)
    mol_df.to_csv(out_dir / "molecule_attributions.csv", index=False)
    atom_df.to_csv(out_dir / "atom_attributions.csv", index=False)
    desc_df.to_csv(out_dir / "descriptor_attributions.csv", index=False)

    # Dataset-level reading: mean atom IG by chemical environment, per seed, then across seeds.
    per_seed = atom_df.groupby(["group", "seed"])["ig"].mean().unstack("seed")
    group_tbl = pd.DataFrame(
        {
            "n_atoms_total": atom_df.groupby("group")["ig"].size(),
            "mean_ig": atom_df.groupby("group")["ig"].mean(),
            "median_ig": atom_df.groupby("group")["ig"].median(),
            "frac_positive": atom_df.groupby("group")["ig"].apply(lambda s: float((s > 0).mean())),
            "seed_mean_std": per_seed.std(axis=1, ddof=1),
            "n_seeds_present": per_seed.notna().sum(axis=1),
            "n_seeds_positive_mean": (per_seed > 0).sum(axis=1),
            "mean_attention": atom_df.groupby("group")["attention"].mean(),
        }
    ).sort_values("mean_ig")
    group_tbl.to_csv(out_dir / "atom_group_ig_summary.csv")
    desc_tbl = pd.DataFrame(
        {
            "mean_ig": desc_df.groupby("descriptor")["ig"].mean(),
            "mean_abs_ig": desc_df.groupby("descriptor")["ig"].apply(lambda s: float(s.abs().mean())),
        }
    ).sort_values("mean_abs_ig", ascending=False)
    desc_tbl.to_csv(out_dir / "descriptor_ig_summary.csv")
    # Does the MolecularWeight descriptor carry the halogen penalty? Group molecules by halogen content.
    symbols = atom_df.groupby(["seed", "source_index"])["symbol"].apply(set)

    def halogen_class(syms: set) -> str:
        if syms & {"Br", "I"}:
            return "contains Br or I"
        if "Cl" in syms:
            return "contains Cl (no Br/I)"
        if "F" in syms:
            return "contains F only"
        return "no halogen"

    mw = desc_df[desc_df["descriptor"] == "MolecularWeight"].set_index(["seed", "source_index"])["ig"]
    hal = pd.DataFrame({"halogen_class": symbols.map(halogen_class), "mw_ig": mw}).join(
        mol_df.set_index(["seed", "source_index"])[["ig_graph_sum", "ig_desc_sum", "y_true"]]
    )
    hal.groupby("halogen_class").agg(
        n=("mw_ig", "size"), mean_mw_ig=("mw_ig", "mean"), mean_graph_sum=("ig_graph_sum", "mean"),
        mean_desc_sum=("ig_desc_sum", "mean"), mean_y_true=("y_true", "mean"),
    ).to_csv(out_dir / "halogen_mw_check.csv")
    share = pd.DataFrame(
        {
            "graph_abs_share": mol_df["ig_graph_sum"].abs()
            / (mol_df["ig_graph_sum"].abs() + mol_df["ig_desc_sum"].abs()),
        }
    )
    # Attention vs |IG| agreement inside each molecule (Spearman), averaged.
    from scipy.stats import spearmanr

    rhos = []
    for (_, _), g in atom_df.groupby(["seed", "source_index"]):
        if len(g) >= 4 and g["attention"].std() > 0 and g["ig"].abs().std() > 0:
            rhos.append(spearmanr(g["attention"], g["ig"].abs()).correlation)
    stats = {
        "n_molecule_instances": int(len(mol_df)),
        "ig_steps": args.steps,
        "max_abs_completeness_gap": float(mol_df["ig_completeness_gap"].abs().max()),
        "median_abs_completeness_gap": float(mol_df["ig_completeness_gap"].abs().median()),
        "median_abs_delta": float(mol_df["ig_delta"].abs().median()),
        "mean_graph_abs_share": float(share["graph_abs_share"].mean()),
        "mean_spearman_attention_vs_abs_ig": float(np.nanmean(rhos)) if rhos else float("nan"),
        "n_molecules_for_spearman": len(rhos),
    }

    if args.fig_dir:
        fig_dir = Path(args.fig_dir).resolve()
        fig_dir.mkdir(parents=True, exist_ok=True)
        rmse_by_seed = {s: summaries[s]["test"]["rmse"] for s in summaries}
        ordered = sorted(rmse_by_seed, key=lambda s: rmse_by_seed[s])
        show_seed = ordered[len(ordered) // 2]
        details = details_by_seed[show_seed]
        sdf = mol_df[mol_df["seed"] == show_seed]
        best = (
            sdf[sdf["heavy_atoms"] >= args.min_heavy_best]
            .sort_values("abs_error")
            .head(args.n_best)
        )
        worst = sdf.sort_values("abs_error", ascending=False).head(args.n_worst)
        selected, sel_rows = [], []
        for prefix, frame in (("best", best), ("worst", worst)):
            for rank, (_, r) in enumerate(frame.iterrows(), 1):
                tag = f"{prefix}{rank}"
                entry = details[int(r["source_index"])]
                paths = draw_molecule_panels(entry, tag, fig_dir)
                mol = entry["mol"]
                nitro = nitro_atoms(mol)
                top_atoms = sorted(
                    [(a.GetIdx(), a.GetSymbol(), atom_group(a, nitro), float(entry["ig"]["atom"][a.GetIdx()]))
                     for a in mol.GetAtoms()],
                    key=lambda t: t[3],
                )
                sel_rows.append(
                    {
                        "tag": tag,
                        **{k: entry["row"][k] for k in ("source_index", "SMILES", "heavy_atoms", "y_true", "mu", "sigma", "abs_error", "ig_delta", "ig_graph_sum", "ig_desc_sum", "mu_baseline")},
                        "most_negative_atoms": "; ".join(f"{s}{i} {g} {v:+.3f}" for i, s, g, v in top_atoms[:3]),
                        "most_positive_atoms": "; ".join(f"{s}{i} {g} {v:+.3f}" for i, s, g, v in top_atoms[::-1][:3]),
                        "top_attention_atoms": "; ".join(
                            f"{mol.GetAtomWithIdx(int(i)).GetSymbol()}{int(i)} {entry['attn'][int(i)]:.2f}"
                            for i in np.argsort(-entry["attn"])[:3]
                        ),
                        "desc_ig": "; ".join(
                            f"{n} {v:+.3f}" for n, v in zip(entry["desc_names"], entry["ig"]["desc"])
                        ),
                        "ig_svg": paths["ig"]["svg"],
                        "ig_png": paths["ig"]["png"],
                        "attention_svg": paths["attention"]["svg"],
                        "attention_png": paths["attention"]["png"],
                    }
                )
                selected.append((tag, entry))
        sel_df = pd.DataFrame(sel_rows)
        sel_df.to_csv(out_dir / "showcase_molecules.csv", index=False)
        ov = overview_figure(
            selected,
            fig_dir,
            f"gine_desc + Gaussian NLL head, ESOL {summaries[show_seed]['split']} split, seed {show_seed} "
            f"(median test RMSE {rmse_by_seed[show_seed]:.3f})",
        )
        stats.update({"showcase_seed": int(show_seed), "overview_png": str(ov)})

    with (out_dir / "interpretability_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(group_tbl.to_string())
    print(desc_tbl.to_string())


if __name__ == "__main__":
    main()
