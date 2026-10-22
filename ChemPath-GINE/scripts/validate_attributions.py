"""Quantitative validation of the IG atom attributions of the NLL-head gine_desc models.

Reads the attributions written by scripts/run_interpretability.py (symmetry-averaged
``ig`` column of atom_attributions.csv) for the 5 seeds of each split and runs:

1. Crippen check (independent chemistry reference).
   RDKit Wildman-Crippen per-atom logP contributions. ``_CalcCrippenContribs`` on a
   molecule with implicit hydrogens returns heavy-atom terms only: the H terms are
   dropped (their sum is not MolLogP). We therefore add explicit Hs, compute all
   contributions and fold every H term onto its heavy-atom neighbour; the folded
   heavy-atom values sum to Crippen.MolLogP (asserted). Higher logP means lower
   solubility, so IG (logS) is compared with -Crippen. MolLogP is not among the
   model's descriptors, so this is an independent check.

2. Deletion (faithfulness) test.
   Each test molecule is re-run through the trained model with the atom embeddings
   h0 of selected atoms set to 0 (the IG baseline); bond embeddings and descriptors
   are unchanged. Masked atoms stay in the graph (they still exchange messages over
   their bond embeddings and are pooled), so this is "replace by the baseline", not
   "remove the atom". For k in {1, 3, 25% of atoms} masking the top-k atoms by |IG|,
   by readout attention, and k random atoms (10 draws, fixed RNG) is compared by
   |delta mu|; the signed test masks the top-k most positive (most negative) IG atoms
   and expects delta mu < 0 (> 0). Supplementary: single-atom occlusion rank
   agreement and the sensitivity-n correlation between -sum(IG of masked atoms) and
   delta mu over the random subsets (Ancona et al., 2018).

3. Sanity: the unmasked forward must reproduce predictions_test.csv mu (tol 1e-4),
   and the recomputed readout attention must match atom_attributions.csv.

Nothing existing is modified: datasets are rebuilt from the run directory's saved
split CSVs into a temporary directory (build_datasets would otherwise rewrite the
run's splits/ and scalers/), and every output goes to --out / --fig-dir.

Example
-------
python scripts/validate_attributions.py --threads 2
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.gridspec import GridSpec
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, rdMolDescriptors
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chempath_gine.data.dataset import collate_molecules
from chempath_gine.data.scaler import load_scaler
from chempath_gine.training.train import _ns, build_datasets, build_model, load_checkpoint
from chempath_gine.extensions import apply_model_extensions
from chempath_gine.uncertainty import GaussianGINE
from chempath_gine.uncertainty.attribution import readout_attention

RDLogger.DisableLog("rdApp.*")

K_SETTINGS = ("k=1", "k=3", "k=25%")
HALOGENS = {"F", "Cl", "Br", "I"}
MIN_ATOMS_PER_MOL_SPEARMAN = 4
MU_TOL = 1e-4
ATTN_TOL = 1e-5

# Figure styling (same tokens as scripts/summarize_calibration.py).
INK, INK2, GRID, SPINE, SURFACE, NEUTRAL = "#0b0b0b", "#52514e", "#e6e5e0", "#b5b3aa", "#fcfcfb", "#7f7d76"
C_BLUE, C_ORANGE = "#2a78d6", "#eb6834"
SPLIT_STYLE = {"random": (C_BLUE, "o"), "scaffold": (C_ORANGE, "s")}
GROUP_SHORT = {
    "C (aromatic)": "C arom",
    "C (aliphatic sp3/sp)": "C sp3/sp",
    "C (sp2, non-aromatic)": "C sp2",
    "O-H (hydroxyl / acid OH)": "O-H",
    "O= (carbonyl C=O)": "O=C",
    "O= (S=O / P=O)": "O=S/P",
    "O (ether / ester)": "O ether/ester",
    "O (aromatic ring)": "O arom",
    "O (nitro)": "O nitro",
    "O (N-oxide / nitrate)": "O N-oxide",
    "N (aromatic)": "N arom",
    "N-H (amine / amide)": "N-H",
    "N (nitro)": "N nitro",
    "N (tertiary / nitrile / other)": "N tert/nitrile",
    "N (N-oxide / nitrate)": "N N-oxide",
    "F (halogen)": "F",
    "Cl (halogen)": "Cl",
    "Br (halogen)": "Br",
    "I (halogen)": "I",
}
LABEL_OFFSETS = {"C (aromatic)": (6, 5), "C (aliphatic sp3/sp)": (16, -22), "O (ether / ester)": (-6, 24),
                 "N (tertiary / nitrile / other)": (-14, -30), "N (aromatic)": (12, -16),
                 "O= (S=O / P=O)": (10, 12), "O= (carbonyl C=O)": (4, -24), "O (aromatic ring)": (-10, 0)}


# ----------------------------------------------------------------------------- helpers
def mean_std(values) -> dict:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n_seeds": 0}
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(arr.mean()), "std": std, "n_seeds": int(arr.size)}


def fmt(ms: dict, nd: int = 3) -> str:
    return f"{ms['mean']:.{nd}f} +/- {ms['std']:.{nd}f}"


def safe_spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def safe_pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(pearsonr(a, b).statistic)


def k_for(label: str, n_atoms: int) -> int:
    if label == "k=1":
        return 1
    if label == "k=3":
        return 3
    return max(1, int(np.floor(0.25 * n_atoms + 0.5)))  # 25% of atoms, round half up, at least 1


# ----------------------------------------------------------------------------- Crippen
def crippen_heavy_atom_contribs(mol: Chem.Mol) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-heavy-atom Crippen logP with attached-H terms folded in.

    Returns (with_h, heavy_only, max |heavy-atom term change after AddHs|).
    ``heavy_only`` is what _CalcCrippenContribs gives on the implicit-H molecule.
    """
    n = mol.GetNumAtoms()
    heavy_only = np.array([c[0] for c in rdMolDescriptors._CalcCrippenContribs(mol)], dtype=float)
    molh = Chem.AddHs(mol)
    contribs = np.array([c[0] for c in rdMolDescriptors._CalcCrippenContribs(molh)], dtype=float)
    folded = contribs[:n].copy()  # AddHs keeps the original atom indices and appends the Hs
    for atom in molh.GetAtoms():
        if atom.GetIdx() >= n:
            nbrs = atom.GetNeighbors()
            if atom.GetAtomicNum() != 1 or len(nbrs) != 1:
                raise ValueError("unexpected added atom")
            folded[nbrs[0].GetIdx()] += contribs[atom.GetIdx()]
    type_shift = float(np.max(np.abs(contribs[:n] - heavy_only))) if n else 0.0
    return folded, heavy_only, type_shift


def crippen_metrics(df: pd.DataFrame) -> dict:
    """df: atoms of one (split, seed) with columns ig, neg_crippen, neg_crippen_heavy_only, group, symbol."""
    out = {
        "n_atoms": int(len(df)),
        "n_molecules": int(df["source_index"].nunique()),
        "pooled_spearman": safe_spearman(df["ig"], df["neg_crippen"]),
        "pooled_pearson": safe_pearson(df["ig"], df["neg_crippen"]),
    }
    nohal = df[~df["symbol"].isin(HALOGENS)]
    out["pooled_spearman_no_halogen"] = safe_spearman(nohal["ig"], nohal["neg_crippen"])
    out["pooled_pearson_no_halogen"] = safe_pearson(nohal["ig"], nohal["neg_crippen"])
    # Beyond atom-type identity: remove the group means of both quantities first.
    resid_ig = df["ig"] - df.groupby("group")["ig"].transform("mean")
    resid_cr = df["neg_crippen"] - df.groupby("group")["neg_crippen"].transform("mean")
    out["within_group_spearman"] = safe_spearman(resid_ig, resid_cr)
    rhos, rhos_h, n_const = [], [], 0
    for _, g in df.groupby("source_index"):
        if len(g) < MIN_ATOMS_PER_MOL_SPEARMAN:
            continue
        r = safe_spearman(g["ig"], g["neg_crippen"])
        if np.isnan(r):
            n_const += 1
        else:
            rhos.append(r)
        rh = safe_spearman(g["ig"], g["neg_crippen_heavy_only"])
        if not np.isnan(rh):
            rhos_h.append(rh)
    out["per_mol_spearman_mean"] = float(np.mean(rhos)) if rhos else float("nan")
    out["per_mol_spearman_median"] = float(np.median(rhos)) if rhos else float("nan")
    out["per_mol_frac_positive"] = float(np.mean(np.array(rhos) > 0)) if rhos else float("nan")
    out["per_mol_n_used"] = len(rhos)
    out["per_mol_n_constant_excluded"] = n_const
    out["heavy_only_pooled_spearman"] = safe_spearman(df["ig"], df["neg_crippen_heavy_only"])
    out["heavy_only_per_mol_spearman_mean"] = float(np.mean(rhos_h)) if rhos_h else float("nan")
    mol = df.groupby("source_index").agg(ig_sum=("ig", "sum"), neg_logp=("neg_crippen", "sum"))
    out["molecule_spearman_atom_ig_sum_vs_neg_MolLogP"] = safe_spearman(mol["ig_sum"], mol["neg_logp"])
    return out


# ----------------------------------------------------------------------------- model
def load_model(run_dir: Path, scratch: Path):
    """Rebuild GaussianGINE + test set from a run dir without writing into it."""
    ckpt = load_checkpoint(run_dir / "best.pt", map_location="cpu")
    cfg_raw = copy.deepcopy(ckpt["config"])
    for part in ("train", "val", "test"):
        cfg_raw["data"][f"{part}_csv"] = str(run_dir / "splits" / f"{part}.csv")
    cfg_raw["data"]["cache_dir"] = str(scratch / "cache")
    cfg = _ns(cfg_raw)
    train_ds, _, test_ds = build_datasets(cfg, scratch)
    for name in ("y", "desc"):
        refit = load_scaler(scratch / "scalers" / f"split.{name}.npz")
        saved = ckpt[f"{name}_scaler"]
        for key in ("mean", "scale"):
            if not np.allclose(refit[key], np.asarray(saved[key]), rtol=1e-5, atol=1e-6):
                raise RuntimeError(f"{run_dir}: refit {name} scaler {key} differs from checkpoint")
    model = GaussianGINE(
        apply_model_extensions(build_model(cfg, train_ds), cfg),
        logvar_min=float(ckpt.get("logvar_min", -9.0)),
        logvar_max=float(ckpt.get("logvar_max", 4.0)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    y_mean = float(ckpt["y_scaler"]["mean"][0])
    y_scale = float(ckpt["y_scaler"]["scale"][0])
    return model, test_ds, y_mean, y_scale


@torch.no_grad()
def masked_mu(model, h0, e0, edge_index, desc, masks: np.ndarray) -> np.ndarray:
    """mu (standardized units) for each row of masks (True = set that atom's h0 to 0)."""
    m, n = masks.shape
    H = h0.unsqueeze(0).repeat(m, 1, 1)
    H[torch.from_numpy(masks)] = 0.0
    H = H.reshape(m * n, -1)
    E = e0.repeat(m, 1)
    n_e = edge_index.size(1)
    offsets = (torch.arange(m) * n).repeat_interleave(n_e)
    ei = edge_index.repeat(1, m) + offsets.unsqueeze(0)
    batch_vec = torch.arange(m).repeat_interleave(n)
    h = model.node_states_from_embeddings(H, E, ei)
    mu, _ = model.readout(h, batch_vec, desc.repeat(m, 1))
    return mu.numpy().astype(np.float64)


def evaluate_split_seed(split, seed, split_code, summ, atom_csv, n_random, rng_seed, scratch_root):
    run_dir = Path(summ["run_dir"])
    scratch = Path(tempfile.mkdtemp(dir=scratch_root, prefix=f"{split}_seed{seed}_"))
    model, test_ds, y_mean, y_scale = load_model(run_dir, scratch)
    saved = pd.read_csv(run_dir / "predictions_test.csv").set_index("source_index")
    atoms_seed = atom_csv[atom_csv["seed"] == seed]
    ig_by_mol = {k: g.sort_values("atom_idx") for k, g in atoms_seed.groupby("source_index")}

    crippen_rows, del_rows, occ_rows, sens_rows = [], [], [], []
    max_mu_dev = max_mu_dev_path = max_attn_dev = max_y_dev = max_logp_dev = max_type_shift = 0.0
    max_logp_csv_dev = 0.0
    seen = set()
    for idx in range(len(test_ds)):
        data = test_ds[idx]
        row = test_ds.df.iloc[int(data.row_idx.item())]
        src = int(row["source_index"])
        seen.add(src)
        mol = Chem.MolFromSmiles(data.smiles)
        n = mol.GetNumAtoms()
        g = ig_by_mol.get(src)
        if g is None or len(g) != n or n != data.num_nodes:
            raise RuntimeError(f"{split} seed {seed}: atom table mismatch for source_index {src}")
        ig = g["ig"].to_numpy(float)
        attn_csv = g["attention"].to_numpy(float)

        # ---- sanity: unmasked forward reproduces saved predictions
        batch = collate_molecules([data])
        with torch.no_grad():
            mu_s, _ = model(batch)
            h0, e0 = model.embed(batch)
        mu_full_s = float(mu_s.item())
        mu_full = mu_full_s * y_scale + y_mean
        mu_path_s = masked_mu(model, h0, e0, batch.edge_index, batch.desc, np.zeros((1, n), bool))[0]
        max_mu_dev = max(max_mu_dev, abs(mu_full - float(saved.loc[src, "mu"])))
        max_mu_dev_path = max(max_mu_dev_path, abs(mu_path_s - mu_full_s) * y_scale)
        max_y_dev = max(max_y_dev, abs(float(data.y.item()) * y_scale + y_mean - float(saved.loc[src, "y_true"])))
        attn = readout_attention(model, data).astype(float)
        max_attn_dev = max(max_attn_dev, float(np.max(np.abs(attn - attn_csv))))

        # ---- Crippen
        cr, cr_heavy, type_shift = crippen_heavy_atom_contribs(mol)
        max_type_shift = max(max_type_shift, type_shift)
        max_logp_dev = max(max_logp_dev, abs(cr.sum() - Crippen.MolLogP(mol)))
        if "MolLogP" in row.index:
            max_logp_csv_dev = max(max_logp_csv_dev, abs(cr.sum() - float(row["MolLogP"])))
        for i in range(n):
            crippen_rows.append(
                {
                    "split": split, "seed": seed, "source_index": src, "atom_idx": i,
                    "symbol": g["symbol"].iloc[i], "group": g["group"].iloc[i],
                    "ig": ig[i], "attention": attn_csv[i],
                    "crippen_logp": cr[i], "crippen_logp_heavy_only": cr_heavy[i],
                    "neg_crippen": -cr[i], "neg_crippen_heavy_only": -cr_heavy[i],
                }
            )

        # ---- deletion masks (one batched forward per molecule)
        masks, tags = [], []

        def add(mask_idx, tag):
            m = np.zeros(n, dtype=bool)
            m[np.asarray(mask_idx, dtype=int)] = True
            masks.append(m)
            tags.append(tag)

        for i in range(n):
            add([i], ("single", i))
        order_abs = np.argsort(-np.abs(ig), kind="stable")
        order_attn = np.argsort(-attn, kind="stable")
        order_pos = np.argsort(-ig, kind="stable")
        order_neg = np.argsort(ig, kind="stable")
        rand_sets = {}
        for label in K_SETTINGS:
            k = k_for(label, n)
            if k >= n:
                continue
            add(order_abs[:k], (label, "ig_abs"))
            add(order_attn[:k], (label, "attention"))
            add(order_pos[:k], (label, "ig_pos"))
            add(order_neg[:k], (label, "ig_neg"))
            rng = np.random.default_rng([rng_seed, split_code, seed, src, k])
            rand_sets[label] = []
            for r in range(n_random):
                pick = rng.choice(n, size=k, replace=False)
                rand_sets[label].append(pick)
                add(pick, (label, "random", r))
        mu_m = masked_mu(model, h0, e0, batch.edge_index, batch.desc, np.stack(masks))
        delta = (mu_m - mu_full_s) * y_scale  # logS; masked minus full
        by_tag = dict(zip(tags, delta))

        single = np.array([by_tag[("single", i)] for i in range(n)])
        occ_rows.append(
            {
                "split": split, "seed": seed, "source_index": src, "n_atoms": n,
                "spearman_ig_vs_neg_delta": safe_spearman(ig, -single) if n >= MIN_ATOMS_PER_MOL_SPEARMAN else np.nan,
                "spearman_absig_vs_absdelta": safe_spearman(np.abs(ig), np.abs(single)) if n >= MIN_ATOMS_PER_MOL_SPEARMAN else np.nan,
                "spearman_attention_vs_absdelta": safe_spearman(attn, np.abs(single)) if n >= MIN_ATOMS_PER_MOL_SPEARMAN else np.nan,
                "pearson_ig_vs_neg_delta": safe_pearson(ig, -single) if n >= MIN_ATOMS_PER_MOL_SPEARMAN else np.nan,
            }
        )
        n_pos, n_neg = int((ig > 0).sum()), int((ig < 0).sum())
        for label in K_SETTINGS:
            k = k_for(label, n)
            rec = {"split": split, "seed": seed, "source_index": src, "n_atoms": n,
                   "k_setting": label, "k": k, "eligible": k < n,
                   "n_pos_ig_atoms": n_pos, "n_neg_ig_atoms": n_neg, "mu_full": mu_full}
            if k < n:
                rd = np.array([by_tag[(label, "random", r)] for r in range(n_random)])
                rec.update(
                    {
                        "delta_ig_abs": by_tag[(label, "ig_abs")],
                        "delta_attention": by_tag[(label, "attention")],
                        "delta_ig_pos": by_tag[(label, "ig_pos")],
                        "delta_ig_neg": by_tag[(label, "ig_neg")],
                        "random_mean_abs_delta": float(np.abs(rd).mean()),
                        "random_mean_delta": float(rd.mean()),
                        "sum_ig_masked_ig_abs": float(ig[order_abs[:k]].sum()),
                        "sum_ig_masked_ig_pos": float(ig[order_pos[:k]].sum()),
                        "sum_ig_masked_ig_neg": float(ig[order_neg[:k]].sum()),
                    }
                )
                for r, pick in enumerate(rand_sets[label]):
                    sens_rows.append(
                        {"split": split, "seed": seed, "k_setting": label, "source_index": src,
                         "neg_sum_ig": -float(ig[pick].sum()), "delta": float(rd[r])}
                    )
            del_rows.append(rec)

    missing = set(ig_by_mol) - seen
    if missing:
        raise RuntimeError(f"{split} seed {seed}: {len(missing)} attributed molecules not in rebuilt test set")
    sanity = {
        "split": split, "seed": seed, "run_dir": str(run_dir), "n_test": len(test_ds),
        "n_saved_predictions": int(len(saved)),
        "max_abs_mu_vs_predictions_test": max_mu_dev,
        "max_abs_mu_embedding_path_vs_forward": max_mu_dev_path,
        "max_abs_y_true_vs_predictions_test": max_y_dev,
        "max_abs_attention_vs_atom_csv": max_attn_dev,
        "max_abs_folded_crippen_sum_vs_MolLogP": max_logp_dev,
        "max_abs_folded_crippen_sum_vs_csv_MolLogP": max_logp_csv_dev,
        "max_abs_heavy_atom_crippen_term_change_after_AddHs": max_type_shift,
        "test_rmse_summary": float(summ["test"]["rmse"]),
    }
    if len(saved) != len(test_ds) or max_mu_dev > MU_TOL or max_mu_dev_path > MU_TOL or max_y_dev > MU_TOL:
        raise AssertionError(f"sanity failed: {sanity}")
    if max_attn_dev > ATTN_TOL:
        raise AssertionError(f"attention does not match atom_attributions.csv: {sanity}")
    if max_logp_dev > 1e-6:
        raise AssertionError(f"folded Crippen contributions do not sum to MolLogP: {sanity}")
    return (pd.DataFrame(crippen_rows), pd.DataFrame(del_rows), pd.DataFrame(occ_rows),
            pd.DataFrame(sens_rows), sanity)


def deletion_metrics(d: pd.DataFrame, sens: pd.DataFrame) -> dict:
    d = d[d["eligible"]]
    out = {"n_molecules": int(len(d))}
    if d.empty:
        return out
    a_ig, a_at, a_rd = d["delta_ig_abs"].abs(), d["delta_attention"].abs(), d["random_mean_abs_delta"]
    out.update(
        {
            "mean_k": float(d["k"].mean()),
            "mean_abs_delta_ig": float(a_ig.mean()),
            "mean_abs_delta_attention": float(a_at.mean()),
            "mean_abs_delta_random": float(a_rd.mean()),
            "ratio_ig_over_random": float(a_ig.mean() / a_rd.mean()),
            "ratio_attention_over_random": float(a_at.mean() / a_rd.mean()),
            "frac_ig_beats_random": float((a_ig > a_rd).mean()),
            "frac_attention_beats_random": float((a_at > a_rd).mean()),
            "frac_ig_beats_attention": float((a_ig > a_at).mean()),
            "mean_delta_top_pos_ig": float(d["delta_ig_pos"].mean()),
            "frac_delta_top_pos_ig_negative": float((d["delta_ig_pos"] < 0).mean()),
            "mean_delta_top_neg_ig": float(d["delta_ig_neg"].mean()),
            "frac_delta_top_neg_ig_positive": float((d["delta_ig_neg"] > 0).mean()),
            "mean_delta_random": float(d["random_mean_delta"].mean()),
        }
    )
    strict_pos = d[d["n_pos_ig_atoms"] >= d["k"]]
    strict_neg = d[d["n_neg_ig_atoms"] >= d["k"]]
    out["strict_n_molecules_k_pos_atoms"] = int(len(strict_pos))
    out["strict_mean_delta_top_pos_ig"] = float(strict_pos["delta_ig_pos"].mean()) if len(strict_pos) else np.nan
    out["strict_n_molecules_k_neg_atoms"] = int(len(strict_neg))
    out["strict_mean_delta_top_neg_ig"] = float(strict_neg["delta_ig_neg"].mean()) if len(strict_neg) else np.nan
    # Additivity check on the selected sets: delta ~= -sum(IG of masked atoms)?
    out["pearson_delta_vs_neg_sum_ig_topabs"] = safe_pearson(d["delta_ig_abs"], -d["sum_ig_masked_ig_abs"])
    if not sens.empty:
        out["sensitivity_n_pearson_random_sets"] = safe_pearson(sens["delta"], sens["neg_sum_ig"])
    return out


# ----------------------------------------------------------------------------- figure
def style_axes(ax):
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
    ax.tick_params(colors=INK2, labelsize=8)


def make_figure(summary: dict, group_df: pd.DataFrame, fig_dir: Path, min_group_atoms: int) -> list[Path]:
    splits = [s for s in ("random", "scaffold") if s in summary["crippen"]]
    n_seeds = max(int(summary["crippen"][s]["pooled_spearman"]["n_seeds"]) for s in splits)
    fig = plt.figure(figsize=(15.5, 10.2))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.12, 1.0], hspace=0.42, wspace=0.27, top=0.92)

    # A: group means, IG vs -Crippen
    ax = fig.add_subplot(gs[0, 0:2])
    style_axes(ax)
    ax.axhline(0, color=SPINE, lw=0.9, zorder=1)
    ax.axvline(0, color=SPINE, lw=0.9, zorder=1)
    for split in splits:
        color, marker = SPLIT_STYLE[split]
        g = group_df[(group_df["split"] == split) & (group_df["n_atoms_all_seeds"] >= min_group_atoms)]
        ax.errorbar(g["neg_crippen_mean"], g["ig_mean"], xerr=g["neg_crippen_std"], yerr=g["ig_std"],
                    fmt=marker, ms=7.5, color=color, ecolor=color, elinewidth=1.1, capsize=0,
                    markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3,
                    label=f"{split} split (pooled Spearman {fmt(summary['crippen'][split]['pooled_spearman'], 2).replace('+/-', '±')})")
        labelled = set() if split == splits[0] else labelled
        for _, r in g.iterrows():
            if r["group"] in labelled:
                continue
            labelled.add(r["group"])
            dx, dy = LABEL_OFFSETS.get(r["group"], (6, 4))
            leader = r["group"] in LABEL_OFFSETS and abs(dx) + abs(dy) > 16
            ax.annotate(GROUP_SHORT.get(r["group"], r["group"]), (r["neg_crippen_mean"], r["ig_mean"]),
                        xytext=(dx, dy), textcoords="offset points", ha="right" if dx < 0 else "left",
                        va="center", fontsize=7.6, color=INK2, zorder=4,
                        arrowprops=dict(arrowstyle="-", color=SPINE, lw=0.7, shrinkA=1, shrinkB=4) if leader else None)
    ax.set_xlabel("mean −Crippen logP contribution per atom (H folded onto heavy atom)", fontsize=9, color=INK2)
    ax.set_ylabel("mean IG per atom (logS)", fontsize=9, color=INK2)
    ax.set_title(f"A  Atom-group means: IG vs −Crippen (groups with ≥{min_group_atoms} atoms; bars = seed std; "
                 "each group labelled once)", fontsize=9.5, color=INK, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    # B: correlation summary
    ax = fig.add_subplot(gs[0, 2])
    style_axes(ax)
    metrics = [("pooled_spearman", "pooled\nSpearman"), ("pooled_pearson", "pooled\nPearson"),
               ("per_mol_spearman_mean", "per-molecule\nSpearman (mean)"),
               ("within_group_spearman", "within-group\nSpearman")]
    x = np.arange(len(metrics))
    width = 0.36
    for j, split in enumerate(splits):
        color, _ = SPLIT_STYLE[split]
        vals = [summary["crippen"][split][m]["mean"] for m, _ in metrics]
        errs = [summary["crippen"][split][m]["std"] for m, _ in metrics]
        pos = x + (j - 0.5) * width
        ax.bar(pos, vals, width - 0.04, yerr=errs, color=color, error_kw={"ecolor": INK2, "elinewidth": 1},
               label=split, zorder=3)
        for p, v, e in zip(pos, vals, errs):
            ax.text(p, v + (e + 0.012 if v >= 0 else -e - 0.012), f"{v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=7.2, color=INK)
    ax.axhline(0, color=SPINE, lw=0.9)
    ax.margins(y=0.14)
    ax.set_xticks(x, [lbl for _, lbl in metrics], fontsize=7.6)
    ax.set_ylabel("correlation of IG with −Crippen", fontsize=9, color=INK2)
    ax.set_title(f"B  Agreement with Crippen (mean ± std, {n_seeds} seeds)", fontsize=9.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # C: |delta mu| deletion
    def split_clusters(ax):
        centers = []
        for j, split in enumerate(splits):
            centers.append(np.arange(len(K_SETTINGS)) + j * (len(K_SETTINGS) + 1))
        ticks = np.concatenate(centers)
        labels = [k.replace("k=25%", "25%") for _ in splits for k in K_SETTINGS]
        ax.set_xticks(ticks, labels, fontsize=8)
        for j, split in enumerate(splits):
            ax.text(centers[j].mean(), -0.16, f"{split} split", transform=ax.get_xaxis_transform(),
                    ha="center", fontsize=8.5, color=INK)
        return centers

    ax = fig.add_subplot(gs[1, 0])
    style_axes(ax)
    centers = split_clusters(ax)
    series = [("mean_abs_delta_ig", "top-k by |IG|", C_BLUE), ("mean_abs_delta_attention", "top-k by attention", C_ORANGE),
              ("mean_abs_delta_random", "k random atoms (10 draws)", NEUTRAL)]
    w = 0.27
    for s_i, (key, lbl, color) in enumerate(series):
        for j, split in enumerate(splits):
            vals = [summary["deletion"][split][k][key]["mean"] for k in K_SETTINGS]
            errs = [summary["deletion"][split][k][key]["std"] for k in K_SETTINGS]
            ax.bar(centers[j] + (s_i - 1) * w, vals, w - 0.03, yerr=errs, color=color,
                   error_kw={"ecolor": INK2, "elinewidth": 1}, label=lbl if j == 0 else None, zorder=3)
    ax.set_ylabel("mean |Δμ| after masking (logS)", fontsize=9, color=INK2)
    ax.set_title("C  Deletion: |Δμ| when atom embeddings → 0", fontsize=9.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")

    # D: fraction beating random
    ax = fig.add_subplot(gs[1, 1])
    style_axes(ax)
    centers = split_clusters(ax)
    series = [("frac_ig_beats_random", "|IG| ranking > random", C_BLUE),
              ("frac_attention_beats_random", "attention ranking > random", C_ORANGE)]
    w = 0.36
    for s_i, (key, lbl, color) in enumerate(series):
        for j, split in enumerate(splits):
            vals = [summary["deletion"][split][k][key]["mean"] for k in K_SETTINGS]
            errs = [summary["deletion"][split][k][key]["std"] for k in K_SETTINGS]
            ax.bar(centers[j] + (s_i - 0.5) * w, vals, w - 0.04, yerr=errs, color=color,
                   error_kw={"ecolor": INK2, "elinewidth": 1}, label=lbl if j == 0 else None, zorder=3)
    ax.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(centers[0][-1] + 1.0, 0.51, "chance", fontsize=7.5, color=INK2, ha="center", va="bottom")
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("fraction of molecules (|Δμ| larger than random)", fontsize=9, color=INK2)
    ax.set_title("D  Per-molecule win rate vs random masking", fontsize=9.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.8, loc="upper left", ncol=2)

    # E: signed deletion
    ax = fig.add_subplot(gs[1, 2])
    style_axes(ax)
    centers = split_clusters(ax)
    series = [("mean_delta_top_pos_ig", "mask top-k positive IG (expect < 0)", C_BLUE),
              ("mean_delta_top_neg_ig", "mask top-k negative IG (expect > 0)", C_ORANGE),
              ("mean_delta_random", "mask k random atoms", NEUTRAL)]
    w = 0.27
    for s_i, (key, lbl, color) in enumerate(series):
        for j, split in enumerate(splits):
            vals = [summary["deletion"][split][k][key]["mean"] for k in K_SETTINGS]
            errs = [summary["deletion"][split][k][key]["std"] for k in K_SETTINGS]
            ax.bar(centers[j] + (s_i - 1) * w, vals, w - 0.03, yerr=errs, color=color,
                   error_kw={"ecolor": INK2, "elinewidth": 1}, label=lbl if j == 0 else None, zorder=3)
    ax.axhline(0, color=SPINE, lw=0.9)
    ax.set_ylabel("mean signed Δμ (logS)", fontsize=9, color=INK2)
    ax.set_title("E  Signed deletion: direction of the effect", fontsize=9.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.6, loc="lower left")

    fig.suptitle("Validating IG atom attributions of gine_desc + Gaussian NLL head on ESOL test sets "
                 f"({n_seeds} seeds per split; error bars = std over seeds)", fontsize=11.5, color=INK, y=0.985)
    fig.patch.set_facecolor(SURFACE)
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = [fig_dir / "attribution_validation.png", fig_dir / "attribution_validation.svg"]
    fig.savefig(paths[0], dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(paths[1], facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------- main
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--splits", default="random,scaffold")
    parser.add_argument("--mode", default="nll")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--runs-template", default=str(ROOT / "calibration_runs" / "esol_{split}"))
    parser.add_argument("--interp-template", default=str(ROOT / "calibration_runs" / "interpretability" / "esol_{split}"))
    parser.add_argument("--out", default=str(ROOT / "calibration_runs" / "interpretability" / "validation"))
    parser.add_argument("--fig-dir", default=str(ROOT.parent / "reports" / "figures" / "calibration_interpretability_2026-10-17"))
    parser.add_argument("--n-random", type=int, default=10)
    parser.add_argument("--rng-seed", type=int, default=20261018)
    parser.add_argument("--min-group-atoms", type=int, default=30, help="figure only")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cr, all_del, all_occ, all_sens, sanity_rows = [], [], [], [], []
    with tempfile.TemporaryDirectory(prefix="validate_attr_") as scratch_root:
        for split_code, split in enumerate(splits):
            atom_csv = pd.read_csv(Path(args.interp_template.format(split=split)) / "atom_attributions.csv")
            runs_root = Path(args.runs_template.format(split=split))
            for seed in seeds:
                with (runs_root / "summaries" / f"{args.mode}_seed{seed}.json").open() as f:
                    summ = json.load(f)
                if summ.get("mode") != args.mode or int(summ.get("seed", seed)) != seed or int(summ.get("member", 0)) != 0:
                    raise RuntimeError(f"unexpected summary for {split} seed {seed}")
                cr, de, oc, se, san = evaluate_split_seed(
                    split, seed, split_code, summ, atom_csv, args.n_random, args.rng_seed, scratch_root
                )
                all_cr.append(cr)
                all_del.append(de)
                all_occ.append(oc)
                all_sens.append(se)
                sanity_rows.append(san)
                print(f"[{split} seed {seed}] n_test={san['n_test']} max|mu-saved|={san['max_abs_mu_vs_predictions_test']:.2e} "
                      f"max|attn-csv|={san['max_abs_attention_vs_atom_csv']:.2e} "
                      f"max|sum crippen - MolLogP|={san['max_abs_folded_crippen_sum_vs_MolLogP']:.1e}", flush=True)

    cr_df = pd.concat(all_cr, ignore_index=True)
    del_df = pd.concat(all_del, ignore_index=True)
    occ_df = pd.concat(all_occ, ignore_index=True)
    sens_df = pd.concat(all_sens, ignore_index=True)
    sanity_df = pd.DataFrame(sanity_rows)

    # ---------------- Crippen: per seed, then mean +/- std
    cr_seed_rows = []
    for (split, seed), g in cr_df.groupby(["split", "seed"]):
        cr_seed_rows.append({"split": split, "seed": seed, **crippen_metrics(g)})
    cr_seed = pd.DataFrame(cr_seed_rows)
    metric_cols = [c for c in cr_seed.columns if c not in ("split", "seed")]
    crippen_summary = {
        split: {m: {**mean_std(g[m]), "per_seed": dict(zip(g["seed"].astype(int).astype(str), g[m].astype(float)))}
                for m in metric_cols}
        for split, g in cr_seed.groupby("split")
    }
    for split in crippen_summary:
        crippen_summary[split]["all_seeds_pooled_spearman"] = safe_spearman(
            cr_df.loc[cr_df["split"] == split, "ig"], cr_df.loc[cr_df["split"] == split, "neg_crippen"])

    # Group means side by side (per seed -> mean/std over seeds)
    per = cr_df.groupby(["split", "group", "seed"]).agg(
        n=("ig", "size"), ig=("ig", "mean"), neg_crippen=("neg_crippen", "mean"))
    grp_rows = []
    for (split, group), g in per.groupby(level=["split", "group"]):
        ig_ms, cr_ms = mean_std(g["ig"]), mean_std(g["neg_crippen"])
        grp_rows.append(
            {
                "split": split, "group": group, "n_atoms_all_seeds": int(g["n"].sum()),
                "n_seeds_present": int(len(g)),
                "ig_mean": ig_ms["mean"], "ig_std": ig_ms["std"],
                "neg_crippen_mean": cr_ms["mean"], "neg_crippen_std": cr_ms["std"],
                "n_seeds_ig_positive": int((g["ig"] > 0).sum()),
                "sign_agrees": bool(np.sign(ig_ms["mean"]) == np.sign(cr_ms["mean"])),
                **{f"ig_seed{int(s)}": float(v) for s, v in zip(g.index.get_level_values("seed"), g["ig"])},
            }
        )
    group_df = pd.DataFrame(grp_rows).sort_values(["split", "neg_crippen_mean"])
    group_level = {}
    for split, g in group_df.groupby("split"):
        big = g[g["n_atoms_all_seeds"] >= args.min_group_atoms]
        group_level[split] = {
            "n_groups": int(len(big)),
            "spearman_group_means": safe_spearman(big["ig_mean"], big["neg_crippen_mean"]),
            "pearson_group_means": safe_pearson(big["ig_mean"], big["neg_crippen_mean"]),
            "n_sign_agree": int(big["sign_agrees"].sum()),
            "sign_disagree_groups": big.loc[~big["sign_agrees"], "group"].tolist(),
        }

    # ---------------- Deletion: per seed, then mean +/- std
    del_seed_rows = []
    for (split, seed, label), d in del_df.groupby(["split", "seed", "k_setting"]):
        s = sens_df[(sens_df["split"] == split) & (sens_df["seed"] == seed) & (sens_df["k_setting"] == label)]
        del_seed_rows.append({"split": split, "seed": seed, "k_setting": label, **deletion_metrics(d, s)})
    del_seed = pd.DataFrame(del_seed_rows)
    dmetric_cols = [c for c in del_seed.columns if c not in ("split", "seed", "k_setting")]
    deletion_summary = {}
    for (split, label), g in del_seed.groupby(["split", "k_setting"]):
        deletion_summary.setdefault(split, {})[label] = {
            m: {**mean_std(g[m]), "per_seed": dict(zip(g["seed"].astype(int).astype(str), g[m].astype(float)))}
            for m in dmetric_cols
        }

    # ---------------- Single-atom occlusion agreement
    occ_seed = occ_df.groupby(["split", "seed"]).agg(
        n_molecules=("spearman_ig_vs_neg_delta", lambda s: int(s.notna().sum())),
        spearman_ig_vs_neg_delta=("spearman_ig_vs_neg_delta", "mean"),
        pearson_ig_vs_neg_delta=("pearson_ig_vs_neg_delta", "mean"),
        spearman_absig_vs_absdelta=("spearman_absig_vs_absdelta", "mean"),
        spearman_attention_vs_absdelta=("spearman_attention_vs_absdelta", "mean"),
    ).reset_index()
    occlusion_summary = {
        split: {m: mean_std(g[m]) for m in occ_seed.columns if m not in ("split", "seed")}
        for split, g in occ_seed.groupby("split")
    }

    # ---------------- Crippen implicit-H verification (documented numbers)
    hv = cr_df.drop_duplicates(["split", "source_index", "atom_idx"])
    mol_h = hv.groupby(["split", "source_index"]).agg(
        with_h=("crippen_logp", "sum"), heavy_only=("crippen_logp_heavy_only", "sum"))
    crippen_verification = {
        "note": ("rdMolDescriptors._CalcCrippenContribs on the implicit-H molecule returns heavy-atom terms only; "
                 "H terms are missing (their sum != MolLogP). Contributions were computed on Chem.AddHs(mol) and "
                 "each H term was added to its heavy-atom neighbour; heavy-atom types/terms are unchanged by AddHs."),
        "max_abs_folded_sum_minus_MolLogP": float(sanity_df["max_abs_folded_crippen_sum_vs_MolLogP"].max()),
        "max_abs_folded_sum_minus_dataset_MolLogP_column": float(sanity_df["max_abs_folded_crippen_sum_vs_csv_MolLogP"].max()),
        "max_abs_heavy_atom_term_change_after_AddHs": float(sanity_df["max_abs_heavy_atom_crippen_term_change_after_AddHs"].max()),
        "mean_abs_missing_H_logp_per_molecule_if_not_folded": float((mol_h["with_h"] - mol_h["heavy_only"]).abs().mean()),
    }

    summary = {
        "description": "IG atom attribution validation, gine_desc + Gaussian NLL head (mode=%s), ESOL test sets" % args.mode,
        "settings": {
            "splits": splits, "seeds": seeds, "ig_column": "ig (symmetry-averaged)",
            "k_settings": {"k=1": 1, "k=3": 3, "k=25%": "max(1, round_half_up(0.25 * n_atoms))"},
            "eligibility": "a molecule enters a k setting only if k < n_atoms",
            "n_random_draws": args.n_random, "rng_seed": args.rng_seed,
            "mask": "atom embedding h0 := 0 (IG baseline); bond embeddings and descriptors unchanged",
            "delta_mu": "mu(masked) - mu(full), logS units",
            "per_molecule_spearman_min_atoms": MIN_ATOMS_PER_MOL_SPEARMAN,
            "aggregation": "metric per seed (over molecules/atoms), then mean +/- std (ddof=1) over seeds",
        },
        "sanity": {
            "max_abs_mu_vs_predictions_test": float(sanity_df["max_abs_mu_vs_predictions_test"].max()),
            "max_abs_mu_embedding_path_vs_forward": float(sanity_df["max_abs_mu_embedding_path_vs_forward"].max()),
            "max_abs_attention_vs_atom_csv": float(sanity_df["max_abs_attention_vs_atom_csv"].max()),
            "tolerance_mu": MU_TOL, "passed": True,
        },
        "crippen_verification": crippen_verification,
        "crippen": crippen_summary,
        "crippen_group_level": group_level,
        "deletion": deletion_summary,
        "single_atom_occlusion": occlusion_summary,
    }

    files = {
        "crippen_atom_table": out_dir / "crippen_atom_table.csv",
        "crippen_per_seed": out_dir / "crippen_per_seed.csv",
        "crippen_group_means": out_dir / "crippen_group_means.csv",
        "deletion_per_molecule": out_dir / "deletion_per_molecule.csv",
        "deletion_per_seed": out_dir / "deletion_per_seed.csv",
        "occlusion_per_molecule": out_dir / "occlusion_single_atom_per_molecule.csv",
        "sanity_checks": out_dir / "sanity_checks.csv",
        "summary_json": out_dir / "validation_summary.json",
    }
    cr_df.to_csv(files["crippen_atom_table"], index=False)
    cr_seed.to_csv(files["crippen_per_seed"], index=False)
    group_df.to_csv(files["crippen_group_means"], index=False)
    del_df.to_csv(files["deletion_per_molecule"], index=False)
    del_seed.to_csv(files["deletion_per_seed"], index=False)
    occ_df.to_csv(files["occlusion_per_molecule"], index=False)
    sanity_df.to_csv(files["sanity_checks"], index=False)
    fig_paths = make_figure(summary, group_df, Path(args.fig_dir).resolve(), args.min_group_atoms)
    summary["files"] = {k: str(v) for k, v in files.items()}
    summary["figures"] = [str(p) for p in fig_paths]
    with files["summary_json"].open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    # ---------------- console report
    print("\n== Crippen check (IG vs -Crippen, mean +/- std over seeds) ==")
    for split in splits:
        c = crippen_summary[split]
        print(f"[{split}] pooled Spearman {fmt(c['pooled_spearman'])} | pooled Pearson {fmt(c['pooled_pearson'])} | "
              f"per-mol Spearman {fmt(c['per_mol_spearman_mean'])} (n={c['per_mol_n_used']['mean']:.0f}/seed) | "
              f"within-group {fmt(c['within_group_spearman'])} | no-halogen Spearman {fmt(c['pooled_spearman_no_halogen'])}")
        print(f"   group level: {group_level[split]}")
    print(group_df[["split", "group", "n_atoms_all_seeds", "ig_mean", "ig_std", "neg_crippen_mean",
                    "neg_crippen_std", "sign_agrees"]].to_string(index=False))
    print("\n== Deletion test ==")
    for split in splits:
        for label in K_SETTINGS:
            d = deletion_summary[split][label]
            print(f"[{split} {label}] n={d['n_molecules']['mean']:.0f} |dmu| IG {fmt(d['mean_abs_delta_ig'])} "
                  f"attn {fmt(d['mean_abs_delta_attention'])} rand {fmt(d['mean_abs_delta_random'])} | "
                  f"IG>rand {fmt(d['frac_ig_beats_random'])} attn>rand {fmt(d['frac_attention_beats_random'])} | "
                  f"dmu top+IG {fmt(d['mean_delta_top_pos_ig'])} top-IG {fmt(d['mean_delta_top_neg_ig'])} "
                  f"rand {fmt(d['mean_delta_random'])} | sens-n r {fmt(d['sensitivity_n_pearson_random_sets'])}")
    print("\n== Single-atom occlusion ==")
    for split in splits:
        print(split, {k: fmt(v) for k, v in occlusion_summary[split].items() if k != "n_molecules"})
    print("\nfiles:", json.dumps(summary["files"], indent=1), "\nfigures:", summary["figures"])


if __name__ == "__main__":
    main()
