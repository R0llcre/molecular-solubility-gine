"""Embedding-level integrated gradients and readout attention for GaussianGINE.

Integrated gradients (Sundararajan et al., 2017) are computed jointly over
three inputs of the predicted mean mu:

* atom embeddings  h0 = AtomEncoder(atom features)        baseline 0
* bond embeddings  e0 = EdgeEncoder(bond features)        baseline 0
* descriptors      d  (train-standardized, so 0 = train mean)  baseline 0

The path x(alpha) = alpha * x is integrated with a midpoint Riemann sum.
Attributions are summed over the hidden dimension to give one number per atom,
per bond (sum of both directed edges) and per descriptor, then converted to
original target units (multiplied by the target scaler's scale). By the
completeness axiom, atoms + bonds + descriptors ~= mu(x) - mu(baseline).

GraphPool contains a max-pool branch. For chemically equivalent atoms (graph
automorphisms) the max is tied along the whole IG path, so the gradient goes to
whichever tied atom torch.max happens to pick and IG splits credit between
equivalent atoms arbitrarily. symmetrize_attributions() averages atom IG within
RDKit symmetry classes (CanonicalRankAtoms with breakTies=False) and bond IG
within classes of (end-atom classes, bond type). The per-molecule sum is
unchanged, so completeness still holds.

Readout attention is the softmax weight of the attention-pooling branch in
GraphPool over the final node states (it sums to 1 over atoms). gine_desc has
no transformer attention layers, so this is the only attention in the model.
"""

from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem

from chempath_gine.data.dataset import collate_molecules


@torch.no_grad()
def readout_attention(model, data) -> np.ndarray:
    model.eval()
    batch = collate_molecules([data])
    h0, e0 = model.embed(batch)
    h = model.node_states_from_embeddings(h0, e0, batch.edge_index)
    _, _, attn, node_mask = model.readout(h, batch.batch, batch.desc, return_attention=True)
    return attn[0, node_mask[0]].cpu().numpy()


def integrated_gradients(model, data, n_bonds: int, steps: int = 64, y_scale: float = 1.0) -> dict:
    model.eval()
    batch = collate_molecules([data])
    with torch.no_grad():
        h0, e0 = model.embed(batch)
    d0 = batch.desc
    n_atoms, n_edges = h0.size(0), e0.size(0)

    alphas = (torch.arange(steps, dtype=h0.dtype) + 0.5) / steps
    H = (h0.repeat(steps, 1) * alphas.repeat_interleave(n_atoms)[:, None]).requires_grad_(True)
    E = (e0.repeat(steps, 1) * alphas.repeat_interleave(n_edges)[:, None]).requires_grad_(True)
    D = (d0.repeat(steps, 1) * alphas[:, None]).requires_grad_(True)
    edge_index = torch.cat([batch.edge_index + k * n_atoms for k in range(steps)], dim=1)
    batch_vec = torch.arange(steps).repeat_interleave(n_atoms)

    h = model.node_states_from_embeddings(H, E, edge_index)
    mu, _ = model.readout(h, batch_vec, D)
    mu.sum().backward()

    g_h = H.grad.view(steps, n_atoms, -1).mean(0)
    g_e = E.grad.view(steps, n_edges, -1).mean(0)
    g_d = D.grad.mean(0)

    atom_attr = ((h0 * g_h).sum(-1) * y_scale).detach().numpy()
    edge_attr = ((e0 * g_e).sum(-1) * y_scale).detach().numpy()
    desc_attr = ((d0[0] * g_d) * y_scale).detach().numpy()
    if n_bonds > 0:
        if n_edges != 2 * n_bonds:
            raise ValueError(f"expected {2 * n_bonds} directed edges, got {n_edges}")
        bond_attr = edge_attr.reshape(n_bonds, 2).sum(-1)
        sentinel = 0.0
    else:
        bond_attr = np.zeros((0,), dtype=np.float64)
        sentinel = float(edge_attr.sum())

    with torch.no_grad():
        mu_x, _ = model.readout(
            model.node_states_from_embeddings(h0, e0, batch.edge_index), batch.batch, d0
        )
        mu_b, _ = model.readout(
            model.node_states_from_embeddings(
                torch.zeros_like(h0), torch.zeros_like(e0), batch.edge_index
            ),
            batch.batch,
            torch.zeros_like(d0),
        )
    delta = float((mu_x - mu_b).item() * y_scale)
    total = float(atom_attr.sum() + bond_attr.sum() + desc_attr.sum() + sentinel)
    return {
        "atom": atom_attr.astype(np.float64),
        "bond": bond_attr.astype(np.float64),
        "desc": desc_attr.astype(np.float64),
        "sentinel_edge": sentinel,
        "delta": delta,
        "attr_sum": total,
        "completeness_gap": total - delta,
        "mu_baseline_scaled": float(mu_b.item()),
    }


def symmetrize_attributions(mol: Chem.Mol, atom_attr, bond_attr) -> tuple[np.ndarray, np.ndarray]:
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    atom_attr = np.asarray(atom_attr, dtype=np.float64)
    atom_out = atom_attr.copy()
    for r in set(ranks):
        idx = [i for i, rr in enumerate(ranks) if rr == r]
        atom_out[idx] = atom_attr[idx].mean()
    bond_attr = np.asarray(bond_attr, dtype=np.float64)
    bond_out = bond_attr.copy()
    keys = []
    for bond in mol.GetBonds():
        a, b = ranks[bond.GetBeginAtomIdx()], ranks[bond.GetEndAtomIdx()]
        keys.append((min(a, b), max(a, b), str(bond.GetBondType())))
    for key in set(keys):
        idx = [i for i, k in enumerate(keys) if k == key]
        bond_out[idx] = bond_attr[idx].mean()
    return atom_out, bond_out
