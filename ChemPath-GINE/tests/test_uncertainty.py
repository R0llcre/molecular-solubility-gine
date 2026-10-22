import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from rdkit import Chem

from chempath_gine.data import MoleculeDataset, collate_molecules
from chempath_gine.models import ChemPathGINE
from chempath_gine.uncertainty import (
    GaussianGINE,
    calibration_metrics,
    fit_variance_scale,
)
from chempath_gine.uncertainty.attribution import (
    integrated_gradients,
    readout_attention,
    symmetrize_attributions,
)


def test_calibration_metrics_on_well_specified_gaussian():
    rng = np.random.default_rng(0)
    n = 20000
    mu = rng.normal(size=n)
    sigma = rng.uniform(0.2, 2.0, size=n)
    y = mu + sigma * rng.normal(size=n)
    m = calibration_metrics(y, mu, sigma)
    assert m["ece"] < 0.01
    assert abs(m["coverage_90"] - 0.90) < 0.01
    assert abs(m["width_90"] - 2 * 1.6448536 * sigma.mean()) < 1e-6
    # Overconfident sigma: coverage drops and the val scale recovers the factor.
    m_over = calibration_metrics(y, mu, sigma / 2)
    assert m_over["coverage_90"] < 0.65
    assert abs(fit_variance_scale(y, mu, sigma / 2) - 2.0) < 0.05


def _toy_dataset(tmp_path):
    csv = tmp_path / "toy.csv"
    pd.DataFrame(
        {
            "SMILES": ["CCO", "c1ccccc1O", "O=C1c2ccccc2C(=O)c2ccccc21", "C"],
            "Solubility": [0.1, -0.2, -5.2, -0.9],
            "MolWt": [46.0, 94.0, 208.0, 16.0],
        }
    ).to_csv(csv, index=False)
    return MoleculeDataset(
        csv,
        mode="train",
        smiles_col="SMILES",
        target_col="Solubility",
        descriptor_cols=["MolWt"],
        scaler_dir=tmp_path / "scalers",
        scaler_key="toy",
        cache_dir=tmp_path / "cache",
        skip_path_features=True,
    )


def _models(ds, seed=0):
    torch.manual_seed(seed)
    base = ChemPathGINE(
        hidden_dim=32, num_gnn_layers=2, num_attention_layers=1, heads=4, dropout=0.0,
        path_dim=ds.path_dim, descriptor_dim=ds.descriptor_dim, max_path_distance=12,
        variant="gine_desc",
    )
    reference_state = {k: v.clone() for k, v in base.state_dict().items()}
    reference = ChemPathGINE(
        hidden_dim=32, num_gnn_layers=2, num_attention_layers=1, heads=4, dropout=0.0,
        path_dim=ds.path_dim, descriptor_dim=ds.descriptor_dim, max_path_distance=12,
        variant="gine_desc",
    )
    reference.load_state_dict(reference_state)
    return GaussianGINE(base).eval(), reference.eval()


def test_gaussian_head_mean_matches_original_forward(tmp_path):
    ds = _toy_dataset(tmp_path)
    model, reference = _models(ds)
    batch = collate_molecules([ds[i] for i in range(len(ds))])
    with torch.no_grad():
        mu, logvar = model(batch)
        ref = reference(batch)
    assert torch.allclose(mu, ref, atol=1e-6)
    assert torch.allclose(logvar, torch.zeros_like(logvar))  # sigma=1 at init


def test_integrated_gradients_completeness_and_symmetry(tmp_path):
    ds = _toy_dataset(tmp_path)
    model, _ = _models(ds, seed=1)
    for p in model.parameters():
        p.requires_grad_(False)
    for data in ds:
        mol = Chem.MolFromSmiles(data.smiles)
        ig = integrated_gradients(model, data, n_bonds=mol.GetNumBonds(), steps=256)
        assert abs(ig["completeness_gap"]) < 0.02 * max(1.0, abs(ig["delta"]))
        attn = readout_attention(model, data)
        assert attn.shape == (mol.GetNumAtoms(),)
        assert abs(attn.sum() - 1.0) < 1e-5
        atom_sym, bond_sym = symmetrize_attributions(mol, ig["atom"], ig["bond"])
        assert np.isclose(atom_sym.sum(), ig["atom"].sum())
        assert np.isclose(bond_sym.sum(), ig["bond"].sum())
        if data.smiles.startswith("O=C1"):  # anthraquinone: the two carbonyl O are equivalent
            o_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "O"]
            assert np.isclose(atom_sym[o_idx[0]], atom_sym[o_idx[1]])
