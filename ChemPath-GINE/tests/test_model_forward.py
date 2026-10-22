import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader

from chempath_gine.data import MoleculeDataset, collate_molecules
from chempath_gine.models import ChemPathGINE


@pytest.mark.parametrize(
    "variant",
    [
        "desc_mlp",
        "gine",
        "gine_desc",
        "gine_desc_add_gate",
        "gine_desc_concat",
        "gine_desc_residual_add",
        "gine_desc_film_graph",
        "gine_desc_lowrank_bilinear",
        "gine_desc_ddrop_005",
        "gine_desc_ddrop_010",
        "gine_desc_ddrop_020",
        "gine_attn",
        "gine_attn_desc",
        "gine_spd",
        "chempath_gine",
        "chempath_desc",
        "chempath_desc_full",
        "chempath_desc_fusion",
        "chempath_desc_gate",
        "chempath_desc_zero_path",
        "chempath_desc_zero_attr_same_params",
        "chempath_desc_shuffle_path",
        "chempath_desc_shuffle_bucket",
        "chempath_desc_distance_only",
        "chempath_desc_chem_only",
        "chempath_desc_film",
        "chempath_desc_film_sparse",
        "chempath_desc_film_sparse_ls",
        "chempath_desc_film_sparse_ls_ddrop",
    ],
)
def test_chempath_gine_forward(tmp_path, variant):
    csv = tmp_path / "toy.csv"
    pd.DataFrame(
        {
            "SMILES": ["CCO", "c1ccccc1"],
            "Solubility": [0.1, -0.2],
            "MolWt": [46.0, 78.0],
        }
    ).to_csv(csv, index=False)

    ds = MoleculeDataset(
        csv,
        mode="train",
        smiles_col="SMILES",
        target_col="Solubility",
        descriptor_cols=["MolWt"],
        scaler_dir=tmp_path / "scalers",
        scaler_key="toy",
        cache_dir=tmp_path / "cache",
        max_path_distance=8,
    )
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_molecules)
    batch = next(iter(loader))

    model = ChemPathGINE(
        hidden_dim=32,
        num_gnn_layers=2,
        num_attention_layers=1,
        heads=4,
        dropout=0.0,
        path_dim=ds.path_dim,
        descriptor_dim=ds.descriptor_dim,
        max_path_distance=8,
        variant=variant,
    )
    out = model(batch)
    assert out.shape == (2,)
    assert torch.isfinite(out).all()
