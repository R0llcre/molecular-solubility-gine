import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader

from molfuse_gine.data import MoleculeDataset, collate_molecules
from molfuse_gine.features.chem_path import CHEM_PATH_FIELDS


def test_collate_keeps_path_attr_aligned_with_dense_node_order(tmp_path):
    csv = tmp_path / "toy.csv"
    pd.DataFrame(
        {
            "SMILES": ["CCO", "C"],
            "Solubility": [0.1, 0.2],
        }
    ).to_csv(csv, index=False)
    ds = MoleculeDataset(
        csv,
        mode="train",
        smiles_col="SMILES",
        target_col="Solubility",
        descriptor_cols=[],
        scaler_dir=tmp_path / "scalers",
        scaler_key="toy",
        cache_dir=tmp_path / "cache",
        max_path_distance=8,
    )

    batch = next(iter(DataLoader(ds, batch_size=2, collate_fn=collate_molecules)))
    dist_idx = CHEM_PATH_FIELDS.index("distance")

    assert batch.node_mask.tolist() == [[True, True, True], [True, False, False]]
    assert batch.path_attr[0, 0, 2, dist_idx].item() == 2
    assert batch.path_attr[1, 0, 0, dist_idx].item() == 0
    assert batch.path_attr[1, 0, 1, dist_idx].item() == 0
