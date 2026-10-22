import pytest

torch = pytest.importorskip("torch")

from chempath_gine.features.chem_path import CHEM_PATH_DIM, CHEM_PATH_FIELDS
from chempath_gine.models.chem_path_attention import ChemPathAttentionBlock


def idx(name: str) -> int:
    return CHEM_PATH_FIELDS.index(name)


def test_structured_chem_path_bias_components_are_active():
    block = ChemPathAttentionBlock(
        hidden_dim=16,
        heads=4,
        path_dim=CHEM_PATH_DIM,
        dropout=0.0,
        max_path_distance=8,
        mode="chem",
    )
    path = torch.zeros((1, 2, 2, CHEM_PATH_DIM))
    path[..., idx("distance")] = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.float32)
    path[..., idx("same_ring")] = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float32)

    bias = block._path_bias(path)

    assert bias.shape == (1, 2, 2, 4)
    assert block.relation_emb is not None
    assert block.count_proj is not None
    assert abs(float(torch.tanh(block.path_scale))) < 0.02


def test_attention_diagnostics_include_path_scale():
    block = ChemPathAttentionBlock(
        hidden_dim=16,
        heads=4,
        path_dim=CHEM_PATH_DIM,
        dropout=0.0,
        max_path_distance=8,
        mode="chem",
        sparse_attention=True,
    )
    diag = block.diagnostics(prefix="layer0_")
    assert diag["layer0_attention_mode"] == "chem"
    assert "layer0_path_scale_tanh" in diag
    assert diag["layer0_sparse_attention"] == "True"
