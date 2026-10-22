import pandas as pd

from chempath_gine.data.split import make_splits
from chempath_gine.training.metrics import regression_metrics


def test_clean_before_split_drops_invalid_and_keeps_source_index(tmp_path):
    csv = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "SMILES": ["CCO", "not_a_smiles", "CCN", "CCC"],
            "Solubility": [0.1, 99.0, 0.2, 0.3],
        }
    ).to_csv(csv, index=False)

    paths = make_splits(csv, tmp_path / "splits", "SMILES", "random", 0.5, 0.25, seed=3)
    clean = pd.read_csv(tmp_path / "splits" / "clean.csv")

    assert len(clean) == 3
    assert "source_index" in clean.columns
    assert 1 not in set(clean["source_index"])
    assert all(path.exists() for path in paths.values())


def test_scaffold_split_puts_largest_scaffold_in_train(tmp_path):
    csv = tmp_path / "scaffold.csv"
    pd.DataFrame(
        {
            "SMILES": ["c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "CCO", "CCN", "CCC"],
            "Solubility": [0.1, 0.2, 0.3, -0.1, -0.2, -0.3],
        }
    ).to_csv(csv, index=False)

    paths = make_splits(csv, tmp_path / "splits", "SMILES", "scaffold", 0.5, 0.25, seed=11)
    train = pd.read_csv(paths["train"])

    assert any("c1ccccc1" in smi for smi in train["SMILES"])
    assert len(train) >= 3


def test_constant_target_r2_is_not_perfect_when_prediction_is_wrong():
    metrics = regression_metrics([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])

    assert metrics["r2"] == 0.0
