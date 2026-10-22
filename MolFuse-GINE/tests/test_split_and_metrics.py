import pandas as pd
import pytest

from molfuse_gine.data.clean import clean_molecules
from molfuse_gine.data.split import make_splits
from molfuse_gine.training.metrics import regression_metrics
from scripts.audit_splits import audit_config, main as audit_main


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


def test_clean_rejects_duplicate_existing_source_index(tmp_path):
    csv = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "source_index": [1, 1],
            "SMILES": ["CCO", "CCN"],
            "Solubility": [0.1, 0.2],
        }
    ).to_csv(csv, index=False)

    with pytest.raises(ValueError, match="source_index"):
        clean_molecules(csv, tmp_path / "clean.csv", "SMILES")


def test_random_split_keeps_duplicate_canonical_smiles_together(tmp_path):
    csv = tmp_path / "dupes.csv"
    pd.DataFrame(
        {
            "SMILES": ["CCO", "OCC", "CCN", "CCC", "CCCl", "CCBr"],
            "Solubility": [0.1, 0.1, 0.2, 0.3, -0.1, -0.2],
        }
    ).to_csv(csv, index=False)

    paths = make_splits(csv, tmp_path / "dupe_splits", "SMILES", "random", 0.5, 0.25, seed=2)
    split_for_smiles = {}
    for split, path in paths.items():
        frame = pd.read_csv(path)
        for smiles in frame["SMILES"]:
            split_for_smiles.setdefault(smiles, set()).add(split)

    assert split_for_smiles["CCO"] in ({"train"}, {"val"}, {"test"})


def test_scaffold_split_puts_largest_scaffold_in_train(tmp_path):
    csv = tmp_path / "scaffold.csv"
    pd.DataFrame(
        {
            "SMILES": [
                "c1ccccc1",
                "Cc1ccccc1",
                "Oc1ccccc1",
                "CCO",
                "CCN",
                "CCC",
                "C1CCCCC1",
            ],
            "Solubility": [0.1, 0.2, 0.3, -0.1, -0.2, -0.3, 0.4],
        }
    ).to_csv(csv, index=False)

    paths = make_splits(csv, tmp_path / "splits", "SMILES", "scaffold", 0.5, 0.25, seed=11)
    train = pd.read_csv(paths["train"])

    assert any("c1ccccc1" in smi for smi in train["SMILES"])
    assert len(train) >= 3


def test_scaffold_split_does_not_group_all_acyclic_molecules_together(tmp_path):
    csv = tmp_path / "acyclic.csv"
    pd.DataFrame(
        {
            "SMILES": ["CCO", "CCN", "CCC", "CCCC", "CCCl", "CCBr"],
            "Solubility": [0.1, 0.2, 0.3, 0.4, -0.1, -0.2],
        }
    ).to_csv(csv, index=False)

    paths = make_splits(csv, tmp_path / "acyclic_splits", "SMILES", "scaffold", 0.5, 0.25, seed=4)
    train = pd.read_csv(paths["train"])
    val = pd.read_csv(paths["val"])
    test = pd.read_csv(paths["test"])

    assert len(train) < 6
    assert len(val) > 0
    assert len(test) > 0


def test_constant_target_r2_is_not_perfect_when_prediction_is_wrong():
    metrics = regression_metrics([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])

    assert metrics["r2"] == 0.0


def test_split_audit_writes_distribution_tables(tmp_path):
    csv = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "SMILES": ["CCO", "CCN", "CCC", "c1ccccc1", "Cc1ccccc1", "CCCl"],
            "Solubility": [0.1, 0.2, 0.3, -0.1, -0.2, 0.4],
        }
    ).to_csv(csv, index=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"""
data:
  csv: "{csv}"
  smiles_col: "SMILES"
  target_col: "Solubility"
  split: "scaffold"
  train_ratio: 0.5
  val_ratio: 0.25
""",
        encoding="utf-8",
    )

    audit_config(cfg, tmp_path / "audit", seeds=[1, 2])

    assert (tmp_path / "audit" / "cfg" / "split_overlap_matrix.csv").exists()
    assert (tmp_path / "audit" / "cfg" / "target_distribution_by_split.csv").exists()
    assert (tmp_path / "audit" / "cfg" / "scaffold_size_distribution.csv").exists()
    assert (tmp_path / "audit" / "cfg" / "acyclic_fraction_by_split.csv").exists()
    assert (tmp_path / "audit" / "cfg" / "canonical_duplicate_overlap.csv").exists()


def test_split_audit_cli_writes_manifest_and_summary(tmp_path):
    csv = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "SMILES": ["CCO", "CCN", "CCC", "c1ccccc1", "Cc1ccccc1", "CCCl"],
            "Solubility": [0.1, 0.2, 0.3, -0.1, -0.2, 0.4],
        }
    ).to_csv(csv, index=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"""
data:
  csv: "{csv}"
  smiles_col: "SMILES"
  target_col: "Solubility"
  split: "random"
  train_ratio: 0.5
  val_ratio: 0.25
""",
        encoding="utf-8",
    )
    out = tmp_path / "audit_cli"

    audit_main(["--configs", str(cfg), "--seeds", "1,2", "--out", str(out)])

    assert (out / "split_audit_summary.csv").exists()
    assert (out / "split_audit_manifest.json").exists()
