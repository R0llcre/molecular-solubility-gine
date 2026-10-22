import yaml
import pandas as pd
import pytest

pytest.importorskip("torch")

from molfuse_gine.training.predict import run_prediction
from molfuse_gine.training.train import main


def test_training_smoke(tmp_path):
    csv = tmp_path / "toy.csv"
    pd.DataFrame(
        {
            "SMILES": ["C", "CC", "CCC", "CCO", "CCN", "c1ccccc1"],
            "Solubility": [0.0, 0.1, 0.2, -0.1, -0.2, 0.3],
        }
    ).to_csv(csv, index=False)
    cfg = {
        "run": {"name": "smoke", "output_dir": str(tmp_path / "runs"), "seed": 7},
        "data": {
            "csv": str(csv),
            "smiles_col": "SMILES",
            "target_col": "Solubility",
            "descriptor_cols": [],
            "split": "random",
            "train_ratio": 0.5,
            "val_ratio": 0.25,
            "cache_dir": "cache",
        },
        "model": {
            "variant": "molfuse_gine",
            "hidden_dim": 32,
            "num_gnn_layers": 1,
            "num_attention_layers": 1,
            "heads": 4,
            "dropout": 0.0,
            "use_descriptor_fusion": False,
            "use_descriptor_path_gate": False,
            "max_path_distance": 8,
        },
        "train": {
            "epochs": 1,
            "batch_size": 2,
            "lr": 0.001,
            "weight_decay": 0.0,
            "accum_steps": 1,
            "max_grad_norm": 5.0,
            "patience": 2,
            "num_workers": 0,
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    summary = main(["--config", str(cfg_path)])
    assert (tmp_path / "runs").exists()
    assert "rmse" in summary

    pred_csv = tmp_path / "predictions.csv"
    predict_input = tmp_path / "predict.csv"
    pd.DataFrame({"SMILES": ["CCO", "not_a_smiles", "CCCl"]}).to_csv(predict_input, index=False)
    meta = run_prediction(
        checkpoint=tmp_path / summary["run_dir"] / "best.pt",
        csv_path=predict_input,
        output=pred_csv,
        batch_size=2,
    )
    preds = pd.read_csv(pred_csv)
    assert meta["num_predictions"] == 2
    assert "source_index" in preds.columns
    assert "y_pred" in preds.columns
