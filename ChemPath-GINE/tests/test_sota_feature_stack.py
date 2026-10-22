import numpy as np
import pandas as pd
import pytest

from chempath_gine.baselines.tabular import make_tabular_features, parse_feature_spec
from chempath_gine.distill import attach_teacher_predictions
from chempath_gine.features.conformer3d_bank import compute_conformer3d_frame
from chempath_gine.features.descriptor_bank import DESC16_NAMES, compute_descriptor_frame
from chempath_gine.features.fingerprint_bank import compute_fingerprint_frame
from chempath_gine.selection import FeaturePreprocessor, FeatureSelectionSpec


def test_chem_no_dist_variant_requires_real_path_features():
    pytest.importorskip("torch")
    from chempath_gine.models import ChemPathGINE

    assert ChemPathGINE.variant_uses_real_path("chempath_desc_chem_only")


def test_desc16_and_fingerprint_shapes():
    smiles = ["CCO", "c1ccccc1"]
    desc = compute_descriptor_frame(smiles, "desc16")
    fp = compute_fingerprint_frame(smiles, "morgan2,maccs")

    assert list(desc.columns) == DESC16_NAMES
    assert desc.shape == (2, 16)
    assert fp.shape == (2, 2048 + 167)
    assert np.isfinite(desc.to_numpy()).all()


def test_train_only_feature_preprocessor_selects_top_k():
    x = pd.DataFrame(
        {
            "signal": [0.0, 1.0, 2.0, 3.0],
            "duplicate": [0.0, 1.0, 2.0, 3.0],
            "noise": [1.0, 0.0, 1.0, 0.0],
            "missing": [1.0, np.nan, 1.0, np.nan],
        }
    )
    y = np.array([0.0, 1.0, 2.0, 3.0])
    pre = FeaturePreprocessor(FeatureSelectionSpec(method="pearson", top_k=1, corr_threshold=0.95))
    xt = pre.fit_transform(x, y)

    assert xt.shape == (4, 1)
    assert pre.selected_columns_ == ["signal"]


def test_tabular_feature_builder_uses_train_brics_vocab():
    train = pd.DataFrame({"SMILES": ["CCO", "CCN"], "y": [0.0, 1.0]})
    val = pd.DataFrame({"SMILES": ["CCC"], "y": [0.5]})
    test = pd.DataFrame({"SMILES": ["c1ccccc1"], "y": [0.2]})
    spec = parse_feature_spec("desc16_morgan2_brics4_fg")
    x_train, x_val, x_test, vocab = make_tabular_features(train, val, test, "SMILES", spec)

    assert len(vocab) <= 4
    assert x_train.shape[0] == 2
    assert x_val.shape[0] == 1
    assert x_test.shape[0] == 1
    assert any(col.startswith("fg_") for col in x_train.columns)


def test_conformer3d_feature_frame_smoke():
    frame = compute_conformer3d_frame(["CCO"], num_confs=1, seed=1)

    assert frame.shape[0] == 1
    assert any(col.startswith("conf3d_RadiusOfGyration") for col in frame.columns)
    assert np.isfinite(frame.to_numpy()).all()


def test_attach_teacher_predictions_by_source_index(tmp_path):
    split = tmp_path / "split.csv"
    pred = tmp_path / "pred.csv"
    out = tmp_path / "out.csv"
    pd.DataFrame({"source_index": [2, 5], "SMILES": ["CCO", "CCN"], "y": [0.1, 0.2]}).to_csv(
        split, index=False
    )
    pd.DataFrame({"source_index": [5, 2], "stack_pred": [0.25, 0.15]}).to_csv(pred, index=False)

    attach_teacher_predictions(split, pred, out)
    df = pd.read_csv(out)

    assert df["teacher_pred"].tolist() == [0.15, 0.25]
