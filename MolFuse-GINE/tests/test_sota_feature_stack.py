import numpy as np
import pandas as pd
import pytest

from molfuse_gine.baselines.tabular import make_tabular_features, parse_feature_spec
from molfuse_gine.distill import attach_teacher_predictions
from molfuse_gine.features.conformer3d_bank import compute_conformer3d_frame
from molfuse_gine.features.descriptor_bank import DESC16_NAMES, compute_descriptor_frame
from molfuse_gine.features.fingerprint_bank import compute_fingerprint_frame
from molfuse_gine.features.pretrained_smiles_bank import smiles_views
from molfuse_gine.selection import FeaturePreprocessor, FeatureSelectionSpec
from molfuse_gine.stacking.oof import ExpertSpec, run_tabular_oof_stack
from scripts.run_pretrained_smiles_oof import (
    _expected_protocol,
    _smiles_fingerprint,
    _summary_matches_protocol,
)


def test_chem_no_dist_variant_requires_real_path_features():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from molfuse_gine.models import ChemPathGINE

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


def test_feature_preprocessor_transform_aligns_by_column_name():
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    y = np.array([1.0, 2.0, 3.0])
    pre = FeaturePreprocessor(
        FeatureSelectionSpec(method="none", top_k=None, corr_threshold=1.0),
        scale=False,
    )
    pre.fit(x, y)

    transformed = pre.transform(pd.DataFrame({"b": [100.0], "a": [5.0]}))

    assert transformed.tolist() == [[5.0, 100.0]]


def test_feature_preprocessor_can_select_only_descriptor_candidates():
    x = pd.DataFrame(
        {
            "MolWt": [1.0, 2.0, 3.0, 4.0],
            "TPSA": [0.0, 0.0, 1.0, 1.0],
            "morgan2_0": [1.0, 0.0, 1.0, 0.0],
            "morgan2_constant": [0.0, 0.0, 0.0, 0.0],
            "maccs_1": [0.0, 1.0, 0.0, 1.0],
        }
    )
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pre = FeaturePreprocessor(
        FeatureSelectionSpec(
            method="pearson",
            top_k=1,
            corr_threshold=1.0,
            candidate_columns=("MolWt", "TPSA"),
            keep_non_candidates=True,
        ),
        scale=False,
    )

    pre.fit(x, y)

    assert pre.selected_columns_ == ["MolWt", "morgan2_0", "morgan2_constant", "maccs_1"]


def test_feature_preprocessor_fails_when_all_features_are_dropped():
    x = pd.DataFrame({"constant": [1.0, 1.0, 1.0], "missing": [np.nan, np.nan, np.nan]})
    y = np.array([0.0, 1.0, 2.0])
    pre = FeaturePreprocessor(FeatureSelectionSpec(method="none", top_k=None))

    with pytest.raises(ValueError, match="No usable tabular features"):
        pre.fit(x, y)


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


def test_tabular_feature_parser_rejects_substring_typos():
    with pytest.raises(ValueError, match="nomorgan2"):
        parse_feature_spec("desc16_nomorgan2")


def test_tabular_feature_parser_supports_descriptor_only_selection_scope():
    spec = parse_feature_spec("rdkit2d_descsel64_morgan2_maccs")

    assert spec.descriptor_set == "rdkit2d"
    assert spec.selection_method == "pearson"
    assert spec.selected_k == 64
    assert spec.selection_scope == "descriptor"
    assert spec.fingerprints == ("morgan2", "maccs")

    mi_spec = parse_feature_spec("rdkit2d_midesc32_morgan2")
    enet_spec = parse_feature_spec("rdkit2d_enetdesc16_maccs")

    assert mi_spec.selection_method == "mutual_info"
    assert mi_spec.selected_k == 32
    assert mi_spec.selection_scope == "descriptor"
    assert enet_spec.selection_method == "elasticnet"
    assert enet_spec.selected_k == 16
    assert enet_spec.selection_scope == "descriptor"


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


def test_attach_teacher_predictions_rejects_duplicate_source_index(tmp_path):
    split = tmp_path / "split.csv"
    pred = tmp_path / "pred.csv"
    out = tmp_path / "out.csv"
    pd.DataFrame({"source_index": [2, 2], "SMILES": ["CCO", "CCN"], "y": [0.1, 0.2]}).to_csv(
        split, index=False
    )
    pd.DataFrame({"source_index": [2, 5], "stack_pred": [0.25, 0.15]}).to_csv(pred, index=False)

    with pytest.raises(ValueError, match="duplicate source_index"):
        attach_teacher_predictions(split, pred, out)


def test_smiles_views_are_deterministic_for_same_seed():
    assert smiles_views("CCOC", n_random=8, seed=123) == smiles_views("CCOC", n_random=8, seed=123)


def test_pretrained_oof_protocol_and_smiles_fingerprints_are_strict():
    cfg = {"data": {"csv": "data.csv", "split": "random", "target_col": "y"}}
    expected = _expected_protocol(cfg, "model/a", "ridge", "desc16", 1, 5, 16, 32, "auto")

    assert _summary_matches_protocol({**expected, "test_rmse": 1.0}, expected)
    assert not _summary_matches_protocol({**expected, "random_smiles": 8}, expected)
    assert _smiles_fingerprint(pd.Series(["CCO", "CCN"])) != _smiles_fingerprint(
        pd.Series(["CCN", "CCO"])
    )


def test_tabular_oof_stack_writes_train_stack_prediction(tmp_path):
    train = pd.DataFrame(
        {
            "source_index": list(range(8)),
            "SMILES": ["C", "CC", "CCC", "CCCC", "CCO", "CCN", "CCCl", "c1ccccc1"],
            "y": [0.0, 0.1, 0.2, 0.3, -0.1, -0.2, 0.4, 0.5],
        }
    )
    val = pd.DataFrame(
        {"source_index": [8, 9], "SMILES": ["CO", "CN"], "y": [0.05, -0.05]}
    )
    test = pd.DataFrame(
        {"source_index": [10, 11], "SMILES": ["CCBr", "CCF"], "y": [0.15, 0.25]}
    )
    out = tmp_path / "oof"
    run_tabular_oof_stack(
        train,
        val,
        test,
        smiles_col="SMILES",
        target_col="y",
        expert_specs=[ExpertSpec("desc16", "ridge")],
        out_dir=out,
        seed=7,
        folds=2,
        meta_model_name="ridge",
    )

    train_pred = pd.read_csv(out / "expert_oof_train.csv")
    assert "stack_pred" in train_pred.columns
    assert np.isfinite(train_pred["stack_pred"]).all()
