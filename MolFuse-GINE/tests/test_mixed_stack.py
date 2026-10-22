from __future__ import annotations

import pandas as pd
import pytest

from molfuse_gine.stacking.mixed import run_mixed_prediction_stack
from molfuse_gine.stacking.pruned import ExpertPruningSpec, run_pruned_prediction_stack


def _write_expert(path, train_pred, val_pred, test_pred):
    path.mkdir(parents=True)
    pd.DataFrame(
        {"source_index": [0, 1, 2, 3], "y_true": [0.0, 1.0, 2.0, 3.0], "pred": train_pred}
    ).to_csv(path / "expert_oof_train.csv", index=False)
    pd.DataFrame({"source_index": [4, 5], "y_true": [1.0, 2.0], "pred": val_pred}).to_csv(
        path / "expert_val_predictions.csv", index=False
    )
    pd.DataFrame({"source_index": [6, 7], "y_true": [2.0, 3.0], "pred": test_pred}).to_csv(
        path / "expert_test_predictions.csv", index=False
    )


def test_mixed_prediction_stack_ignores_upstream_in_sample_stack_pred(tmp_path):
    expert = tmp_path / "expert"
    _write_expert(expert, [0.1, 0.9, 2.1, 2.9], [1.1, 1.9], [2.1, 2.9])
    for name in ["expert_oof_train.csv", "expert_val_predictions.csv", "expert_test_predictions.csv"]:
        df = pd.read_csv(expert / name)
        df["stack_pred"] = [999.0] * len(df)
        df.to_csv(expert / name, index=False)

    out = tmp_path / "mixed"
    result = run_mixed_prediction_stack([expert], out_dir=out, meta_model_name="average", seed=1)

    assert result.summary["n_expert_columns"] == 1
    assert not any("stack_pred" in col for col in result.coefficients["expert"])


def test_mixed_prediction_stack_outputs(tmp_path):
    expert_a = tmp_path / "expert_a"
    expert_b = tmp_path / "expert_b"
    _write_expert(expert_a, [0.1, 0.9, 2.1, 2.9], [1.1, 1.9], [2.1, 2.9])
    _write_expert(expert_b, [0.0, 1.2, 1.8, 3.2], [0.9, 2.2], [1.8, 3.1])

    out = tmp_path / "mixed"
    result = run_mixed_prediction_stack(
        [expert_a, expert_b],
        out_dir=out,
        meta_model_name="average",
        seed=1,
    )

    assert result.summary["n_expert_columns"] == 2
    assert (out / "expert_oof_train.csv").exists()
    assert (out / "expert_val_predictions.csv").exists()
    assert (out / "expert_test_predictions.csv").exists()
    assert (out / "meta_coefficients.csv").exists()
    assert (out / "expert_correlation.csv").exists()


def test_mixed_prediction_stack_rejects_mismatched_sources(tmp_path):
    expert_a = tmp_path / "expert_a"
    expert_b = tmp_path / "expert_b"
    _write_expert(expert_a, [0.1, 0.9, 2.1, 2.9], [1.1, 1.9], [2.1, 2.9])
    _write_expert(expert_b, [0.0, 1.2, 1.8, 3.2], [0.9, 2.2], [1.8, 3.1])
    df = pd.read_csv(expert_b / "expert_oof_train.csv")
    df.loc[0, "source_index"] = 99
    df.to_csv(expert_b / "expert_oof_train.csv", index=False)

    with pytest.raises(ValueError, match="source_index"):
        run_mixed_prediction_stack([expert_a, expert_b], tmp_path / "mixed", meta_model_name="average")


def test_mixed_prediction_stack_rejects_y_mismatch(tmp_path):
    expert_a = tmp_path / "expert_a"
    expert_b = tmp_path / "expert_b"
    _write_expert(expert_a, [0.1, 0.9, 2.1, 2.9], [1.1, 1.9], [2.1, 2.9])
    _write_expert(expert_b, [0.0, 1.2, 1.8, 3.2], [0.9, 2.2], [1.8, 3.1])
    df = pd.read_csv(expert_b / "expert_val_predictions.csv")
    df.loc[0, "y_true"] = 100.0
    df.to_csv(expert_b / "expert_val_predictions.csv", index=False)

    with pytest.raises(ValueError, match="y_true"):
        run_mixed_prediction_stack([expert_a, expert_b], tmp_path / "mixed", meta_model_name="average")


def test_pruned_prediction_stack_writes_selection_and_loo(tmp_path):
    expert_a = tmp_path / "expert_a"
    expert_b = tmp_path / "expert_b"
    expert_c = tmp_path / "expert_c"
    _write_expert(expert_a, [0.0, 1.0, 2.0, 3.0], [1.0, 2.0], [2.0, 3.0])
    _write_expert(expert_b, [0.2, 0.8, 2.2, 2.8], [1.2, 1.8], [2.2, 2.8])
    _write_expert(expert_c, [3.0, 2.0, 1.0, 0.0], [2.0, 1.0], [3.0, 2.0])

    out = tmp_path / "pruned"
    result = run_pruned_prediction_stack(
        [expert_a, expert_b, expert_c],
        out,
        pruning=ExpertPruningSpec(top_k=2, max_experts=3, corr_threshold=0.8, oof_rmse_slack=10.0),
        meta_model_name="average",
        seed=1,
    )

    assert result.summary["n_selected_experts"] >= 2
    assert (out / "expert_selection_report.csv").exists()
    assert (out / "leave_one_expert_out.csv").exists()
    assert (out / "meta_coefficients_ranked.csv").exists()
    assert (out / "expert_correlation.csv").exists()
    assert (out / "pruned_stack_summary.json").exists()
    assert not pd.read_csv(out / "leave_one_expert_out.csv").empty


def test_pruned_prediction_stack_rejects_split_overlap(tmp_path):
    expert = tmp_path / "expert"
    _write_expert(expert, [0.0, 1.0, 2.0, 3.0], [1.0, 2.0], [2.0, 3.0])
    val = pd.read_csv(expert / "expert_val_predictions.csv")
    val.loc[0, "source_index"] = 0
    val.to_csv(expert / "expert_val_predictions.csv", index=False)

    with pytest.raises(ValueError, match="train/val source_index overlap"):
        run_pruned_prediction_stack([expert], tmp_path / "pruned", meta_model_name="average", seed=1)
