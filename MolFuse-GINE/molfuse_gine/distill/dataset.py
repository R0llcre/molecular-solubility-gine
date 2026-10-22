from __future__ import annotations

from pathlib import Path

import pandas as pd


def attach_teacher_predictions(
    split_csv: str | Path,
    prediction_csv: str | Path,
    out_csv: str | Path,
    teacher_col: str = "teacher_pred",
    prediction_col: str = "stack_pred",
) -> Path:
    split_csv = Path(split_csv)
    prediction_csv = Path(prediction_csv)
    out_csv = Path(out_csv)
    split_df = pd.read_csv(split_csv)
    pred_df = pd.read_csv(prediction_csv)

    if prediction_col not in pred_df.columns:
        if "y_pred" in pred_df.columns:
            prediction_col = "y_pred"
        else:
            raise KeyError(f"{prediction_csv} must contain {prediction_col!r} or 'y_pred'")

    if "source_index" in split_df.columns and "source_index" in pred_df.columns:
        if split_df["source_index"].duplicated().any():
            raise ValueError(f"{split_csv} contains duplicate source_index values")
        if pred_df["source_index"].duplicated().any():
            raise ValueError(f"{prediction_csv} contains duplicate source_index values")
        teacher = pred_df[["source_index", prediction_col]].rename(columns={prediction_col: teacher_col})
        out = split_df.merge(teacher, on="source_index", how="left", validate="one_to_one")
    else:
        raise ValueError("Teacher distillation requires source_index in both split and prediction CSVs")

    if len(out) != len(split_df):
        raise ValueError("Teacher merge changed row count")

    if out[teacher_col].isna().any():
        missing = int(out[teacher_col].isna().sum())
        raise ValueError(f"Missing {missing} teacher predictions for {split_csv}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out_csv
