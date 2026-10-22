from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_stack_diagnostics(stack_dir: Path, out_dir: Path | None = None) -> dict[str, Path]:
    out_dir = out_dir or stack_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    train_file = stack_dir / "expert_oof_train.csv"
    if train_file.exists():
        train = pd.read_csv(train_file)
        cols = [
            col
            for col in train.columns
            if col not in {"source_index", "y_true", "stack_absolute_error", "absolute_error"}
            and not col.endswith("_true")
        ]
        if len(cols) >= 2:
            corr_path = out_dir / "expert_correlation.csv"
            train[cols].corr().to_csv(corr_path)
            paths["expert_correlation"] = corr_path

    coef_file = stack_dir / "meta_coefficients.csv"
    if coef_file.exists():
        coef = pd.read_csv(coef_file)
        if {"expert", "coefficient"}.issubset(coef.columns):
            ranked = coef.reindex(coef["coefficient"].abs().sort_values(ascending=False).index)
            ranked_path = out_dir / "meta_coefficients_ranked.csv"
            ranked.to_csv(ranked_path, index=False)
            paths["meta_coefficients_ranked"] = ranked_path

    return paths
