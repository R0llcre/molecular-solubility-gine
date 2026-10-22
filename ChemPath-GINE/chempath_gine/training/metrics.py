from __future__ import annotations

import math

import numpy as np
from scipy.stats import spearmanr as _spearmanr


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    err = pred - target
    mse = float(np.mean(err ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    ss_res = float(np.sum(err ** 2))
    if ss_tot < 1e-12:
        r2 = 1.0 if ss_res < 1e-12 else 0.0
    else:
        r2 = float(1.0 - ss_res / ss_tot)
    pred_const = float(np.std(pred)) < 1e-12
    target_const = float(np.std(target)) < 1e-12
    if len(pred) < 2 or pred_const or target_const:
        pearson = 0.0
        spearman = 0.0
    else:
        pearson = float(np.corrcoef(pred, target)[0, 1])
        if not np.isfinite(pearson):
            pearson = 0.0
        spearman = float(_spearmanr(pred, target).correlation)
        if not np.isfinite(spearman):
            spearman = 0.0
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
    }
