"""Calibration metrics for Gaussian regression predictions.

All functions take arrays in the *original* target units (for ESOL: log10 mol/L).

Definitions
-----------
NLL         mean over molecules of 0.5*log(2*pi*sigma^2) + (y - mu)^2 / (2*sigma^2)
coverage(p) fraction of molecules whose y falls inside the central p-interval
            mu +/- z_{(1+p)/2} * sigma
ECE         mean over nominal levels p in DEFAULT_LEVELS of |coverage(p) - p|
            (interval-based regression calibration error, Kuleshov et al. 2018 style)
width(p)    mean full width 2 * z_{(1+p)/2} * sigma of the central p-interval
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm, spearmanr

# 19 nominal central-interval levels: 0.05, 0.10, ..., 0.95
DEFAULT_LEVELS = np.round(np.arange(1, 20) * 0.05, 2)


def _as_1d(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def gaussian_nll(y, mu, sigma) -> float:
    y, mu, sigma = _as_1d(y), _as_1d(mu), _as_1d(sigma)
    var = sigma ** 2
    return float(np.mean(0.5 * np.log(2.0 * math.pi * var) + (y - mu) ** 2 / (2.0 * var)))


def coverage_curve(y, mu, sigma, levels=DEFAULT_LEVELS) -> tuple[np.ndarray, np.ndarray]:
    y, mu, sigma = _as_1d(y), _as_1d(mu), _as_1d(sigma)
    levels = np.asarray(levels, dtype=np.float64)
    abs_z = np.abs(y - mu) / sigma
    half_widths = norm.ppf(0.5 + levels / 2.0)
    empirical = np.array([np.mean(abs_z <= hw) for hw in half_widths])
    return levels, empirical


def interval_coverage_width(y, mu, sigma, level: float) -> tuple[float, float]:
    y, mu, sigma = _as_1d(y), _as_1d(mu), _as_1d(sigma)
    hw = norm.ppf(0.5 + level / 2.0) * sigma
    covered = np.abs(y - mu) <= hw
    return float(np.mean(covered)), float(np.mean(2.0 * hw))


def expected_calibration_error(y, mu, sigma, levels=DEFAULT_LEVELS) -> float:
    levels, empirical = coverage_curve(y, mu, sigma, levels)
    return float(np.mean(np.abs(empirical - levels)))


def fit_variance_scale(y_val, mu_val, sigma_val) -> float:
    """Single scale s minimising validation Gaussian NLL of (mu, s * sigma).

    Closed form: s^2 = mean((y - mu)^2 / sigma^2) on the validation split.
    """
    y, mu, sigma = _as_1d(y_val), _as_1d(mu_val), _as_1d(sigma_val)
    return float(math.sqrt(np.mean(((y - mu) / sigma) ** 2)))


def calibration_metrics(y, mu, sigma, levels=DEFAULT_LEVELS) -> dict[str, float]:
    y, mu, sigma = _as_1d(y), _as_1d(mu), _as_1d(sigma)
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("sigma must be finite and positive")
    err = mu - y
    abs_err = np.abs(err)
    out = {
        "n": int(len(y)),
        "rmse": float(math.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(abs_err)),
        "nll": gaussian_nll(y, mu, sigma),
        "ece": expected_calibration_error(y, mu, sigma, levels),
        "mean_sigma": float(np.mean(sigma)),
        "rms_z": float(math.sqrt(np.mean((err / sigma) ** 2))),
    }
    for level in (0.5, 0.68, 0.9, 0.95):
        cov, width = interval_coverage_width(y, mu, sigma, level)
        tag = f"{int(round(level * 100))}"
        out[f"coverage_{tag}"] = cov
        out[f"width_{tag}"] = width
    if np.std(sigma) > 1e-12 and np.std(abs_err) > 1e-12:
        out["spearman_sigma_abs_err"] = float(spearmanr(sigma, abs_err).correlation)
    else:
        out["spearman_sigma_abs_err"] = float("nan")
    return out


def conformal_quantile(scores, level: float) -> float:
    """Split-conformal quantile: the ceil((n+1)*level)-th smallest calibration score."""
    scores = np.sort(_as_1d(scores))
    n = len(scores)
    k = int(math.ceil((n + 1) * level))
    if k > n:
        return float("inf")
    return float(scores[k - 1])


def conformal_metrics(y_val, mu_val, sigma_val, y_test, mu_test, sigma_test, levels=DEFAULT_LEVELS) -> dict:
    """Normalized split conformal prediction intervals mu +/- q_p * sigma.

    Calibration scores are |y - mu| / sigma on the validation split; q_p is their
    conformal quantile. The finite-sample guarantee (marginal test coverage >= p)
    needs validation and test molecules to be exchangeable AND the score model to
    be fixed independently of the calibration data. In this project the same
    validation split also picks checkpoints and configurations, so the guarantee
    is only approximate (validation residuals are slightly optimistic), and it
    does not hold under the scaffold shift. With constant sigma this reduces to
    plain residual split conformal.
    """
    y_val, mu_val, sigma_val = _as_1d(y_val), _as_1d(mu_val), _as_1d(sigma_val)
    y_test, mu_test, sigma_test = _as_1d(y_test), _as_1d(mu_test), _as_1d(sigma_test)
    levels = np.asarray(levels, dtype=np.float64)
    scores = np.abs(y_val - mu_val) / sigma_val
    abs_z = np.abs(y_test - mu_test) / sigma_test
    q = np.array([conformal_quantile(scores, p) for p in levels])
    coverage = np.array([np.mean(abs_z <= qq) for qq in q])
    width = np.array([np.mean(2.0 * qq * sigma_test) for qq in q])
    i90 = int(np.argmin(np.abs(levels - 0.9)))
    return {
        "ece": float(np.mean(np.abs(coverage - levels))),
        "coverage_90": float(coverage[i90]),
        "width_90": float(width[i90]),
        "q_90": float(q[i90]),
        "n_calibration": int(len(scores)),
        "curve": {
            "levels": [float(v) for v in levels],
            "empirical": [float(v) for v in coverage],
            "width": [float(v) for v in width],
        },
    }
