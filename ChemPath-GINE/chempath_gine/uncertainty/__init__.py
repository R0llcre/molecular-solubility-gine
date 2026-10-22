"""Heteroscedastic (Gaussian NLL) head, calibration metrics and attribution for gine_desc.

This package is additive: it reuses the existing ChemPathGINE trunk and data
pipeline without changing the original training entry point.
"""

from .calibration import (
    DEFAULT_LEVELS,
    calibration_metrics,
    conformal_metrics,
    coverage_curve,
    fit_variance_scale,
    gaussian_nll,
)
from .gaussian import GaussianGINE, beta_nll_loss, gaussian_nll_loss

__all__ = [
    "DEFAULT_LEVELS",
    "GaussianGINE",
    "beta_nll_loss",
    "calibration_metrics",
    "conformal_metrics",
    "coverage_curve",
    "fit_variance_scale",
    "gaussian_nll",
    "gaussian_nll_loss",
]
