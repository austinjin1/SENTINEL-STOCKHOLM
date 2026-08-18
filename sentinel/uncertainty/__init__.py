"""SENTINEL uncertainty-quantification utilities.

Split-conformal and *weighted* split-conformal prediction (Tibshirani, Barber,
Candes & Ramdas 2019, "Conformal Prediction Under Covariate Shift").
"""
from sentinel.uncertainty.weighted_conformal import (
    domain_classifier_weights,
    weighted_quantile,
    conformal_classification_sets,
    conformal_regression_intervals,
    coverage_and_size_classification,
    coverage_and_width_regression,
)

__all__ = [
    "domain_classifier_weights",
    "weighted_quantile",
    "conformal_classification_sets",
    "conformal_regression_intervals",
    "coverage_and_size_classification",
    "coverage_and_width_regression",
]
