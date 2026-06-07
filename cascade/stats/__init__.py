"""
CASCADE Statistical Primitives
===============================

General-purpose statistical tools for survival analysis, interaction modeling,
multiple testing correction, equivalence testing, bootstrapping, and
cross-validation.

All modules are dataset-agnostic and operate on pandas DataFrames with
standard column conventions.
"""

from cascade.stats.cox import PenalizedCox
from cascade.stats.competing_risks import CauseSpecificCox, cause_specific_screen
from cascade.stats.interaction import InteractionResult, InteractionCox, stratified_effect
from cascade.stats.multiple_testing import apply_fdr, apply_bonferroni, apply_holm
from cascade.stats.equivalence import (
    phi_coefficient,
    tost_equivalence,
    bayes_factor_independence,
    independence_suite,
)
from cascade.stats.bootstrap import BootstrapCI, bootstrap_delta_c
from cascade.stats.cross_validate import cv_concordance, cv_delta_c

__all__ = [
    # Cox modeling
    "PenalizedCox",
    # Competing risks
    "CauseSpecificCox",
    "cause_specific_screen",
    # Interaction models
    "InteractionResult",
    "InteractionCox",
    "stratified_effect",
    # Multiple testing
    "apply_fdr",
    "apply_bonferroni",
    "apply_holm",
    # Equivalence / independence
    "phi_coefficient",
    "tost_equivalence",
    "bayes_factor_independence",
    "independence_suite",
    # Bootstrap
    "BootstrapCI",
    "bootstrap_delta_c",
    # Cross-validation
    "cv_concordance",
    "cv_delta_c",
]
