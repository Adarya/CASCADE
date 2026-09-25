"""
Check: Separation Problems in Regression Estimates
====================================================

Detects quasi-complete or complete separation in regression results
by examining confidence interval ratios. When a CI spans several
orders of magnitude, the estimate is unreliable due to sparse data
in one or more cells of the contingency table.

Pitfall related to Study A v3 (4 separation-problem estimates filtered).
"""

from dataclasses import replace
from typing import List, Optional

import numpy as np
import pandas as pd

from ..library import PitfallWarning, Severity, PITFALL_LIBRARY

# Separation is not one of the 11 canonical pitfalls; it is filed under
# pitfall #7 (numerical instability of the fit) but carries its own
# name so reports do not mislabel it as "singular matrix from constant
# covariate".
_PITFALL = replace(
    PITFALL_LIBRARY[6],  # id=7, numerical-instability family
    name="Separation / extreme confidence interval (sparse cells)",
    description=(
        "Quasi-complete or complete separation: an estimate whose "
        "confidence interval spans orders of magnitude on the hazard- or "
        "odds-ratio scale because one or more cells contain (almost) no "
        "events.  Related to, but distinct from, a singular design matrix."
    ),
)


def check_separation_problems(
    results_df: pd.DataFrame,
    ci_lower_col: str = "ci_lower",
    ci_upper_col: str = "ci_upper",
    threshold: float = 100.0,
    estimate_col: Optional[str] = None,
    log_scale: Optional[bool] = None,
) -> List[PitfallWarning]:
    """Check for separation problems by examining confidence interval ratios.

    The CI ratio is always computed on the **ratio (HR / OR) scale**:
    ``upper / lower`` for ratio-scale bounds, or ``exp(upper - lower)``
    for log-scale (coefficient) bounds.  This is a width criterion --
    the ratio equals ``exp(width of the log-scale CI)`` -- so a narrow
    CI close to zero on the log scale (e.g. [-2.0, -0.01], HR CI
    [0.14, 0.99], ratio 7.3) is not flagged, while a CI spanning
    orders of magnitude is.

    Parameters
    ----------
    results_df : pd.DataFrame
        Dataframe of model results, one row per estimate.
    ci_lower_col : str
        Column name for the lower confidence bound (default 'ci_lower').
    ci_upper_col : str
        Column name for the upper confidence bound (default 'ci_upper').
    threshold : float
        CI ratio threshold for flagging (default 100.0). Results with
        CI_ratio > threshold are flagged.
    estimate_col : str, optional
        Column name for the point estimate. If provided, included in
        warning messages for context.
    log_scale : bool, optional
        Whether the bounds are log-scale coefficients.  ``None``
        (default) infers log scale when any bound is negative.

    Returns
    -------
    list of PitfallWarning
        One warning per result with an extreme CI ratio.
    """
    warnings: List[PitfallWarning] = []

    # Validate columns exist
    required_cols = [ci_lower_col, ci_upper_col]
    missing = [c for c in required_cols if c not in results_df.columns]
    if missing:
        warnings.append(
            PitfallWarning(
                pitfall=_PITFALL,
                message=f"Missing columns for CI check: {missing}",
                location=", ".join(missing),
                severity=Severity.WARNING,
                suggestion="Verify column names for CI bounds.",
            )
        )
        return warnings

    lower = pd.to_numeric(results_df[ci_lower_col], errors="coerce")
    upper = pd.to_numeric(results_df[ci_upper_col], errors="coerce")
    if log_scale is None:
        log_scale = bool((lower < 0).any() or (upper < 0).any())

    for idx in results_df.index:
        lo = lower.get(idx)
        hi = upper.get(idx)

        if pd.isna(lo) or pd.isna(hi):
            continue

        # Compute CI ratio on the HR / OR scale
        if log_scale:
            with np.errstate(over="ignore"):
                ratio = float(np.exp(abs(hi - lo)))
        elif lo > 0 and hi > 0:
            ratio = hi / lo
        else:
            # Ratio-scale lower bound at 0 (e.g. exp(-inf)) -- degenerate
            ratio = float("inf")

        if ratio > threshold:
            # Build informative message
            estimate_info = ""
            if estimate_col and estimate_col in results_df.columns:
                est = results_df.at[idx, estimate_col]
                estimate_info = f" (point estimate: {est})"

            # Try to get a row identifier
            row_label = str(idx)
            if "name" in results_df.columns:
                row_label = str(results_df.at[idx, "name"])
            elif "gene" in results_df.columns:
                row_label = str(results_df.at[idx, "gene"])

            warnings.append(
                PitfallWarning(
                    pitfall=_PITFALL,
                    message=(
                        f"Result '{row_label}' has CI ratio = {ratio:.1f} "
                        f"(CI: [{lo:.3f}, {hi:.3f}]){estimate_info}. "
                        f"This indicates possible separation or extreme "
                        f"data sparsity."
                    ),
                    location=f"row {idx}: {row_label}",
                    severity=Severity.CRITICAL if ratio > 1000 else Severity.WARNING,
                    suggestion=(
                        f"Consider filtering this estimate (CI ratio > "
                        f"{threshold}). The result is numerically unstable "
                        f"and should not be interpreted. Check the cell "
                        f"counts in the underlying contingency table."
                    ),
                )
            )

    return warnings
