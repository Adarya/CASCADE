"""
Check: Separation Problems in Regression Estimates
====================================================

Detects quasi-complete or complete separation in regression results
by examining confidence interval ratios. When a CI spans several
orders of magnitude, the estimate is unreliable due to sparse data
in one or more cells of the contingency table.

Pitfall related to Study A v3 (4 separation-problem estimates filtered).
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from ..library import PitfallWarning, Severity, PITFALL_LIBRARY

# Use pitfall #3 (collinearity) as a close proxy; separation is related
# but distinct. We'll reference the correct pitfall in the warning.
# Since separation isn't one of the 9 canonical pitfalls, we create
# warnings referencing the most relevant one or using a generic approach.
# For this module, we reference pitfall #7 (singular matrix) since
# separation causes similar numerical issues.
_PITFALL = PITFALL_LIBRARY[6]  # id=7, singular matrix


def check_separation_problems(
    results_df: pd.DataFrame,
    ci_lower_col: str = "ci_lower",
    ci_upper_col: str = "ci_upper",
    threshold: float = 100.0,
    estimate_col: Optional[str] = None,
) -> List[PitfallWarning]:
    """Check for separation problems by examining confidence interval ratios.

    Computes CI_ratio = upper / lower for each row. When both bounds
    are positive (as with hazard ratios or odds ratios on the same
    side of 1), a very large ratio indicates quasi-separation or
    sparse data.

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

    for idx in results_df.index:
        lo = lower.get(idx)
        hi = upper.get(idx)

        if pd.isna(lo) or pd.isna(hi):
            continue

        # Compute CI ratio
        if lo > 0 and hi > 0:
            ratio = hi / lo
        elif lo < 0 and hi < 0:
            ratio = abs(lo) / abs(hi)
        elif lo == 0 or hi == 0:
            # One bound at zero -- likely degenerate
            ratio = float("inf")
        else:
            # CI spans zero (includes null for log-scale estimates)
            # Compute width as a proxy
            width = hi - lo
            # For HR/OR on log scale, this is less informative.
            # Flag if the absolute range is extreme.
            ratio = abs(hi - lo) if abs(hi - lo) > threshold else 0.0

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
