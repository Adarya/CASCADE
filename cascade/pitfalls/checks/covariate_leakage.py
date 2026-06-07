"""
Check: Covariate Leakage in Landmark Models
============================================

Detects whether binary covariates in a landmark survival analysis
were computed using events that occur after the landmark date,
which constitutes future-data leakage.

Pitfall #2 in the CASCADE library.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from ..library import PitfallWarning, Severity, PITFALL_LIBRARY

_PITFALL = PITFALL_LIBRARY[1]  # id=2, covariate leakage


def check_landmark_leakage(
    df: pd.DataFrame,
    landmark_date_col: str,
    covariate_cols: List[str],
    event_date_cols: Optional[List[str]] = None,
) -> List[PitfallWarning]:
    """Check for future-data leakage in landmark model covariates.

    This check addresses two scenarios:

    1. **With event dates** (``event_date_cols`` provided): For each
       covariate, verifies that no contributing event occurs after the
       landmark date.
    2. **Without event dates** (heuristic mode): Checks if binary
       "ever received" covariates are constant across different landmark
       dates for the same patient, which suggests they were computed
       over the full follow-up rather than up to the landmark.

    Parameters
    ----------
    df : pd.DataFrame
        The analysis dataframe. Must contain at least the landmark date
        column and the covariate columns.
    landmark_date_col : str
        Column name containing the landmark date (numeric, in days).
    covariate_cols : list of str
        Column names of covariates to check for leakage.
    event_date_cols : list of str, optional
        Column names containing dates of events that contribute to the
        covariates. If provided, direct date-based leakage detection is
        performed. If None, heuristic checks are used.

    Returns
    -------
    list of PitfallWarning
        One warning per covariate suspected of leakage.
    """
    warnings: List[PitfallWarning] = []

    # Validate inputs
    missing_cols = [c for c in [landmark_date_col] + covariate_cols if c not in df.columns]
    if missing_cols:
        warnings.append(
            PitfallWarning(
                pitfall=_PITFALL,
                message=f"Missing columns in dataframe: {missing_cols}",
                location=", ".join(missing_cols),
                severity=Severity.WARNING,
                suggestion="Verify column names match the dataframe.",
            )
        )
        return warnings

    landmark_dates = df[landmark_date_col]

    # --- Strategy 1: Direct date-based check ---
    if event_date_cols is not None:
        valid_event_cols = [c for c in event_date_cols if c in df.columns]
        for event_col in valid_event_cols:
            # Find rows where event date is after the landmark date
            # and the associated covariate is nonzero
            event_dates = pd.to_numeric(df[event_col], errors="coerce")
            landmark_numeric = pd.to_numeric(landmark_dates, errors="coerce")

            post_landmark_mask = event_dates > landmark_numeric
            post_landmark_count = post_landmark_mask.sum()

            if post_landmark_count > 0:
                # Check which covariates are nonzero for these rows
                for cov_col in covariate_cols:
                    cov_values = pd.to_numeric(df[cov_col], errors="coerce")
                    leaking_rows = post_landmark_mask & (cov_values != 0)
                    n_leaking = leaking_rows.sum()

                    if n_leaking > 0:
                        pct = 100.0 * n_leaking / len(df)
                        warnings.append(
                            PitfallWarning(
                                pitfall=_PITFALL,
                                message=(
                                    f"Covariate '{cov_col}' has nonzero values in "
                                    f"{n_leaking} rows ({pct:.1f}%) where event date "
                                    f"'{event_col}' is after the landmark date "
                                    f"'{landmark_date_col}'. This indicates future "
                                    f"data leakage."
                                ),
                                location=cov_col,
                                severity=Severity.CRITICAL,
                                suggestion=(
                                    f"Recompute '{cov_col}' using only events with "
                                    f"'{event_col}' <= '{landmark_date_col}'. Filter "
                                    f"the source timeline to dates on or before the "
                                    f"landmark before aggregating."
                                ),
                            )
                        )
        return warnings

    # --- Strategy 2: Heuristic check for "ever received" variables ---
    for cov_col in covariate_cols:
        cov_values = df[cov_col]

        # Skip non-binary columns for heuristic check
        unique_vals = cov_values.dropna().unique()
        if not set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False}):
            continue

        # Heuristic: if a binary "ever received" variable is 1 for
        # patients whose landmark date is very early (e.g., bottom 10%),
        # it may be leaking future treatments.
        cov_numeric = pd.to_numeric(cov_values, errors="coerce")
        landmark_numeric = pd.to_numeric(landmark_dates, errors="coerce")

        positive_mask = cov_numeric == 1
        if positive_mask.sum() == 0:
            continue

        # Compare median landmark date for positive vs negative cases
        median_landmark_positive = landmark_numeric[positive_mask].median()
        median_landmark_negative = landmark_numeric[~positive_mask].median()

        # If positive cases have earlier landmarks on average, the
        # variable may span the entire follow-up
        if pd.notna(median_landmark_positive) and pd.notna(median_landmark_negative):
            # Suspicious if the rate of positivity doesn't vary with
            # landmark date. Check correlation between landmark date
            # and covariate.
            valid_mask = cov_numeric.notna() & landmark_numeric.notna()
            if valid_mask.sum() > 10:
                corr = cov_numeric[valid_mask].corr(landmark_numeric[valid_mask])
                # If correlation is near zero, covariate doesn't depend
                # on landmark date at all -- suspicious for a time-dependent
                # variable
                if pd.notna(corr) and abs(corr) < 0.05:
                    warnings.append(
                        PitfallWarning(
                            pitfall=_PITFALL,
                            message=(
                                f"Binary covariate '{cov_col}' shows near-zero "
                                f"correlation (r={corr:.3f}) with landmark date "
                                f"'{landmark_date_col}'. For a legitimately "
                                f"time-restricted variable, we would expect the "
                                f"positive rate to increase with later landmarks. "
                                f"This pattern is consistent with an 'ever received' "
                                f"variable computed over the full follow-up."
                            ),
                            location=cov_col,
                            severity=Severity.WARNING,
                            suggestion=(
                                f"Verify that '{cov_col}' was computed using only "
                                f"events up to the landmark date. If it represents "
                                f"'ever received treatment X', recompute from the "
                                f"treatment timeline filtered to dates <= landmark."
                            ),
                        )
                    )

    return warnings
