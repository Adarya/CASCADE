"""
Check: Constant Variables
=========================

Detects features with zero variance in a feature matrix. These can
arise when a variable that varies across the full cohort becomes
constant within a subgroup or at a particular landmark time.

Pitfall #4 in the CASCADE library.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from ..library import PitfallWarning, Severity, PITFALL_LIBRARY

_PITFALL = PITFALL_LIBRARY[3]  # id=4, constant variable at short landmark


def check_constant_variables(
    X: pd.DataFrame,
    subgroup_label: Optional[str] = None,
) -> List[PitfallWarning]:
    """Check each column for zero variance (constant values).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples x features). All columns are checked.
    subgroup_label : str, optional
        If provided, adds context to warnings indicating this may be
        a subgroup-specific issue (e.g., "within ER+ patients" or
        "at 3-month landmark").

    Returns
    -------
    list of PitfallWarning
        One warning per constant column found.
    """
    warnings: List[PitfallWarning] = []

    if X.empty:
        return warnings

    for col in X.columns:
        series = X[col]

        # Drop NaN for variance check
        non_null = series.dropna()

        if len(non_null) == 0:
            warnings.append(
                PitfallWarning(
                    pitfall=_PITFALL,
                    message=(
                        f"Column '{col}' is entirely NaN "
                        f"({len(series)} rows, all missing)."
                    ),
                    location=col,
                    severity=Severity.WARNING,
                    suggestion=f"Remove '{col}' from the model or impute values.",
                )
            )
            continue

        unique_values = non_null.unique()
        n_unique = len(unique_values)

        if n_unique == 1:
            constant_value = unique_values[0]
            context = ""
            if subgroup_label:
                context = (
                    f" This may be specific to the current subgroup "
                    f"({subgroup_label}); the variable may have variance "
                    f"in the full cohort."
                )

            # Detect specific patterns
            is_binary = _is_binary_column(series)
            pattern_info = ""
            if is_binary:
                if constant_value in (0, 0.0, False):
                    pattern_info = (
                        " This is a binary flag that is always 0 (never "
                        "true) in this subset."
                    )
                elif constant_value in (1, 1.0, True):
                    pattern_info = (
                        " This is a binary flag that is always 1 (always "
                        "true) in this subset."
                    )

            warnings.append(
                PitfallWarning(
                    pitfall=_PITFALL,
                    message=(
                        f"Column '{col}' is constant (value = {constant_value}) "
                        f"across all {len(non_null)} non-null rows.{pattern_info}"
                        f"{context}"
                    ),
                    location=col,
                    severity=Severity.WARNING,
                    suggestion=(
                        f"Remove '{col}' from the model before fitting. "
                        f"A constant predictor cannot contribute to any "
                        f"regression model and will cause fitting issues."
                    ),
                )
            )

    return warnings


def _is_binary_column(series: pd.Series) -> bool:
    """Check if a series contains only binary values (0/1, True/False).

    Parameters
    ----------
    series : pd.Series
        The series to check.

    Returns
    -------
    bool
        True if all non-null values are in {0, 1, True, False}.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    unique_vals = set(non_null.unique())
    binary_vals = {0, 1, 0.0, 1.0, True, False}
    return unique_vals.issubset(binary_vals)
