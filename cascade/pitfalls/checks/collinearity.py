"""
Check: Collinearity Detection
==============================

Detects perfect or near-perfect collinearity in feature matrices
via pairwise Pearson correlations and variance inflation factors
(VIF). Also checks for algebraic identities where one feature is a
deterministic function of others.

Pitfall #3 in the CASCADE library.
"""

import warnings as _pywarnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..library import PitfallWarning, Severity, PITFALL_LIBRARY

_PITFALL = PITFALL_LIBRARY[2]  # id=3, perfect collinearity


def check_collinearity(
    X: pd.DataFrame,
    threshold: float = 0.99,
    vif_threshold: float = 10.0,
    check_algebraic: bool = True,
) -> List[PitfallWarning]:
    """Check for collinearity in a feature matrix.

    Performs three checks:

    1. **Pairwise correlation**: Flags pairs with |r| > ``threshold``.
    2. **Variance Inflation Factor**: Flags features with VIF > ``vif_threshold``.
    3. **Algebraic identity** (optional): Checks whether any feature
       can be perfectly predicted as a linear combination of others.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples x features). All columns should be
        numeric.
    threshold : float
        Pearson correlation threshold for flagging pairs (default 0.99).
    vif_threshold : float
        VIF threshold for flagging individual features (default 10.0).
    check_algebraic : bool
        If True, also check for algebraic identities (default True).

    Returns
    -------
    list of PitfallWarning
        Warnings for detected collinearity issues.
    """
    warnings: List[PitfallWarning] = []

    if X.shape[1] < 2:
        return warnings  # Need at least 2 features

    # Ensure numeric
    X_numeric = X.select_dtypes(include=[np.number])
    if X_numeric.shape[1] < 2:
        return warnings

    # Drop incomplete rows (VIF / SVD cannot handle NaN or inf)
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
    n_incomplete = int(X_numeric.isna().any(axis=1).sum())
    if n_incomplete:
        _pywarnings.warn(
            f"check_collinearity: dropped {n_incomplete} of {len(X_numeric)} "
            f"rows with missing values before VIF / rank checks.",
            RuntimeWarning,
            stacklevel=2,
        )
        X_numeric = X_numeric.dropna()
    if len(X_numeric) < 3:
        return warnings

    # Drop columns with zero variance (handled by constant_variable check)
    non_constant = X_numeric.loc[:, X_numeric.std() > 0]
    if non_constant.shape[1] < 2:
        return warnings

    # --- Check 1: Pairwise correlations ---
    corr_warnings = _check_pairwise_correlation(non_constant, threshold)
    warnings.extend(corr_warnings)

    # --- Check 2: VIF ---
    vif_warnings = _check_vif(non_constant, vif_threshold)
    warnings.extend(vif_warnings)

    # --- Check 3: Algebraic identities ---
    if check_algebraic:
        algebraic_warnings = _check_algebraic_identity(non_constant)
        warnings.extend(algebraic_warnings)

    return warnings


def _check_pairwise_correlation(
    X: pd.DataFrame,
    threshold: float,
) -> List[PitfallWarning]:
    """Flag feature pairs with |correlation| above threshold."""
    warnings: List[PitfallWarning] = []

    corr_matrix = X.corr()
    cols = X.columns.tolist()

    # Check upper triangle only
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if pd.notna(r) and abs(r) > threshold:
                warnings.append(
                    PitfallWarning(
                        pitfall=_PITFALL,
                        message=(
                            f"Features '{cols[i]}' and '{cols[j]}' have "
                            f"Pearson correlation r = {r:.4f} (threshold: "
                            f"{threshold}). Including both in a regression "
                            f"model will cause unstable coefficient estimates "
                            f"or fitting failures."
                        ),
                        location=f"{cols[i]}, {cols[j]}",
                        severity=Severity.CRITICAL if abs(r) > 0.999 else Severity.WARNING,
                        suggestion=(
                            f"Drop one of '{cols[i]}' or '{cols[j]}' from the "
                            f"model. Prefer retaining the more interpretable "
                            f"or clinically relevant variable."
                        ),
                    )
                )

    return warnings


def _check_vif(
    X: pd.DataFrame,
    threshold: float,
) -> List[PitfallWarning]:
    """Compute VIF for each feature and flag high values."""
    warnings: List[PitfallWarning] = []
    cols = X.columns.tolist()

    if len(cols) < 2:
        return warnings

    # Standardize to avoid numerical issues
    X_arr = X.values.astype(float)
    X_centered = X_arr - X_arr.mean(axis=0)
    stds = X_centered.std(axis=0)

    # Skip columns with zero std (shouldn't happen after pre-filtering)
    valid_mask = stds > 0
    if valid_mask.sum() < 2:
        return warnings

    X_std = X_centered[:, valid_mask] / stds[valid_mask]
    valid_cols = [c for c, v in zip(cols, valid_mask) if v]

    for idx, col in enumerate(valid_cols):
        # Regress feature idx on all other features
        others = np.delete(X_std, idx, axis=1)

        try:
            # OLS: R^2 = 1 - SS_res / SS_tot
            # Use least squares solution
            coeffs, residuals, rank, sv = np.linalg.lstsq(others, X_std[:, idx], rcond=None)

            y_pred = others @ coeffs
            ss_res = np.sum((X_std[:, idx] - y_pred) ** 2)
            ss_tot = np.sum(X_std[:, idx] ** 2)  # Already centered

            if ss_tot == 0:
                continue

            r_squared = 1.0 - ss_res / ss_tot
            r_squared = min(r_squared, 1.0 - 1e-15)  # Avoid division by zero

            vif = 1.0 / (1.0 - r_squared)

            if vif > threshold:
                warnings.append(
                    PitfallWarning(
                        pitfall=_PITFALL,
                        message=(
                            f"Feature '{col}' has VIF = {vif:.1f} "
                            f"(threshold: {threshold}). This feature is "
                            f"highly predictable from other features in "
                            f"the model (R^2 = {r_squared:.4f})."
                        ),
                        location=col,
                        severity=Severity.CRITICAL if vif > 100 else Severity.WARNING,
                        suggestion=(
                            f"Consider removing '{col}' or one of the "
                            f"features it is collinear with."
                        ),
                    )
                )
        except np.linalg.LinAlgError:
            # If lstsq fails, the matrix is likely singular
            warnings.append(
                PitfallWarning(
                    pitfall=_PITFALL,
                    message=(
                        f"Could not compute VIF for '{col}' due to "
                        f"singular design matrix. This indicates severe "
                        f"multicollinearity."
                    ),
                    location=col,
                    severity=Severity.CRITICAL,
                    suggestion=f"Inspect '{col}' for linear dependencies.",
                )
            )

    return warnings


def _check_algebraic_identity(
    X: pd.DataFrame,
) -> List[PitfallWarning]:
    """Check whether any feature is a perfect linear function of others.

    Uses the rank of the feature matrix: if rank < n_features, there
    is at least one algebraic dependency.
    """
    warnings: List[PitfallWarning] = []

    X_arr = X.values.astype(float)
    n_features = X_arr.shape[1]

    # Check rank on the centred, unit-scaled matrix (implicit intercept,
    # scale-invariant tolerance)
    X_arr = (X_arr - X_arr.mean(axis=0)) / X_arr.std(axis=0)
    rank = np.linalg.matrix_rank(X_arr)

    if rank < n_features:
        n_redundant = n_features - rank
        warnings.append(
            PitfallWarning(
                pitfall=_PITFALL,
                message=(
                    f"Feature matrix has rank {rank} but {n_features} "
                    f"columns. There are {n_redundant} algebraic "
                    f"dependenc{'y' if n_redundant == 1 else 'ies'} "
                    f"(one or more features are exact linear combinations "
                    f"of others)."
                ),
                location=", ".join(X.columns.tolist()),
                severity=Severity.CRITICAL,
                suggestion=(
                    f"Identify and remove {n_redundant} redundant "
                    f"feature(s). Check for derived variables that are "
                    f"algebraic functions of other features in the matrix."
                ),
            )
        )

    return warnings
