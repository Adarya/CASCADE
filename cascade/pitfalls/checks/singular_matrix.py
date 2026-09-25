"""
Check: Singular Matrix Detection
==================================

Detects singular or near-singular design matrices that will cause
linear algebra failures in regression models. Identifies which
columns contribute to rank deficiency.

Pitfall #7 in the CASCADE library.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from ..library import PitfallWarning, Severity, PITFALL_LIBRARY

_PITFALL = PITFALL_LIBRARY[6]  # id=7, singular matrix from constant covariate


def check_singular_matrix(
    X: pd.DataFrame,
    condition_threshold: float = 1e10,
) -> List[PitfallWarning]:
    """Check for singularity or near-singularity in a feature matrix.

    All checks are run on the **centred and unit-variance scaled**
    matrix, which is equivalent to including an intercept (a Cox
    baseline hazard absorbs any constant) and makes the checks
    scale-invariant: a constant column, an exact linear combination
    that involves the intercept (e.g. ``RATE = (N - 1) / 6``) or a full
    set of dummy indicators are caught, while a column that is merely
    rescaled (e.g. days vs. years) is not flagged.

    Performs three checks:

    1. **Condition number**: If the condition number of the standardised
       matrix exceeds ``condition_threshold``, it is ill-conditioned.
    2. **Rank deficiency**: If rank < n_columns, identifies which
       columns are linearly dependent.
    3. **Specific column diagnosis**: Uses SVD to identify which
       columns contribute most to rank deficiency.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples x features). All columns should be
        numeric.
    condition_threshold : float
        Condition number above which the matrix is flagged as
        ill-conditioned (default 1e10).

    Returns
    -------
    list of PitfallWarning
        Warnings about singularity issues with suggestions for which
        columns to drop.
    """
    warnings: List[PitfallWarning] = []

    if X.empty or X.shape[1] == 0:
        return warnings

    X_numeric = X.select_dtypes(include=[np.number])
    if X_numeric.empty:
        return warnings

    X_arr = X_numeric.values.astype(float)
    n_samples, n_features = X_arr.shape
    col_names = X_numeric.columns.tolist()

    # Handle NaN: drop rows with any NaN for matrix checks
    valid_rows = ~np.any(np.isnan(X_arr), axis=1)
    X_clean = X_arr[valid_rows]

    if X_clean.shape[0] < 2:
        warnings.append(
            PitfallWarning(
                pitfall=_PITFALL,
                message=(
                    f"Feature matrix has only {X_clean.shape[0]} valid "
                    f"(non-NaN) rows. Cannot assess singularity."
                ),
                location="entire matrix",
                severity=Severity.WARNING,
                suggestion="Check for excessive missing data.",
            )
        )
        return warnings

    # Standardise: centre (absorbs intercept) and scale (scale-invariance).
    # Constant columns become exact zero columns (rank-deficient).
    means = X_clean.mean(axis=0)
    stds = X_clean.std(axis=0)
    const_mask = ~(stds > 1e-12 * np.maximum(np.abs(means), 1.0))
    Z = np.zeros_like(X_clean)
    Z[:, ~const_mask] = (X_clean[:, ~const_mask] - means[~const_mask]) / stds[~const_mask]
    X_clean = Z

    # --- Check 1: Condition number ---
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            cond = np.linalg.cond(X_clean)
        if not np.isfinite(cond):
            cond = np.inf
        if cond > condition_threshold:
            warnings.append(
                PitfallWarning(
                    pitfall=_PITFALL,
                    message=(
                        f"Feature matrix has condition number = {cond:.2e} "
                        f"(threshold: {condition_threshold:.0e}). The matrix "
                        f"is ill-conditioned and regression fits may be "
                        f"numerically unstable."
                    ),
                    location=", ".join(col_names),
                    severity=Severity.CRITICAL,
                    suggestion=(
                        "Remove or combine collinear features. See column-level "
                        "diagnostics below for specific recommendations."
                    ),
                )
            )
    except np.linalg.LinAlgError:
        warnings.append(
            PitfallWarning(
                pitfall=_PITFALL,
                message="Could not compute condition number (matrix is singular).",
                location=", ".join(col_names),
                severity=Severity.CRITICAL,
                suggestion="The design matrix is exactly singular. Remove redundant columns.",
            )
        )

    # --- Check 2: Rank deficiency (relative SVD tolerance on unit-scaled
    # columns, so the test is scale-invariant) ---
    rank = np.linalg.matrix_rank(X_clean)
    if rank < n_features:
        n_redundant = n_features - rank

        # --- Check 3: Identify problematic columns via SVD ---
        problematic_cols = _identify_redundant_columns(X_clean, col_names, rank)
        # Constant columns are always redundant with the intercept
        const_cols = [c for c, m in zip(col_names, const_mask) if m]
        problematic_cols = const_cols + [c for c in problematic_cols if c not in const_cols]

        warnings.append(
            PitfallWarning(
                pitfall=_PITFALL,
                message=(
                    f"Feature matrix has rank {rank} but {n_features} columns. "
                    f"{n_redundant} column(s) are linearly dependent. "
                    f"Suspected redundant column(s): {problematic_cols}"
                ),
                location=", ".join(problematic_cols),
                severity=Severity.CRITICAL,
                suggestion=(
                    f"Drop the following column(s) to resolve singularity: "
                    f"{problematic_cols}. These columns can be expressed as "
                    f"linear combinations of the remaining columns."
                ),
            )
        )

    return warnings


def _identify_redundant_columns(
    X: np.ndarray,
    col_names: List[str],
    matrix_rank: int,
) -> List[str]:
    """Identify which columns are redundant using pivoted QR decomposition.

    Uses column-pivoted QR factorization to determine a maximal
    linearly independent subset of columns. The columns not in this
    subset are the ones to drop.

    Parameters
    ----------
    X : np.ndarray
        The feature matrix (already cleaned of NaN).
    col_names : list of str
        Column names corresponding to X.
    matrix_rank : int
        The rank of X.

    Returns
    -------
    list of str
        Names of columns that should be dropped.
    """
    n_features = X.shape[1]

    try:
        # QR decomposition with column pivoting
        from scipy.linalg import qr

        _, R, pivot = qr(X, pivoting=True)

        # The first `matrix_rank` pivot indices are the independent columns.
        # The rest are redundant.
        independent_indices = set(pivot[:matrix_rank])
        redundant_indices = [i for i in range(n_features) if i not in independent_indices]

        return [col_names[i] for i in redundant_indices]

    except ImportError:
        # Fallback without scipy: use SVD
        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        # Columns of V corresponding to near-zero singular values
        # indicate the directions of redundancy
        redundant = []
        tol = max(X.shape) * S[0] * np.finfo(float).eps

        for i in range(len(S)):
            if S[i] < tol:
                # Find which original column loads most heavily
                # on this null-space direction
                v = Vt[i, :]
                max_col_idx = int(np.argmax(np.abs(v)))
                col_name = col_names[max_col_idx]
                if col_name not in redundant:
                    redundant.append(col_name)

        # If we haven't found enough, add columns by process of
        # greedy elimination
        if len(redundant) < (n_features - matrix_rank):
            for idx in range(n_features - 1, -1, -1):
                if col_names[idx] not in redundant:
                    test_cols = [i for i in range(n_features)
                                 if i != idx and col_names[i] not in redundant]
                    if len(test_cols) == 0:
                        continue
                    sub_rank = np.linalg.matrix_rank(X[:, test_cols])
                    if sub_rank == matrix_rank:
                        redundant.append(col_names[idx])
                if len(redundant) >= (n_features - matrix_rank):
                    break

        return redundant
