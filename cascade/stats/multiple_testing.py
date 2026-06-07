"""
Multiple Testing Corrections
=============================

Thin wrappers around ``statsmodels.stats.multitest`` and
``scipy.stats`` for Benjamini-Hochberg FDR, Bonferroni, and
Holm-Bonferroni corrections.

All functions accept a 1-D array-like of raw p-values and return
a tuple of (adjusted p-values, boolean significance mask).
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from statsmodels.stats.multitest import multipletests


def apply_fdr(
    p_values: ArrayLike,
    alpha: float = 0.05,
    method: str = "fdr_bh",
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply FDR correction (Benjamini-Hochberg by default).

    Parameters
    ----------
    p_values : array-like
        Raw (unadjusted) p-values.  NaN values are preserved in the
        output with ``significant = False``.
    alpha : float, default 0.05
        Family-wise error rate / FDR threshold.
    method : str, default "fdr_bh"
        Any method accepted by ``statsmodels.stats.multitest.multipletests``:
        ``"fdr_bh"`` (Benjamini-Hochberg), ``"fdr_by"`` (Benjamini-Yekutieli),
        ``"fdr_tsbh"`` (two-stage BH), etc.

    Returns
    -------
    adjusted : ndarray of float
        Adjusted p-values (same length as input).
    significant : ndarray of bool
        ``True`` where adjusted p < *alpha*.
    """
    return _correct(p_values, alpha=alpha, method=method)


def apply_bonferroni(
    p_values: ArrayLike,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Bonferroni correction.

    Parameters
    ----------
    p_values : array-like
        Raw p-values.
    alpha : float, default 0.05

    Returns
    -------
    adjusted : ndarray
    significant : ndarray of bool
    """
    return _correct(p_values, alpha=alpha, method="bonferroni")


def apply_holm(
    p_values: ArrayLike,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Holm-Bonferroni (step-down) correction.

    Parameters
    ----------
    p_values : array-like
        Raw p-values.
    alpha : float, default 0.05

    Returns
    -------
    adjusted : ndarray
    significant : ndarray of bool
    """
    return _correct(p_values, alpha=alpha, method="holm")


# ------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------

def _correct(
    p_values: ArrayLike,
    alpha: float,
    method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a multipletests correction, handling NaN gracefully."""
    p = np.asarray(p_values, dtype=float)

    if p.ndim != 1:
        raise ValueError(f"p_values must be 1-D, got shape {p.shape}")

    if len(p) == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)

    # Identify valid (non-NaN) entries
    valid = ~np.isnan(p)

    if valid.sum() == 0:
        return np.full_like(p, np.nan), np.zeros_like(p, dtype=bool)

    # Run correction only on valid values
    reject, adj_p, _, _ = multipletests(p[valid], alpha=alpha, method=method)

    # Place results back
    adjusted = np.full_like(p, np.nan)
    significant = np.zeros_like(p, dtype=bool)
    adjusted[valid] = adj_p
    significant[valid] = reject

    return adjusted, significant
