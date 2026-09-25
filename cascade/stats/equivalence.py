"""
Independence and Equivalence Testing
======================================

Tools for demonstrating that two variables are **not associated**,
which is the complementary problem to standard hypothesis testing.

Includes:
  - Phi coefficient with Fisher z-transform confidence interval
  - TOST (Two One-Sided Tests) equivalence procedure
  - Bayes factor for independence in 2x2 tables
  - Comprehensive independence test suite
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ======================================================================
# Phi coefficient
# ======================================================================

def phi_coefficient(
    x: np.ndarray,
    y: np.ndarray,
    ci_level: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute the phi coefficient between two binary variables with CI.

    The confidence interval is computed via Fisher z-transform of the
    Pearson correlation.

    Parameters
    ----------
    x, y : array-like
        Binary (0/1) arrays of equal length.
    ci_level : float, default 0.95
        Confidence level for the interval.

    Returns
    -------
    phi : float
        Phi coefficient (equivalent to Pearson r for binary data).
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    n = len(x)
    if n < 4:
        raise ValueError("Need at least 4 observations.")

    # Phi = Pearson r for binary variables (NaN if either is constant)
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan, np.nan, np.nan
    phi = np.corrcoef(x, y)[0, 1]

    if np.isnan(phi):
        return np.nan, np.nan, np.nan

    # Fisher z-transform for CI
    z = np.arctanh(phi)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf((1 + ci_level) / 2)

    ci_lower = np.tanh(z - z_crit * se)
    ci_upper = np.tanh(z + z_crit * se)

    return float(phi), float(ci_lower), float(ci_upper)


# ======================================================================
# TOST equivalence
# ======================================================================

def tost_equivalence(
    x: np.ndarray,
    y: np.ndarray,
    equivalence_bound: float,
    alpha: float = 0.05,
) -> Tuple[bool, float, float]:
    """Two One-Sided Tests (TOST) for equivalence of means.

    Tests whether the difference in means of *x* and *y* falls within
    [-equivalence_bound, +equivalence_bound].

    Parameters
    ----------
    x, y : array-like
        Two samples to compare.
    equivalence_bound : float
        Maximum allowable difference in means (symmetric bound).
    alpha : float, default 0.05
        Significance level of each one-sided test.

    Returns
    -------
    reject_null : bool
        ``True`` if we can reject the null of non-equivalence
        (i.e., the means are equivalent within the bound).
    p1 : float
        P-value for the lower bound test (mean diff > -bound).
    p2 : float
        P-value for the upper bound test (mean diff < +bound).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        raise ValueError("Each sample must have at least 2 observations.")

    mean_diff = x.mean() - y.mean()
    se = np.sqrt(x.var(ddof=1) / n1 + y.var(ddof=1) / n2)
    df = _welch_df(x, y)

    if se == 0:
        # Identical samples
        p1 = 0.0 if mean_diff >= -equivalence_bound else 1.0
        p2 = 0.0 if mean_diff <= equivalence_bound else 1.0
        return (p1 < alpha and p2 < alpha), p1, p2

    # Test 1: H0: mean_diff <= -bound  vs  H1: mean_diff > -bound
    t1 = (mean_diff - (-equivalence_bound)) / se
    p1 = stats.t.sf(t1, df)  # upper tail

    # Test 2: H0: mean_diff >= +bound  vs  H1: mean_diff < +bound
    t2 = (mean_diff - equivalence_bound) / se
    p2 = stats.t.cdf(t2, df)  # lower tail

    reject = (p1 < alpha) and (p2 < alpha)
    return bool(reject), float(p1), float(p2)


# ======================================================================
# Bayes factor for independence
# ======================================================================

def bayes_factor_independence(
    contingency_table: np.ndarray,
) -> float:
    """Approximate Bayes factor for independence in a 2x2 table.

    Uses the BIC approximation:
        BF10 ~ exp((BIC_H0 - BIC_H1) / 2) = exp((G2 - dof * ln n) / 2)
    where H0 is independence and H1 is association, and G2 is the
    likelihood-ratio (deviance) statistic of the independence model.
    The BIC difference is defined in terms of the log-likelihood ratio,
    so G2 (not Pearson's chi-square, which only approximates it) is used.

    Returns NaN when the table is degenerate (empty, or a zero row/column
    margin, i.e. one variable is constant), because the association
    model is then not identifiable.

    A BF10 > 1 supports association; BF10 < 1 (i.e., 1/BF10 > 1)
    supports independence.

    Parameters
    ----------
    contingency_table : 2-D array-like, shape (2, 2)
        Observed frequencies.

    Returns
    -------
    bf10 : float
        Bayes factor in favor of association (H1) over independence (H0).
        Values < 1 support independence.
    """
    table = np.asarray(contingency_table, dtype=float)
    if table.shape != (2, 2):
        raise ValueError("contingency_table must be 2x2.")

    n = table.sum()
    if n == 0 or (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        return np.nan

    # Likelihood-ratio G^2 statistic (deviance of the independence model)
    g2, p, dof, expected = stats.chi2_contingency(
        table, correction=False, lambda_="log-likelihood"
    )

    # BIC approximation: BIC_H0 - BIC_H1 = G2 - dof * ln(n)
    # For a 2x2 table, dof = 1
    bic_diff = g2 - dof * np.log(n)
    bf10 = np.exp(bic_diff / 2.0)

    return float(bf10)


# ======================================================================
# Comprehensive independence suite
# ======================================================================

def independence_suite(
    x_binary: np.ndarray,
    y_binary: np.ndarray,
    x_continuous: Optional[np.ndarray] = None,
    y_continuous: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run a comprehensive battery of independence / association tests.

    Parameters
    ----------
    x_binary, y_binary : array-like
        Binary (0/1) variables (required).
    x_continuous, y_continuous : array-like, optional
        Continuous versions of x and y for rank-based tests.
        If not provided, the binary versions are used.

    Returns
    -------
    dict
        Keys:
        - ``chi2_stat``, ``chi2_p`` : Chi-squared test on 2x2 table
        - ``phi``, ``phi_ci_lower``, ``phi_ci_upper`` : Phi coefficient + CI
        - ``spearman_rho``, ``spearman_p`` : Spearman rank correlation
        - ``kendall_tau``, ``kendall_p`` : Kendall's tau
        - ``cramers_v`` : Cramer's V statistic
        - ``discordance_rate`` : Fraction of pairs where x != y
        - ``bf10`` : Bayes factor for association (BIC approximation
          based on the likelihood-ratio G^2)
        - ``n`` : Sample size
        - ``phi_p`` : p-value for phi (equal to ``chi2_p`` for a 2x2 table)
        - ``independent`` : summary flag using the same rule as
          ``ArtifactGuard``'s fallback: ``True`` if ``chi2_p > 0.05`` or
          ``|phi| < 0.1``; ``False`` otherwise; ``None`` when not
          evaluable (too few observations or a constant variable)
        - ``note`` : str, explanation when a statistic is NaN (e.g. a
          constant variable); empty string otherwise

        Statistics that cannot be computed (e.g. because one variable is
        constant) are returned as NaN rather than raising.
    """
    xb = np.asarray(x_binary, dtype=float)
    yb = np.asarray(y_binary, dtype=float)

    if len(xb) != len(yb):
        raise ValueError("x_binary and y_binary must have the same length.")

    # Remove NaN pairs
    mask = ~(np.isnan(xb) | np.isnan(yb))
    xb = xb[mask]
    yb = yb[mask]
    n = len(xb)

    result: Dict[str, Any] = {"n": n, "independent": None, "note": ""}

    if n < 4:
        # Not enough data for meaningful tests
        for key in [
            "chi2_stat", "chi2_p", "phi", "phi_ci_lower", "phi_ci_upper",
            "spearman_rho", "spearman_p", "kendall_tau", "kendall_p",
            "cramers_v", "discordance_rate", "bf10", "phi_p",
        ]:
            result[key] = np.nan
        result["note"] = f"Too few observations (n={n}) for independence tests."
        return result

    notes = []
    constant = [name for name, v in (("x", xb), ("y", yb)) if np.unique(v).size < 2]
    if constant:
        notes.append(
            f"Variable(s) {constant} constant; association statistics are "
            f"undefined (NaN)."
        )

    # Contingency table
    table = np.array([
        [(xb == 0) & (yb == 0), (xb == 0) & (yb == 1)],
        [(xb == 1) & (yb == 0), (xb == 1) & (yb == 1)],
    ])
    table = np.array([[s.sum() for s in row] for row in table], dtype=float)

    # Chi-squared
    try:
        if constant:
            raise ValueError("constant variable")
        chi2, chi2_p, _, _ = stats.chi2_contingency(table, correction=True)
    except Exception:
        chi2, chi2_p = np.nan, np.nan
    result["chi2_stat"] = float(chi2)
    result["chi2_p"] = float(chi2_p)
    result["phi_p"] = float(chi2_p)

    # Phi coefficient
    phi_val, phi_lo, phi_hi = phi_coefficient(xb, yb)
    result["phi"] = phi_val
    result["phi_ci_lower"] = phi_lo
    result["phi_ci_upper"] = phi_hi

    # Cramer's V (for 2x2, equals |phi|)
    result["cramers_v"] = abs(phi_val) if not np.isnan(phi_val) else np.nan

    # Discordance rate
    result["discordance_rate"] = float(np.mean(xb != yb))

    # Rank-based correlations (use continuous if available)
    xc = np.asarray(x_continuous, dtype=float) if x_continuous is not None else xb
    yc = np.asarray(y_continuous, dtype=float) if y_continuous is not None else yb

    # Align continuous arrays with binary NaN mask
    if x_continuous is not None or y_continuous is not None:
        if len(xc) == len(mask):
            xc = xc[mask]
        if len(yc) == len(mask):
            yc = yc[mask]
        # Remove any remaining NaN
        cmask = ~(np.isnan(xc) | np.isnan(yc))
        xc = xc[cmask]
        yc = yc[cmask]

    if len(xc) >= 4 and len(yc) >= 4 and np.unique(xc).size > 1 and np.unique(yc).size > 1:
        rho, sp = stats.spearmanr(xc, yc)
        result["spearman_rho"] = float(rho)
        result["spearman_p"] = float(sp)

        tau, tp = stats.kendalltau(xc, yc)
        result["kendall_tau"] = float(tau)
        result["kendall_p"] = float(tp)
    else:
        result["spearman_rho"] = np.nan
        result["spearman_p"] = np.nan
        result["kendall_tau"] = np.nan
        result["kendall_p"] = np.nan

    # Bayes factor
    try:
        result["bf10"] = bayes_factor_independence(table)
    except Exception as exc:
        result["bf10"] = np.nan
        notes.append(f"Bayes factor failed: {exc}")

    if np.isfinite(result["chi2_p"]) and np.isfinite(result["phi"]):
        # Same rule as ArtifactGuard.independence_decision: P > 0.05 and |phi| < 0.1
        result["independent"] = bool(
            result["chi2_p"] > 0.05 and abs(result["phi"]) < 0.1
        )
    result["note"] = " ".join(notes)
    return result


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _welch_df(x: np.ndarray, y: np.ndarray) -> float:
    """Welch-Satterthwaite degrees of freedom."""
    n1, n2 = len(x), len(y)
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    num = (v1 / n1 + v2 / n2) ** 2
    denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    if denom == 0:
        return float(n1 + n2 - 2)
    return float(num / denom)
