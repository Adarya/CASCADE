"""
Layer 1: Biomarker Discovery Screen
=====================================

High-throughput screening of candidate biomarkers using survival and
cross-sectional models, with automatic multiple-testing correction and
separation-problem detection.

Classes
-------
BiomarkerScreen
    Screens many biomarkers against an outcome using Cox, logistic, or
    competing-risks models with FDR or Bonferroni correction.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

try:
    from cascade.stats.cox import PenalizedCox
except ImportError:
    PenalizedCox = None  # type: ignore[assignment,misc]

try:
    from cascade.stats.competing_risks import cause_specific_screen
except ImportError:
    cause_specific_screen = None  # type: ignore[assignment,misc]

try:
    from cascade.stats.multiple_testing import apply_fdr, apply_bonferroni, apply_holm
except ImportError:
    apply_fdr = None  # type: ignore[assignment,misc]
    apply_bonferroni = None  # type: ignore[assignment,misc]
    apply_holm = None  # type: ignore[assignment,misc]


def _apply_correction_fallback(
    p_values: pd.Series,
    method: str,
    alpha: float,
) -> pd.Series:
    """Apply multiple-testing correction using scipy as fallback."""
    from statsmodels.stats.multitest import multipletests

    valid = p_values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=p_values.index)

    reject, pvals_corrected, _, _ = multipletests(
        valid.values, alpha=alpha, method=method
    )
    result = pd.Series(np.nan, index=p_values.index)
    result[valid.index] = pvals_corrected
    return result


def _is_continuous(series: pd.Series) -> bool:
    """True for a numeric (non-bool) series whose values are not all in {0, 1}."""
    if series.dtype == bool or not pd.api.types.is_numeric_dtype(series):
        return False
    vals = pd.unique(series.dropna())
    return not set(np.asarray(vals, dtype=float).tolist()).issubset({0.0, 1.0})



def _parse_event_column(df: pd.DataFrame, event_col: Optional[str]) -> pd.DataFrame:
    """Return *df* with a non-numeric event column parsed to 1/0/NaN.

    Accepts cBioPortal status strings (``'1:DECEASED'``, ``'0:LIVING'``)
    and booleans; numeric columns are returned unchanged.
    """
    if event_col is None or event_col not in df.columns:
        return df
    if pd.api.types.is_numeric_dtype(df[event_col]) and not pd.api.types.is_bool_dtype(df[event_col]):
        return df
    from cascade.core.cohort import parse_event_status

    df = df.copy()
    df[event_col] = df[event_col].map(parse_event_status)
    return df


def _encode_categorical(sub: pd.DataFrame, covariates: List[str]):
    """Dummy-encode non-numeric covariates (first level as reference).

    Returns the frame with encoded columns and the updated covariate list.
    """
    cat = [c for c in covariates
           if c in sub.columns and not pd.api.types.is_numeric_dtype(sub[c])]
    if not cat:
        return sub, covariates
    dummies = pd.get_dummies(sub[cat].astype(str), prefix=cat, drop_first=True, dtype=float)
    sub = pd.concat([sub.drop(columns=cat), dummies], axis=1)
    covs = [c for c in covariates if c not in cat] + list(dummies.columns)
    return sub, covs

class BiomarkerScreen:
    """High-throughput biomarker screening with multiple-testing correction.

    Screens a set of candidate biomarkers (binary or continuous) against a
    clinical outcome using one of several statistical methods.

    Parameters
    ----------
    method : str, default 'cox'
        Statistical method: ``'cox'`` (survival), ``'logistic'``
        (cross-sectional binary outcome), or ``'competing_risks'``
        (cause-specific hazard).
    correction : str, default 'fdr_bh'
        Multiple-testing correction: ``'fdr_bh'``, ``'bonferroni'``,
        ``'holm'``.
    alpha : float, default 0.05
        Significance threshold after correction.
    min_exposed : int, default 20
        Minimum number of patients with biomarker == 1 to attempt fitting
        (binary biomarkers only; not applied to continuous biomarkers).
    min_events : int, default 5
        Minimum events in the exposed group.
    penalizer : float, default 0.01
        Ridge penalty for Cox models.
    separation_ci_ratio : float, default 100
        Maximum allowed ratio of upper/lower CI bounds.  Results exceeding
        this are flagged as separation problems and excluded.
    """

    def __init__(
        self,
        method: str = "cox",
        correction: str = "fdr_bh",
        alpha: float = 0.05,
        min_exposed: int = 20,
        min_events: int = 5,
        penalizer: float = 0.01,
        separation_ci_ratio: float = 100,
    ) -> None:
        valid_methods = {"cox", "logistic", "competing_risks"}
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")

        self.method = method
        self.correction = correction
        self.alpha = alpha
        self.min_exposed = min_exposed
        self.min_events = min_events
        self.penalizer = penalizer
        self.separation_ci_ratio = separation_ci_ratio

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def screen(
        self,
        df: pd.DataFrame,
        biomarker_cols: Sequence[str],
        outcome_col: str,
        event_col: Optional[str] = None,
        covariates: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Screen biomarkers against a single outcome.

        Parameters
        ----------
        df : DataFrame
            Analysis-ready dataset.
        biomarker_cols : sequence of str
            Columns to test as candidate biomarkers.
        outcome_col : str
            Duration column (for Cox) or binary 0/1 outcome (for logistic).
        event_col : str, optional
            Event indicator column.  Required for ``'cox'`` /
            ``'competing_risks'``; ignored for ``'logistic'`` (the binary
            outcome is *outcome_col*).
        covariates : sequence of str, optional
            Adjustment covariates.

        Returns
        -------
        DataFrame
            One row per biomarker with columns: ``biomarker``, ``hr``
            (for logistic this holds the odds ratio), ``ci_lower``,
            ``ci_upper``, ``p``, ``p_adjusted``, ``significant``,
            ``n_exposed``, ``n_events_exposed``.  Binary (0/1) biomarkers
            must have at least *min_exposed* carriers; continuous biomarkers
            are not subject to *min_exposed* and report ``n_exposed`` /
            ``n_events_exposed`` as NaN (the *min_events* check is then
            applied to all events).
        """
        if self.method != "logistic" and event_col is None:
            raise ValueError(
                f"event_col is required for method='{self.method}'."
            )
        df = _parse_event_column(df, event_col)
        results: List[Dict[str, Any]] = []

        for col in biomarker_cols:
            row = self._fit_single(df, col, outcome_col, event_col, covariates)
            results.append(row)

        results_df = pd.DataFrame(results)

        if results_df.empty:
            return results_df

        # Filter separation problems
        results_df = self._filter_separation(results_df)

        # Multiple-testing correction
        results_df["p_adjusted"] = self._correct(results_df["p"])
        results_df["significant"] = results_df["p_adjusted"] < self.alpha

        return results_df

    def screen_competing_risks(
        self,
        df: pd.DataFrame,
        biomarker_cols: Sequence[str],
        site_cols: Sequence[str],
        duration_col: str,
        event_col: str,
        covariates: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Screen gene-site pairs using cause-specific Cox models.

        Parameters
        ----------
        df : DataFrame
        biomarker_cols : sequence of str
            Gene mutation columns (binary 0/1).
        site_cols : sequence of str
            Site-specific event columns (binary 0/1).
        duration_col : str
        event_col : str
        covariates : sequence of str, optional

        Returns
        -------
        DataFrame
            One row per gene-site pair with standard result columns plus
            ``p_adjusted`` and ``significant``.
        """
        all_results: List[pd.DataFrame] = []
        covs = list(covariates) if covariates else []

        for gene in biomarker_cols:
            if cause_specific_screen is not None:
                res = cause_specific_screen(
                    df=df,
                    gene_col=gene,
                    site_cols=site_cols,
                    duration_col=duration_col,
                    event_col=event_col,
                    covariates=covs,
                    penalizer=self.penalizer,
                    min_exposed=self.min_exposed,
                    min_events=self.min_events,
                )
                all_results.append(res)
            else:
                # Fallback: iterate manually
                for site in site_cols:
                    row = self._fit_single(df, gene, duration_col, site, covs)
                    row["site"] = site
                    row["gene"] = gene
                    all_results.append(pd.DataFrame([row]))

        if not all_results:
            return pd.DataFrame()

        results_df = pd.concat(all_results, ignore_index=True)
        results_df = self._filter_separation(results_df)

        results_df["p_adjusted"] = self._correct(results_df["p"])
        results_df["significant"] = results_df["p_adjusted"] < self.alpha

        return results_df

    # ------------------------------------------------------------------
    # Single model fitting
    # ------------------------------------------------------------------

    def _fit_single(
        self,
        df: pd.DataFrame,
        biomarker_col: str,
        outcome_col: str,
        event_col: str,
        covariates: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Fit a single biomarker model and extract key statistics.

        Parameters
        ----------
        df : DataFrame
        biomarker_col : str
        outcome_col : str
        event_col : str
        covariates : sequence of str, optional

        Returns
        -------
        dict
            Result dictionary with standard keys.
        """
        covs = list(covariates) if covariates else []
        # For logistic models the binary outcome *is* the event; never use
        # the same column twice (duplicate columns break the fit).
        ev_col = outcome_col if (self.method == "logistic" or event_col is None) else event_col
        all_cols = list(dict.fromkeys([outcome_col, ev_col, biomarker_col] + covs))
        sub = df[all_cols].dropna()
        sub, covs = _encode_categorical(sub, covs)

        row: Dict[str, Any] = {"biomarker": biomarker_col}

        if _is_continuous(sub[biomarker_col]):
            # Continuous biomarker: "exposed" is undefined -> min_exposed
            # does not apply; require min_events over all observations.
            row["n_exposed"] = np.nan
            row["n_events_exposed"] = np.nan
            too_small = (
                len(sub) == 0
                or float(pd.to_numeric(sub[ev_col], errors="coerce").sum()) < self.min_events
            )
        else:
            exposed = sub[biomarker_col].astype(bool) if sub[biomarker_col].dtype == object else (sub[biomarker_col] == 1)
            n_exposed = int(exposed.sum())
            n_events_exposed = int(sub.loc[exposed, ev_col].sum()) if n_exposed > 0 else 0
            row["n_exposed"] = n_exposed
            row["n_events_exposed"] = n_events_exposed
            too_small = n_exposed < self.min_exposed or n_events_exposed < self.min_events

        if too_small:
            row.update(hr=np.nan, ci_lower=np.nan, ci_upper=np.nan, p=np.nan)
            return row

        if self.method == "cox":
            row.update(self._fit_cox(sub, biomarker_col, outcome_col, ev_col, covs))
        elif self.method == "logistic":
            # Binary outcome is outcome_col (per the screen() docstring)
            row.update(self._fit_logistic(sub, biomarker_col, outcome_col, covs))
        else:
            # Competing risks handled via screen_competing_risks
            row.update(self._fit_cox(sub, biomarker_col, outcome_col, ev_col, covs))

        return row

    def _fit_cox(
        self,
        sub: pd.DataFrame,
        biomarker_col: str,
        duration_col: str,
        event_col: str,
        covariates: List[str],
    ) -> Dict[str, Any]:
        """Fit a Cox PH model for a single biomarker."""
        from lifelines import CoxPHFitter

        all_covs = [biomarker_col] + [c for c in covariates if c != biomarker_col]

        # Drop constant columns
        fit_covs = [c for c in all_covs if sub[c].nunique() > 1]
        if biomarker_col not in fit_covs:
            return dict(hr=np.nan, ci_lower=np.nan, ci_upper=np.nan, p=np.nan)

        fit_data = sub[[duration_col, event_col] + fit_covs]

        try:
            fitter = CoxPHFitter(penalizer=self.penalizer)
            fitter.fit(fit_data, duration_col=duration_col, event_col=event_col, show_progress=False)
            s = fitter.summary.loc[biomarker_col]
            return dict(
                hr=s["exp(coef)"],
                ci_lower=s["exp(coef) lower 95%"],
                ci_upper=s["exp(coef) upper 95%"],
                p=s["p"],
            )
        except Exception as exc:
            return dict(hr=np.nan, ci_lower=np.nan, ci_upper=np.nan, p=np.nan,
                        fit_error=f"{type(exc).__name__}: {exc}")

    def _fit_logistic(
        self,
        sub: pd.DataFrame,
        biomarker_col: str,
        outcome_col: str,
        covariates: List[str],
    ) -> Dict[str, Any]:
        """Fit a logistic regression for a single biomarker."""
        import statsmodels.api as sm

        all_covs = [biomarker_col] + [c for c in covariates if c != biomarker_col]
        fit_covs = [c for c in all_covs if sub[c].nunique() > 1]
        if biomarker_col not in fit_covs:
            return dict(hr=np.nan, ci_lower=np.nan, ci_upper=np.nan, p=np.nan)

        X = sm.add_constant(sub[fit_covs].astype(float))
        y = sub[outcome_col].astype(float)

        try:
            model = sm.Logit(y, X).fit(disp=0, maxiter=100)
            coef = model.params[biomarker_col]
            ci = model.conf_int().loc[biomarker_col]
            return dict(
                hr=np.exp(coef),  # OR, stored as hr for uniformity
                ci_lower=np.exp(ci[0]),
                ci_upper=np.exp(ci[1]),
                p=model.pvalues[biomarker_col],
            )
        except Exception as exc:
            return dict(hr=np.nan, ci_lower=np.nan, ci_upper=np.nan, p=np.nan,
                        fit_error=f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _filter_separation(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Remove results with quasi-complete separation.

        Flagged when the CI ratio (upper / lower) exceeds
        *separation_ci_ratio*.

        Parameters
        ----------
        results_df : DataFrame
            Must have ``ci_lower`` and ``ci_upper`` columns.

        Returns
        -------
        DataFrame
            Rows with separation problems have hr/ci/p set to NaN.
        """
        if "ci_lower" not in results_df.columns or "ci_upper" not in results_df.columns:
            return results_df

        mask = results_df["ci_lower"] > 0
        ratio = results_df["ci_upper"] / results_df["ci_lower"].replace(0, np.nan)
        separation = mask & (ratio > self.separation_ci_ratio)

        if separation.any():
            n_sep = separation.sum()
            warnings.warn(
                f"BiomarkerScreen: {n_sep} result(s) flagged as separation "
                f"problems (CI ratio > {self.separation_ci_ratio}).",
                stacklevel=2,
            )
            results_df.loc[separation, ["hr", "ci_lower", "ci_upper", "p"]] = np.nan

        return results_df

    def _correct(self, p_values: pd.Series) -> pd.Series:
        """Apply the configured multiple-testing correction.

        Returns
        -------
        pd.Series
            Adjusted p-values (same index as input).
        """
        # Try CASCADE stats functions first — they return (adjusted, significant) tuples
        if self.correction == "fdr_bh" and apply_fdr is not None:
            adjusted, _ = apply_fdr(p_values.values, alpha=self.alpha)
            return pd.Series(adjusted, index=p_values.index)
        if self.correction == "bonferroni" and apply_bonferroni is not None:
            adjusted, _ = apply_bonferroni(p_values.values, alpha=self.alpha)
            return pd.Series(adjusted, index=p_values.index)
        if self.correction == "holm" and apply_holm is not None:
            adjusted, _ = apply_holm(p_values.values, alpha=self.alpha)
            return pd.Series(adjusted, index=p_values.index)

        # Fallback to statsmodels
        return _apply_correction_fallback(p_values, self.correction, self.alpha)
