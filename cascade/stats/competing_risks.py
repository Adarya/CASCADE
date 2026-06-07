"""
Competing Risks Analysis
=========================

Cause-specific Cox regression and cumulative incidence estimation.

* ``CauseSpecificCox`` fits a Cox model for a specific cause while
  treating all other event types as censored observations.
* ``cause_specific_screen`` runs cause-specific Cox regressions over
  many gene-site combinations and returns a tidy results DataFrame.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

try:
    from lifelines import AalenJohansenFitter

    _HAS_AJ = True
except ImportError:
    _HAS_AJ = False


class CauseSpecificCox:
    """Cause-specific Cox proportional hazards model.

    For a given cause of interest, events from other causes are treated
    as censored observations.  This yields cause-specific hazard ratios
    that are directly interpretable as the instantaneous rate ratio for
    the event of interest.

    Parameters
    ----------
    penalizer : float, default 0.01
        Ridge penalty forwarded to ``CoxPHFitter``.
    """

    def __init__(self, penalizer: float = 0.01) -> None:
        self.penalizer = penalizer
        self._fitter: Optional[CoxPHFitter] = None
        self._cause: Optional[Any] = None
        self._converged: bool = False
        self._error_message: Optional[str] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        cause_of_interest: Any,
        covariates: Sequence[str],
    ) -> "CauseSpecificCox":
        """Fit a cause-specific Cox model.

        Parameters
        ----------
        df : DataFrame
            Must contain *duration_col*, *event_col*, and all *covariates*.
        duration_col : str
            Column with follow-up time.
        event_col : str
            Column encoding the event type.  Values equal to
            *cause_of_interest* are treated as events; all others
            (including 0 / censored) are treated as censored.
        cause_of_interest : any
            The value in *event_col* that denotes the event of interest.
        covariates : sequence of str
            Predictor columns.

        Returns
        -------
        self
        """
        self._cause = cause_of_interest
        data = df[[duration_col, event_col] + list(covariates)].copy()
        data = data.dropna()

        if len(data) == 0:
            raise ValueError("No rows remaining after dropping NaN values.")

        # Recode event: 1 if cause_of_interest, 0 otherwise
        cs_event_col = f"__cs_event_{cause_of_interest}"
        data[cs_event_col] = (data[event_col] == cause_of_interest).astype(int)

        # Drop constant covariates
        final_covs: List[str] = []
        dropped: List[str] = []
        for c in covariates:
            if data[c].nunique() <= 1:
                dropped.append(c)
            else:
                final_covs.append(c)

        if dropped:
            warnings.warn(
                f"CauseSpecificCox: dropped constant columns {dropped}",
                stacklevel=2,
            )

        if len(final_covs) == 0:
            raise ValueError("All covariates are constant.")

        fit_cols = [duration_col, cs_event_col] + final_covs
        data = data[fit_cols]

        self._fitter = CoxPHFitter(penalizer=self.penalizer)
        try:
            self._fitter.fit(
                data,
                duration_col=duration_col,
                event_col=cs_event_col,
                show_progress=False,
            )
            self._converged = True
            self._error_message = None
        except Exception as exc:
            self._converged = False
            self._error_message = str(exc)
            raise

        return self

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    @property
    def summary(self) -> pd.DataFrame:
        """Coefficient summary table."""
        if self._fitter is None:
            raise RuntimeError("Model not fitted.")
        return self._fitter.summary

    @property
    def concordance_index_(self) -> float:
        """Training concordance index."""
        if self._fitter is None:
            raise RuntimeError("Model not fitted.")
        return self._fitter.concordance_index_

    @property
    def converged(self) -> bool:
        """Whether the model converged."""
        return self._converged

    # ------------------------------------------------------------------
    # Cumulative incidence
    # ------------------------------------------------------------------

    def cumulative_incidence(
        self,
        durations: pd.Series,
        event_observed: pd.Series,
        times: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Estimate cumulative incidence function via Aalen-Johansen.

        This is a non-parametric estimator that accounts for competing
        risks and does not use the fitted Cox model directly.

        Parameters
        ----------
        durations : Series
            Follow-up times.
        event_observed : Series
            Event type indicator (0 = censored, other values = causes).
        times : array-like, optional
            Time points at which to evaluate the CIF.  If ``None``,
            all unique event times are used.

        Returns
        -------
        DataFrame
            Cumulative incidence at each time point, indexed by time,
            with one column per cause.
        """
        if not _HAS_AJ:
            raise ImportError(
                "AalenJohansenFitter is not available in your version of "
                "lifelines.  Please upgrade to lifelines >= 0.27."
            )

        if self._cause is None:
            raise RuntimeError("Model not fitted; cause of interest unknown.")

        aj = AalenJohansenFitter(calculate_variance=False)
        aj.fit(durations, event_observed=event_observed, event_of_interest=self._cause)

        cif = aj.cumulative_density_

        if times is not None:
            # Interpolate to requested times
            times = np.asarray(times)
            idx = np.searchsorted(cif.index.values, times, side="right") - 1
            idx = np.clip(idx, 0, len(cif) - 1)
            result = cif.iloc[idx].copy()
            result.index = pd.Index(times, name="timeline")
            return result

        return cif


def cause_specific_screen(
    df: pd.DataFrame,
    gene_col: str,
    site_cols: Sequence[str],
    duration_col: str,
    event_col: str,
    covariates: Sequence[str],
    penalizer: float = 0.01,
    min_exposed: int = 10,
    min_events: int = 5,
) -> pd.DataFrame:
    """Screen many gene-site pairs with cause-specific Cox models.

    For each site column, a binary event indicator is constructed
    (1 = site-specific event, 0 = censored or other), and a Cox model
    is fitted with *gene_col* as the predictor of interest.

    Parameters
    ----------
    df : DataFrame
        Analysis-ready dataset.
    gene_col : str
        Binary (0/1) column for gene mutation status.
    site_cols : sequence of str
        Binary (0/1) columns for site-specific events.
    duration_col : str
        Follow-up time column.
    event_col : str
        Overall event indicator (used as baseline; overridden by each
        site column when constructing cause-specific events).
    covariates : sequence of str
        Adjustment covariates (in addition to *gene_col*).
    penalizer : float, default 0.01
        Ridge penalty.
    min_exposed : int, default 10
        Minimum patients with gene_col == 1 to attempt the model.
    min_events : int, default 5
        Minimum events of the cause of interest to attempt the model.

    Returns
    -------
    DataFrame
        One row per site with columns: gene, site, n, n_events,
        n_exposed, coef, hr, se, z, p, ci_lower, ci_upper, converged.
    """
    results: List[Dict[str, Any]] = []
    all_covs = [gene_col] + [c for c in covariates if c != gene_col]

    for site in site_cols:
        row: Dict[str, Any] = {
            "gene": gene_col,
            "site": site,
        }

        sub = df[[duration_col, site, gene_col] + list(covariates)].dropna()
        n_events = int(sub[site].sum())
        n_exposed = int(sub[gene_col].sum())
        row["n"] = len(sub)
        row["n_events"] = n_events
        row["n_exposed"] = n_exposed

        if n_exposed < min_exposed or n_events < min_events:
            row.update(
                coef=np.nan, hr=np.nan, se=np.nan, z=np.nan,
                p=np.nan, ci_lower=np.nan, ci_upper=np.nan,
                converged=False,
            )
            results.append(row)
            continue

        # Build cause-specific event column
        cs_event = f"__cs_{site}"
        sub = sub.copy()
        sub[cs_event] = sub[site].astype(int)

        # Drop constant covariates
        fit_covs = [c for c in all_covs if sub[c].nunique() > 1]
        if gene_col not in fit_covs:
            row.update(
                coef=np.nan, hr=np.nan, se=np.nan, z=np.nan,
                p=np.nan, ci_lower=np.nan, ci_upper=np.nan,
                converged=False,
            )
            results.append(row)
            continue

        fit_data = sub[[duration_col, cs_event] + fit_covs]
        fitter = CoxPHFitter(penalizer=penalizer)
        try:
            fitter.fit(
                fit_data,
                duration_col=duration_col,
                event_col=cs_event,
                show_progress=False,
            )
            s = fitter.summary.loc[gene_col]
            row.update(
                coef=s["coef"],
                hr=s["exp(coef)"],
                se=s["se(coef)"],
                z=s["z"],
                p=s["p"],
                ci_lower=s["exp(coef) lower 95%"],
                ci_upper=s["exp(coef) upper 95%"],
                converged=True,
            )
        except Exception as exc:
            row.update(
                coef=np.nan, hr=np.nan, se=np.nan, z=np.nan,
                p=np.nan, ci_lower=np.nan, ci_upper=np.nan,
                converged=False,
            )

        results.append(row)

    return pd.DataFrame(results)
