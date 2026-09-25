"""
Penalized Cox Proportional Hazards Wrapper
==========================================

Provides a thin, opinionated wrapper around lifelines CoxPHFitter with:
  - Automatic detection and removal of constant columns
  - Ridge penalization (default 0.01) for numerical stability
  - Convergence detection and graceful error handling
  - Proportional hazards diagnostics
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from lifelines.utils import concordance_index


def fit_recording_convergence(fitter: Any, data: pd.DataFrame, **fit_kwargs: Any) -> List[str]:
    """Fit a lifelines model and return any ``ConvergenceWarning`` messages.

    ``fitter.fit(data, **fit_kwargs)`` is called with warnings recorded.
    Lifelines ``ConvergenceWarning`` messages (Newton-Raphson failure,
    near-separation, near-singular / low-variance columns) are returned
    instead of being silently printed, so callers can mark the fit as
    not converged.  All other warnings are re-emitted unchanged.
    Exceptions raised by ``fit`` propagate.
    """
    caught: List[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitter.fit(data, **fit_kwargs)
    finally:
        messages: List[str] = []
        for w in caught:
            if issubclass(w.category, ConvergenceWarning):
                messages.append(str(w.message).strip())
            else:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    return messages


class PenalizedCox:
    """Ridge-penalized Cox proportional hazards model.

    Wraps ``lifelines.CoxPHFitter`` with sensible defaults, automatic
    constant-column removal, and convergence safeguards.

    Parameters
    ----------
    penalizer : float, default 0.01
        L2 (ridge) penalty strength.  Increasing this value improves
        numerical stability at the cost of some bias.
    l1_ratio : float, default 0.0
        Elastic-net mixing parameter.  0.0 = pure ridge, 1.0 = pure lasso.

    Examples
    --------
    >>> model = PenalizedCox(penalizer=0.05)
    >>> model.fit(df, duration_col="T", event_col="E",
    ...           covariates=["age", "sex", "gene_mut"])
    >>> print(model.concordance_index_)
    """

    def __init__(self, penalizer: float = 0.01, l1_ratio: float = 0.0) -> None:
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self._fitter: Optional[CoxPHFitter] = None
        self._training_data: Optional[pd.DataFrame] = None
        self._dropped_columns: List[str] = []
        self._converged: bool = False
        self._error_message: Optional[str] = None
        self._convergence_warnings: List[str] = []

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        covariates: Optional[Sequence[str]] = None,
        penalizer: Optional[float] = None,
    ) -> "PenalizedCox":
        """Fit the penalized Cox model.

        Parameters
        ----------
        df : DataFrame
            Must contain *duration_col*, *event_col*, and all *covariates*.
        duration_col : str
            Column with follow-up time (non-negative).
        event_col : str
            Column with event indicator (1 = event, 0 = censored).
        covariates : sequence of str, optional
            Columns to use as predictors.  If ``None``, every column except
            *duration_col* and *event_col* is used.
        penalizer : float, optional
            Override instance-level penalizer for this fit.

        Returns
        -------
        self
            Fitted model instance (for method chaining).

        Raises
        ------
        ValueError
            If the resulting covariate matrix has zero columns after
            constant-column removal.
        """
        pen = penalizer if penalizer is not None else self.penalizer

        # Determine covariates
        if covariates is not None:
            cols = list(covariates)
        else:
            cols = [c for c in df.columns if c not in (duration_col, event_col)]

        # Subset and copy
        keep = [duration_col, event_col] + cols
        data = df[keep].copy()

        # Drop rows with NaN in any required column
        data = data.dropna(subset=keep)

        if len(data) == 0:
            raise ValueError("No rows remaining after dropping NaN values.")

        # Identify and remove constant columns
        self._dropped_columns = []
        for c in cols:
            if data[c].nunique() <= 1:
                self._dropped_columns.append(c)

        remaining_covariates = [c for c in cols if c not in self._dropped_columns]

        if len(remaining_covariates) == 0:
            raise ValueError(
                "All covariates are constant after NaN removal. "
                f"Dropped columns: {self._dropped_columns}"
            )

        if self._dropped_columns:
            warnings.warn(
                f"PenalizedCox: dropped {len(self._dropped_columns)} constant "
                f"column(s): {self._dropped_columns}",
                stacklevel=2,
            )

        fit_cols = [duration_col, event_col] + remaining_covariates
        data = data[fit_cols]

        # Fit
        self._training_data = data.copy()
        self._fitter = CoxPHFitter(penalizer=pen, l1_ratio=self.l1_ratio)
        try:
            self._convergence_warnings = fit_recording_convergence(
                self._fitter,
                data,
                duration_col=duration_col,
                event_col=event_col,
                show_progress=False,
            )
            # A lifelines ConvergenceWarning means the estimates may be
            # unreliable even though no exception was raised.
            self._converged = not self._convergence_warnings
            self._error_message = (
                "; ".join(self._convergence_warnings)
                if self._convergence_warnings else None
            )
        except Exception as exc:
            self._converged = False
            self._error_message = str(exc)
            raise

        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def concordance_index_(self) -> float:
        """Harrell's concordance index on the training data."""
        self._check_fitted()
        return self._fitter.concordance_index_

    @property
    def summary(self) -> pd.DataFrame:
        """Coefficient summary table (coef, HR, p, CI)."""
        self._check_fitted()
        return self._fitter.summary

    @property
    def log_likelihood_(self) -> float:
        """Log partial likelihood of the fitted model."""
        self._check_fitted()
        return self._fitter.log_likelihood_

    @property
    def converged(self) -> bool:
        """Whether the most recent fit converged successfully."""
        return self._converged

    @property
    def convergence_warnings(self) -> List[str]:
        """Lifelines ``ConvergenceWarning`` messages raised during the last fit."""
        return list(self._convergence_warnings)

    @property
    def dropped_columns(self) -> List[str]:
        """Columns removed because they were constant."""
        return list(self._dropped_columns)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_hazard(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict cumulative hazard for new observations.

        Parameters
        ----------
        df : DataFrame
            Must contain the same covariates used during fit (minus any
            that were dropped as constant).

        Returns
        -------
        DataFrame
            Predicted cumulative hazard at each unique event time.
        """
        self._check_fitted()
        return self._fitter.predict_cumulative_hazard(df)

    def predict_partial_hazard(self, df: pd.DataFrame) -> pd.Series:
        """Predict the partial hazard (exp(X @ beta)) for new data.

        Parameters
        ----------
        df : DataFrame
            Must contain the fitted covariates.

        Returns
        -------
        Series
            Partial hazard values.
        """
        self._check_fitted()
        return self._fitter.predict_partial_hazard(df)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def check_proportional_hazards(self) -> Dict[str, Any]:
        """Test the proportional hazards assumption for every covariate.

        Uses the Schoenfeld residual test via
        ``lifelines.statistics.proportional_hazard_test``.

        Returns
        -------
        dict
            Mapping of covariate name to p-value from the PH test.
            A small p-value suggests violation of the PH assumption.
            If the test itself fails, returns ``{"__error__": <message>}``
            (an explicit marker, so a failure is never mistaken for
            "no violations").
        """
        self._check_fitted()
        try:
            from lifelines.statistics import proportional_hazard_test

            # Use stored training data for the PH test
            result = proportional_hazard_test(
                self._fitter,
                self._training_data,
                time_transform="rank",
            )
            summary_df = result.summary
            return dict(
                zip(summary_df.index.get_level_values(0), summary_df["p"])
            )
        except Exception as exc:
            return {"__error__": f"PH test failed: {type(exc).__name__}: {exc}"}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise if the model has not been fitted yet."""
        if self._fitter is None:
            raise RuntimeError(
                "Model has not been fitted. Call .fit() first."
            )
