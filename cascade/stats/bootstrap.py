"""
Bootstrap Utilities
====================

Non-parametric bootstrap for arbitrary statistics and a specialised
routine for bootstrapping the concordance index difference (delta-C)
between two nested Cox models.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index


class BootstrapCI:
    """Non-parametric bootstrap confidence interval estimator.

    Parameters
    ----------
    n_bootstrap : int, default 1000
        Number of bootstrap resamples.
    ci_level : float, default 0.95
        Confidence level (e.g. 0.95 for 95% CI).
    seed : int, default 42
        Random seed for reproducibility.

    Attributes
    ----------
    n_failed_ : int
        Number of resamples in the last :meth:`run` whose statistic
        raised or returned NaN (excluded from the interval; a
        ``RuntimeWarning`` is emitted when this is non-zero).
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
        seed: int = 42,
    ) -> None:
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        self.seed = seed
        self.n_failed_ = 0

    def run(
        self,
        statistic_fn: Callable[..., float],
        data: Any,
        **kwargs: Any,
    ) -> Tuple[float, float, float]:
        """Run the bootstrap and return the point estimate with CI.

        Parameters
        ----------
        statistic_fn : callable
            A function that takes ``data`` (a DataFrame or array) as
            its first argument, plus any ``**kwargs``, and returns a
            scalar float.
        data : DataFrame or array-like
            The dataset to resample.  Resampling is done **with
            replacement** along axis 0 (rows).
        **kwargs
            Additional keyword arguments passed to *statistic_fn*.

        Returns
        -------
        point_estimate : float
            The statistic computed on the original (full) data.
        ci_lower : float
            Lower confidence bound.
        ci_upper : float
            Upper confidence bound.
        """
        rng = np.random.RandomState(self.seed)
        point = statistic_fn(data, **kwargs)

        if isinstance(data, pd.DataFrame):
            n = len(data)
            boot_values = np.full(self.n_bootstrap, np.nan)
            for i in range(self.n_bootstrap):
                idx = rng.choice(n, size=n, replace=True)
                sample = data.iloc[idx].reset_index(drop=True)
                try:
                    boot_values[i] = statistic_fn(sample, **kwargs)
                except Exception:
                    boot_values[i] = np.nan
        else:
            arr = np.asarray(data)
            n = len(arr)
            boot_values = np.full(self.n_bootstrap, np.nan)
            for i in range(self.n_bootstrap):
                idx = rng.choice(n, size=n, replace=True)
                sample = arr[idx]
                try:
                    boot_values[i] = statistic_fn(sample, **kwargs)
                except Exception:
                    boot_values[i] = np.nan

        # Remove NaN values (failed resamples), reporting how many
        valid = boot_values[~np.isnan(boot_values)]
        self.n_failed_ = int(self.n_bootstrap - len(valid))
        _warn_failed(self.n_failed_, self.n_bootstrap, "BootstrapCI")
        if len(valid) == 0:
            return point, np.nan, np.nan

        alpha = 1.0 - self.ci_level
        ci_lower = float(np.percentile(valid, 100 * alpha / 2))
        ci_upper = float(np.percentile(valid, 100 * (1 - alpha / 2)))

        return float(point), ci_lower, ci_upper


def bootstrap_delta_c(
    df: pd.DataFrame,
    model_vars_base: Sequence[str],
    model_vars_full: Sequence[str],
    duration_col: str,
    event_col: str,
    n_bootstrap: int = 1000,
    penalizer: float = 0.01,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """Descriptive bootstrap interval for the apparent concordance difference.

    .. warning::
       Each resample fits **and** scores both models on the same data
       (training / apparent C-index), so the estimate and its interval
       are optimistic: adding pure-noise covariates yields a positive
       delta-C.  The interval is therefore **descriptive only** and no
       p-value is computed (the fourth return value is always NaN and
       is kept for backward compatibility).  Use
       :func:`cascade.stats.cross_validate.cv_delta_c` (out-of-sample
       concordance) for inference on added discriminative value.

    Parameters
    ----------
    df : DataFrame
        Analysis-ready dataset.
    model_vars_base : sequence of str
        Covariates for the base (restricted) model.
    model_vars_full : sequence of str
        Covariates for the full model (superset of base).
    duration_col : str
        Follow-up time column.
    event_col : str
        Event indicator column.
    n_bootstrap : int, default 1000
        Number of bootstrap resamples.
    penalizer : float, default 0.01
        Ridge penalty.
    seed : int, default 42
        Random seed.

    Returns
    -------
    delta_c : float
        Apparent (in-sample) C_full - C_base on the original data.
    ci_lower : float
        Lower 2.5th percentile of the bootstrap apparent delta-C
        (descriptive, not optimism-corrected).
    ci_upper : float
        Upper 97.5th percentile of the bootstrap apparent delta-C.
    p_value : float
        Always NaN.  An in-sample bootstrap p-value is biased toward
        significance; use ``cv_delta_c`` instead.

    Failed resamples (fit errors, too few rows) are excluded and their
    number is reported via a ``RuntimeWarning``.
    """
    base_vars = list(model_vars_base)
    full_vars = list(model_vars_full)

    def _fit_c(data: pd.DataFrame, covariates: list) -> float:
        """Fit Cox on data and return training C-index."""
        # Drop constant columns
        fit_covs = [c for c in covariates if data[c].nunique() > 1]
        if len(fit_covs) == 0:
            return 0.5
        fit_data = data[[duration_col, event_col] + fit_covs].dropna()
        if len(fit_data) < 10:
            return np.nan
        fitter = CoxPHFitter(penalizer=penalizer)
        fitter.fit(fit_data, duration_col=duration_col, event_col=event_col,
                    show_progress=False)
        return fitter.concordance_index_

    def _delta_c(data: pd.DataFrame) -> float:
        c_base = _fit_c(data, base_vars)
        c_full = _fit_c(data, full_vars)
        if np.isnan(c_base) or np.isnan(c_full):
            return np.nan
        return c_full - c_base

    # Point estimate
    all_needed = list(set([duration_col, event_col] + base_vars + full_vars))
    clean = df[all_needed].dropna()

    delta_c = _delta_c(clean)

    # Bootstrap
    rng = np.random.RandomState(seed)
    n = len(clean)
    boot_deltas = np.full(n_bootstrap, np.nan)

    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        sample = clean.iloc[idx].reset_index(drop=True)
        try:
            boot_deltas[i] = _delta_c(sample)
        except Exception:
            boot_deltas[i] = np.nan

    valid = boot_deltas[~np.isnan(boot_deltas)]
    _warn_failed(n_bootstrap - len(valid), n_bootstrap, "bootstrap_delta_c")
    if len(valid) == 0:
        return delta_c, np.nan, np.nan, np.nan

    ci_lower = float(np.percentile(valid, 2.5))
    ci_upper = float(np.percentile(valid, 97.5))

    # No p-value: the apparent-C bootstrap is optimistic (see docstring).
    return float(delta_c), ci_lower, ci_upper, np.nan


def _warn_failed(n_failed: int, n_total: int, where: str) -> None:
    """Emit a RuntimeWarning reporting failed bootstrap resamples."""
    if n_failed > 0:
        warnings.warn(
            f"{where}: {n_failed}/{n_total} bootstrap resamples failed and "
            f"were excluded from the interval.",
            RuntimeWarning,
            stacklevel=3,
        )
