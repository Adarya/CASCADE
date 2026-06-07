"""
Cross-Validated Concordance
============================

Stratified K-fold cross-validation for Cox model concordance indices,
including delta-C (improvement over a baseline model).

Uses ``sklearn.model_selection.StratifiedKFold`` to ensure each fold
has a proportional number of events and censored observations.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold


def cv_concordance(
    df: pd.DataFrame,
    covariates: Sequence[str],
    duration_col: str,
    event_col: str,
    n_folds: int = 10,
    penalizer: float = 0.01,
    seed: int = 42,
) -> Tuple[float, float, List[float]]:
    """Cross-validated concordance index for a Cox model.

    Parameters
    ----------
    df : DataFrame
        Must contain *duration_col*, *event_col*, and all *covariates*.
    covariates : sequence of str
        Predictor columns.
    duration_col : str
        Follow-up time column.
    event_col : str
        Event indicator column (1 = event, 0 = censored).
    n_folds : int, default 10
        Number of stratified cross-validation folds.
    penalizer : float, default 0.01
        Ridge penalty.
    seed : int, default 42
        Random seed for fold assignment.

    Returns
    -------
    mean_c : float
        Mean concordance index across folds.
    std_c : float
        Standard deviation of concordance across folds.
    fold_cs : list of float
        Per-fold concordance values.
    """
    cov_list = list(covariates)
    all_cols = [duration_col, event_col] + cov_list
    data = df[all_cols].dropna().reset_index(drop=True)

    if len(data) < n_folds * 2:
        raise ValueError(
            f"Not enough data ({len(data)} rows) for {n_folds}-fold CV."
        )

    events = data[event_col].astype(int).values
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_cs: List[float] = []

    for train_idx, test_idx in skf.split(data, events):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]

        # Drop constant covariates in the training set
        fit_covs = [c for c in cov_list if train[c].nunique() > 1]
        if len(fit_covs) == 0:
            fold_cs.append(0.5)
            continue

        fit_data = train[[duration_col, event_col] + fit_covs]
        fitter = CoxPHFitter(penalizer=penalizer)

        try:
            fitter.fit(
                fit_data,
                duration_col=duration_col,
                event_col=event_col,
                show_progress=False,
            )
        except Exception:
            fold_cs.append(np.nan)
            continue

        # Predict on test set
        try:
            partial_hazard = fitter.predict_partial_hazard(test[fit_covs])
            c = concordance_index(
                test[duration_col],
                -partial_hazard.values.ravel(),  # negate: higher hazard = shorter time
                test[event_col],
            )
            fold_cs.append(float(c))
        except Exception:
            fold_cs.append(np.nan)

    valid = [c for c in fold_cs if not np.isnan(c)]
    if len(valid) == 0:
        return np.nan, np.nan, fold_cs

    mean_c = float(np.mean(valid))
    std_c = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0

    return mean_c, std_c, fold_cs


def cv_delta_c(
    df: pd.DataFrame,
    base_vars: Sequence[str],
    full_vars: Sequence[str],
    duration_col: str,
    event_col: str,
    n_folds: int = 10,
    penalizer: float = 0.01,
    seed: int = 42,
) -> Tuple[float, float, List[float]]:
    """Cross-validated difference in concordance between two Cox models.

    For each fold, both a base model (with *base_vars*) and a full model
    (with *full_vars*) are fitted on the training split and evaluated on
    the test split.  The delta-C is ``C_full - C_base``.

    Parameters
    ----------
    df : DataFrame
    base_vars : sequence of str
        Covariates for the restricted (base) model.
    full_vars : sequence of str
        Covariates for the full model.
    duration_col, event_col : str
    n_folds : int, default 10
    penalizer : float, default 0.01
    seed : int, default 42

    Returns
    -------
    delta_c_mean : float
        Mean delta-C across folds.
    delta_c_std : float
        Standard deviation across folds.
    fold_deltas : list of float
        Per-fold delta-C values.
    """
    base_list = list(base_vars)
    full_list = list(full_vars)
    all_cols = list(set([duration_col, event_col] + base_list + full_list))
    data = df[all_cols].dropna().reset_index(drop=True)

    if len(data) < n_folds * 2:
        raise ValueError(
            f"Not enough data ({len(data)} rows) for {n_folds}-fold CV."
        )

    events = data[event_col].astype(int).values
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_deltas: List[float] = []

    for train_idx, test_idx in skf.split(data, events):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]

        c_base = _fit_and_score(train, test, base_list, duration_col, event_col, penalizer)
        c_full = _fit_and_score(train, test, full_list, duration_col, event_col, penalizer)

        if np.isnan(c_base) or np.isnan(c_full):
            fold_deltas.append(np.nan)
        else:
            fold_deltas.append(c_full - c_base)

    valid = [d for d in fold_deltas if not np.isnan(d)]
    if len(valid) == 0:
        return np.nan, np.nan, fold_deltas

    mean_d = float(np.mean(valid))
    std_d = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0

    return mean_d, std_d, fold_deltas


# ------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------

def _fit_and_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    covariates: List[str],
    duration_col: str,
    event_col: str,
    penalizer: float,
) -> float:
    """Fit on train, evaluate C-index on test."""
    fit_covs = [c for c in covariates if train[c].nunique() > 1]
    if len(fit_covs) == 0:
        return 0.5

    fit_data = train[[duration_col, event_col] + fit_covs]
    fitter = CoxPHFitter(penalizer=penalizer)

    try:
        fitter.fit(
            fit_data,
            duration_col=duration_col,
            event_col=event_col,
            show_progress=False,
        )
    except Exception:
        return np.nan

    try:
        ph = fitter.predict_partial_hazard(test[fit_covs])
        c = concordance_index(
            test[duration_col],
            -ph.values.ravel(),
            test[event_col],
        )
        return float(c)
    except Exception:
        return np.nan
