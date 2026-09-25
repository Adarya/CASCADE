"""
Check: Informative (Biomarker-Dependent) Censoring
===================================================

Detects whether the censoring process depends on the biomarker under
study. Cox models assume non-informative censoring; when carriers and
non-carriers are censored at different rates (differential follow-up,
later panel adoption, referral patterns), hazard ratios can be biased.

Pitfall #10 in the CASCADE library.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from ..library import PitfallWarning, Severity, PITFALL_LIBRARY

_PITFALL = PITFALL_LIBRARY[9]  # id=10, informative censoring


def check_informative_censoring(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    biomarker_cols: List[str],
    covariates: Optional[List[str]] = None,
    alpha: float = 0.05,
    min_hr: float = 1.25,
    min_censored: int = 20,
    penalizer: float = 0.01,
) -> List[PitfallWarning]:
    """Test whether each biomarker predicts the hazard of being censored.

    A reverse-censoring Cox model is fitted per biomarker: censoring is
    treated as the event (``1 - event``) and the biomarker, plus any
    covariates, as predictors. A biomarker is flagged when the censoring
    hazard ratio is both statistically significant (Bonferroni-adjusted
    across biomarkers) and at least ``min_hr``-fold in either direction.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis dataframe, one row per patient.
    duration_col : str
        Follow-up time column.
    event_col : str
        Event indicator (1 = event, 0 = censored).
    biomarker_cols : list of str
        Binary biomarker columns to test.
    covariates : list of str, optional
        Adjustment covariates for the censoring model.
    alpha : float
        Family-wise significance level (default 0.05).
    min_hr : float
        Minimum censoring hazard ratio (or its reciprocal) to flag
        (default 1.25).
    min_censored : int
        Minimum number of censored patients required to run the check
        (default 20).
    penalizer : float
        L2 penalizer for the Cox model (default 0.01).

    Returns
    -------
    list of PitfallWarning
        One warning per biomarker with biomarker-dependent censoring.
    """
    from lifelines import CoxPHFitter

    warnings: List[PitfallWarning] = []
    covariates = list(covariates or [])

    required = [duration_col, event_col] + list(biomarker_cols) + covariates
    missing = [c for c in required if c not in df.columns]
    if missing:
        warnings.append(
            PitfallWarning(
                pitfall=_PITFALL,
                message=f"Missing columns for censoring check: {missing}",
                location=", ".join(missing),
                severity=Severity.WARNING,
                suggestion="Verify column names for duration, event and biomarkers.",
            )
        )
        return warnings

    n_tests = max(len(biomarker_cols), 1)
    for biomarker in biomarker_cols:
        sub = df[[duration_col, event_col, biomarker] + covariates].dropna().copy()
        sub["_censored"] = 1 - sub[event_col].astype(int)
        if sub["_censored"].sum() < min_censored or sub[biomarker].nunique() < 2:
            continue

        try:
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(
                sub[[duration_col, "_censored", biomarker] + covariates],
                duration_col=duration_col,
                event_col="_censored",
            )
        except Exception:
            continue

        hr = float(np.exp(cph.params_[biomarker]))
        p = float(cph.summary.loc[biomarker, "p"])
        if p < alpha / n_tests and max(hr, 1 / hr) >= min_hr:
            direction = "earlier" if hr > 1 else "later"
            warnings.append(
                PitfallWarning(
                    pitfall=_PITFALL,
                    message=(
                        f"Censoring depends on '{biomarker}': censoring HR = "
                        f"{hr:.2f} (P = {p:.2e}); carriers are censored "
                        f"{direction} than non-carriers."
                    ),
                    location=biomarker,
                    severity=Severity.WARNING,
                    suggestion=(
                        "Compare follow-up distributions by biomarker status, "
                        "adjust for the driver of differential follow-up "
                        "(e.g. sequencing date, institution), or use inverse "
                        "probability of censoring weights as a sensitivity "
                        "analysis."
                    ),
                )
            )

    return warnings
