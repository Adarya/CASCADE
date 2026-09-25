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
        Event indicator (1 = event, 0 = censored).  cBioPortal-style
        strings such as ``'1:DECEASED'`` / ``'0:LIVING'`` and booleans are
        parsed to 0/1.
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
        One warning per biomarker with biomarker-dependent censoring,
        plus an INFO warning for each biomarker whose censoring model
        could not be fitted (so a failure is not read as "no problem").
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

    try:
        event = parse_event_indicator(df[event_col])
    except ValueError as exc:
        warnings.append(
            PitfallWarning(
                pitfall=_PITFALL,
                message=f"Could not parse event column '{event_col}': {exc}",
                location=event_col,
                severity=Severity.WARNING,
                suggestion="Encode the event column as 0/1 (or 'N:LABEL').",
            )
        )
        return warnings

    n_tests = max(len(biomarker_cols), 1)
    for biomarker in biomarker_cols:
        sub = df[[duration_col, biomarker] + covariates].assign(_event=event)
        sub = sub.dropna().copy()
        sub["_censored"] = 1 - sub["_event"].astype(int)
        if sub["_censored"].sum() < min_censored or sub[biomarker].nunique() < 2:
            continue

        try:
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(
                sub[[duration_col, "_censored", biomarker] + covariates],
                duration_col=duration_col,
                event_col="_censored",
            )
        except Exception as exc:
            warnings.append(
                PitfallWarning(
                    pitfall=_PITFALL,
                    message=(
                        f"Censoring model for '{biomarker}' could not be "
                        f"fitted ({type(exc).__name__}: {exc}); informative "
                        f"censoring was NOT assessed for this biomarker."
                    ),
                    location=biomarker,
                    severity=Severity.INFO,
                    suggestion="Inspect the biomarker / covariates for separation or collinearity.",
                )
            )
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


def parse_event_indicator(series: pd.Series) -> pd.Series:
    """Parse an event column to float 0/1 (NaN preserved).

    Accepts numeric / boolean values and cBioPortal-style strings such as
    ``'1:DECEASED'`` or ``'0:LIVING'`` (the integer before ``':'`` is used).

    Raises
    ------
    ValueError
        If non-missing values cannot be parsed or are not 0/1.
    """
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        parsed = pd.to_numeric(series, errors="coerce").astype(float)
    else:
        text = series.astype("string").str.strip().str.split(":").str[0]
        parsed = pd.to_numeric(text, errors="coerce").astype(float)
        bad = series.notna() & parsed.isna()
        if bad.any():
            raise ValueError(
                f"unparseable event values, e.g. {series[bad].iloc[0]!r}"
            )
    vals = set(parsed.dropna().unique())
    if not vals.issubset({0.0, 1.0}):
        raise ValueError(f"event values must be 0/1, got {sorted(vals)[:5]}")
    return parsed
