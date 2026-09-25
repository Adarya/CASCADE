"""
Check: Centre / Batch Effects in Multi-Institutional Cohorts
=============================================================

Detects two ways in which the contributing centre (institution,
sequencing panel, batch) can distort a biomarker association:

1. biomarker prevalence differs across centres (panel coverage,
   referral patterns), so centre confounds the association;
2. the biomarker effect itself differs across centres
   (biomarker-by-centre interaction).

Pitfall #11 in the CASCADE library.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..library import PitfallWarning, Severity, PITFALL_LIBRARY

_PITFALL = PITFALL_LIBRARY[10]  # id=11, centre / batch effect


def check_center_effect(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    biomarker_cols: List[str],
    center_col: str,
    covariates: Optional[List[str]] = None,
    alpha: float = 0.05,
    prevalence_range: float = 0.15,
    min_per_center: int = 20,
    penalizer: float = 0.01,
) -> List[PitfallWarning]:
    """Check biomarker prevalence and effect heterogeneity across centres.

    For each biomarker, (a) a chi-square test compares prevalence across
    centres and the absolute prevalence range is computed; (b) a
    likelihood-ratio test compares a centre-stratified Cox model with
    and without biomarker-by-centre interaction terms. P-values are
    Bonferroni-adjusted across biomarkers.

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
    center_col : str
        Column identifying the contributing centre or batch.
    covariates : list of str, optional
        Adjustment covariates for the Cox models.
    alpha : float
        Family-wise significance level (default 0.05).
    prevalence_range : float
        Absolute prevalence range across centres required, together
        with a significant chi-square test, to flag prevalence
        heterogeneity (default 0.15).
    min_per_center : int
        Centres with fewer patients are excluded (default 20).
    penalizer : float
        L2 penalizer for the Cox models (default 0.01).

    Returns
    -------
    list of PitfallWarning
        Warnings for prevalence heterogeneity and effect heterogeneity.
    """
    from lifelines import CoxPHFitter

    warnings: List[PitfallWarning] = []
    covariates = list(covariates or [])

    required = [duration_col, event_col, center_col] + list(biomarker_cols) + covariates
    missing = [c for c in required if c not in df.columns]
    if missing:
        warnings.append(
            PitfallWarning(
                pitfall=_PITFALL,
                message=f"Missing columns for centre-effect check: {missing}",
                location=", ".join(missing),
                severity=Severity.WARNING,
                suggestion="Verify column names for centre, duration, event and biomarkers.",
            )
        )
        return warnings

    counts = df[center_col].value_counts()
    centers = list(counts[counts >= min_per_center].index)
    if len(centers) < 2:
        return warnings
    data = df[df[center_col].isin(centers)]
    n_tests = max(len(biomarker_cols), 1)

    for biomarker in biomarker_cols:
        sub = data[[duration_col, event_col, center_col, biomarker] + covariates].dropna()
        if sub[biomarker].nunique() < 2 or sub[center_col].nunique() < 2:
            continue

        # (a) prevalence heterogeneity
        table = pd.crosstab(sub[center_col], sub[biomarker])
        _, p_prev, _, _ = stats.chi2_contingency(table)
        prev = sub.groupby(center_col)[biomarker].mean()
        spread = float(prev.max() - prev.min())
        if p_prev < alpha / n_tests and spread >= prevalence_range:
            warnings.append(
                PitfallWarning(
                    pitfall=_PITFALL,
                    message=(
                        f"Prevalence of '{biomarker}' differs across centres "
                        f"(range {prev.min():.1%}-{prev.max():.1%}, "
                        f"chi-square P = {p_prev:.2e})."
                    ),
                    location=f"{biomarker} x {center_col}",
                    severity=Severity.WARNING,
                    suggestion=(
                        "Stratify or adjust the model by centre and check "
                        "panel coverage; report a leave-one-centre-out "
                        "sensitivity analysis."
                    ),
                )
            )

        # (b) effect heterogeneity: LRT for biomarker-by-centre interaction
        inter = sub.copy()
        inter_cols = []
        for c in sorted(inter[center_col].unique())[1:]:
            col = f"_{biomarker}_x_{c}"
            inter[col] = inter[biomarker] * (inter[center_col] == c).astype(int)
            if inter[col].sum() > 0:
                inter_cols.append(col)
        if not inter_cols:
            continue
        try:
            base = CoxPHFitter(penalizer=penalizer).fit(
                inter[[duration_col, event_col, center_col, biomarker] + covariates],
                duration_col=duration_col, event_col=event_col, strata=[center_col],
            )
            full = CoxPHFitter(penalizer=penalizer).fit(
                inter[[duration_col, event_col, center_col, biomarker] + covariates + inter_cols],
                duration_col=duration_col, event_col=event_col, strata=[center_col],
            )
        except Exception:
            continue
        lr = 2 * (full.log_likelihood_ - base.log_likelihood_)
        p_int = float(stats.chi2.sf(max(lr, 0.0), df=len(inter_cols)))
        if p_int < alpha / n_tests:
            warnings.append(
                PitfallWarning(
                    pitfall=_PITFALL,
                    message=(
                        f"Effect of '{biomarker}' differs across centres "
                        f"(biomarker-by-centre interaction LRT P = {p_int:.2e})."
                    ),
                    location=f"{biomarker} x {center_col}",
                    severity=Severity.WARNING,
                    suggestion=(
                        "Report centre-specific estimates and a "
                        "centre-stratified pooled estimate; do not pool "
                        "without acknowledging heterogeneity."
                    ),
                )
            )

    return warnings
