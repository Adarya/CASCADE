"""
Check: Covariate Leakage in Landmark Models
============================================

Detects whether binary covariates in a landmark survival analysis
were computed using events that occur after the landmark date,
which constitutes future-data leakage.

Pitfall #2 in the CASCADE library.
"""

import re
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..library import PitfallWarning, Severity, PITFALL_LIBRARY

_PITFALL = PITFALL_LIBRARY[1]  # id=2, covariate leakage

# Name fragments typical of covariates aggregated over the whole
# follow-up ("ever received X", "any X", counts / totals).
_LEAKY_NAME_RE = re.compile(
    r"(^|_)(EVER|ANY|TOTAL|CUMULATIVE|CUM|COUNT)(_|$)",
    re.IGNORECASE,
)


def check_landmark_leakage(
    df: pd.DataFrame,
    landmark_date_col: str,
    covariate_cols: List[str],
    event_date_cols: Optional[Union[List[str], Dict[str, str]]] = None,
    followup_col: Optional[str] = None,
    corr_threshold: float = 0.2,
) -> List[PitfallWarning]:
    """Check for future-data leakage in landmark model covariates.

    This check addresses two scenarios:

    1. **Date mode** (``event_date_cols`` provided): each covariate is
       compared only against **its own** exposure-date column.  Pass a
       mapping ``{covariate: exposure_date_col}``; if a plain list is
       given, a covariate is paired with a date column only when one
       name contains the other (e.g. ``EVER_ICI`` / ``EVER_ICI_DATE``).
       A covariate is flagged (CRITICAL) when it is nonzero in rows
       whose exposure date is after the landmark / anchor date.
       Covariates with no mapped date column are not tested (an INFO
       warning lists them) -- e.g. ``AGE`` is never compared with an
       unrelated event date.
    2. **Heuristic mode** (no dates): a covariate is flagged when its
       name matches an "ever / any / total / count" pattern (WARNING),
       and additionally -- if ``followup_col`` is given -- when it is
       positively correlated with follow-up time above
       ``corr_threshold`` (longer follow-up => more opportunity to become
       "ever exposed"); both together are CRITICAL.  The landmark date
       itself is not used, so a constant landmark is handled.

    Parameters
    ----------
    df : pd.DataFrame
        The analysis dataframe. Must contain at least the landmark date
        column and the covariate columns.
    landmark_date_col : str
        Column name containing the landmark / anchor date (numeric, in
        days).  May be constant.
    covariate_cols : list of str
        Column names of covariates to check for leakage.
    event_date_cols : dict or list of str, optional
        Exposure-date columns (see date mode).  If None, heuristic checks
        are used.
    followup_col : str, optional
        Follow-up time column used by the heuristic correlation test.
    corr_threshold : float
        Minimum positive correlation with follow-up time to flag in
        heuristic mode (default 0.2).

    Returns
    -------
    list of PitfallWarning
        One warning per covariate suspected of leakage.
    """
    warnings: List[PitfallWarning] = []

    # Validate inputs
    needed = [landmark_date_col] + covariate_cols
    if followup_col is not None:
        needed.append(followup_col)
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        warnings.append(
            PitfallWarning(
                pitfall=_PITFALL,
                message=f"Missing columns in dataframe: {missing_cols}",
                location=", ".join(missing_cols),
                severity=Severity.WARNING,
                suggestion="Verify column names match the dataframe.",
            )
        )
        return warnings

    landmark_numeric = pd.to_numeric(df[landmark_date_col], errors="coerce")

    # --- Strategy 1: Direct date-based check (covariate -> own date) ---
    if event_date_cols is not None:
        mapping = _resolve_date_mapping(covariate_cols, event_date_cols, df)
        unmapped = [c for c in covariate_cols if c not in mapping]
        for cov_col, date_col in mapping.items():
            event_dates = pd.to_numeric(df[date_col], errors="coerce")
            cov_values = pd.to_numeric(df[cov_col], errors="coerce")
            leaking_rows = (event_dates > landmark_numeric) & (cov_values != 0) & cov_values.notna()
            n_leaking = int(leaking_rows.sum())

            if n_leaking > 0:
                pct = 100.0 * n_leaking / len(df)
                warnings.append(
                    PitfallWarning(
                        pitfall=_PITFALL,
                        message=(
                            f"Covariate '{cov_col}' has nonzero values in "
                            f"{n_leaking} rows ({pct:.1f}%) where its exposure "
                            f"date '{date_col}' is after the landmark date "
                            f"'{landmark_date_col}'. This indicates future "
                            f"data leakage."
                        ),
                        location=cov_col,
                        severity=Severity.CRITICAL,
                        suggestion=(
                            f"Recompute '{cov_col}' using only events with "
                            f"'{date_col}' <= '{landmark_date_col}'. Filter "
                            f"the source timeline to dates on or before the "
                            f"landmark before aggregating."
                        ),
                    )
                )
        if unmapped:
            warnings.append(
                PitfallWarning(
                    pitfall=_PITFALL,
                    message=(
                        f"No exposure-date column mapped for covariate(s) "
                        f"{unmapped}; they were not checked in date mode."
                    ),
                    location=", ".join(unmapped),
                    severity=Severity.INFO,
                    suggestion=(
                        "Pass event_date_cols as {covariate: exposure_date_col} "
                        "for time-dependent covariates."
                    ),
                )
            )
        return warnings

    # --- Strategy 2: Heuristic check for "ever received" variables ---
    followup = (
        pd.to_numeric(df[followup_col], errors="coerce")
        if followup_col is not None else None
    )
    for cov_col in covariate_cols:
        cov_numeric = pd.to_numeric(df[cov_col], errors="coerce")
        name_hit = bool(_LEAKY_NAME_RE.search(str(cov_col)))

        corr = np.nan
        if followup is not None:
            valid = cov_numeric.notna() & followup.notna()
            if (
                valid.sum() > 10
                and cov_numeric[valid].nunique() > 1
                and followup[valid].nunique() > 1
            ):
                corr = float(cov_numeric[valid].corr(followup[valid]))
        corr_hit = bool(np.isfinite(corr) and corr >= corr_threshold)

        if not (name_hit or corr_hit):
            continue

        reasons = []
        if name_hit:
            reasons.append(
                "its name suggests an 'ever / any / total' aggregate over "
                "the full follow-up"
            )
        if corr_hit:
            reasons.append(
                f"it is positively correlated with follow-up time "
                f"'{followup_col}' (r={corr:.3f} >= {corr_threshold})"
            )
        warnings.append(
            PitfallWarning(
                pitfall=_PITFALL,
                message=(
                    f"Covariate '{cov_col}' may leak post-landmark "
                    f"information: " + " and ".join(reasons) + "."
                ),
                location=cov_col,
                severity=Severity.CRITICAL if (name_hit and corr_hit) else Severity.WARNING,
                suggestion=(
                    f"Verify that '{cov_col}' was computed using only "
                    f"events up to the landmark date. If it represents "
                    f"'ever received treatment X', recompute from the "
                    f"treatment timeline filtered to dates <= landmark."
                ),
            )
        )

    return warnings


def _resolve_date_mapping(
    covariate_cols: List[str],
    event_date_cols: Union[List[str], Dict[str, str]],
    df: pd.DataFrame,
) -> Dict[str, str]:
    """Map each covariate to its own exposure-date column."""
    if isinstance(event_date_cols, dict):
        return {
            c: d for c, d in event_date_cols.items()
            if c in covariate_cols and d in df.columns
        }
    mapping: Dict[str, str] = {}
    date_cols = [d for d in event_date_cols if d in df.columns]
    for cov in covariate_cols:
        matches = [
            d for d in date_cols
            if cov.lower() in d.lower() or d.lower() in cov.lower()
        ]
        if len(matches) == 1:
            mapping[cov] = matches[0]
    return mapping
