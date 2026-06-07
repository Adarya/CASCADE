"""
Layer 2: Orthogonal Confirmation
=================================

Validates discovery-screen hits using an independent statistical method.
For example, cause-specific Cox results (temporal) can be confirmed with
cross-sectional logistic regression.

Classes
-------
OrthogonalConfirm
    Runs a confirmatory analysis on primary screen results and quantifies
    cross-method concordance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from cascade.core.discovery import BiomarkerScreen


class OrthogonalConfirm:
    """Confirm discovery-screen hits with an orthogonal statistical method.

    The principle is that genuine biological signals should be detectable
    by more than one analytical approach.  This class pairs a primary
    analysis (e.g. cause-specific Cox) with a confirmatory analysis
    (e.g. logistic regression) and measures concordance.

    Parameters
    ----------
    primary_method : str, default 'competing_risks'
        Method used in the primary screen.
    confirm_method : str, default 'logistic'
        Method used for confirmation.
    alpha : float, default 0.05
        Significance threshold for both primary and confirmatory results.
    """

    def __init__(
        self,
        primary_method: str = "competing_risks",
        confirm_method: str = "logistic",
        alpha: float = 0.05,
    ) -> None:
        self.primary_method = primary_method
        self.confirm_method = confirm_method
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def confirm(
        self,
        primary_results: pd.DataFrame,
        df: pd.DataFrame,
        biomarker_cols: Sequence[str],
        outcome_col: str,
        event_col: Optional[str] = None,
        covariates: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Run confirmatory analysis and merge with primary results.

        Parameters
        ----------
        primary_results : DataFrame
            Results from the primary screen.  Must contain a ``biomarker``
            column and an effect-size column (``hr`` or ``or``).
        df : DataFrame
            Analysis-ready dataset for the confirmatory analysis.
        biomarker_cols : sequence of str
            Biomarker columns to test.
        outcome_col : str
            Outcome variable for the confirmatory method.
        event_col : str, optional
            Event indicator (needed for survival confirmatory methods).
        covariates : sequence of str, optional
            Adjustment covariates.

        Returns
        -------
        DataFrame
            Merged table with columns from both primary and confirmatory
            analyses, plus ``concordance_direction`` and
            ``confidence_tier``.
        """
        screen = BiomarkerScreen(
            method=self.confirm_method,
            alpha=self.alpha,
        )

        if event_col is not None:
            confirm_results = screen.screen(
                df, biomarker_cols, outcome_col, event_col, covariates
            )
        else:
            # Logistic: outcome_col is the binary outcome, no event_col
            confirm_results = screen.screen(
                df, biomarker_cols, outcome_col, outcome_col, covariates
            )

        # Rename confirmatory columns to avoid collision
        confirm_rename = {}
        for col in confirm_results.columns:
            if col != "biomarker":
                confirm_rename[col] = f"confirm_{col}"
        confirm_results = confirm_results.rename(columns=confirm_rename)

        # Merge on biomarker
        merged = primary_results.merge(
            confirm_results, on="biomarker", how="left"
        )

        # Direction concordance
        primary_effect = "hr" if "hr" in primary_results.columns else "or"
        confirm_effect = f"confirm_hr" if "confirm_hr" in merged.columns else "confirm_or"

        if primary_effect in merged.columns and confirm_effect in merged.columns:
            p_dir = np.sign(np.log(merged[primary_effect].replace(0, np.nan)))
            c_dir = np.sign(np.log(merged[confirm_effect].replace(0, np.nan)))
            merged["concordance_direction"] = (p_dir == c_dir).astype(int)
        else:
            merged["concordance_direction"] = np.nan

        # Classify confidence
        merged = self.classify_confidence(merged)

        return merged

    # ------------------------------------------------------------------
    # Concordance metrics
    # ------------------------------------------------------------------

    @staticmethod
    def concordance_metrics(
        primary_results: pd.DataFrame,
        confirm_results: pd.DataFrame,
        effect_col_primary: str = "hr",
        effect_col_confirm: str = "or",
        merge_on: str = "biomarker",
    ) -> Dict[str, Any]:
        """Compute cross-method concordance metrics.

        Parameters
        ----------
        primary_results : DataFrame
            Must contain *merge_on* and *effect_col_primary*.
        confirm_results : DataFrame
            Must contain *merge_on* and *effect_col_confirm*.
        effect_col_primary : str, default 'hr'
            Effect-size column in primary results.
        effect_col_confirm : str, default 'or'
            Effect-size column in confirmatory results.
        merge_on : str, default 'biomarker'
            Column to merge the two result sets on.

        Returns
        -------
        dict
            Keys: ``spearman_rho``, ``spearman_p``,
            ``direction_agreement_pct``, ``n_both_significant``.
        """
        merged = primary_results.merge(
            confirm_results, on=merge_on, how="inner", suffixes=("_primary", "_confirm")
        )

        # Resolve column names after merge
        pcol = effect_col_primary if effect_col_primary in merged.columns else f"{effect_col_primary}_primary"
        ccol = effect_col_confirm if effect_col_confirm in merged.columns else f"{effect_col_confirm}_confirm"

        valid = merged.dropna(subset=[pcol, ccol])

        if len(valid) < 3:
            return dict(
                spearman_rho=np.nan,
                spearman_p=np.nan,
                direction_agreement_pct=np.nan,
                n_both_significant=0,
            )

        # Log-transform for Spearman on effect sizes
        log_p = np.log(valid[pcol].replace(0, np.nan))
        log_c = np.log(valid[ccol].replace(0, np.nan))
        both_valid = log_p.notna() & log_c.notna()

        if both_valid.sum() < 3:
            rho, rho_p = np.nan, np.nan
        else:
            rho, rho_p = sp_stats.spearmanr(
                log_p[both_valid], log_c[both_valid]
            )

        # Direction agreement
        dir_p = np.sign(log_p)
        dir_c = np.sign(log_c)
        agree = (dir_p == dir_c) & both_valid
        direction_pct = float(agree.sum() / both_valid.sum() * 100) if both_valid.sum() > 0 else np.nan

        # Both significant
        sig_p_col = "significant" if "significant" in merged.columns else "significant_primary"
        sig_c_col = "confirm_significant" if "confirm_significant" in merged.columns else "significant_confirm"

        n_both = 0
        if sig_p_col in merged.columns and sig_c_col in merged.columns:
            n_both = int((merged[sig_p_col] & merged[sig_c_col]).sum())

        return dict(
            spearman_rho=rho,
            spearman_p=rho_p,
            direction_agreement_pct=direction_pct,
            n_both_significant=n_both,
        )

    # ------------------------------------------------------------------
    # Confidence classification
    # ------------------------------------------------------------------

    @staticmethod
    def classify_confidence(merged_results: pd.DataFrame) -> pd.DataFrame:
        """Assign a confidence tier based on primary and confirmatory significance.

        Parameters
        ----------
        merged_results : DataFrame
            Must contain ``significant`` (primary) and
            ``confirm_significant`` (confirmatory) boolean columns.

        Returns
        -------
        DataFrame
            Input with an added ``confidence_tier`` column.
        """
        df = merged_results.copy()

        sig_primary = df.get("significant", pd.Series(False, index=df.index))
        sig_confirm = df.get("confirm_significant", pd.Series(False, index=df.index))

        # Fill NaN as False
        sig_primary = sig_primary.fillna(False).astype(bool)
        sig_confirm = sig_confirm.fillna(False).astype(bool)

        conditions = [
            sig_primary & sig_confirm,
            sig_primary & ~sig_confirm,
            ~sig_primary & sig_confirm,
            ~sig_primary & ~sig_confirm,
        ]
        labels = ["both_significant", "primary_only", "confirm_only", "neither"]

        df["confidence_tier"] = np.select(conditions, labels, default="neither")

        return df
