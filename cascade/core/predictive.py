"""
Layer 3: Predictive vs Prognostic Distinction
==============================================

Tests whether a biomarker's association with outcome differs by treatment
arm (predictive) or is treatment-independent (prognostic), via
gene-treatment interaction modeling.

Classes
-------
PredictiveTest
    Tests a single gene-treatment interaction in a Cox model.
PredictiveResult
    Dataclass holding the result of a predictive test.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from cascade.core.cohort import parse_event_status

try:
    from cascade.stats.interaction import InteractionCox, InteractionResult
except ImportError:
    InteractionCox = None  # type: ignore[assignment,misc]
    InteractionResult = None  # type: ignore[assignment,misc]

try:
    from cascade.stats.multiple_testing import apply_fdr
except ImportError:
    apply_fdr = None  # type: ignore[assignment,misc]


@dataclass
class PredictiveResult:
    """Result of a predictive vs prognostic test.

    Attributes
    ----------
    gene : str
        Biomarker column name.
    treatment : str
        Treatment column name.
    interaction_hr : float
        Hazard ratio for the gene-treatment interaction term.
    interaction_p : float
        P-value for the interaction term.
    interaction_ci : tuple of float
        95% CI for the interaction HR (lower, upper).
    hr_treated : float
        Biomarker HR within the treated subgroup.
    hr_untreated : float
        Biomarker HR within the untreated subgroup.
    classification : str
        ``'predictive'`` if interaction significant, ``'prognostic'`` if the
        interaction was estimated but not significant, or
        ``'not_evaluable'`` if the test could not be performed (see
        *reason*).
    n_treated : int
        Number of treated patients.
    n_untreated : int
        Number of untreated patients.
    converged : bool
        Whether the model converged.
    reason : str
        Why the result is ``'not_evaluable'`` (empty otherwise).
    """

    gene: str
    treatment: str
    interaction_hr: float = np.nan
    interaction_p: float = np.nan
    interaction_ci: tuple = (np.nan, np.nan)
    hr_treated: float = np.nan
    hr_untreated: float = np.nan
    classification: str = "not_evaluable"
    n_treated: int = 0
    n_untreated: int = 0
    converged: bool = False
    reason: str = ""


class PredictiveTest:
    """Test whether a biomarker is predictive (treatment-modifying) or prognostic.

    Fits a Cox model with gene, treatment, gene*treatment interaction,
    and optional covariates.  If the interaction term is significant,
    the biomarker is classified as predictive; otherwise prognostic.

    Parameters
    ----------
    gene_col : str
        Binary (0/1) column for gene mutation status.
    treatment_col : str
        Binary (0/1) column for treatment received.
    duration_col : str, default 'OS_MONTHS'
        Follow-up time column.
    event_col : str, default 'OS_STATUS'
        Event indicator column.
    covariates : sequence of str, optional
        Additional adjustment covariates.
    penalizer : float, default 0.01
        Ridge penalty for numerical stability.
    alpha : float, default 0.05
        Significance threshold for the interaction term.
    min_arm_size : int, default 10
        Minimum number of patients in each treatment arm; smaller arms
        yield ``classification='not_evaluable'``.

    Notes
    -----
    *event_col* may hold 0/1, booleans or cBioPortal status strings such as
    ``'1:DECEASED'`` / ``'0:LIVING'``; they are parsed to 0/1 before fitting.
    """

    def __init__(
        self,
        gene_col: str,
        treatment_col: str,
        duration_col: str = "OS_MONTHS",
        event_col: str = "OS_STATUS",
        covariates: Optional[Sequence[str]] = None,
        penalizer: float = 0.01,
        alpha: float = 0.05,
        min_arm_size: int = 10,
    ) -> None:
        self.gene_col = gene_col
        self.treatment_col = treatment_col
        self.duration_col = duration_col
        self.event_col = event_col
        self.covariates = list(covariates) if covariates else []
        self.penalizer = penalizer
        self.alpha = alpha
        self.min_arm_size = min_arm_size

    # ------------------------------------------------------------------
    # Single test
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> PredictiveResult:
        """Run the predictive vs prognostic test.

        Parameters
        ----------
        df : DataFrame
            Must contain gene_col, treatment_col, duration_col, event_col,
            and all covariates.

        Returns
        -------
        PredictiveResult
        """
        from lifelines import CoxPHFitter

        result = PredictiveResult(
            gene=self.gene_col, treatment=self.treatment_col
        )

        # Prepare data
        required = [self.duration_col, self.event_col, self.gene_col, self.treatment_col]
        all_cols = required + self.covariates
        sub = df[all_cols].copy()
        sub[self.event_col] = sub[self.event_col].map(parse_event_status)
        sub = sub.dropna()

        n_treated = int((sub[self.treatment_col] == 1).sum())
        n_untreated = int((sub[self.treatment_col] == 0).sum())
        result.n_treated = n_treated
        result.n_untreated = n_untreated

        if n_treated < self.min_arm_size or n_untreated < self.min_arm_size:
            result.reason = (
                f"arm size below min_arm_size={self.min_arm_size} "
                f"(treated={n_treated}, untreated={n_untreated})"
            )
            return result

        # Create interaction term
        interaction_col = f"{self.gene_col}_x_{self.treatment_col}"
        sub = sub.copy()
        sub[interaction_col] = sub[self.gene_col] * sub[self.treatment_col]

        # All predictors
        predictors = [self.gene_col, self.treatment_col, interaction_col] + self.covariates

        # Drop constant columns
        fit_predictors = [c for c in predictors if sub[c].nunique() > 1]
        if interaction_col not in fit_predictors:
            # Interaction is constant (no variation)
            result.reason = "no variation in gene x treatment interaction term"
            return result

        fit_data = sub[[self.duration_col, self.event_col] + fit_predictors]

        try:
            fitter = CoxPHFitter(penalizer=self.penalizer)
            fitter.fit(
                fit_data,
                duration_col=self.duration_col,
                event_col=self.event_col,
                show_progress=False,
            )
            result.converged = True

            s = fitter.summary
            if interaction_col in s.index:
                row = s.loc[interaction_col]
                result.interaction_hr = row["exp(coef)"]
                result.interaction_p = row["p"]
                result.interaction_ci = (
                    row["exp(coef) lower 95%"],
                    row["exp(coef) upper 95%"],
                )
        except Exception as exc:
            result.reason = f"model fit failed: {type(exc).__name__}: {exc}"
            return result

        # Classify
        if pd.isna(result.interaction_p):
            result.reason = "interaction term not estimated"
            return result
        result.classification = (
            "predictive" if result.interaction_p < self.alpha else "prognostic"
        )

        # Stratified effects
        result.hr_treated = self._stratified_hr(
            sub.loc[sub[self.treatment_col] == 1]
        )
        result.hr_untreated = self._stratified_hr(
            sub.loc[sub[self.treatment_col] == 0]
        )

        return result

    def _stratified_hr(self, sub: pd.DataFrame) -> float:
        """Fit gene-only Cox in a treatment subgroup and return HR."""
        from lifelines import CoxPHFitter

        covs = [c for c in self.covariates if c != self.treatment_col]
        predictors = [self.gene_col] + covs
        fit_preds = [c for c in predictors if sub[c].nunique() > 1]

        if self.gene_col not in fit_preds or len(sub) < 10:
            return np.nan

        try:
            fitter = CoxPHFitter(penalizer=self.penalizer)
            fitter.fit(
                sub[[self.duration_col, self.event_col] + fit_preds],
                duration_col=self.duration_col,
                event_col=self.event_col,
                show_progress=False,
            )
            return fitter.summary.loc[self.gene_col, "exp(coef)"]
        except Exception:
            return np.nan

    # ------------------------------------------------------------------
    # Batch screening
    # ------------------------------------------------------------------

    @staticmethod
    def screen_predictive(
        df: pd.DataFrame,
        gene_cols: Sequence[str],
        treatment_col: str,
        duration_col: str,
        event_col: str,
        covariates: Optional[Sequence[str]] = None,
        penalizer: float = 0.01,
        alpha: float = 0.05,
        correction: str = "fdr_bh",
        min_arm_size: int = 10,
    ) -> pd.DataFrame:
        """Screen multiple genes for predictive interactions with treatment.

        Parameters
        ----------
        df : DataFrame
        gene_cols : sequence of str
            Binary gene columns to test.
        treatment_col : str
        duration_col : str
        event_col : str
        covariates : sequence of str, optional
        penalizer : float, default 0.01
        alpha : float, default 0.05
        correction : str, default 'fdr_bh'
        min_arm_size : int, default 10
            Minimum patients per treatment arm (see :class:`PredictiveTest`).

        Returns
        -------
        DataFrame
            One row per gene with interaction test results,
            ``classification`` (``'predictive'`` / ``'prognostic'`` /
            ``'not_evaluable'``) and ``reason``.  Genes whose interaction
            could not be tested stay ``'not_evaluable'``; they are never
            relabelled prognostic.
        """
        results: List[Dict[str, Any]] = []

        for gene in gene_cols:
            test = PredictiveTest(
                gene_col=gene,
                treatment_col=treatment_col,
                duration_col=duration_col,
                event_col=event_col,
                covariates=covariates,
                penalizer=penalizer,
                alpha=alpha,
                min_arm_size=min_arm_size,
            )
            res = test.run(df)
            results.append({
                "gene": res.gene,
                "treatment": res.treatment,
                "interaction_hr": res.interaction_hr,
                "interaction_p": res.interaction_p,
                "interaction_ci_lower": res.interaction_ci[0],
                "interaction_ci_upper": res.interaction_ci[1],
                "hr_treated": res.hr_treated,
                "hr_untreated": res.hr_untreated,
                "classification": res.classification,
                "n_treated": res.n_treated,
                "n_untreated": res.n_untreated,
                "converged": res.converged,
                "reason": res.reason,
            })

        results_df = pd.DataFrame(results)

        if results_df.empty:
            return results_df

        # Multiple-testing correction on interaction p-values
        from cascade.core.discovery import _apply_correction_fallback

        if apply_fdr is not None and correction == "fdr_bh":
            adjusted, _ = apply_fdr(
                results_df["interaction_p"].values, alpha=alpha
            )
            results_df["interaction_p_adjusted"] = adjusted
        else:
            results_df["interaction_p_adjusted"] = _apply_correction_fallback(
                results_df["interaction_p"], correction, alpha
            )

        results_df["significant"] = results_df["interaction_p_adjusted"] < alpha

        # Reclassify based on adjusted p-value; untestable genes stay
        # not_evaluable (a NaN p-value is not evidence of prognostic-only).
        evaluable = results_df["interaction_p_adjusted"].notna() & (
            results_df["classification"] != "not_evaluable"
        )
        results_df["classification"] = np.select(
            [evaluable & results_df["significant"], evaluable],
            ["predictive", "prognostic"],
            default="not_evaluable",
        )
        results_df.loc[
            (results_df["classification"] == "not_evaluable")
            & (results_df["reason"] == ""),
            "reason",
        ] = "interaction p-value not available"

        return results_df
