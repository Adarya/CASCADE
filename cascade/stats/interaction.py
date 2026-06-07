"""
Interaction Models for Survival Analysis
=========================================

Two-way and three-way interaction Cox models for testing whether
the effect of a genomic biomarker on survival is modified by treatment,
clinical phenotype, or their combination.

The core output is ``InteractionResult``, a structured dataclass that
captures coefficients, hazard ratios, convergence status, and model
fit statistics.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


# ======================================================================
# Result container
# ======================================================================

@dataclass
class InteractionResult:
    """Structured result from a two-way or three-way interaction Cox model.

    Attributes
    ----------
    gene : str
        Name of the genomic factor column.
    treatment : str
        Name of the treatment factor column.
    phenotype : str
        Name of the phenotype factor column (empty string for two-way).
    converged : bool
        Whether the model converged.
    n_patients : int
        Number of patients in the analysis.
    n_events : int
        Number of events observed.
    main_effects : dict
        ``{covariate: {"coef", "hr", "p", "ci_lower", "ci_upper"}}``
        for each main-effect factor.
    two_way_effects : dict
        ``{interaction_term: {"coef", "hr", "p", "ci_lower", "ci_upper"}}``
        for each two-way interaction.
    three_way_effect : dict
        ``{"coef", "hr", "p", "ci_lower", "ci_upper"}`` for the
        three-way interaction term.  Empty dict for two-way models.
    concordance : float
        Harrell's C-index on the training data.
    log_likelihood : float
        Log partial likelihood.
    aic : float
        Akaike information criterion.
    error_message : str
        Error message if the model did not converge; empty string otherwise.
    """

    gene: str = ""
    treatment: str = ""
    phenotype: str = ""
    converged: bool = False
    n_patients: int = 0
    n_events: int = 0
    main_effects: Dict[str, Dict[str, float]] = field(default_factory=dict)
    two_way_effects: Dict[str, Dict[str, float]] = field(default_factory=dict)
    three_way_effect: Dict[str, float] = field(default_factory=dict)
    concordance: float = np.nan
    log_likelihood: float = np.nan
    aic: float = np.nan
    error_message: str = ""


# ======================================================================
# Interaction Cox fitter
# ======================================================================

class InteractionCox:
    """Fit Cox models with two-way or three-way interaction terms.

    Parameters
    ----------
    penalizer : float, default 0.01
        Ridge penalty for CoxPHFitter.
    min_subgroup_n : int, default 20
        Minimum patients per interaction subgroup cell.
    min_subgroup_events : int, default 5
        Minimum events per interaction subgroup cell.
    """

    def __init__(
        self,
        penalizer: float = 0.01,
        min_subgroup_n: int = 20,
        min_subgroup_events: int = 5,
    ) -> None:
        self.penalizer = penalizer
        self.min_subgroup_n = min_subgroup_n
        self.min_subgroup_events = min_subgroup_events

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_two_way(
        self,
        df: pd.DataFrame,
        factor1: str,
        factor2: str,
        duration_col: str,
        event_col: str,
        covariates: Optional[Sequence[str]] = None,
    ) -> InteractionResult:
        """Fit a Cox model with a two-way interaction.

        Model: h(t) = h0(t) * exp(b1*F1 + b2*F2 + b3*F1:F2 + covariates)

        Parameters
        ----------
        df : DataFrame
        factor1, factor2 : str
            Binary (0/1) columns for the two factors.
        duration_col, event_col : str
        covariates : sequence of str, optional
            Additional adjustment covariates.

        Returns
        -------
        InteractionResult
        """
        factors = [factor1, factor2]
        cov_list = list(covariates) if covariates else []
        return self._fit_interaction(
            df, factors, duration_col, event_col, cov_list,
            gene=factor1, treatment=factor2, phenotype="",
        )

    def fit_three_way(
        self,
        df: pd.DataFrame,
        factor1: str,
        factor2: str,
        factor3: str,
        duration_col: str,
        event_col: str,
        covariates: Optional[Sequence[str]] = None,
    ) -> InteractionResult:
        """Fit a Cox model with a three-way interaction.

        Model: h(t) = h0(t) * exp(b1*F1 + b2*F2 + b3*F3
                                   + b4*F1:F2 + b5*F1:F3 + b6*F2:F3
                                   + b7*F1:F2:F3 + covariates)

        Parameters
        ----------
        df : DataFrame
        factor1, factor2, factor3 : str
            Binary (0/1) columns for gene, treatment, phenotype.
        duration_col, event_col : str
        covariates : sequence of str, optional

        Returns
        -------
        InteractionResult
        """
        factors = [factor1, factor2, factor3]
        cov_list = list(covariates) if covariates else []
        return self._fit_interaction(
            df, factors, duration_col, event_col, cov_list,
            gene=factor1, treatment=factor2, phenotype=factor3,
        )

    # ------------------------------------------------------------------
    # Interaction term creation
    # ------------------------------------------------------------------

    @staticmethod
    def _create_interaction_terms(
        df: pd.DataFrame, factors: List[str]
    ) -> Tuple[pd.DataFrame, List[str], List[str], Optional[str]]:
        """Create all interaction columns for a set of factors.

        Parameters
        ----------
        df : DataFrame
            Must contain the factor columns.
        factors : list of str
            Factor column names (length 2 or 3).

        Returns
        -------
        df : DataFrame
            Copy with interaction columns appended.
        two_way_names : list of str
            Names of two-way interaction columns.
        all_interaction_names : list of str
            Names of all created interaction columns.
        three_way_name : str or None
            Name of the three-way interaction column, if applicable.
        """
        df = df.copy()
        two_way_names: List[str] = []
        three_way_name: Optional[str] = None

        # Two-way interactions
        for f1, f2 in combinations(factors, 2):
            name = f"{f1}:{f2}"
            df[name] = df[f1] * df[f2]
            two_way_names.append(name)

        # Three-way interaction (if 3 factors)
        if len(factors) == 3:
            three_way_name = ":".join(factors)
            df[three_way_name] = df[factors[0]] * df[factors[1]] * df[factors[2]]

        all_names = list(two_way_names)
        if three_way_name is not None:
            all_names.append(three_way_name)

        return df, two_way_names, all_names, three_way_name

    # ------------------------------------------------------------------
    # Subgroup size check
    # ------------------------------------------------------------------

    def _check_subgroup_sizes(
        self,
        df: pd.DataFrame,
        factors: List[str],
        event_col: str,
        min_n: Optional[int] = None,
        min_events: Optional[int] = None,
    ) -> bool:
        """Check that every interaction subgroup has adequate size.

        Parameters
        ----------
        df : DataFrame
        factors : list of str
            Binary factor columns.
        event_col : str
        min_n : int, optional
            Minimum patients per cell (defaults to ``self.min_subgroup_n``).
        min_events : int, optional
            Minimum events per cell (defaults to ``self.min_subgroup_events``).

        Returns
        -------
        bool
            ``True`` if all cells meet the minimums.
        """
        min_n = min_n if min_n is not None else self.min_subgroup_n
        min_events = min_events if min_events is not None else self.min_subgroup_events

        grouped = df.groupby(factors)
        for _, group in grouped:
            if len(group) < min_n:
                return False
            if group[event_col].sum() < min_events:
                return False
        return True

    # ------------------------------------------------------------------
    # Internal fitting
    # ------------------------------------------------------------------

    def _fit_interaction(
        self,
        df: pd.DataFrame,
        factors: List[str],
        duration_col: str,
        event_col: str,
        covariates: List[str],
        gene: str,
        treatment: str,
        phenotype: str,
    ) -> InteractionResult:
        """Core fitting logic shared by two-way and three-way fits."""
        result = InteractionResult(
            gene=gene, treatment=treatment, phenotype=phenotype
        )

        # Prepare columns
        all_needed = [duration_col, event_col] + factors + covariates
        data = df[list(set(all_needed))].dropna()
        result.n_patients = len(data)
        result.n_events = int(data[event_col].sum())

        if result.n_patients == 0:
            result.error_message = "No rows after dropping NaN."
            return result

        # Check subgroup sizes
        if not self._check_subgroup_sizes(data, factors, event_col):
            result.error_message = (
                f"Insufficient subgroup sizes (min_n={self.min_subgroup_n}, "
                f"min_events={self.min_subgroup_events})."
            )
            return result

        # Create interaction terms
        data, two_way_names, all_int_names, three_way_name = (
            self._create_interaction_terms(data, factors)
        )

        # Assemble model columns
        model_covs = factors + all_int_names + covariates

        # Drop constant columns
        dropped: List[str] = []
        final_covs: List[str] = []
        for c in model_covs:
            if data[c].nunique() <= 1:
                dropped.append(c)
            else:
                final_covs.append(c)

        if dropped:
            warnings.warn(
                f"InteractionCox: dropped constant columns {dropped}",
                stacklevel=2,
            )

        if len(final_covs) == 0:
            result.error_message = "All model columns are constant."
            return result

        fit_data = data[[duration_col, event_col] + final_covs]

        # Fit
        fitter = CoxPHFitter(penalizer=self.penalizer)
        try:
            fitter.fit(
                fit_data,
                duration_col=duration_col,
                event_col=event_col,
                show_progress=False,
            )
        except Exception as exc:
            result.error_message = str(exc)
            return result

        result.converged = True
        result.concordance = fitter.concordance_index_
        result.log_likelihood = fitter.log_likelihood_
        result.aic = fitter.AIC_partial_

        summary = fitter.summary

        # Extract main effects
        for f in factors:
            if f in summary.index:
                result.main_effects[f] = _extract_effect(summary, f)

        # Extract two-way effects
        for tw in two_way_names:
            if tw in summary.index:
                result.two_way_effects[tw] = _extract_effect(summary, tw)

        # Extract three-way effect
        if three_way_name is not None and three_way_name in summary.index:
            result.three_way_effect = _extract_effect(summary, three_way_name)

        return result


# ======================================================================
# Stratified effect helper
# ======================================================================

def stratified_effect(
    df: pd.DataFrame,
    gene_col: str,
    treatment_col: str,
    duration_col: str,
    event_col: str,
    covariates: Optional[Sequence[str]] = None,
    penalizer: float = 0.01,
) -> Dict[str, Any]:
    """Estimate the gene effect separately in treated and untreated groups.

    Fits the interaction model ``gene + treatment + gene:treatment`` and
    derives the gene HR within each treatment stratum.

    Parameters
    ----------
    df : DataFrame
    gene_col : str
        Binary gene mutation column.
    treatment_col : str
        Binary treatment column.
    duration_col, event_col : str
    covariates : sequence of str, optional
    penalizer : float, default 0.01

    Returns
    -------
    dict
        Keys: hr_treated, hr_untreated, interaction_hr, interaction_p,
        hr_treated_ci, hr_untreated_ci, converged, error_message.
    """
    cov_list = list(covariates) if covariates else []
    out: Dict[str, Any] = {
        "hr_treated": np.nan,
        "hr_untreated": np.nan,
        "interaction_hr": np.nan,
        "interaction_p": np.nan,
        "hr_treated_ci": (np.nan, np.nan),
        "hr_untreated_ci": (np.nan, np.nan),
        "converged": False,
        "error_message": "",
    }

    data = df[[duration_col, event_col, gene_col, treatment_col] + cov_list].dropna()
    if len(data) == 0:
        out["error_message"] = "No rows after dropping NaN."
        return out

    # Create interaction
    int_name = f"{gene_col}:{treatment_col}"
    data = data.copy()
    data[int_name] = data[gene_col] * data[treatment_col]

    model_cols = [gene_col, treatment_col, int_name] + cov_list

    # Drop constants
    model_cols = [c for c in model_cols if data[c].nunique() > 1]
    if gene_col not in model_cols:
        out["error_message"] = "Gene column is constant."
        return out

    fit_data = data[[duration_col, event_col] + model_cols]
    fitter = CoxPHFitter(penalizer=penalizer)
    try:
        fitter.fit(fit_data, duration_col=duration_col, event_col=event_col,
                    show_progress=False)
    except Exception as exc:
        out["error_message"] = str(exc)
        return out

    out["converged"] = True
    s = fitter.summary

    # Gene HR in untreated stratum = exp(beta_gene)
    if gene_col in s.index:
        out["hr_untreated"] = s.loc[gene_col, "exp(coef)"]
        out["hr_untreated_ci"] = (
            s.loc[gene_col, "exp(coef) lower 95%"],
            s.loc[gene_col, "exp(coef) upper 95%"],
        )

    # Gene HR in treated stratum = exp(beta_gene + beta_interaction)
    if gene_col in s.index and int_name in s.index:
        beta_gene = s.loc[gene_col, "coef"]
        beta_int = s.loc[int_name, "coef"]
        out["hr_treated"] = np.exp(beta_gene + beta_int)

        # Approximate CI using variance-covariance
        try:
            vcov = fitter.variance_matrix_
            idx_gene = list(vcov.index).index(gene_col)
            idx_int = list(vcov.index).index(int_name)
            var_sum = (
                vcov.iloc[idx_gene, idx_gene]
                + vcov.iloc[idx_int, idx_int]
                + 2 * vcov.iloc[idx_gene, idx_int]
            )
            se_sum = np.sqrt(var_sum)
            out["hr_treated_ci"] = (
                np.exp(beta_gene + beta_int - 1.96 * se_sum),
                np.exp(beta_gene + beta_int + 1.96 * se_sum),
            )
        except Exception:
            pass

    # Interaction term
    if int_name in s.index:
        out["interaction_hr"] = s.loc[int_name, "exp(coef)"]
        out["interaction_p"] = s.loc[int_name, "p"]

    return out


# ======================================================================
# Helpers
# ======================================================================

def _extract_effect(summary: pd.DataFrame, name: str) -> Dict[str, float]:
    """Extract HR, p-value, and CI from a lifelines summary row."""
    row = summary.loc[name]
    return {
        "coef": row["coef"],
        "hr": row["exp(coef)"],
        "p": row["p"],
        "ci_lower": row["exp(coef) lower 95%"],
        "ci_upper": row["exp(coef) upper 95%"],
    }
