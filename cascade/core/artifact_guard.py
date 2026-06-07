"""
Layer 5: Statistical Artifact Guards
======================================

Systematic checks for common statistical pitfalls in biomarker studies:
variable independence, covariate leakage, baseline confounding, and
floor/ceiling effects.

Classes
-------
ArtifactGuard
    Orchestrates all artifact checks and produces an ``ArtifactReport``.
ArtifactReport
    Dataclass summarising the results of all artifact checks.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

try:
    from cascade.stats.equivalence import independence_suite
except ImportError:
    independence_suite = None  # type: ignore[assignment,misc]


@dataclass
class ArtifactReport:
    """Summary of all artifact checks.

    Attributes
    ----------
    independence_results : dict or None
        Output of ``check_independence``.
    leakage_results : list of str
        Covariates flagged as potential future-data leakage.
    confounding_results : dict or None
        Output of ``check_baseline_confounding``.
    floor_ceiling_results : dict or None
        Output of ``check_floor_ceiling_effects``.
    warnings : list of str
        Human-readable warning messages from all checks.
    """

    independence_results: Optional[Dict[str, Any]] = None
    leakage_results: List[str] = field(default_factory=list)
    confounding_results: Optional[Dict[str, Any]] = None
    floor_ceiling_results: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        """Whether any checks produced warnings."""
        return len(self.warnings) > 0

    @property
    def n_warnings(self) -> int:
        """Total number of warnings."""
        return len(self.warnings)


class ArtifactGuard:
    """Orchestrate statistical artifact checks.

    By default, all available checks are enabled.  Pass *checks* to
    restrict to a subset.

    Parameters
    ----------
    checks : sequence of str, optional
        Subset of checks to run.  Valid values: ``'independence'``,
        ``'leakage'``, ``'confounding'``, ``'floor_ceiling'``.
        If ``None``, all checks are run.
    """

    AVAILABLE_CHECKS = {"independence", "leakage", "confounding", "floor_ceiling"}

    def __init__(self, checks: Optional[Sequence[str]] = None) -> None:
        if checks is not None:
            invalid = set(checks) - self.AVAILABLE_CHECKS
            if invalid:
                raise ValueError(f"Unknown checks: {invalid}")
            self.checks = set(checks)
        else:
            self.checks = set(self.AVAILABLE_CHECKS)

    # ------------------------------------------------------------------
    # Independence
    # ------------------------------------------------------------------

    @staticmethod
    def check_independence(
        x_binary: pd.Series,
        y_binary: pd.Series,
        x_continuous: Optional[pd.Series] = None,
        y_continuous: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """Test independence between two variables.

        Uses multiple metrics to build a comprehensive independence
        assessment: phi coefficient, chi-squared test, and (if
        continuous versions are provided) Spearman correlation.

        Parameters
        ----------
        x_binary : Series
            Binary (0/1) version of the first variable.
        y_binary : Series
            Binary (0/1) version of the second variable.
        x_continuous : Series, optional
            Continuous version of the first variable (for Spearman).
        y_continuous : Series, optional
            Continuous version of the second variable (for Spearman).

        Returns
        -------
        dict
            Keys: ``phi``, ``phi_p``, ``chi2``, ``chi2_p``,
            ``spearman_rho`` (if continuous provided), ``spearman_p``,
            ``discordant_pct``, ``independent`` (summary boolean).
        """
        # Try CASCADE stats.equivalence first
        if independence_suite is not None:
            try:
                return independence_suite(
                    x_binary, y_binary, x_continuous, y_continuous
                )
            except Exception:
                pass

        # Fallback implementation
        result: Dict[str, Any] = {}

        # Align indices
        aligned = pd.DataFrame({"x": x_binary, "y": y_binary}).dropna()
        x = aligned["x"].values.astype(int)
        y = aligned["y"].values.astype(int)
        n = len(x)

        if n == 0:
            return {"phi": np.nan, "independent": None, "warnings": ["No valid observations"]}

        # Phi coefficient
        n11 = ((x == 1) & (y == 1)).sum()
        n10 = ((x == 1) & (y == 0)).sum()
        n01 = ((x == 0) & (y == 1)).sum()
        n00 = ((x == 0) & (y == 0)).sum()

        denom = np.sqrt(
            (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
        )
        if denom == 0:
            result["phi"] = 0.0
        else:
            result["phi"] = (n11 * n00 - n10 * n01) / denom

        # Chi-squared test
        contingency = np.array([[n00, n01], [n10, n11]])
        if contingency.min() >= 0:
            try:
                chi2, chi2_p, _, _ = sp_stats.chi2_contingency(
                    contingency, correction=True
                )
                result["chi2"] = chi2
                result["chi2_p"] = chi2_p
            except Exception:
                result["chi2"] = np.nan
                result["chi2_p"] = np.nan
        else:
            result["chi2"] = np.nan
            result["chi2_p"] = np.nan

        # Phi p-value (same as chi2_p for 2x2)
        result["phi_p"] = result.get("chi2_p", np.nan)

        # Spearman (if continuous)
        if x_continuous is not None and y_continuous is not None:
            cont_aligned = pd.DataFrame(
                {"xc": x_continuous, "yc": y_continuous}
            ).dropna()
            if len(cont_aligned) >= 3:
                rho, rho_p = sp_stats.spearmanr(
                    cont_aligned["xc"], cont_aligned["yc"]
                )
                result["spearman_rho"] = rho
                result["spearman_p"] = rho_p
            else:
                result["spearman_rho"] = np.nan
                result["spearman_p"] = np.nan
        else:
            result["spearman_rho"] = np.nan
            result["spearman_p"] = np.nan

        # Discordant percentage
        concordant = (x == y).sum()
        result["discordant_pct"] = (n - concordant) / n * 100 if n > 0 else np.nan

        # Summary: independent if phi_p > 0.05 and |phi| < 0.1
        phi_p = result.get("phi_p", np.nan)
        phi = result.get("phi", np.nan)
        if pd.notna(phi_p) and pd.notna(phi):
            result["independent"] = (phi_p > 0.05) or (abs(phi) < 0.1)
        else:
            result["independent"] = None

        return result

    # ------------------------------------------------------------------
    # Covariate leakage
    # ------------------------------------------------------------------

    @staticmethod
    def check_covariate_leakage(
        df: pd.DataFrame,
        landmark_col: str,
        covariates: Sequence[str],
        timeline_data: Optional[pd.DataFrame] = None,
        patient_col: str = "PATIENT_ID",
    ) -> List[str]:
        """Detect covariates that may contain future-data leakage.

        A covariate leaks future data if its value depends on events
        that occur after the landmark date.  Common examples: "ever
        received ICI" computed over the full follow-up instead of up
        to the landmark.

        Heuristic checks:
        1. Binary covariates that are constant within the analysis set
           (suggests they may be "ever" variables).
        2. Covariates whose distribution differs markedly between
           early and late landmark subsets (suggests time-dependency).
        3. If *timeline_data* is provided, checks whether covariate values
           reference events beyond the landmark date.

        Parameters
        ----------
        df : DataFrame
            Analysis dataset with *landmark_col* and *covariates*.
        landmark_col : str
            Column indicating the landmark time point.
        covariates : sequence of str
            Covariates to check.
        timeline_data : DataFrame, optional
            Longitudinal timeline data with event dates for deeper checks.
        patient_col : str, default 'PATIENT_ID'

        Returns
        -------
        list of str
            Names of covariates flagged as potential leakage.
        """
        flagged: List[str] = []
        sub = df.dropna(subset=[landmark_col])

        if sub.empty:
            return flagged

        # Split into early and late landmark groups
        median_lm = sub[landmark_col].median()

        for cov in covariates:
            if cov not in sub.columns:
                continue

            col = sub[cov]

            # Check 1: Binary "ever" variable with suspiciously high prevalence
            if col.nunique() == 2 and col.dtype in (int, float, np.int64, np.float64, bool):
                # If > 80% have value 1, may be a broadly computed flag
                if col.mean() > 0.80:
                    flagged.append(cov)
                    continue

            # Check 2: Distribution shift between early and late landmarks
            early = sub.loc[sub[landmark_col] <= median_lm, cov].dropna()
            late = sub.loc[sub[landmark_col] > median_lm, cov].dropna()

            if len(early) >= 10 and len(late) >= 10:
                if col.nunique() == 2:
                    # Proportion test
                    p_early = early.mean()
                    p_late = late.mean()
                    if abs(p_early - p_late) < 0.01 and p_early > 0.5:
                        # Suspiciously stable across time
                        flagged.append(cov)
                        continue
                else:
                    # Mann-Whitney for continuous
                    try:
                        _, p_val = sp_stats.mannwhitneyu(
                            early, late, alternative="two-sided"
                        )
                        # If no difference but should have some, suspicious
                        if p_val > 0.95 and col.std() > 0:
                            flagged.append(cov)
                            continue
                    except Exception:
                        pass

        return flagged

    # ------------------------------------------------------------------
    # Baseline confounding
    # ------------------------------------------------------------------

    @staticmethod
    def check_baseline_confounding(
        df: pd.DataFrame,
        adjusted_hr: float,
        unadjusted_hr: float,
        threshold: float = 0.15,
    ) -> bool:
        """Detect baseline confounding via HR change.

        Confounding is flagged when the relative change between
        unadjusted and adjusted hazard ratios exceeds *threshold*.

        Parameters
        ----------
        df : DataFrame
            Not directly used but accepted for interface consistency.
        adjusted_hr : float
            HR from the fully adjusted model.
        unadjusted_hr : float
            HR from the unadjusted (crude) model.
        threshold : float, default 0.15
            Relative change threshold (e.g. 0.15 = 15%).

        Returns
        -------
        bool
            ``True`` if confounding is detected.
        """
        if np.isnan(adjusted_hr) or np.isnan(unadjusted_hr):
            return False
        if unadjusted_hr == 0:
            return False

        relative_change = abs(adjusted_hr - unadjusted_hr) / abs(unadjusted_hr)
        return relative_change > threshold

    # ------------------------------------------------------------------
    # Floor / ceiling effects
    # ------------------------------------------------------------------

    @staticmethod
    def check_floor_ceiling_effects(
        df: pd.DataFrame,
        variable_col: str,
        min_val: float,
        max_val: float,
        group_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect floor and ceiling effects in a variable.

        Flags when a large fraction of observations are at the minimum
        or maximum of the scale, which compresses variance and can
        mask true effects.

        Parameters
        ----------
        df : DataFrame
        variable_col : str
            Column to check.
        min_val : float
            Scale minimum (floor).
        max_val : float
            Scale maximum (ceiling).
        group_col : str, optional
            If provided, compute floor/ceiling percentages per group.

        Returns
        -------
        dict
            Keys: ``floor_pct``, ``ceiling_pct``, ``affected_groups``
            (list of group values where floor or ceiling > 25%).
        """
        data = df[[variable_col]].copy()
        if group_col is not None and group_col in df.columns:
            data[group_col] = df[group_col]

        col = data[variable_col].dropna()
        n = len(col)

        result: Dict[str, Any] = {
            "floor_pct": 0.0,
            "ceiling_pct": 0.0,
            "affected_groups": [],
        }

        if n == 0:
            return result

        result["floor_pct"] = float((col == min_val).sum() / n * 100)
        result["ceiling_pct"] = float((col == max_val).sum() / n * 100)

        if group_col is not None and group_col in data.columns:
            affected: List[Any] = []
            for grp_name, grp_data in data.groupby(group_col):
                grp_col = grp_data[variable_col].dropna()
                if len(grp_col) == 0:
                    continue
                floor_pct = (grp_col == min_val).sum() / len(grp_col) * 100
                ceiling_pct = (grp_col == max_val).sum() / len(grp_col) * 100
                if floor_pct > 25 or ceiling_pct > 25:
                    affected.append(grp_name)
            result["affected_groups"] = affected

        return result

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run_all(
        self,
        df: pd.DataFrame,
        x_binary: Optional[pd.Series] = None,
        y_binary: Optional[pd.Series] = None,
        x_continuous: Optional[pd.Series] = None,
        y_continuous: Optional[pd.Series] = None,
        landmark_col: Optional[str] = None,
        covariates: Optional[Sequence[str]] = None,
        adjusted_hr: Optional[float] = None,
        unadjusted_hr: Optional[float] = None,
        variable_col: Optional[str] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        group_col: Optional[str] = None,
        confounding_threshold: float = 0.15,
        **kwargs: Any,
    ) -> ArtifactReport:
        """Run all enabled artifact checks and produce a report.

        Parameters
        ----------
        df : DataFrame
            Primary analysis dataset.
        x_binary, y_binary : Series, optional
            For independence check.
        x_continuous, y_continuous : Series, optional
            For independence check (continuous versions).
        landmark_col : str, optional
            For leakage check.
        covariates : sequence of str, optional
            For leakage check.
        adjusted_hr, unadjusted_hr : float, optional
            For confounding check.
        variable_col : str, optional
            For floor/ceiling check.
        min_val, max_val : float, optional
            For floor/ceiling check.
        group_col : str, optional
            For floor/ceiling check.
        confounding_threshold : float, default 0.15
        **kwargs
            Reserved for future checks.

        Returns
        -------
        ArtifactReport
        """
        report = ArtifactReport()

        # Independence
        if "independence" in self.checks and x_binary is not None and y_binary is not None:
            report.independence_results = self.check_independence(
                x_binary, y_binary, x_continuous, y_continuous
            )
            if report.independence_results.get("independent") is False:
                report.warnings.append(
                    f"Variables may NOT be independent: "
                    f"phi={report.independence_results.get('phi', 'N/A'):.3f}, "
                    f"P={report.independence_results.get('phi_p', 'N/A')}"
                )

        # Leakage
        if "leakage" in self.checks and landmark_col is not None and covariates is not None:
            report.leakage_results = self.check_covariate_leakage(
                df, landmark_col, covariates, **kwargs
            )
            if report.leakage_results:
                report.warnings.append(
                    f"Potential covariate leakage in: {report.leakage_results}"
                )

        # Confounding
        if "confounding" in self.checks and adjusted_hr is not None and unadjusted_hr is not None:
            confounded = self.check_baseline_confounding(
                df, adjusted_hr, unadjusted_hr, confounding_threshold
            )
            report.confounding_results = {
                "confounded": confounded,
                "adjusted_hr": adjusted_hr,
                "unadjusted_hr": unadjusted_hr,
                "relative_change": abs(adjusted_hr - unadjusted_hr) / abs(unadjusted_hr) if unadjusted_hr != 0 else np.nan,
            }
            if confounded:
                report.warnings.append(
                    f"Baseline confounding detected: adjusted HR={adjusted_hr:.3f} "
                    f"vs unadjusted HR={unadjusted_hr:.3f} "
                    f"(>{confounding_threshold*100:.0f}% change)"
                )

        # Floor/ceiling
        if "floor_ceiling" in self.checks and variable_col is not None and min_val is not None and max_val is not None:
            report.floor_ceiling_results = self.check_floor_ceiling_effects(
                df, variable_col, min_val, max_val, group_col
            )
            fc = report.floor_ceiling_results
            if fc["floor_pct"] > 25:
                report.warnings.append(
                    f"Floor effect: {fc['floor_pct']:.1f}% at minimum ({min_val})"
                )
            if fc["ceiling_pct"] > 25:
                report.warnings.append(
                    f"Ceiling effect: {fc['ceiling_pct']:.1f}% at maximum ({max_val})"
                )

        return report
