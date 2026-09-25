"""
CASCADE Pitfall Detector
========================

Orchestrator that runs all applicable pitfall checks against provided
data artifacts and accumulates warnings. The entry point for users
who want a one-call scan of their analysis for known pitfalls.
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .library import PitfallWarning, Severity
from .registry import PitfallRegistry
from .checks.comment_header import check_comment_corruption
from .checks.covariate_leakage import check_landmark_leakage
from .checks.collinearity import check_collinearity as _check_collinearity
from .checks.constant_variable import check_constant_variables as _check_constant
from .checks.separation import check_separation_problems
from .checks.singular_matrix import check_singular_matrix as _check_singular
from .checks.informative_censoring import check_informative_censoring as _check_censoring
from .checks.center_effect import check_center_effect as _check_center


class PitfallDetector:
    """Orchestrator for running pitfall checks.

    Provides both individual check methods and a ``check_all`` sweep
    that runs every applicable check given the supplied artifacts.

    Attributes
    ----------
    registry : PitfallRegistry
        The registry of known pitfalls.

    Examples
    --------
    >>> detector = PitfallDetector()
    >>> warnings = detector.check_data_format("data_timeline_performance_status.txt")
    >>> for w in warnings:
    ...     print(w)

    >>> detector.check_all(data_file="data.txt", feature_matrix=X)
    >>> print(detector.summary())
    """

    def __init__(self, registry: Optional[PitfallRegistry] = None) -> None:
        """Initialize the detector.

        Parameters
        ----------
        registry : PitfallRegistry, optional
            Custom registry to use. If None, creates a default registry
            pre-populated with the canonical pitfall library.
        """
        self.registry = registry if registry is not None else PitfallRegistry()
        self._warnings: List[PitfallWarning] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def warnings(self) -> List[PitfallWarning]:
        """All accumulated warnings from checks run so far.

        Returns
        -------
        list of PitfallWarning
        """
        return list(self._warnings)

    # ------------------------------------------------------------------
    # Master check
    # ------------------------------------------------------------------

    def check_all(
        self,
        data_file: Optional[Union[str, Path]] = None,
        model_df: Optional[pd.DataFrame] = None,
        feature_matrix: Optional[pd.DataFrame] = None,
        landmark_days: Optional[float] = None,
        landmark_date_col: Optional[str] = None,
        covariate_cols: Optional[List[str]] = None,
        event_date_cols: Optional[List[str]] = None,
        results_df: Optional[pd.DataFrame] = None,
        ci_lower_col: str = "ci_lower",
        ci_upper_col: str = "ci_upper",
        subgroup_label: Optional[str] = None,
    ) -> List[PitfallWarning]:
        """Run all applicable checks given the supplied artifacts.

        Each check is only invoked if its required inputs are provided.
        All warnings are both returned and accumulated internally.

        Parameters
        ----------
        data_file : str or Path, optional
            Path to a tab-delimited data file to check for comment
            header corruption (Pitfall #1).
        model_df : pd.DataFrame, optional
            The analysis dataframe for covariate leakage checks
            (Pitfall #2). Requires ``landmark_date_col`` and
            ``covariate_cols``.
        feature_matrix : pd.DataFrame, optional
            Feature matrix for collinearity (#3), constant variable (#4),
            and singular matrix (#7) checks.
        landmark_days : float, optional
            The landmark time in days (context for constant variable
            warnings).
        landmark_date_col : str, optional
            Column name for landmark date in ``model_df``.
        covariate_cols : list of str, optional
            Covariate column names for leakage check.
        event_date_cols : list of str, optional
            Event date columns for direct leakage detection.
        results_df : pd.DataFrame, optional
            Model results dataframe for separation check.
        ci_lower_col : str
            Lower CI column name in results_df (default 'ci_lower').
        ci_upper_col : str
            Upper CI column name in results_df (default 'ci_upper').
        subgroup_label : str, optional
            Label for the current subgroup context.

        Returns
        -------
        list of PitfallWarning
            All warnings emitted by the checks.
        """
        new_warnings: List[PitfallWarning] = []

        # Pitfall #1: Comment header corruption
        if data_file is not None:
            new_warnings.extend(check_comment_corruption(data_file))

        # Pitfall #2: Covariate leakage
        if (
            model_df is not None
            and landmark_date_col is not None
            and covariate_cols is not None
        ):
            new_warnings.extend(
                check_landmark_leakage(
                    model_df, landmark_date_col, covariate_cols, event_date_cols
                )
            )

        # Pitfalls #3, #4, #7: Feature matrix checks
        if feature_matrix is not None:
            new_warnings.extend(_check_collinearity(feature_matrix))
            label = subgroup_label
            if label is None and landmark_days is not None:
                label = f"{landmark_days}-day landmark"
            new_warnings.extend(_check_constant(feature_matrix, label))
            new_warnings.extend(_check_singular(feature_matrix))

        # Separation check (relates to Pitfall #7)
        if results_df is not None:
            new_warnings.extend(
                check_separation_problems(results_df, ci_lower_col, ci_upper_col)
            )

        self._warnings.extend(new_warnings)
        return new_warnings

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_data_format(
        self,
        filepath: Union[str, Path],
        comment_char: str = "#",
    ) -> List[PitfallWarning]:
        """Check a data file for comment header corruption (Pitfall #1).

        Parameters
        ----------
        filepath : str or Path
            Path to the tab-delimited file.
        comment_char : str
            The comment character (default '#').

        Returns
        -------
        list of PitfallWarning
        """
        result = check_comment_corruption(filepath, comment_char)
        self._warnings.extend(result)
        return result

    def check_covariate_leakage(
        self,
        df: pd.DataFrame,
        landmark_col: str,
        covariate_cols: List[str],
        timeline_cols: Optional[List[str]] = None,
    ) -> List[PitfallWarning]:
        """Check for covariate leakage in landmark models (Pitfall #2).

        Parameters
        ----------
        df : pd.DataFrame
            Analysis dataframe.
        landmark_col : str
            Column with landmark dates.
        covariate_cols : list of str
            Covariates to check.
        timeline_cols : list of str, optional
            Event date columns for direct leakage detection.

        Returns
        -------
        list of PitfallWarning
        """
        result = check_landmark_leakage(df, landmark_col, covariate_cols, timeline_cols)
        self._warnings.extend(result)
        return result

    def check_collinearity(
        self,
        X: pd.DataFrame,
        threshold: float = 0.99,
    ) -> List[PitfallWarning]:
        """Check for collinearity in a feature matrix (Pitfall #3).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        threshold : float
            Correlation threshold (default 0.99).

        Returns
        -------
        list of PitfallWarning
        """
        result = _check_collinearity(X, threshold)
        self._warnings.extend(result)
        return result

    def check_constant_variables(
        self,
        X: pd.DataFrame,
        subgroup_label: Optional[str] = None,
    ) -> List[PitfallWarning]:
        """Check for constant (zero-variance) variables (Pitfall #4).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        subgroup_label : str, optional
            Context label for the subgroup.

        Returns
        -------
        list of PitfallWarning
        """
        result = _check_constant(X, subgroup_label)
        self._warnings.extend(result)
        return result

    def check_separation(
        self,
        results_df: pd.DataFrame,
        ci_lower_col: str = "ci_lower",
        ci_upper_col: str = "ci_upper",
        threshold: float = 100.0,
    ) -> List[PitfallWarning]:
        """Check for separation problems in model results.

        Parameters
        ----------
        results_df : pd.DataFrame
            Results dataframe with CI columns.
        ci_lower_col : str
            Lower CI column name.
        ci_upper_col : str
            Upper CI column name.
        threshold : float
            CI ratio threshold (default 100).

        Returns
        -------
        list of PitfallWarning
        """
        result = check_separation_problems(results_df, ci_lower_col, ci_upper_col, threshold)
        self._warnings.extend(result)
        return result

    def check_singular_matrix(
        self,
        X: pd.DataFrame,
    ) -> List[PitfallWarning]:
        """Check for singular design matrix (Pitfall #7).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        list of PitfallWarning
        """
        result = _check_singular(X)
        self._warnings.extend(result)
        return result

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def check_informative_censoring(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        biomarker_cols: List[str],
        covariates: Optional[List[str]] = None,
        alpha: float = 0.05,
        min_hr: float = 1.25,
    ) -> List[PitfallWarning]:
        """Check whether censoring depends on the biomarker (Pitfall #10).

        Parameters
        ----------
        df : pd.DataFrame
            Analysis dataframe, one row per patient.
        duration_col : str
            Follow-up time column.
        event_col : str
            Event indicator column (1 = event, 0 = censored).
        biomarker_cols : list of str
            Binary biomarker columns to test.
        covariates : list of str, optional
            Adjustment covariates for the censoring model.
        alpha : float
            Family-wise significance level (default 0.05).
        min_hr : float
            Minimum censoring hazard ratio to flag (default 1.25).

        Returns
        -------
        list of PitfallWarning
        """
        result = _check_censoring(
            df, duration_col, event_col, biomarker_cols, covariates, alpha, min_hr
        )
        self._warnings.extend(result)
        return result

    def check_center_effect(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        biomarker_cols: List[str],
        center_col: str,
        covariates: Optional[List[str]] = None,
        alpha: float = 0.05,
        prevalence_range: float = 0.15,
    ) -> List[PitfallWarning]:
        """Check for centre / batch effects (Pitfall #11).

        Parameters
        ----------
        df : pd.DataFrame
            Analysis dataframe, one row per patient.
        duration_col : str
            Follow-up time column.
        event_col : str
            Event indicator column (1 = event, 0 = censored).
        biomarker_cols : list of str
            Binary biomarker columns to test.
        center_col : str
            Column identifying the contributing centre or batch.
        covariates : list of str, optional
            Adjustment covariates for the Cox models.
        alpha : float
            Family-wise significance level (default 0.05).
        prevalence_range : float
            Absolute prevalence range across centres to flag (default 0.15).

        Returns
        -------
        list of PitfallWarning
        """
        result = _check_center(
            df, duration_col, event_col, biomarker_cols, center_col,
            covariates, alpha, prevalence_range,
        )
        self._warnings.extend(result)
        return result

    def summary(self) -> str:
        """Generate a formatted summary of all accumulated warnings.

        Returns
        -------
        str
            Multi-line report grouped by severity.
        """
        if not self._warnings:
            return "CASCADE Pitfall Detector: No issues found."

        lines = [
            "=" * 60,
            "CASCADE Pitfall Detector Report",
            "=" * 60,
            "",
        ]

        # Group by severity
        by_severity = {sev: [] for sev in Severity}
        for w in self._warnings:
            by_severity[w.severity].append(w)

        total = len(self._warnings)
        lines.append(
            f"Total warnings: {total} "
            f"({sum(1 for w in self._warnings if w.severity == Severity.CRITICAL)} critical, "
            f"{sum(1 for w in self._warnings if w.severity == Severity.WARNING)} warning, "
            f"{sum(1 for w in self._warnings if w.severity == Severity.INFO)} info)"
        )
        lines.append("")

        for severity in [Severity.CRITICAL, Severity.WARNING, Severity.INFO]:
            group = by_severity[severity]
            if not group:
                continue

            lines.append(f"--- {severity.value.upper()} ({len(group)}) ---")
            lines.append("")

            for i, w in enumerate(group, 1):
                lines.append(f"  {i}. Pitfall #{w.pitfall.id}: {w.pitfall.name}")
                lines.append(f"     Location: {w.location}")
                lines.append(f"     {w.message}")
                lines.append(f"     Suggestion: {w.suggestion}")
                lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all accumulated warnings."""
        self._warnings.clear()
