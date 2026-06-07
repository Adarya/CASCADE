"""
Layer 4: Sensitivity & Robustness Testing
==========================================

Runs the primary analysis under multiple perturbations (different cohort
definitions, time windows, statistical methods, outcome definitions) and
classifies each finding as robust, exploratory, or unstable.

Classes
-------
SensitivitySuite
    Manages a collection of analysis variants and runs them against a
    user-supplied analysis function.
RobustnessScorer
    Classifies findings based on cross-variant concordance.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


class SensitivitySuite:
    """Manage and execute sensitivity analysis variants.

    Each variant represents a perturbation of the primary analysis:
    a different cohort filter, time window, feature definition, or
    statistical approach.

    Example
    -------
    >>> suite = SensitivitySuite()
    >>> suite.add_variant('6mo_landmark', df_6mo, category='time_domain')
    >>> suite.add_variant('12mo_landmark', df_12mo, category='time_domain')
    >>> results = suite.run(my_analysis_fn)
    """

    # Valid categories for variant classification
    VALID_CATEGORIES = {
        "feature_def",
        "stratification",
        "time_domain",
        "statistical",
        "outcome",
        "subgroup",
        "threshold",
    }

    def __init__(self) -> None:
        self._variants: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_variant(
        self,
        name: str,
        df_or_filter: Union[pd.DataFrame, Callable[[pd.DataFrame], pd.DataFrame]],
        category: str,
    ) -> None:
        """Register a sensitivity variant.

        Parameters
        ----------
        name : str
            Unique name for this variant (e.g. ``'adeno_only'``).
        df_or_filter : DataFrame or callable
            Either a pre-filtered DataFrame, or a callable that takes
            the primary DataFrame and returns a filtered version.
        category : str
            One of: ``'feature_def'``, ``'stratification'``,
            ``'time_domain'``, ``'statistical'``, ``'outcome'``,
            ``'subgroup'``, ``'threshold'``.

        Raises
        ------
        ValueError
            If *name* is already registered or *category* is invalid.
        """
        if name in self._variants:
            raise ValueError(f"Variant '{name}' already registered.")
        if category not in self.VALID_CATEGORIES:
            raise ValueError(
                f"category must be one of {self.VALID_CATEGORIES}, got '{category}'"
            )
        self._variants[name] = {
            "data": df_or_filter,
            "category": category,
        }

    @property
    def variant_names(self) -> List[str]:
        """Return the names of all registered variants."""
        return list(self._variants.keys())

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        analysis_fn: Callable[..., pd.DataFrame],
        primary_df: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> Dict[str, pd.DataFrame]:
        """Run the analysis function under each registered variant.

        Parameters
        ----------
        analysis_fn : callable
            A function with signature ``analysis_fn(df, **kwargs) -> DataFrame``.
            It receives the variant DataFrame (or the result of calling
            the variant filter on *primary_df*) as the first argument.
        primary_df : DataFrame, optional
            Required if any variant was registered as a callable filter.
        **kwargs
            Additional keyword arguments forwarded to *analysis_fn*.

        Returns
        -------
        dict
            Mapping of variant name to the results DataFrame.
        """
        results: Dict[str, pd.DataFrame] = {}

        for name, spec in self._variants.items():
            data = spec["data"]

            if callable(data):
                if primary_df is None:
                    raise ValueError(
                        f"Variant '{name}' is a filter callable but "
                        "no primary_df was provided."
                    )
                try:
                    variant_df = data(primary_df)
                except Exception as exc:
                    warnings.warn(
                        f"SensitivitySuite: variant '{name}' filter failed: {exc}",
                        stacklevel=2,
                    )
                    continue
            else:
                variant_df = data

            try:
                results[name] = analysis_fn(variant_df, **kwargs)
            except Exception as exc:
                warnings.warn(
                    f"SensitivitySuite: analysis failed for variant '{name}': {exc}",
                    stacklevel=2,
                )

        return results

    # ------------------------------------------------------------------
    # Robustness classification
    # ------------------------------------------------------------------

    @staticmethod
    def classify_robustness(
        primary_results: pd.DataFrame,
        sensitivity_results: Dict[str, pd.DataFrame],
        biomarker_col: str = "biomarker",
        direction_col: str = "hr",
        threshold: float = 0.75,
    ) -> pd.DataFrame:
        """Classify each primary finding as robust, exploratory, or unstable.

        Parameters
        ----------
        primary_results : DataFrame
            Must contain *biomarker_col* and *direction_col*.
        sensitivity_results : dict
            Mapping of variant name to results DataFrame (same schema).
        biomarker_col : str, default 'biomarker'
            Column identifying findings across variants.
        direction_col : str, default 'hr'
            Effect-size column used to assess direction consistency.
        threshold : float, default 0.75
            Minimum fraction of variants with concordant direction to
            be classified as robust.

        Returns
        -------
        DataFrame
            Primary results with an added ``robustness_class`` column.
        """
        result = primary_results.copy()
        scorer = RobustnessScorer()

        classes: List[str] = []

        for _, row in result.iterrows():
            biomarker = row.get(biomarker_col)
            primary_effect = row.get(direction_col, np.nan)

            variant_effects: List[float] = []
            for _, variant_df in sensitivity_results.items():
                if biomarker_col not in variant_df.columns:
                    continue
                match = variant_df.loc[variant_df[biomarker_col] == biomarker]
                if not match.empty and direction_col in match.columns:
                    val = match[direction_col].iloc[0]
                    if pd.notna(val):
                        variant_effects.append(val)

            cls = scorer.score(
                primary_effect, variant_effects, threshold=threshold
            )
            classes.append(cls)

        result["robustness_class"] = classes
        return result


class RobustnessScorer:
    """Classify a single finding's robustness based on sensitivity results.

    Classification logic:

    - **robust**: Direction concordance >= threshold AND no direction flips.
    - **unstable**: ANY direction flip (primary HR > 1 but variant HR < 1,
      or vice versa).
    - **exploratory**: Everything else.
    """

    @staticmethod
    def score(
        primary_result: float,
        sensitivity_results: Sequence[float],
        threshold: float = 0.75,
    ) -> str:
        """Score a single finding's robustness.

        Parameters
        ----------
        primary_result : float
            Effect size from the primary analysis (e.g. HR).
        sensitivity_results : sequence of float
            Effect sizes from sensitivity variants.
        threshold : float, default 0.75
            Minimum concordance fraction for 'robust'.

        Returns
        -------
        str
            ``'robust'``, ``'exploratory'``, or ``'unstable'``.
        """
        if np.isnan(primary_result) or len(sensitivity_results) == 0:
            return "exploratory"

        primary_direction = 1 if primary_result >= 1.0 else -1
        n_concordant = 0
        n_total = 0
        has_flip = False

        for val in sensitivity_results:
            if np.isnan(val):
                continue
            n_total += 1
            variant_direction = 1 if val >= 1.0 else -1

            if variant_direction == primary_direction:
                n_concordant += 1
            else:
                has_flip = True

        if n_total == 0:
            return "exploratory"

        if has_flip:
            return "unstable"

        concordance = n_concordant / n_total
        if concordance >= threshold:
            return "robust"

        return "exploratory"
