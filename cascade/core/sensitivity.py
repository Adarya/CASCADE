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
        # Variants whose filter or analysis raised during the last ``run``
        # (name -> error message).  Reported as failed variants downstream.
        self.failed_variants: Dict[str, str] = {}

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

    @property
    def variant_categories(self) -> Dict[str, str]:
        """Return a mapping of variant name to category."""
        return {name: spec["category"] for name, spec in self._variants.items()}

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
        self.failed_variants = {}

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
                    self.failed_variants[name] = f"filter failed: {exc}"
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
                self.failed_variants[name] = f"analysis failed: {exc}"
                warnings.warn(
                    f"SensitivitySuite: analysis failed for variant '{name}': {exc}",
                    stacklevel=2,
                )

        return results

    # ------------------------------------------------------------------
    # Robustness classification
    # ------------------------------------------------------------------

    # Columns (besides the biomarker column) that, when present in the
    # primary results, jointly identify a finding (e.g. gene x site).
    DEFAULT_KEY_COLS = ("gene", "site", "treatment", "phenotype", "subgroup",
                        "endpoint", "outcome")

    @staticmethod
    def classify_robustness(
        primary_results: pd.DataFrame,
        sensitivity_results: Dict[str, pd.DataFrame],
        biomarker_col: str = "biomarker",
        direction_col: str = "hr",
        threshold: float = 0.75,
        key_cols: Optional[Sequence[str]] = None,
        min_evaluable: int = 2,
        failed_variants: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Classify each primary finding as robust, exploratory, or unstable.

        Each primary finding is matched to variant rows on its **full key**
        (``biomarker_col`` plus any of :attr:`DEFAULT_KEY_COLS` present in
        the primary results, or *key_cols* if given), so gene x site results
        compare each site with the same site in every variant.

        Parameters
        ----------
        primary_results : DataFrame
            Must contain the key column(s) and *direction_col*.
        sensitivity_results : dict
            Mapping of variant name to results DataFrame (same schema).
        biomarker_col : str, default 'biomarker'
            Column identifying findings across variants.
        direction_col : str, default 'hr'
            Effect-size column used to assess direction consistency.
        threshold : float, default 0.75
            Minimum fraction of variants (failed variants included) that
            reproduce the primary direction for a finding to be classified
            robust; any direction flip classifies the finding as unstable.
        key_cols : sequence of str, optional
            Explicit key columns identifying a finding.  Overrides the
            automatic ``biomarker_col`` + :attr:`DEFAULT_KEY_COLS` detection.
        min_evaluable : int, default 2
            Minimum number of evaluable variants (non-missing effect for the
            finding) required before a finding can be classified robust.
        failed_variants : sequence of str, optional
            Names of variants that failed to run at all (e.g.
            ``SensitivitySuite.failed_variants``); counted as failed for
            every finding.

        Returns
        -------
        DataFrame
            Primary results with added ``robustness_class``, ``n_evaluable``,
            ``n_concordant``, ``n_failed_variants`` and ``concordance``
            columns.
        """
        result = primary_results.copy()
        scorer = RobustnessScorer()

        if key_cols is None:
            keys = [biomarker_col] if biomarker_col in result.columns else []
            keys += [
                c for c in SensitivitySuite.DEFAULT_KEY_COLS
                if c in result.columns and c not in keys
            ]
        else:
            keys = [c for c in key_cols if c in result.columns]
        if not keys:
            raise ValueError(
                f"No key column found to match findings across variants "
                f"(looked for {biomarker_col!r} and {SensitivitySuite.DEFAULT_KEY_COLS})."
            )

        n_run_failures = len(failed_variants) if failed_variants else 0
        classes: List[str] = []
        details: List[Dict[str, Any]] = []

        for _, row in result.iterrows():
            primary_effect = row.get(direction_col, np.nan)

            variant_effects: List[float] = []
            for _, variant_df in sensitivity_results.items():
                val = np.nan
                if (
                    isinstance(variant_df, pd.DataFrame)
                    and all(k in variant_df.columns for k in keys)
                    and direction_col in variant_df.columns
                ):
                    mask = pd.Series(True, index=variant_df.index)
                    for k in keys:
                        mask &= variant_df[k] == row[k]
                    match = variant_df.loc[mask]
                    # Exactly one matching row is required; zero or
                    # ambiguous (duplicate) matches count as failed.
                    if len(match) == 1:
                        val = match[direction_col].iloc[0]
                # Missing / NaN effects are kept so they are counted as
                # failed variants rather than dropped silently.
                variant_effects.append(val)

            detail = scorer.score_detail(
                primary_effect, variant_effects,
                threshold=threshold, min_evaluable=min_evaluable,
            )
            detail["n_failed"] += n_run_failures
            classes.append(detail["robustness_class"])
            details.append(detail)

        result["robustness_class"] = classes
        result["n_evaluable"] = [d["n_evaluable"] for d in details]
        result["n_concordant"] = [d["n_concordant"] for d in details]
        result["n_failed_variants"] = [d["n_failed"] for d in details]
        result["concordance"] = [d["concordance"] for d in details]
        return result


class RobustnessScorer:
    """Classify a single finding's robustness based on sensitivity results.

    Classification logic (a variant is *evaluable* if it produced a
    non-missing effect estimate for the finding):

    - **unstable**: any evaluable variant flips the direction of effect.
    - **robust**: zero direction flips, at least ``min_evaluable``
      evaluable variants, AND concordance >= ``threshold``, where
      concordance is the fraction of *all* variants (failed ones included)
      that produced an estimate in the primary direction.
    - **exploratory**: otherwise (primary effect missing, too few
      evaluable variants, or too many failed variants).

    This is the rule stated in the CASCADE specification (ROBUST if
    concordance >= 0.75 and zero direction flips; UNSTABLE if any
    direction flip; EXPLORATORY otherwise).  Variants with a missing
    effect are counted as failed (``n_failed``) and lower the
    concordance, never silently dropped.
    """

    @staticmethod
    def score_detail(
        primary_result: float,
        sensitivity_results: Sequence[float],
        threshold: float = 0.75,
        min_evaluable: int = 2,
    ) -> Dict[str, Any]:
        """Score a finding and return the counts behind the classification.

        Returns
        -------
        dict
            Keys: ``robustness_class``, ``n_evaluable``, ``n_concordant``,
            ``n_failed``, ``concordance``.
        """
        n_concordant = 0
        n_evaluable = 0
        n_failed = 0
        primary_ok = primary_result is not None and pd.notna(primary_result)
        primary_direction = (1 if primary_result >= 1.0 else -1) if primary_ok else 0

        for val in sensitivity_results:
            if val is None or not pd.notna(val):
                n_failed += 1
                continue
            n_evaluable += 1
            variant_direction = 1 if val >= 1.0 else -1
            if variant_direction == primary_direction:
                n_concordant += 1

        n_total = n_evaluable + n_failed
        concordance = n_concordant / n_total if n_total else np.nan
        if not primary_ok:
            cls = "exploratory"
        elif n_concordant < n_evaluable:
            cls = "unstable"
        elif n_evaluable >= max(int(min_evaluable), 1) and concordance >= threshold:
            cls = "robust"
        else:
            cls = "exploratory"

        return {
            "robustness_class": cls,
            "n_evaluable": n_evaluable,
            "n_concordant": n_concordant,
            "n_failed": n_failed,
            "concordance": concordance,
        }

    @staticmethod
    def score(
        primary_result: float,
        sensitivity_results: Sequence[float],
        threshold: float = 0.75,
        min_evaluable: int = 2,
    ) -> str:
        """Score a single finding's robustness.

        Parameters
        ----------
        primary_result : float
            Effect size from the primary analysis (e.g. HR).
        sensitivity_results : sequence of float
            Effect sizes from sensitivity variants (NaN = failed variant).
        threshold : float, default 0.75
            Minimum concordance fraction (among all variants, failed ones
            included) for 'robust'; any direction flip gives 'unstable'.
        min_evaluable : int, default 2
            Minimum number of evaluable variants for a robust call.

        Returns
        -------
        str
            ``'robust'``, ``'exploratory'``, or ``'unstable'``.
        """
        return RobustnessScorer.score_detail(
            primary_result, sensitivity_results,
            threshold=threshold, min_evaluable=min_evaluable,
        )["robustness_class"]
