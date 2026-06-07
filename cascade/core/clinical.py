"""
Layer 6: Clinical Translation Assessment
==========================================

Tools for landmark survival analysis, discriminative ability assessment
(Delta-C), and risk-group stratification.

Classes
-------
LandmarkAnalysis
    Restricts cohorts to patients alive at a landmark and computes
    residual survival and trajectory features.
DeltaC
    Cross-validated concordance index comparison between a base model
    and a model augmented with candidate features.
RiskGroupAnalysis
    Creates risk groups from a continuous score and evaluates their
    separation via Kaplan-Meier and log-rank tests.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

try:
    from cascade.stats.cross_validate import cv_concordance, cv_delta_c
except ImportError:
    cv_concordance = None  # type: ignore[assignment,misc]
    cv_delta_c = None  # type: ignore[assignment,misc]


class LandmarkAnalysis:
    """Landmark survival analysis utilities.

    Implements the landmark approach: restrict to patients alive at a
    fixed time point, then analyse residual survival from that point.
    This avoids immortal-time bias when using time-varying features.

    Parameters
    ----------
    landmarks_months : list of int, default [3, 6, 12]
        Landmark time points in months.
    """

    def __init__(
        self,
        landmarks_months: Optional[List[int]] = None,
    ) -> None:
        self.landmarks_months = landmarks_months or [3, 6, 12]

    # ------------------------------------------------------------------
    # Restriction
    # ------------------------------------------------------------------

    @staticmethod
    def restrict_to_alive(
        df: pd.DataFrame,
        os_col: str,
        landmark_months: float,
    ) -> pd.DataFrame:
        """Restrict to patients alive at the landmark time.

        Parameters
        ----------
        df : DataFrame
            Must contain *os_col* (overall survival in months).
        os_col : str
            Column with OS duration in months.
        landmark_months : float
            Landmark time in months.

        Returns
        -------
        DataFrame
            Rows where os_col >= landmark_months.
        """
        return df.loc[df[os_col] >= landmark_months].copy()

    # ------------------------------------------------------------------
    # Residual survival
    # ------------------------------------------------------------------

    @staticmethod
    def compute_residual_survival(
        df: pd.DataFrame,
        os_col: str,
        landmark_months: float,
        os_from_lm_col: str = "OS_FROM_LM",
    ) -> pd.DataFrame:
        """Compute residual survival from the landmark.

        Parameters
        ----------
        df : DataFrame
        os_col : str
        landmark_months : float
        os_from_lm_col : str, default 'OS_FROM_LM'
            Name for the new column.

        Returns
        -------
        DataFrame
            Copy with *os_from_lm_col* added.
        """
        result = df.copy()
        result[os_from_lm_col] = result[os_col] - landmark_months
        # Floor at zero
        result[os_from_lm_col] = result[os_from_lm_col].clip(lower=0)
        return result

    # ------------------------------------------------------------------
    # Trajectory at landmark
    # ------------------------------------------------------------------

    @staticmethod
    def compute_trajectory_at_landmark(
        longitudinal_df: pd.DataFrame,
        patient_col: str,
        date_col: str,
        value_col: str,
        landmark_days: float,
    ) -> pd.DataFrame:
        """Compute per-patient trajectory features at a landmark.

        For each patient, uses all observations up to *landmark_days*
        to compute baseline, current value, slope, and a trajectory
        label.

        Parameters
        ----------
        longitudinal_df : DataFrame
            Longitudinal measurements with *patient_col*, *date_col*,
            *value_col*.
        patient_col : str
        date_col : str
            Measurement date in days from a reference.
        value_col : str
        landmark_days : float
            Landmark time in days.

        Returns
        -------
        DataFrame
            One row per patient with columns: *patient_col*,
            ``baseline``, ``current``, ``slope``, ``trajectory_label``.
        """
        records: List[Dict[str, Any]] = []

        for pid, grp in longitudinal_df.groupby(patient_col):
            eligible = grp.loc[grp[date_col] <= landmark_days].sort_values(date_col)

            if eligible.empty:
                records.append({
                    patient_col: pid,
                    "baseline": np.nan,
                    "current": np.nan,
                    "slope": np.nan,
                    "trajectory_label": "insufficient_data",
                })
                continue

            baseline = eligible[value_col].iloc[0]
            current = eligible[value_col].iloc[-1]

            # Slope via linear regression if >= 2 points
            if len(eligible) >= 2:
                x = eligible[date_col].values.astype(float)
                y = eligible[value_col].values.astype(float)
                # Remove NaN pairs
                valid = ~(np.isnan(x) | np.isnan(y))
                if valid.sum() >= 2:
                    slope_val, _, _, _, _ = sp_stats.linregress(x[valid], y[valid])
                else:
                    slope_val = np.nan
            else:
                slope_val = np.nan

            # Classify trajectory
            if np.isnan(slope_val):
                label = "insufficient_data"
            elif slope_val > 0.01:
                label = "worsening"
            elif slope_val < -0.01:
                label = "improving"
            else:
                label = "stable"

            records.append({
                patient_col: pid,
                "baseline": baseline,
                "current": current,
                "slope": slope_val,
                "trajectory_label": label,
            })

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Convenience: run all landmarks
    # ------------------------------------------------------------------

    def run_all_landmarks(
        self,
        df: pd.DataFrame,
        os_col: str,
        event_col: str,
    ) -> Dict[int, pd.DataFrame]:
        """Restrict and compute residual survival at each landmark.

        Parameters
        ----------
        df : DataFrame
        os_col : str
        event_col : str

        Returns
        -------
        dict
            Mapping of landmark months to the restricted DataFrame.
        """
        results: Dict[int, pd.DataFrame] = {}
        for lm in self.landmarks_months:
            restricted = self.restrict_to_alive(df, os_col, lm)
            restricted = self.compute_residual_survival(restricted, os_col, lm)
            results[lm] = restricted
        return results


class DeltaC:
    """Cross-validated concordance index comparison.

    Computes the improvement in Harrell's C-index when candidate features
    are added to a base model (Delta-C), using K-fold cross-validation
    and bootstrap confidence intervals.

    Parameters
    ----------
    n_folds : int, default 10
        Number of CV folds.
    n_bootstrap : int, default 1000
        Number of bootstrap iterations for the CI.
    penalizer : float, default 0.01
        Ridge penalty for Cox models.
    seed : int, default 42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_folds: int = 10,
        n_bootstrap: int = 1000,
        penalizer: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.n_folds = n_folds
        self.n_bootstrap = n_bootstrap
        self.penalizer = penalizer
        self.seed = seed

    def compute(
        self,
        df: pd.DataFrame,
        base_vars: Sequence[str],
        full_vars: Sequence[str],
        duration_col: str,
        event_col: str,
    ) -> Dict[str, Any]:
        """Compute cross-validated Delta-C.

        Parameters
        ----------
        df : DataFrame
        base_vars : sequence of str
            Covariates in the base (reference) model.
        full_vars : sequence of str
            Covariates in the full (augmented) model.  Must be a
            superset of *base_vars*.
        duration_col : str
        event_col : str

        Returns
        -------
        dict
            Keys: ``delta_c``, ``ci`` (tuple of lower, upper),
            ``p_value``, ``base_c``, ``full_c``.
        """
        # Try CASCADE stats first
        if cv_delta_c is not None:
            try:
                return cv_delta_c(
                    df=df,
                    base_vars=list(base_vars),
                    full_vars=list(full_vars),
                    duration_col=duration_col,
                    event_col=event_col,
                    n_folds=self.n_folds,
                    n_bootstrap=self.n_bootstrap,
                    penalizer=self.penalizer,
                    seed=self.seed,
                )
            except Exception:
                pass

        # Fallback: manual K-fold CV
        return self._compute_manual(
            df, list(base_vars), list(full_vars), duration_col, event_col
        )

    def _compute_manual(
        self,
        df: pd.DataFrame,
        base_vars: List[str],
        full_vars: List[str],
        duration_col: str,
        event_col: str,
    ) -> Dict[str, Any]:
        """Manual cross-validated Delta-C computation."""
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index
        from sklearn.model_selection import KFold

        rng = np.random.RandomState(self.seed)
        all_cols = list(set([duration_col, event_col] + full_vars))
        clean = df[all_cols].dropna()

        if len(clean) < self.n_folds * 2:
            return dict(
                delta_c=np.nan, ci=(np.nan, np.nan),
                p_value=np.nan, base_c=np.nan, full_c=np.nan,
            )

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=rng.randint(1e6))

        base_cs: List[float] = []
        full_cs: List[float] = []

        for train_idx, test_idx in kf.split(clean):
            train = clean.iloc[train_idx]
            test = clean.iloc[test_idx]

            for var_set, c_list in [(base_vars, base_cs), (full_vars, full_cs)]:
                # Drop constant cols
                fit_vars = [v for v in var_set if train[v].nunique() > 1]
                if not fit_vars:
                    c_list.append(0.5)
                    continue

                try:
                    fitter = CoxPHFitter(penalizer=self.penalizer)
                    fitter.fit(
                        train[[duration_col, event_col] + fit_vars],
                        duration_col=duration_col,
                        event_col=event_col,
                        show_progress=False,
                    )
                    preds = fitter.predict_partial_hazard(test[fit_vars])
                    c = concordance_index(
                        test[duration_col], -preds.values.ravel(), test[event_col]
                    )
                    c_list.append(c)
                except Exception:
                    c_list.append(0.5)

        base_c = float(np.mean(base_cs))
        full_c = float(np.mean(full_cs))
        delta_c = full_c - base_c

        # Bootstrap CI for Delta-C
        deltas: List[float] = []
        for _ in range(self.n_bootstrap):
            idx = rng.choice(len(base_cs), size=len(base_cs), replace=True)
            b_base = np.mean([base_cs[i] for i in idx])
            b_full = np.mean([full_cs[i] for i in idx])
            deltas.append(b_full - b_base)

        ci = (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))

        # Approximate p-value
        p_value = float((np.array(deltas) <= 0).sum() / len(deltas))
        if p_value == 0:
            p_value = 1.0 / (len(deltas) + 1)

        return dict(
            delta_c=delta_c,
            ci=ci,
            p_value=p_value,
            base_c=base_c,
            full_c=full_c,
        )


class RiskGroupAnalysis:
    """Create and evaluate risk groups from a continuous score.

    Parameters
    ----------
    n_groups : int, default 3
        Number of risk groups (e.g. 3 for tertiles).
    method : str, default 'quantile'
        Grouping method: ``'quantile'`` or ``'equal_width'``.
    """

    def __init__(
        self,
        n_groups: int = 3,
        method: str = "quantile",
    ) -> None:
        self.n_groups = n_groups
        self.method = method

    # ------------------------------------------------------------------
    # Group creation
    # ------------------------------------------------------------------

    def create_groups(
        self,
        df: pd.DataFrame,
        score_col: str,
        group_col: str = "risk_group",
    ) -> pd.DataFrame:
        """Assign risk groups based on a continuous score.

        Parameters
        ----------
        df : DataFrame
        score_col : str
            Column with the continuous risk score.
        group_col : str, default 'risk_group'
            Name for the new group column.

        Returns
        -------
        DataFrame
            Copy with *group_col* added (labels: 1 to n_groups,
            with 1 = lowest risk).
        """
        result = df.copy()
        valid = result[score_col].dropna()

        if len(valid) < self.n_groups:
            result[group_col] = np.nan
            return result

        if self.method == "quantile":
            result[group_col] = pd.qcut(
                result[score_col],
                q=self.n_groups,
                labels=list(range(1, self.n_groups + 1)),
                duplicates="drop",
            )
        elif self.method == "equal_width":
            result[group_col] = pd.cut(
                result[score_col],
                bins=self.n_groups,
                labels=list(range(1, self.n_groups + 1)),
            )
        else:
            raise ValueError(f"method must be 'quantile' or 'equal_width', got '{self.method}'")

        return result

    # ------------------------------------------------------------------
    # KM by group
    # ------------------------------------------------------------------

    @staticmethod
    def km_by_group(
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        group_col: str,
    ) -> Dict[str, Any]:
        """Fit Kaplan-Meier curves per risk group.

        Parameters
        ----------
        df : DataFrame
        duration_col, event_col, group_col : str

        Returns
        -------
        dict
            Keys: ``group_medians`` (dict of group -> median survival),
            ``group_n`` (dict of group -> sample size),
            ``kmf_results`` (dict of group -> KaplanMeierFitter).
        """
        from lifelines import KaplanMeierFitter

        group_medians: Dict[Any, float] = {}
        group_n: Dict[Any, int] = {}
        kmf_results: Dict[Any, Any] = {}

        for grp_name, grp_data in df.groupby(group_col):
            grp_clean = grp_data.dropna(subset=[duration_col, event_col])
            if grp_clean.empty:
                continue

            kmf = KaplanMeierFitter()
            kmf.fit(
                grp_clean[duration_col],
                event_observed=grp_clean[event_col],
                label=str(grp_name),
            )
            group_medians[grp_name] = kmf.median_survival_time_
            group_n[grp_name] = len(grp_clean)
            kmf_results[grp_name] = kmf

        return dict(
            group_medians=group_medians,
            group_n=group_n,
            kmf_results=kmf_results,
        )

    # ------------------------------------------------------------------
    # Pairwise log-rank
    # ------------------------------------------------------------------

    @staticmethod
    def pairwise_logrank(
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        group_col: str,
    ) -> pd.DataFrame:
        """Run pairwise log-rank tests between risk groups.

        Parameters
        ----------
        df : DataFrame
        duration_col, event_col, group_col : str

        Returns
        -------
        DataFrame
            Columns: ``group_a``, ``group_b``, ``test_statistic``,
            ``p_value``.
        """
        from lifelines.statistics import logrank_test

        groups = sorted(df[group_col].dropna().unique())
        records: List[Dict[str, Any]] = []

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                ga = df.loc[df[group_col] == groups[i]].dropna(subset=[duration_col, event_col])
                gb = df.loc[df[group_col] == groups[j]].dropna(subset=[duration_col, event_col])

                if ga.empty or gb.empty:
                    continue

                try:
                    result = logrank_test(
                        ga[duration_col], gb[duration_col],
                        event_observed_A=ga[event_col],
                        event_observed_B=gb[event_col],
                    )
                    records.append({
                        "group_a": groups[i],
                        "group_b": groups[j],
                        "test_statistic": result.test_statistic,
                        "p_value": result.p_value,
                    })
                except Exception:
                    records.append({
                        "group_a": groups[i],
                        "group_b": groups[j],
                        "test_statistic": np.nan,
                        "p_value": np.nan,
                    })

        return pd.DataFrame(records)
