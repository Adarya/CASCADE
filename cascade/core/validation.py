"""
Layer 7: External Validation
==============================

Tools for validating a model or classification trained on one dataset
against an independent external dataset: feature alignment, performance
evaluation, hierarchical grouping, and calibrated consensus.

Classes
-------
ExternalValidator
    Evaluates a trained model on external data with feature alignment
    and comprehensive metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


class ExternalValidator:
    """Validate a trained model on an external dataset.

    Handles the practical challenges of external validation: mismatched
    feature sets, performance evaluation at multiple levels of
    granularity, and consensus prediction from multiple models.

    Parameters
    ----------
    model : object, optional
        A fitted model with a ``predict`` and optionally ``predict_proba``
        method.  If ``None``, evaluation methods can still be used directly.
    """

    def __init__(self, model: Optional[Any] = None) -> None:
        self.model = model

    # ------------------------------------------------------------------
    # Feature alignment
    # ------------------------------------------------------------------

    @staticmethod
    def align_features(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        fill_value: float = 0.0,
    ) -> pd.DataFrame:
        """Align test features to match training feature columns.

        Adds missing columns (filled with *fill_value*) and drops extra
        columns so that X_test has exactly the same columns in the same
        order as X_train.

        Parameters
        ----------
        X_train : DataFrame
            Training feature matrix (defines the expected columns).
        X_test : DataFrame
            Test feature matrix (may have missing or extra columns).
        fill_value : float, default 0.0
            Value to fill in for missing columns.

        Returns
        -------
        DataFrame
            Aligned test matrix with the same columns as X_train.
        """
        train_cols = X_train.columns.tolist()
        result = X_test.copy()

        # Add missing columns
        for col in train_cols:
            if col not in result.columns:
                result[col] = fill_value

        # Reorder and drop extra columns
        result = result[train_cols]
        return result

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        y_proba: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Compute comprehensive classification metrics.

        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted labels.
        y_proba : array-like, optional
            Predicted class probabilities (n_samples x n_classes).
            Used for confidence-weighted metrics if provided.

        Returns
        -------
        dict
            Keys: ``balanced_accuracy``, ``kappa``, ``per_class_f1``
            (dict), ``confusion_matrix`` (ndarray), ``n_samples``,
            ``n_classes``.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Remove NaN
        valid = ~(pd.isna(y_true) | pd.isna(y_pred))
        y_true = y_true[valid]
        y_pred = y_pred[valid]

        if len(y_true) == 0:
            return dict(
                balanced_accuracy=np.nan, kappa=np.nan,
                per_class_f1={}, confusion_matrix=np.array([]),
                n_samples=0, n_classes=0,
            )

        labels = sorted(set(y_true) | set(y_pred))
        ba = balanced_accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Per-class F1
        f1_per_class: Dict[str, float] = {}
        for lbl in labels:
            binary_true = (y_true == lbl).astype(int)
            binary_pred = (y_pred == lbl).astype(int)
            if binary_true.sum() == 0 and binary_pred.sum() == 0:
                f1_per_class[str(lbl)] = np.nan
            else:
                f1_per_class[str(lbl)] = float(
                    f1_score(binary_true, binary_pred, zero_division=0)
                )

        return dict(
            balanced_accuracy=float(ba),
            kappa=float(kappa),
            per_class_f1=f1_per_class,
            confusion_matrix=cm,
            n_samples=int(len(y_true)),
            n_classes=len(labels),
        )

    # ------------------------------------------------------------------
    # Hierarchical evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def hierarchical_evaluate(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        group_map: Dict[str, str],
    ) -> Dict[str, Any]:
        """Evaluate at a coarser group level.

        Maps fine-grained classes to broader groups (e.g. molecular
        subtypes to prognosis groups) and evaluates at the group level.

        Parameters
        ----------
        y_true : array-like
            True fine-grained labels.
        y_pred : array-like
            Predicted fine-grained labels.
        group_map : dict
            Mapping from fine label (str) to group label (str).

        Returns
        -------
        dict
            Keys: ``group_balanced_accuracy``, ``group_kappa``,
            ``group_confusion_matrix``, ``per_group_f1``,
            ``unmapped_true``, ``unmapped_pred``.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Map to groups
        def _map(val: Any) -> str:
            return group_map.get(str(val), str(val))

        y_true_grp = np.array([_map(v) for v in y_true])
        y_pred_grp = np.array([_map(v) for v in y_pred])

        # Track unmapped labels
        unmapped_true = [str(v) for v in y_true if str(v) not in group_map]
        unmapped_pred = [str(v) for v in y_pred if str(v) not in group_map]

        # Remove NaN
        valid = ~(pd.isna(y_true_grp) | pd.isna(y_pred_grp))
        y_true_grp = y_true_grp[valid]
        y_pred_grp = y_pred_grp[valid]

        if len(y_true_grp) == 0:
            return dict(
                group_balanced_accuracy=np.nan, group_kappa=np.nan,
                group_confusion_matrix=np.array([]),
                per_group_f1={}, unmapped_true=[], unmapped_pred=[],
            )

        labels = sorted(set(y_true_grp) | set(y_pred_grp))
        ba = balanced_accuracy_score(y_true_grp, y_pred_grp)
        kappa = cohen_kappa_score(y_true_grp, y_pred_grp)
        cm = confusion_matrix(y_true_grp, y_pred_grp, labels=labels)

        f1_per_group: Dict[str, float] = {}
        for lbl in labels:
            bt = (y_true_grp == lbl).astype(int)
            bp = (y_pred_grp == lbl).astype(int)
            f1_per_group[str(lbl)] = float(
                f1_score(bt, bp, zero_division=0)
            )

        return dict(
            group_balanced_accuracy=float(ba),
            group_kappa=float(kappa),
            group_confusion_matrix=cm,
            per_group_f1=f1_per_group,
            unmapped_true=sorted(set(unmapped_true)),
            unmapped_pred=sorted(set(unmapped_pred)),
        )

    # ------------------------------------------------------------------
    # Calibrated consensus
    # ------------------------------------------------------------------

    @staticmethod
    def calibrated_consensus(
        predictions_list: List[pd.DataFrame],
        weights: List[float],
        confidence_threshold: float = 0.6,
        class_col: str = "predicted_class",
        proba_prefix: str = "proba_",
    ) -> pd.DataFrame:
        """Combine predictions from multiple models via weighted averaging.

        Parameters
        ----------
        predictions_list : list of DataFrame
            Each DataFrame has a *class_col* column and probability
            columns (named ``proba_<class_label>``).  All DataFrames
            must share the same index.
        weights : list of float
            Model weights (will be normalised to sum to 1).
        confidence_threshold : float, default 0.6
            Minimum weighted probability for 'High' confidence.
            Below this: 'Low' if < 0.4, else 'Moderate'.
        class_col : str, default 'predicted_class'
        proba_prefix : str, default 'proba_'

        Returns
        -------
        DataFrame
            Columns: ``consensus_pred``, ``consensus_max_proba``,
            ``confidence_tier`` ('High', 'Moderate', 'Low').
        """
        if not predictions_list:
            return pd.DataFrame()

        w = np.array(weights, dtype=float)
        w = w / w.sum()

        # Identify probability columns from first DataFrame
        first = predictions_list[0]
        proba_cols = [c for c in first.columns if c.startswith(proba_prefix)]

        if not proba_cols:
            raise ValueError(
                f"No probability columns found with prefix '{proba_prefix}'."
            )

        # Weighted average of probabilities
        avg_proba = pd.DataFrame(0.0, index=first.index, columns=proba_cols)
        for pred_df, weight in zip(predictions_list, w):
            for col in proba_cols:
                if col in pred_df.columns:
                    avg_proba[col] += pred_df[col].values * weight

        # Consensus prediction = class with highest averaged probability
        consensus_pred = avg_proba.idxmax(axis=1).str.replace(proba_prefix, "", regex=False)
        consensus_max_proba = avg_proba.max(axis=1)

        # Confidence tiers
        def _tier(p: float) -> str:
            if p >= confidence_threshold:
                return "High"
            elif p < 0.4:
                return "Low"
            else:
                return "Moderate"

        confidence_tier = consensus_max_proba.apply(_tier)

        return pd.DataFrame({
            "consensus_pred": consensus_pred,
            "consensus_max_proba": consensus_max_proba,
            "confidence_tier": confidence_tier,
        }, index=first.index)
