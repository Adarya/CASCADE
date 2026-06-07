"""
Gate Evaluation System
======================

Programmatic evaluation of CASCADE decision gates after each layer.
Each layer has quantitative pass/fail criteria; gate failures trigger
remediation paths rather than terminating the analysis.

Classes
-------
GateResult
    Structured result of evaluating a single layer's decision gate.
GateEvaluator
    Evaluates decision gates for all CASCADE layers with study-type
    awareness and configurable thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of evaluating a single layer's decision gate.

    Attributes
    ----------
    layer_name : str
        CASCADE layer identifier.
    layer_number : int
        Numeric layer index (0-8).
    passed : bool
        Whether the gate criteria were satisfied.
    criteria : dict
        Criterion name to threshold mapping.
    actual : dict
        Criterion name to observed value mapping.
    details : dict
        Additional context (e.g., which criteria failed).
    remediation_attempted : bool
        Whether remediation was tried after failure.
    remediation_succeeded : bool
        Whether remediation resolved the failure.
    error_message : str or None
        Description of gate failure or evaluation error.
    """

    layer_name: str
    layer_number: int
    passed: bool
    criteria: Dict[str, Any] = field(default_factory=dict)
    actual: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    remediation_attempted: bool = False
    remediation_succeeded: bool = False
    error_message: Optional[str] = None


# Default gate configurations per layer
DEFAULT_GATES: Dict[str, Dict[str, Any]] = {
    "discovery": {
        "min_fdr_significant": 1,
    },
    "confirmation": {
        "min_direction_concordance": 0.70,
        "min_both_significant": 1,
    },
    "predictive": {
        "all_classified": True,
    },
    "sensitivity": {
        "all_scored": True,
    },
    "artifact_guard": {
        "no_unresolved_critical": True,
    },
    "clinical": {
        "delta_c_positive": True,
        "delta_c_ci_excludes_zero": True,
    },
    "validation": {
        "above_random_baseline": True,
    },
}

# Study-type overrides: None means skip that layer's gate
STUDY_TYPE_OVERRIDES: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = {
    "tropism": {
        "predictive": None,
    },
    "burden": {
        "discovery": None,
        "confirmation": None,
        "predictive": None,
    },
    "predictive": {},
    "general": {},
}

# Layer number lookup
_LAYER_NUMBERS = {
    "cohort": 0,
    "discovery": 1,
    "confirmation": 2,
    "predictive": 3,
    "sensitivity": 4,
    "artifact_guard": 5,
    "clinical": 6,
    "validation": 7,
    "manuscript": 8,
}


class GateEvaluator:
    """Evaluate decision gates for CASCADE layers.

    Each layer has quantitative pass/fail criteria defined in
    ``DEFAULT_GATES``.  Study-type overrides allow skipping gates
    that don't apply (e.g., Layer 3 for tropism studies).

    Parameters
    ----------
    study_type : str, default ``'general'``
        One of: ``'tropism'``, ``'predictive'``, ``'burden'``,
        ``'general'``.  Selects study-type-specific gate overrides.
    custom_gates : dict, optional
        Override default gate criteria.  Maps layer_name to dict of
        criteria (e.g., ``{'discovery': {'min_fdr_significant': 3}}``).

    Examples
    --------
    >>> ge = GateEvaluator(study_type='tropism')
    >>> result = ge.evaluate('discovery', layer_result)
    >>> result.passed
    True
    """

    def __init__(
        self,
        study_type: str = "general",
        custom_gates: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        if study_type not in STUDY_TYPE_OVERRIDES:
            raise ValueError(
                f"Unknown study_type '{study_type}'. "
                f"Valid: {list(STUDY_TYPE_OVERRIDES.keys())}"
            )
        self.study_type = study_type
        self._gates = self._build_gate_config(custom_gates)

    def _build_gate_config(
        self, custom_gates: Optional[Dict[str, Dict[str, Any]]]
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Merge default gates with study-type overrides and custom gates."""
        gates: Dict[str, Optional[Dict[str, Any]]] = {}
        for layer_name, default_criteria in DEFAULT_GATES.items():
            gates[layer_name] = dict(default_criteria)

        overrides = STUDY_TYPE_OVERRIDES.get(self.study_type, {})
        for layer_name, override in overrides.items():
            if override is None:
                gates[layer_name] = None  # Skip this gate
            else:
                if layer_name in gates and gates[layer_name] is not None:
                    gates[layer_name].update(override)

        if custom_gates:
            for layer_name, custom in custom_gates.items():
                if custom is None:
                    gates[layer_name] = None
                elif layer_name in gates and gates[layer_name] is not None:
                    gates[layer_name].update(custom)
                else:
                    gates[layer_name] = dict(custom)

        return gates

    def evaluate(self, layer_name: str, layer_result: Any) -> GateResult:
        """Evaluate the decision gate for a completed layer.

        Parameters
        ----------
        layer_name : str
            CASCADE layer identifier (e.g., ``'discovery'``).
        layer_result : LayerResult
            Completed layer result containing ``.results``.

        Returns
        -------
        GateResult
        """
        layer_num = _LAYER_NUMBERS.get(layer_name, -1)

        gate_config = self._gates.get(layer_name)
        if gate_config is None:
            return GateResult(
                layer_name=layer_name,
                layer_number=layer_num,
                passed=True,
                details={"skipped": True, "reason": f"Gate skipped for study_type={self.study_type}"},
            )

        if not hasattr(layer_result, "results") or layer_result.results is None:
            return GateResult(
                layer_name=layer_name,
                layer_number=layer_num,
                passed=False,
                criteria=gate_config,
                error_message="Layer produced no results.",
            )

        evaluator = {
            "discovery": self._evaluate_discovery,
            "confirmation": self._evaluate_confirmation,
            "predictive": self._evaluate_predictive,
            "sensitivity": self._evaluate_sensitivity,
            "artifact_guard": self._evaluate_artifact_guard,
            "clinical": self._evaluate_clinical,
            "validation": self._evaluate_validation,
        }.get(layer_name)

        if evaluator is None:
            return GateResult(
                layer_name=layer_name,
                layer_number=layer_num,
                passed=True,
                details={"info": f"No gate defined for layer '{layer_name}'."},
            )

        try:
            return evaluator(layer_result.results, gate_config, layer_num)
        except Exception as e:
            logger.error(f"Gate evaluation error for {layer_name}: {e}")
            return GateResult(
                layer_name=layer_name,
                layer_number=layer_num,
                passed=False,
                criteria=gate_config,
                error_message=f"Gate evaluation failed: {e}",
            )

    # ------------------------------------------------------------------
    # Per-layer evaluators
    # ------------------------------------------------------------------

    def _evaluate_discovery(
        self, results: Any, config: Dict[str, Any], layer_num: int
    ) -> GateResult:
        """Layer 1: At least min_fdr_significant findings with adjusted P < 0.05."""
        min_sig = config.get("min_fdr_significant", 1)
        criteria = {"min_fdr_significant": min_sig}

        if isinstance(results, pd.DataFrame):
            if "significant" in results.columns:
                n_sig = int(results["significant"].sum())
            elif "p_adjusted" in results.columns:
                n_sig = int((results["p_adjusted"] < 0.05).sum())
            else:
                return GateResult(
                    layer_name="discovery", layer_number=layer_num,
                    passed=False, criteria=criteria,
                    error_message="Results missing 'significant' or 'p_adjusted' column.",
                )
        elif isinstance(results, dict) and "n_significant" in results:
            n_sig = results["n_significant"]
        else:
            return GateResult(
                layer_name="discovery", layer_number=layer_num,
                passed=False, criteria=criteria,
                error_message=f"Unexpected result type: {type(results).__name__}",
            )

        passed = n_sig >= min_sig
        return GateResult(
            layer_name="discovery",
            layer_number=layer_num,
            passed=passed,
            criteria=criteria,
            actual={"n_fdr_significant": n_sig},
            error_message=None if passed else f"Only {n_sig} FDR-significant (need {min_sig}).",
        )

    def _evaluate_confirmation(
        self, results: Any, config: Dict[str, Any], layer_num: int
    ) -> GateResult:
        """Layer 2: Direction concordance >= threshold AND at least 1 both-significant."""
        min_conc = config.get("min_direction_concordance", 0.70)
        min_both = config.get("min_both_significant", 1)
        criteria = {"min_direction_concordance": min_conc, "min_both_significant": min_both}
        actual: Dict[str, Any] = {}
        failures: List[str] = []

        if isinstance(results, pd.DataFrame):
            if "direction_match" in results.columns:
                actual["direction_concordance"] = float(results["direction_match"].mean())
            elif "concordance_direction" in results.columns:
                actual["direction_concordance"] = float(results["concordance_direction"].mean())
            else:
                actual["direction_concordance"] = np.nan

            if "confidence_tier" in results.columns:
                actual["n_both_significant"] = int(
                    (results["confidence_tier"] == "both_significant").sum()
                )
            elif "both_significant" in results.columns:
                actual["n_both_significant"] = int(results["both_significant"].sum())
            else:
                actual["n_both_significant"] = 0
        elif isinstance(results, dict):
            actual["direction_concordance"] = results.get("direction_concordance", np.nan)
            actual["n_both_significant"] = results.get("n_both_significant", 0)
        else:
            return GateResult(
                layer_name="confirmation", layer_number=layer_num,
                passed=False, criteria=criteria,
                error_message=f"Unexpected result type: {type(results).__name__}",
            )

        conc = actual.get("direction_concordance", np.nan)
        n_both = actual.get("n_both_significant", 0)

        if pd.notna(conc) and conc < min_conc:
            failures.append(f"Direction concordance {conc:.2f} < {min_conc:.2f}")
        if n_both < min_both:
            failures.append(f"Both-significant count {n_both} < {min_both}")

        passed = len(failures) == 0 and pd.notna(conc)
        return GateResult(
            layer_name="confirmation",
            layer_number=layer_num,
            passed=passed,
            criteria=criteria,
            actual=actual,
            error_message="; ".join(failures) if failures else None,
        )

    def _evaluate_predictive(
        self, results: Any, config: Dict[str, Any], layer_num: int
    ) -> GateResult:
        """Layer 3: Every significant biomarker classified as PREDICTIVE or PROGNOSTIC."""
        criteria = {"all_classified": True}

        if isinstance(results, pd.DataFrame) and "classification" in results.columns:
            n_total = len(results)
            n_classified = results["classification"].notna().sum()
            valid_classes = {"predictive", "prognostic", "underpowered",
                            "PREDICTIVE", "PROGNOSTIC", "UNDERPOWERED"}
            n_valid = results["classification"].isin(valid_classes).sum()
            actual = {"n_total": n_total, "n_classified": int(n_classified), "n_valid": int(n_valid)}
            passed = bool(n_classified == n_total and n_valid == n_total)
        elif isinstance(results, dict) and "all_classified" in results:
            actual = {"all_classified": results["all_classified"]}
            passed = bool(results["all_classified"])
        else:
            actual = {}
            passed = False

        return GateResult(
            layer_name="predictive",
            layer_number=layer_num,
            passed=passed,
            criteria=criteria,
            actual=actual,
            error_message=None if passed else "Not all biomarkers classified.",
        )

    def _evaluate_sensitivity(
        self, results: Any, config: Dict[str, Any], layer_num: int
    ) -> GateResult:
        """Layer 4: All findings have robustness classifications."""
        criteria = {"all_scored": True}

        if isinstance(results, pd.DataFrame) and "robustness_class" in results.columns:
            n_total = len(results)
            n_scored = results["robustness_class"].notna().sum()
            actual = {"n_total": n_total, "n_scored": int(n_scored)}
            passed = bool(n_scored == n_total)
        elif isinstance(results, dict) and "all_scored" in results:
            actual = {"all_scored": results["all_scored"]}
            passed = bool(results["all_scored"])
        else:
            actual = {}
            passed = True  # Permissive if can't check

        return GateResult(
            layer_name="sensitivity",
            layer_number=layer_num,
            passed=passed,
            criteria=criteria,
            actual=actual,
            error_message=None if passed else "Not all findings scored for robustness.",
        )

    def _evaluate_artifact_guard(
        self, results: Any, config: Dict[str, Any], layer_num: int
    ) -> GateResult:
        """Layer 5: No unresolved CRITICAL artifacts."""
        criteria = {"no_unresolved_critical": True}

        if hasattr(results, "warnings"):
            warnings_list = results.warnings if isinstance(results.warnings, list) else []
            critical_warnings = [w for w in warnings_list if "CRITICAL" in str(w).upper()]
            n_critical = len(critical_warnings)
            actual = {
                "n_warnings": len(warnings_list),
                "n_critical": n_critical,
            }
            passed = n_critical == 0
        elif hasattr(results, "has_warnings"):
            actual = {"has_warnings": results.has_warnings}
            passed = not results.has_warnings
        elif isinstance(results, dict):
            n_critical = results.get("n_critical", 0)
            actual = {"n_critical": n_critical}
            passed = n_critical == 0
        else:
            actual = {}
            passed = True

        return GateResult(
            layer_name="artifact_guard",
            layer_number=layer_num,
            passed=passed,
            criteria=criteria,
            actual=actual,
            error_message=None if passed else f"{actual.get('n_critical', '?')} critical artifact(s) unresolved.",
        )

    def _evaluate_clinical(
        self, results: Any, config: Dict[str, Any], layer_num: int
    ) -> GateResult:
        """Layer 6: Cross-validated delta-C > 0 with CI excluding zero."""
        criteria = {"delta_c_positive": True, "delta_c_ci_excludes_zero": True}
        failures: List[str] = []

        if isinstance(results, dict):
            delta_c = results.get("delta_c", results.get("mean_delta_c", np.nan))
            ci = results.get("ci", results.get("ci_95", (np.nan, np.nan)))
            if isinstance(ci, (list, tuple)) and len(ci) >= 2:
                ci_lower, ci_upper = ci[0], ci[1]
            else:
                ci_lower, ci_upper = np.nan, np.nan

            actual = {"delta_c": delta_c, "ci_lower": ci_lower, "ci_upper": ci_upper}

            if pd.notna(delta_c) and delta_c <= 0:
                failures.append(f"delta_c = {delta_c:.4f} <= 0")
            if pd.notna(ci_lower) and ci_lower <= 0:
                failures.append(f"CI lower = {ci_lower:.4f} includes zero")
        else:
            actual = {}
            failures.append(f"Unexpected result type: {type(results).__name__}")

        passed = len(failures) == 0 and bool(actual)
        return GateResult(
            layer_name="clinical",
            layer_number=layer_num,
            passed=passed,
            criteria=criteria,
            actual=actual,
            error_message="; ".join(failures) if failures else None,
        )

    def _evaluate_validation(
        self, results: Any, config: Dict[str, Any], layer_num: int
    ) -> GateResult:
        """Layer 7: Balanced accuracy above random baseline."""
        criteria = {"above_random_baseline": True}

        if isinstance(results, dict):
            bal_acc = results.get("balanced_accuracy", np.nan)
            n_classes = results.get("n_classes", 2)
            random_baseline = 1.0 / n_classes
            actual = {"balanced_accuracy": bal_acc, "random_baseline": random_baseline}
            passed = pd.notna(bal_acc) and bal_acc > random_baseline
        else:
            actual = {}
            passed = False

        return GateResult(
            layer_name="validation",
            layer_number=layer_num,
            passed=passed,
            criteria=criteria,
            actual=actual,
            error_message=None if passed else "Performance at or below random baseline.",
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, gate_results: List[GateResult]) -> str:
        """Generate markdown summary of all gate evaluations.

        Parameters
        ----------
        gate_results : list of GateResult

        Returns
        -------
        str
            Markdown-formatted gate summary table.
        """
        lines = [
            "## CASCADE Gate Evaluation Summary",
            "",
            "| Layer | Name | Status | Details |",
            "|-------|------|--------|---------|",
        ]

        n_passed = 0
        n_evaluated = 0

        for gr in sorted(gate_results, key=lambda g: g.layer_number):
            if gr.details.get("skipped"):
                status = "SKIP"
            elif gr.passed:
                status = "PASS"
                n_passed += 1
                n_evaluated += 1
            else:
                status = "FAIL"
                n_evaluated += 1

            if gr.remediation_attempted:
                status += " (remediated)" if gr.remediation_succeeded else " (remediation failed)"

            detail = gr.error_message or ""
            if not detail and gr.actual:
                detail = ", ".join(f"{k}={v}" for k, v in gr.actual.items())

            lines.append(
                f"| {gr.layer_number} | {gr.layer_name} | {status} | {detail} |"
            )

        lines.append("")
        if n_evaluated > 0:
            lines.append(f"**Result: {n_passed}/{n_evaluated} gates passed.**")
        else:
            lines.append("**No gates evaluated.**")
        lines.append("")

        return "\n".join(lines)
