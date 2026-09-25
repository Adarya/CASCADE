"""
Tests for CASCADE gate evaluation system (cascade.core.gates).

Covers: GateResult creation, GateEvaluator instantiation, per-layer
gate evaluation (discovery, confirmation, predictive, sensitivity,
artifact_guard, clinical, validation), study-type overrides, and
Pipeline integration.
"""

import numpy as np
import pandas as pd
import pytest

from cascade.core.gates import GateEvaluator, GateResult
from cascade.core.artifact_guard import ArtifactReport
from cascade.pipeline import Pipeline, LayerResult


# ---------------------------------------------------------------
# GateResult
# ---------------------------------------------------------------

class TestGateResult:
    def test_creation(self):
        gr = GateResult(layer_name="discovery", layer_number=1, passed=True)
        assert gr.passed is True
        assert gr.layer_name == "discovery"
        assert gr.remediation_attempted is False
        assert gr.remediation_succeeded is False
        assert gr.error_message is None

    def test_defaults(self):
        gr = GateResult(layer_name="x", layer_number=0, passed=False)
        assert gr.criteria == {}
        assert gr.actual == {}
        assert gr.details == {}


# ---------------------------------------------------------------
# GateEvaluator instantiation
# ---------------------------------------------------------------

class TestGateEvaluatorInit:
    def test_default(self):
        ge = GateEvaluator()
        assert ge.study_type == "general"

    def test_tropism(self):
        ge = GateEvaluator(study_type="tropism")
        assert ge.study_type == "tropism"
        # Predictive gate should be skipped
        assert ge._gates.get("predictive") is None

    def test_burden(self):
        ge = GateEvaluator(study_type="burden")
        assert ge._gates.get("discovery") is None
        assert ge._gates.get("confirmation") is None
        assert ge._gates.get("predictive") is None

    def test_custom_gates(self):
        ge = GateEvaluator(custom_gates={"discovery": {"min_fdr_significant": 5}})
        assert ge._gates["discovery"]["min_fdr_significant"] == 5

    def test_invalid_study_type(self):
        with pytest.raises(ValueError, match="Unknown study_type"):
            GateEvaluator(study_type="nonexistent")


# ---------------------------------------------------------------
# Discovery gate (Layer 1)
# ---------------------------------------------------------------

class TestDiscoveryGate:
    def test_pass_with_significant(self):
        results = pd.DataFrame({
            "biomarker": ["A", "B", "C"],
            "p_adjusted": [0.01, 0.15, 0.80],
            "significant": [True, False, False],
        })
        lr = LayerResult("discovery", 1, "completed", results=results)
        gate = GateEvaluator().evaluate("discovery", lr)
        assert gate.passed is True
        assert gate.actual["n_fdr_significant"] == 1

    def test_fail_no_significant(self):
        results = pd.DataFrame({
            "biomarker": ["A", "B"],
            "p_adjusted": [0.15, 0.80],
            "significant": [False, False],
        })
        lr = LayerResult("discovery", 1, "completed", results=results)
        gate = GateEvaluator().evaluate("discovery", lr)
        assert gate.passed is False
        assert "0" in gate.error_message

    def test_pass_via_p_adjusted_column(self):
        results = pd.DataFrame({
            "biomarker": ["A", "B"],
            "p_adjusted": [0.03, 0.80],
        })
        lr = LayerResult("discovery", 1, "completed", results=results)
        gate = GateEvaluator().evaluate("discovery", lr)
        assert gate.passed is True

    def test_custom_threshold(self):
        results = pd.DataFrame({
            "biomarker": ["A", "B", "C"],
            "significant": [True, True, False],
        })
        lr = LayerResult("discovery", 1, "completed", results=results)
        ge = GateEvaluator(custom_gates={"discovery": {"min_fdr_significant": 3}})
        gate = ge.evaluate("discovery", lr)
        assert gate.passed is False  # Only 2, need 3

    def test_dict_results(self):
        lr = LayerResult("discovery", 1, "completed", results={"n_significant": 5})
        gate = GateEvaluator().evaluate("discovery", lr)
        assert gate.passed is True

    def test_no_results(self):
        lr = LayerResult("discovery", 1, "completed", results=None)
        gate = GateEvaluator().evaluate("discovery", lr)
        assert gate.passed is False


# ---------------------------------------------------------------
# Confirmation gate (Layer 2)
# ---------------------------------------------------------------

class TestConfirmationGate:
    def test_pass(self):
        results = pd.DataFrame({
            "biomarker": ["A", "B", "C", "D"],
            "direction_match": [True, True, True, False],
            "confidence_tier": ["both_significant", "primary_only", "primary_only", "neither"],
        })
        lr = LayerResult("confirmation", 2, "completed", results=results)
        gate = GateEvaluator().evaluate("confirmation", lr)
        assert gate.passed is True
        assert gate.actual["direction_concordance"] == 0.75
        assert gate.actual["n_both_significant"] == 1

    def test_fail_low_concordance(self):
        results = pd.DataFrame({
            "biomarker": ["A", "B", "C", "D"],
            "direction_match": [True, False, False, False],
            "confidence_tier": ["both_significant", "neither", "neither", "neither"],
        })
        lr = LayerResult("confirmation", 2, "completed", results=results)
        gate = GateEvaluator().evaluate("confirmation", lr)
        assert gate.passed is False
        assert "concordance" in gate.error_message.lower()

    def test_fail_no_both_significant(self):
        results = pd.DataFrame({
            "biomarker": ["A", "B", "C"],
            "direction_match": [True, True, True],
            "confidence_tier": ["primary_only", "primary_only", "primary_only"],
        })
        lr = LayerResult("confirmation", 2, "completed", results=results)
        gate = GateEvaluator().evaluate("confirmation", lr)
        assert gate.passed is False
        assert "both-significant" in gate.error_message.lower()

    def test_dict_results(self):
        lr = LayerResult("confirmation", 2, "completed", results={
            "direction_concordance": 0.80,
            "n_both_significant": 3,
        })
        gate = GateEvaluator().evaluate("confirmation", lr)
        assert gate.passed is True


# ---------------------------------------------------------------
# Predictive gate (Layer 3)
# ---------------------------------------------------------------

class TestPredictiveGate:
    def test_pass_all_classified(self):
        results = pd.DataFrame({
            "biomarker": ["TP53", "RB1"],
            "classification": ["PROGNOSTIC", "PREDICTIVE"],
        })
        lr = LayerResult("predictive", 3, "completed", results=results)
        gate = GateEvaluator().evaluate("predictive", lr)
        assert gate.passed is True

    def test_fail_missing_classification(self):
        results = pd.DataFrame({
            "biomarker": ["TP53", "RB1"],
            "classification": ["PROGNOSTIC", np.nan],
        })
        lr = LayerResult("predictive", 3, "completed", results=results)
        gate = GateEvaluator().evaluate("predictive", lr)
        assert gate.passed is False

    def test_skipped_for_tropism(self):
        lr = LayerResult("predictive", 3, "completed", results=pd.DataFrame())
        gate = GateEvaluator(study_type="tropism").evaluate("predictive", lr)
        assert gate.passed is True
        assert gate.details.get("skipped") is True


# ---------------------------------------------------------------
# Sensitivity gate (Layer 4)
# ---------------------------------------------------------------

class TestSensitivityGate:
    def test_pass(self):
        results = pd.DataFrame({
            "biomarker": ["A", "B"],
            "robustness_class": ["ROBUST", "EXPLORATORY"],
        })
        lr = LayerResult("sensitivity", 4, "completed", results=results)
        gate = GateEvaluator().evaluate("sensitivity", lr)
        assert gate.passed is True

    def test_fail_unscored(self):
        results = pd.DataFrame({
            "biomarker": ["A", "B"],
            "robustness_class": ["ROBUST", np.nan],
        })
        lr = LayerResult("sensitivity", 4, "completed", results=results)
        gate = GateEvaluator().evaluate("sensitivity", lr)
        assert gate.passed is False


# ---------------------------------------------------------------
# Artifact Guard gate (Layer 5)
# ---------------------------------------------------------------

class TestArtifactGuardGate:
    def test_pass_no_warnings(self):
        # A check must actually have run for the gate to pass (fail-closed).
        report = ArtifactReport(checks_run=["leakage"])
        lr = LayerResult("artifact_guard", 5, "completed", results=report)
        gate = GateEvaluator().evaluate("artifact_guard", lr)
        assert gate.passed is True

    def test_pass_noncritical_warning(self):
        report = ArtifactReport(
            warnings=["WARNING: Minor issue detected"], checks_run=["floor_ceiling"]
        )
        lr = LayerResult("artifact_guard", 5, "completed", results=report)
        gate = GateEvaluator().evaluate("artifact_guard", lr)
        assert gate.passed is True

    def test_fail_critical_warning(self):
        report = ArtifactReport(
            warnings=["CRITICAL: Covariate leakage in RECEIVED_ICI"]
        )
        lr = LayerResult("artifact_guard", 5, "completed", results=report)
        gate = GateEvaluator().evaluate("artifact_guard", lr)
        assert gate.passed is False
        assert gate.actual["n_critical"] == 1

    def test_dict_results(self):
        lr = LayerResult("artifact_guard", 5, "completed", results={"n_critical": 0})
        gate = GateEvaluator().evaluate("artifact_guard", lr)
        assert gate.passed is True


# ---------------------------------------------------------------
# Clinical gate (Layer 6)
# ---------------------------------------------------------------

class TestClinicalGate:
    def test_pass(self):
        lr = LayerResult("clinical", 6, "completed", results={
            "delta_c": 0.033,
            "ci": (0.018, 0.048),
        })
        gate = GateEvaluator().evaluate("clinical", lr)
        assert gate.passed is True

    def test_fail_negative_delta_c(self):
        lr = LayerResult("clinical", 6, "completed", results={
            "delta_c": -0.005,
            "ci": (-0.02, 0.01),
        })
        gate = GateEvaluator().evaluate("clinical", lr)
        assert gate.passed is False

    def test_fail_ci_includes_zero(self):
        lr = LayerResult("clinical", 6, "completed", results={
            "delta_c": 0.01,
            "ci": (-0.005, 0.025),
        })
        gate = GateEvaluator().evaluate("clinical", lr)
        assert gate.passed is False


# ---------------------------------------------------------------
# Validation gate (Layer 7)
# ---------------------------------------------------------------

class TestValidationGate:
    def test_pass(self):
        lr = LayerResult("validation", 7, "completed", results={
            "balanced_accuracy": 0.65,
            "n_classes": 2,
        })
        gate = GateEvaluator().evaluate("validation", lr)
        assert gate.passed is True

    def test_fail_below_random(self):
        lr = LayerResult("validation", 7, "completed", results={
            "balanced_accuracy": 0.09,
            "n_classes": 10,
        })
        gate = GateEvaluator().evaluate("validation", lr)
        assert gate.passed is False


# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------

class TestGateSummary:
    def test_summary_markdown(self):
        ge = GateEvaluator()
        gate_results = [
            GateResult("discovery", 1, True, actual={"n_fdr_significant": 5}),
            GateResult("confirmation", 2, False, error_message="Low concordance"),
        ]
        summary = ge.summary(gate_results)
        assert "PASS" in summary
        assert "FAIL" in summary
        assert "1/2" in summary

    def test_summary_empty(self):
        ge = GateEvaluator()
        summary = ge.summary([])
        assert "No gates evaluated" in summary


# ---------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------

class TestGatePipelineIntegration:
    def test_pipeline_with_gates(self, survival_df, biomarker_cols, covariates):
        from cascade.core import BiomarkerScreen

        ge = GateEvaluator()
        pipeline = Pipeline(gate_evaluator=ge)
        screen = BiomarkerScreen(
            method="cox", correction="fdr_tsbh",
            min_exposed=10, min_events=3,
        )
        pipeline.add_layer("discovery", screen)

        result = pipeline.run(
            data=survival_df,
            biomarker_cols=biomarker_cols,
            outcome_col="OS_MONTHS",
            event_col="OS_STATUS",
            covariates=covariates,
        )
        lr = result.get("discovery")
        assert lr.succeeded
        assert "gate" in lr.metadata
        assert isinstance(lr.metadata["gate"], GateResult)

    def test_pipeline_without_gates(self, survival_df, biomarker_cols, covariates):
        from cascade.core import BiomarkerScreen

        pipeline = Pipeline(gate_evaluator=False)  # gates explicitly off
        screen = BiomarkerScreen(
            method="cox", correction="fdr_tsbh",
            min_exposed=10, min_events=3,
        )
        pipeline.add_layer("discovery", screen)

        result = pipeline.run(
            data=survival_df,
            biomarker_cols=biomarker_cols,
            outcome_col="OS_MONTHS",
            event_col="OS_STATUS",
            covariates=covariates,
        )
        lr = result.get("discovery")
        assert "gate" not in lr.metadata

    def test_gate_summary_method(self, survival_df, biomarker_cols, covariates):
        from cascade.core import BiomarkerScreen

        ge = GateEvaluator()
        pipeline = Pipeline(gate_evaluator=ge)
        screen = BiomarkerScreen(
            method="cox", correction="fdr_tsbh",
            min_exposed=10, min_events=3,
        )
        pipeline.add_layer("discovery", screen)

        pipeline.run(
            data=survival_df,
            biomarker_cols=biomarker_cols,
            outcome_col="OS_MONTHS",
            event_col="OS_STATUS",
            covariates=covariates,
        )
        summary = pipeline.gate_summary()
        assert "CASCADE Gate" in summary

    def test_report_includes_gates(self, survival_df, biomarker_cols, covariates):
        from cascade.core import BiomarkerScreen

        ge = GateEvaluator()
        pipeline = Pipeline(gate_evaluator=ge)
        screen = BiomarkerScreen(
            method="cox", correction="fdr_tsbh",
            min_exposed=10, min_events=3,
        )
        pipeline.add_layer("discovery", screen)

        pipeline.run(
            data=survival_df,
            biomarker_cols=biomarker_cols,
            outcome_col="OS_MONTHS",
            event_col="OS_STATUS",
            covariates=covariates,
        )
        report = pipeline.report()
        assert "Gate Evaluation" in report


# ---------------------------------------------------------------
# Skipped / undefined layers
# ---------------------------------------------------------------

class TestEdgeCases:
    def test_undefined_layer_passes(self):
        lr = LayerResult("cohort", 0, "completed", results=pd.DataFrame())
        gate = GateEvaluator().evaluate("cohort", lr)
        assert gate.passed is True

    def test_failed_layer_not_gated(self):
        """Gate evaluation only runs on succeeded layers; Pipeline handles this."""
        lr = LayerResult("discovery", 1, "failed", results=None)
        # Directly calling evaluate on a failed layer still works gracefully
        gate = GateEvaluator().evaluate("discovery", lr)
        assert gate.passed is False
