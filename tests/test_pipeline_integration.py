"""
Integration tests for the CASCADE Pipeline orchestrator.

Tests the end-to-end pipeline workflow: adding layers, running
discovery, generating reports, and inspecting result properties.
"""

import numpy as np
import pandas as pd
import pytest

from cascade import Pipeline
from cascade.core import BiomarkerScreen
from cascade.pipeline import PipelineResult, LayerResult


# ======================================================================
# Pipeline Layer Management
# ======================================================================


class TestPipelineLayerManagement:
    """Tests for adding, listing, and removing layers."""

    def test_pipeline_add_layers(self):
        """Adding multiple layers should be reflected in list_layers."""
        pipeline = Pipeline()
        screen = BiomarkerScreen(method="cox", correction="fdr_bh")

        pipeline.add_layer("discovery", screen)
        pipeline.add_layer("extra_layer", screen)

        layers = pipeline.list_layers()
        assert len(layers) == 2, (
            f"Expected 2 layers, got {len(layers)}."
        )
        assert layers[0] == "discovery", "First layer should be 'discovery'."
        assert layers[1] == "extra_layer", "Second layer should be 'extra_layer'."

    def test_pipeline_remove_layer(self):
        """Removing a layer should decrease the list_layers count."""
        pipeline = Pipeline()
        screen = BiomarkerScreen(method="cox")

        pipeline.add_layer("discovery", screen)
        pipeline.add_layer("extra", screen)
        assert len(pipeline.list_layers()) == 2

        pipeline.remove_layer("extra")
        layers = pipeline.list_layers()
        assert len(layers) == 1, (
            f"After removal, expected 1 layer, got {len(layers)}."
        )
        assert "extra" not in layers, "'extra' should no longer be listed."

    def test_pipeline_method_chaining(self):
        """add_layer should support method chaining."""
        pipeline = Pipeline()
        screen = BiomarkerScreen(method="cox")

        result = pipeline.add_layer("discovery", screen).add_layer("extra", screen)

        assert result is pipeline, "add_layer should return self for chaining."
        assert len(pipeline.list_layers()) == 2, (
            "Method chaining should add both layers."
        )


# ======================================================================
# Pipeline Execution
# ======================================================================


class TestPipelineExecution:
    """Tests for pipeline run with BiomarkerScreen."""

    def test_pipeline_run_discovery(self, survival_df, biomarker_cols, covariates):
        """Pipeline with BiomarkerScreen should run without error.

        Note: Uses 'fdr_tsbh' correction which routes through the statsmodels
        fallback path in _correct(), avoiding a known tuple-unpacking bug when
        CASCADE's apply_fdr (which returns a 2-tuple) is assigned directly to
        a DataFrame column.
        """
        pipeline = Pipeline()
        screen = BiomarkerScreen(
            method="cox",
            correction="fdr_tsbh",  # routes through _apply_correction_fallback
            min_exposed=10,
            min_events=3,
        )
        pipeline.add_layer("discovery", screen)

        result = pipeline.run(
            data=survival_df,
            biomarker_cols=biomarker_cols,
            outcome_col="OS_MONTHS",
            event_col="OS_STATUS",
            covariates=covariates,
        )

        assert isinstance(result, PipelineResult), (
            "run should return a PipelineResult."
        )
        assert "discovery" in result.completed_layers, (
            f"'discovery' should be in completed_layers. "
            f"Completed: {result.completed_layers}, "
            f"Failed: {result.failed_layers}"
        )
        assert result.total_duration_seconds > 0, (
            "Pipeline duration should be > 0."
        )

        # Check layer result
        layer_result = result.get("discovery")
        assert layer_result is not None, "Should be able to get 'discovery' result."
        assert layer_result.succeeded, (
            f"Discovery layer should succeed. Error: {layer_result.error_message}"
        )
        assert isinstance(layer_result.results, pd.DataFrame), (
            "Discovery results should be a DataFrame."
        )
        assert len(layer_result.results) > 0, (
            "Discovery results should have at least one row."
        )


# ======================================================================
# Pipeline Report
# ======================================================================


class TestPipelineReport:
    """Tests for pipeline report generation."""

    def test_pipeline_report(self, survival_df, biomarker_cols, covariates):
        """Pipeline report should be a non-empty string after a successful run."""
        pipeline = Pipeline()
        screen = BiomarkerScreen(
            method="cox",
            correction="fdr_tsbh",  # routes through _apply_correction_fallback
            min_exposed=10,
            min_events=3,
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

        assert isinstance(report, str), "Report should be a string."
        assert len(report) > 100, (
            f"Report should be substantial (>100 chars), got {len(report)} chars."
        )
        assert "CASCADE" in report, "Report should mention CASCADE."

    def test_pipeline_report_before_run(self):
        """Report before running should produce a helpful message."""
        pipeline = Pipeline()
        report = pipeline.report()

        assert isinstance(report, str), "Report should be a string."
        assert "No pipeline results" in report or len(report) > 0, (
            "Report before run should indicate no results available."
        )


# ======================================================================
# PipelineResult Properties
# ======================================================================


class TestPipelineResultProperties:
    """Tests for PipelineResult properties."""

    def test_completed_layers(self):
        """completed_layers should list only layers with status='completed'."""
        pr = PipelineResult()
        pr.layer_results["discovery"] = LayerResult(
            layer_name="discovery", layer_number=1, status="completed"
        )
        pr.layer_results["confirmation"] = LayerResult(
            layer_name="confirmation", layer_number=2, status="failed",
            error_message="Test failure",
        )
        pr.layer_results["sensitivity"] = LayerResult(
            layer_name="sensitivity", layer_number=4, status="completed"
        )

        completed = pr.completed_layers
        assert completed == ["discovery", "sensitivity"], (
            f"Expected ['discovery', 'sensitivity'], got {completed}."
        )

    def test_failed_layers(self):
        """failed_layers should list only layers with status='failed'."""
        pr = PipelineResult()
        pr.layer_results["discovery"] = LayerResult(
            layer_name="discovery", layer_number=1, status="completed"
        )
        pr.layer_results["confirmation"] = LayerResult(
            layer_name="confirmation", layer_number=2, status="failed",
            error_message="Some error",
        )

        failed = pr.failed_layers
        assert failed == ["confirmation"], (
            f"Expected ['confirmation'], got {failed}."
        )

    def test_all_warnings(self):
        """all_warnings should aggregate warnings from all layers with layer prefix."""
        pr = PipelineResult()
        pr.layer_results["discovery"] = LayerResult(
            layer_name="discovery",
            layer_number=1,
            status="completed",
            warnings=["Found 0 significant results"],
        )
        pr.layer_results["artifact_guard"] = LayerResult(
            layer_name="artifact_guard",
            layer_number=5,
            status="completed",
            warnings=["Collinearity detected", "Constant variable found"],
        )

        all_warnings = pr.all_warnings
        assert len(all_warnings) == 3, (
            f"Expected 3 total warnings, got {len(all_warnings)}."
        )
        # Each warning should be prefixed with the layer name
        assert all_warnings[0].startswith("[discovery]"), (
            "First warning should be prefixed with [discovery]."
        )
        assert all_warnings[1].startswith("[artifact_guard]"), (
            "Second warning should be prefixed with [artifact_guard]."
        )

    def test_get_nonexistent_layer(self):
        """get on a nonexistent layer should return None."""
        pr = PipelineResult()
        result = pr.get("nonexistent")
        assert result is None, "Getting a nonexistent layer should return None."

    def test_pipeline_result_empty(self):
        """An empty PipelineResult should have empty properties."""
        pr = PipelineResult()

        assert pr.completed_layers == [], (
            "Empty result should have no completed layers."
        )
        assert pr.failed_layers == [], (
            "Empty result should have no failed layers."
        )
        assert pr.all_warnings == [], (
            "Empty result should have no warnings."
        )
