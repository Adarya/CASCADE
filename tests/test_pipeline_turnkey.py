"""Tests for turnkey pipeline behavior introduced for the documented
Quick Start: automatic dependency wiring, skip-vs-fail semantics,
fail-loud reporting, and strict mode.
"""

import numpy as np
import pandas as pd
import pytest

from cascade import Pipeline
from cascade.core import (
    BiomarkerScreen,
    OrthogonalConfirm,
    SensitivitySuite,
    ArtifactGuard,
)


def _discovery():
    return BiomarkerScreen(
        method="cox", correction="fdr_bh", min_exposed=10, min_events=3
    )


class _BoomLayer:
    """A layer whose execution always raises (to exercise failure paths)."""

    def run(self, df):
        raise ValueError("boom")


class TestAutoWiring:
    def test_confirmation_autowired_without_depends_on(
        self, survival_df, biomarker_cols
    ):
        """Confirmation needs primary_results; the pipeline must supply it from
        the discovery layer WITHOUT an explicit depends_on declaration."""
        p = Pipeline()
        p.add_layer("discovery", _discovery())
        p.add_layer("confirmation", OrthogonalConfirm(confirm_method="logistic"))
        res = p.run(
            data=survival_df,
            biomarker_cols=biomarker_cols,
            outcome_col="OS_MONTHS",
            event_col="OS_STATUS",
        )
        assert "confirmation" in res.completed_layers
        assert res.failed_layers == []

    def test_readme_quickstart_runs_end_to_end(self, survival_df, biomarker_cols):
        """The exact README Quick Start pattern must complete with no failures."""
        p = Pipeline()
        p.add_layer("discovery", _discovery())
        p.add_layer("confirmation", OrthogonalConfirm(confirm_method="logistic"))
        sens = SensitivitySuite()
        sens.add_variant(
            "complete_cases", lambda d: d.dropna(), category="threshold"
        )
        p.add_layer("sensitivity", sens)
        p.add_layer("artifact_guard", ArtifactGuard())
        res = p.run(
            data=survival_df,
            biomarker_cols=biomarker_cols,
            outcome_col="OS_MONTHS",
            event_col="OS_STATUS",
        )
        assert res.failed_layers == []
        assert {
            "discovery",
            "confirmation",
            "sensitivity",
            "artifact_guard",
        }.issubset(set(res.completed_layers))


class TestSkipVsFail:
    def test_missing_input_is_skipped_not_failed(
        self, survival_df, biomarker_cols
    ):
        """Sensitivity with a variant but no discovery layer cannot get a
        default analysis_fn -> it should be SKIPPED, never failed."""
        p = Pipeline()
        sens = SensitivitySuite()
        sens.add_variant("cc", lambda d: d.dropna(), category="threshold")
        p.add_layer("sensitivity", sens)
        res = p.run(
            data=survival_df,
            biomarker_cols=biomarker_cols,
            outcome_col="OS_MONTHS",
            event_col="OS_STATUS",
        )
        assert "sensitivity" in res.skipped_layers
        assert "sensitivity" not in res.failed_layers


class TestFailLoud:
    def test_failure_recorded_when_not_strict(self, survival_df):
        p = Pipeline()
        p.add_layer("discovery", _BoomLayer())
        res = p.run(data=survival_df)
        assert "discovery" in res.failed_layers
        assert "boom" in res.get("discovery").error_message

    def test_strict_raises_on_failure(self, survival_df):
        p = Pipeline()
        p.add_layer("discovery", _BoomLayer())
        with pytest.raises(RuntimeError):
            p.run(data=survival_df, strict=True)

    def test_report_headlines_failure(self, survival_df):
        p = Pipeline()
        p.add_layer("discovery", _BoomLayer())
        p.run(data=survival_df)
        report = p.report()
        assert "FAILED" in report

    def test_raise_on_failure_helper(self, survival_df):
        p = Pipeline()
        p.add_layer("discovery", _BoomLayer())
        res = p.run(data=survival_df)
        with pytest.raises(RuntimeError):
            res.raise_on_failure()

    def test_clean_run_does_not_raise_in_strict(self, survival_df, biomarker_cols):
        p = Pipeline()
        p.add_layer("discovery", _discovery())
        res = p.run(
            data=survival_df,
            biomarker_cols=biomarker_cols,
            outcome_col="OS_MONTHS",
            event_col="OS_STATUS",
            strict=True,
        )
        assert "discovery" in res.completed_layers
