"""
Tests for CASCADE remediation loop in Pipeline.
"""

import pandas as pd
import pytest

from cascade.core.gates import GateEvaluator, GateResult
from cascade.pipeline import Pipeline, LayerResult


class _MockScreen:
    """Mock discovery screen that returns configurable results."""

    def __init__(self, n_significant=0):
        self.n_significant = n_significant
        self.call_count = 0

    def screen(self, df=None, **kwargs):
        self.call_count += 1
        return pd.DataFrame({
            "biomarker": [f"GENE_{i}" for i in range(5)],
            "p_adjusted": [0.01 if i < self.n_significant else 0.5 for i in range(5)],
            "significant": [i < self.n_significant for i in range(5)],
        })


def _remediation_relax(layer_result, layer_obj, data, dep_results, **kwargs):
    """Remediation: re-run with relaxed threshold (simulate finding 2 hits)."""
    return pd.DataFrame({
        "biomarker": ["GENE_A", "GENE_B", "GENE_C"],
        "p_adjusted": [0.01, 0.03, 0.50],
        "significant": [True, True, False],
    })


def _remediation_still_fails(layer_result, layer_obj, data, dep_results, **kwargs):
    """Remediation that still fails the gate."""
    return pd.DataFrame({
        "biomarker": ["GENE_A"],
        "p_adjusted": [0.50],
        "significant": [False],
    })


class TestRemediation:
    def test_remediation_on_gate_failure(self):
        ge = GateEvaluator()
        pipeline = Pipeline(
            gate_evaluator=ge,
            remediation_strategies={"discovery": _remediation_relax},
        )
        mock_screen = _MockScreen(n_significant=0)  # Will fail gate
        pipeline.add_layer("discovery", mock_screen)

        result = pipeline.run(data=pd.DataFrame({"x": [1, 2, 3]}))
        lr = result.get("discovery")

        assert lr.succeeded
        gate = lr.metadata.get("gate")
        assert gate is not None
        assert gate.passed is True
        assert gate.remediation_attempted is True
        assert gate.remediation_succeeded is True

    def test_remediation_max_retries(self):
        ge = GateEvaluator()
        pipeline = Pipeline(
            gate_evaluator=ge,
            remediation_strategies={"discovery": _remediation_still_fails},
            max_remediation_retries=2,
        )
        mock_screen = _MockScreen(n_significant=0)
        pipeline.add_layer("discovery", mock_screen)

        result = pipeline.run(data=pd.DataFrame({"x": [1]}))
        lr = result.get("discovery")

        # Original result persists (remediation failed)
        gate = lr.metadata.get("gate")
        assert gate is not None
        assert gate.passed is False

    def test_no_remediation_when_gate_passes(self):
        call_count = {"n": 0}

        def should_not_be_called(*args, **kwargs):
            call_count["n"] += 1
            return pd.DataFrame()

        ge = GateEvaluator()
        pipeline = Pipeline(
            gate_evaluator=ge,
            remediation_strategies={"discovery": should_not_be_called},
        )
        mock_screen = _MockScreen(n_significant=3)  # Will pass gate
        pipeline.add_layer("discovery", mock_screen)

        pipeline.run(data=pd.DataFrame({"x": [1]}))
        assert call_count["n"] == 0

    def test_no_remediation_without_strategy(self):
        ge = GateEvaluator()
        pipeline = Pipeline(gate_evaluator=ge)  # No strategies
        mock_screen = _MockScreen(n_significant=0)
        pipeline.add_layer("discovery", mock_screen)

        result = pipeline.run(data=pd.DataFrame({"x": [1]}))
        lr = result.get("discovery")
        gate = lr.metadata.get("gate")
        assert gate is not None
        assert gate.passed is False
        assert gate.remediation_attempted is False

    def test_pipeline_without_gates_unchanged(self):
        pipeline = Pipeline(gate_evaluator=False)  # gates explicitly off
        mock_screen = _MockScreen(n_significant=0)
        pipeline.add_layer("discovery", mock_screen)

        result = pipeline.run(data=pd.DataFrame({"x": [1]}))
        lr = result.get("discovery")
        assert lr.succeeded
        assert "gate" not in lr.metadata
