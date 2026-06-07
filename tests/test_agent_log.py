"""
Tests for CASCADE agent decision logging (cascade.core.agent_log).
"""

import os

import pandas as pd
import pytest

from cascade.core.agent_log import AgentDecision, AgentDecisionLog
from cascade.core.gates import GateResult


class TestAgentDecision:
    def test_creation(self):
        d = AgentDecision(
            timestamp="2026-03-17T10:00:00",
            study="CRC",
            layer=1,
            decision_type="method_selection",
            description="Selected cause-specific Cox",
        )
        assert d.automated is True
        assert d.rationale is None


class TestAgentDecisionLog:
    def test_log_decision(self):
        log = AgentDecisionLog()
        log.log_decision(
            study="CRC", layer=1,
            decision_type="method_selection",
            description="Selected Cox",
        )
        assert log.n_decisions == 1

    def test_log_override(self):
        log = AgentDecisionLog()
        log.log_override(
            study="CRC", layer=5,
            description="Changed independence threshold",
            original_decision="phi < 0.10",
            override_reason="Domain knowledge",
        )
        assert log.n_decisions == 1
        df = log.to_dataframe()
        assert df.iloc[0]["automated"] == False

    def test_log_pitfall_prevention(self):
        log = AgentDecisionLog()
        log.log_pitfall_prevention(
            study="CRC", pitfall_id=1,
            description="Hex color in STYLE_COLOR",
        )
        assert log.n_decisions == 1
        assert log.pitfall_prevention_count() == 1

    def test_log_gate_result(self):
        log = AgentDecisionLog()
        gr = GateResult(layer_name="discovery", layer_number=1, passed=True)
        log.log_gate_result(study="CRC", gate_result=gr)
        assert log.n_decisions == 1

    def test_invalid_decision_type(self):
        log = AgentDecisionLog()
        with pytest.raises(ValueError, match="Invalid decision_type"):
            log.log_decision(
                study="CRC", layer=1,
                decision_type="invalid_type",
                description="Bad",
            )

    def test_to_dataframe(self):
        log = AgentDecisionLog()
        log.log_decision(study="A", layer=1, decision_type="method_selection", description="Cox")
        log.log_decision(study="B", layer=2, decision_type="parameter_choice", description="FDR")
        df = log.to_dataframe()
        assert len(df) == 2
        assert set(df.columns) == {
            "timestamp", "study", "layer", "decision_type",
            "description", "rationale", "outcome", "automated",
        }

    def test_to_dataframe_empty(self):
        log = AgentDecisionLog()
        df = log.to_dataframe()
        assert len(df) == 0
        assert "study" in df.columns

    def test_to_csv(self, tmp_path):
        log = AgentDecisionLog()
        log.log_decision(study="CRC", layer=1, decision_type="method_selection", description="Cox")
        path = str(tmp_path / "decisions.csv")
        log.to_csv(path)
        assert os.path.exists(path)
        df = pd.read_csv(path)
        assert len(df) == 1

    def test_summary_by_study(self):
        log = AgentDecisionLog()
        log.log_decision(study="A", layer=1, decision_type="method_selection", description="Cox")
        log.log_decision(study="A", layer=2, decision_type="parameter_choice", description="FDR")
        log.log_decision(study="B", layer=1, decision_type="method_selection", description="Logistic")
        summary = log.summary_by_study()
        assert summary["A"]["method_selection"] == 1
        assert summary["A"]["parameter_choice"] == 1
        assert summary["B"]["method_selection"] == 1

    def test_summary_by_layer(self):
        log = AgentDecisionLog()
        log.log_decision(study="A", layer=1, decision_type="method_selection", description="Cox")
        log.log_decision(study="B", layer=1, decision_type="parameter_choice", description="Penalizer")
        summary = log.summary_by_layer()
        assert 1 in summary
        assert summary[1]["method_selection"] == 1
        assert summary[1]["parameter_choice"] == 1

    def test_gate_heatmap_data(self):
        log = AgentDecisionLog()
        gr1 = GateResult(layer_name="discovery", layer_number=1, passed=True)
        gr2 = GateResult(layer_name="confirmation", layer_number=2, passed=False)
        log.log_gate_result(study="CRC", gate_result=gr1)
        log.log_gate_result(study="CRC", gate_result=gr2)
        heatmap = log.gate_heatmap_data()
        assert not heatmap.empty
        assert heatmap.loc["CRC", 1] == 1
        assert heatmap.loc["CRC", 2] == 0

    def test_gate_heatmap_empty(self):
        log = AgentDecisionLog()
        heatmap = log.gate_heatmap_data()
        assert heatmap.empty

    def test_override_rate(self):
        log = AgentDecisionLog()
        log.log_decision(study="A", layer=1, decision_type="method_selection", description="Cox")
        log.log_override(study="A", layer=1, description="Changed", original_decision="Cox", override_reason="Bad")
        assert log.override_rate() == 0.5

    def test_override_rate_empty(self):
        log = AgentDecisionLog()
        assert log.override_rate() == 0.0
