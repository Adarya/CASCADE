"""Regression tests for the v0.4.0 audit fixes to gates, artifact guard,
sensitivity scoring, pipeline remediation logging, report and agent log.

Each test fails on the pre-fix behaviour and passes after the fix.
"""

import datetime

import numpy as np
import pandas as pd
import pytest

from cascade.core.agent_log import AgentDecisionLog
from cascade.core.artifact_guard import ArtifactGuard, ArtifactReport
from cascade.core.gates import GateEvaluator, GateResult
from cascade.core.sensitivity import RobustnessScorer, SensitivitySuite
from cascade.pipeline import LayerResult, Pipeline, PipelineResult
from cascade.report import generate_report


def _gate(layer, results, num):
    return GateEvaluator().evaluate(layer, LayerResult(layer, num, "completed", results=results))


# ---------------------------------------------------------------------------
# Bug 1: Layer 5 gate must fail on real ArtifactGuard findings / no checks
# ---------------------------------------------------------------------------

class TestBug1ArtifactGate:
    def test_independence_violation_fails_gate(self, rng):
        x = pd.Series(rng.binomial(1, 0.4, 300))
        report = ArtifactGuard().run_all(pd.DataFrame({"x": x}), x_binary=x, y_binary=x)
        gate = _gate("artifact_guard", report, 5)
        assert gate.passed is False
        assert gate.actual["n_critical"] >= 1

    def test_leakage_fails_gate(self):
        report = ArtifactReport(leakage_results=["EVER_ICI"], checks_run=["leakage"],
                                warnings=["Potential covariate leakage in: ['EVER_ICI']"])
        assert _gate("artifact_guard", report, 5).passed is False

    def test_collinearity_fails_gate(self, rng):
        a = rng.normal(size=200)
        df = pd.DataFrame({"a": a, "b": a * 2 + rng.normal(scale=1e-3, size=200),
                           "c": rng.normal(size=200)})
        report = ArtifactGuard().run_all(df, collinearity_cols=["a", "b", "c"])
        assert "collinearity" in report.checks_run
        assert report.collinearity_results
        assert _gate("artifact_guard", report, 5).passed is False

    def test_no_check_ran_fails(self):
        report = ArtifactGuard().run_all(pd.DataFrame({"x": [1, 2, 3]}))
        gate = _gate("artifact_guard", report, 5)
        assert gate.passed is False
        assert "No artifact check" in gate.error_message

    def test_clean_check_passes(self, rng):
        x = pd.Series(rng.binomial(1, 0.3, 400))
        y = pd.Series(rng.binomial(1, 0.4, 400))
        report = ArtifactGuard().run_all(pd.DataFrame({"x": x}), x_binary=x, y_binary=y)
        if report.independence_results["independent"]:
            assert _gate("artifact_guard", report, 5).passed is True

    def test_unrecognised_result_fails(self):
        assert _gate("artifact_guard", object(), 5).passed is False
        assert _gate("artifact_guard", {}, 5).passed is False


# ---------------------------------------------------------------------------
# Bug 2: independence decision derived from the suite's real keys; AND rule
# ---------------------------------------------------------------------------

class TestBug2Independence:
    def test_x_vs_itself_not_independent(self, rng):
        x = pd.Series(rng.binomial(1, 0.4, 200))
        res = ArtifactGuard.check_independence(x, x)
        assert res["independent"] is False
        report = ArtifactGuard().run_all(pd.DataFrame({"x": x}), x_binary=x, y_binary=x)
        assert any("NOT be independent" in w for w in report.warnings)

    def test_rule_uses_and(self):
        from cascade.core.artifact_guard import independence_decision

        # phi=0.5 with p=0.1: p > 0.05 but |phi| >= 0.1 -> NOT independent
        assert independence_decision({"phi": 0.5, "phi_p": 0.1}) is False
        assert independence_decision({"phi": 0.05, "chi2_p": 0.01}) is False
        assert independence_decision({"phi": 0.05, "chi2_p": 0.5}) is True
        assert independence_decision({"phi": np.nan, "chi2_p": 0.5}) is None

    def test_fallback_uses_and(self, monkeypatch):
        import cascade.core.artifact_guard as ag

        monkeypatch.setattr(ag, "independence_suite", None)
        # 2x2 with n=40: phi ~ 0.3 but p > 0.05 -> must be NOT independent
        x = pd.Series([1] * 10 + [1] * 10 + [0] * 10 + [0] * 10)
        y = pd.Series([1] * 13 + [0] * 7 + [1] * 7 + [0] * 13)
        res = ag.ArtifactGuard.check_independence(x, y)
        assert res["phi_p"] > 0.05 and abs(res["phi"]) >= 0.1
        assert res["independent"] is False


# ---------------------------------------------------------------------------
# Bug 3: Layer 6 gate fails closed
# ---------------------------------------------------------------------------

class TestBug3ClinicalGate:
    @pytest.mark.parametrize("results", [
        {"delta_c": np.nan, "ci": (0.01, 0.02)},
        {"delta_c": None, "ci": (0.01, 0.02)},
        {"ci": (0.01, 0.02)},
        {"delta_c": 0.03},
        {"delta_c": 0.03, "ci": (np.nan, np.nan)},
        {"delta_c": 0.03, "ci": (None, 0.05)},
        {},
    ])
    def test_missing_or_nan_fails(self, results):
        assert _gate("clinical", results, 6).passed is False

    def test_valid_passes(self):
        assert _gate("clinical", {"delta_c": 0.03, "ci": (0.01, 0.05)}, 6).passed is True


# ---------------------------------------------------------------------------
# Bug 4: pipeline produces a robustness classification; gate fails if absent
# ---------------------------------------------------------------------------

class TestBug4SensitivityGate:
    def test_gate_fails_without_classification(self):
        raw = {"v1": pd.DataFrame({"biomarker": ["A"], "hr": [1.2]})}
        gate = _gate("sensitivity", raw, 4)
        assert gate.passed is False

    def test_pipeline_classifies_and_gate_evaluates(self, survival_df, biomarker_cols):
        from cascade.core import BiomarkerScreen

        p = Pipeline(gate_evaluator=GateEvaluator())
        p.add_layer("discovery", BiomarkerScreen(method="cox", min_exposed=10, min_events=3))
        suite = SensitivitySuite()
        suite.add_variant("cc", lambda d: d.dropna(), category="threshold")
        suite.add_variant("young", lambda d: d[d["AGE"] < 70], category="subgroup")
        p.add_layer("sensitivity", suite)
        res = p.run(data=survival_df, biomarker_cols=biomarker_cols,
                    outcome_col="OS_MONTHS", event_col="OS_STATUS")
        lr = res.get("sensitivity")
        assert isinstance(lr.results, pd.DataFrame)
        assert "robustness_class" in lr.results.columns
        assert set(lr.metadata["variant_results"]) == {"cc", "young"}
        gate = lr.metadata["gate"]
        assert "n_scored" in gate.actual  # classification actually evaluated

    def test_pipeline_without_discovery_fails_gate(self, survival_df):
        suite = SensitivitySuite()
        suite.add_variant("cc", lambda d: d.dropna(), category="threshold")
        p = Pipeline(gate_evaluator=GateEvaluator())
        p.add_layer("sensitivity", suite)
        res = p.run(data=survival_df,
                    analysis_fn=lambda d: pd.DataFrame({"biomarker": ["A"], "hr": [1.1]}))
        gate = res.get("sensitivity").metadata["gate"]
        assert gate.passed is False


# ---------------------------------------------------------------------------
# Bug 5: empty predictive / sensitivity input fails
# ---------------------------------------------------------------------------

class TestBug5EmptyInputs:
    def test_predictive_empty_fails(self):
        assert _gate("predictive", pd.DataFrame({"classification": []}), 3).passed is False

    def test_sensitivity_empty_fails(self):
        assert _gate("sensitivity", pd.DataFrame({"robustness_class": []}), 4).passed is False


class TestPredictiveNotEvaluable:
    def test_not_evaluable_accepted(self):
        df = pd.DataFrame({"classification": ["predictive", "not_evaluable", "prognostic"]})
        gate = _gate("predictive", df, 3)
        assert gate.passed is True
        assert gate.actual["n_not_evaluable"] == 1

    def test_all_not_evaluable_fails(self):
        df = pd.DataFrame({"classification": ["not_evaluable", "not_evaluable"]})
        gate = _gate("predictive", df, 3)
        assert gate.passed is False
        assert gate.actual["n_not_evaluable"] == 2

    def test_invalid_class_still_fails(self):
        df = pd.DataFrame({"classification": ["predictive", "bogus"]})
        assert _gate("predictive", df, 3).passed is False


# ---------------------------------------------------------------------------
# Bug 6: full-key matching, threshold, minimum evaluable, n_failed
# ---------------------------------------------------------------------------

class TestBug6Robustness:
    def test_gene_site_matched_on_full_key(self):
        primary = pd.DataFrame({"biomarker": ["G", "G"], "site": ["liver", "bone"],
                                "hr": [2.0, 0.5]})
        variant = pd.DataFrame({"biomarker": ["G", "G"], "site": ["liver", "bone"],
                                "hr": [1.8, 0.6]})
        out = SensitivitySuite.classify_robustness(primary, {"v1": variant, "v2": variant})
        # Old behaviour compared bone to the liver row (hr 1.8) -> unstable.
        assert list(out["robustness_class"]) == ["robust", "robust"]

    def test_specification_rule(self):
        # Spec: UNSTABLE if any direction flip; ROBUST if zero flips and
        # concordance (failed variants included) >= threshold.
        assert RobustnessScorer.score(1.5, [1.3, 1.4, 1.2, 0.8]) == "unstable"
        assert RobustnessScorer.score(1.5, [1.3, 1.4, 1.2, 1.1]) == "robust"
        # 3 of 4 variants concordant, 1 failed: 0.75 passes 0.75 but not 0.9
        assert RobustnessScorer.score(1.5, [1.3, 1.4, 1.2, np.nan], threshold=0.75) == "robust"
        assert RobustnessScorer.score(1.5, [1.3, 1.4, 1.2, np.nan], threshold=0.9) == "exploratory"

    def test_single_variant_insufficient(self):
        assert RobustnessScorer.score(1.5, [1.4]) != "robust"

    def test_nan_variants_counted_as_failed(self):
        d = RobustnessScorer.score_detail(1.5, [1.4, np.nan, np.nan])
        assert d["n_failed"] == 2 and d["n_evaluable"] == 1
        assert d["robustness_class"] != "robust"
        primary = pd.DataFrame({"biomarker": ["A"], "hr": [1.5]})
        out = SensitivitySuite.classify_robustness(
            primary,
            {"v1": pd.DataFrame({"biomarker": ["A"], "hr": [1.4]}),
             "v2": pd.DataFrame({"biomarker": ["B"], "hr": [1.4]})},
            failed_variants=["v3"],
        )
        assert out.loc[0, "n_failed_variants"] == 2
        assert out.loc[0, "robustness_class"] != "robust"


# ---------------------------------------------------------------------------
# Bugs 7-9: remediation logging, attempts recorded, warnings preserved
# ---------------------------------------------------------------------------

class _Screen:
    warnings = ["original layer warning"]

    def screen(self, df=None):
        return pd.DataFrame({"biomarker": ["A"], "p_adjusted": [0.5], "significant": [False]})


def _fix(layer_result, layer_obj, data, dep_results, **kw):
    return pd.DataFrame({"biomarker": ["A"], "p_adjusted": [0.01], "significant": [True]})


def _still_bad(layer_result, layer_obj, data, dep_results, **kw):
    return pd.DataFrame({"biomarker": ["A"], "p_adjusted": [0.5], "significant": [False]})


def _boom(layer_result, layer_obj, data, dep_results, **kw):
    raise RuntimeError("remediation exploded")


class TestBug7to9Remediation:
    def _run(self, strategy, retries=2):
        log = AgentDecisionLog()
        p = Pipeline(gate_evaluator=GateEvaluator(), agent_log=log,
                     remediation_strategies={"discovery": strategy},
                     max_remediation_retries=retries)
        p.add_layer("discovery", _Screen())
        res = p.run(data=pd.DataFrame({"x": [1]}))
        return res.get("discovery"), log

    def test_bug7_final_state_logged_after_remediation(self):
        lr, log = self._run(_fix)
        df = log.to_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["outcome"] == "PASS (after remediation)"

    def test_bug8_failed_remediation_recorded(self):
        lr, log = self._run(_still_bad)
        gate = lr.metadata["gate"]
        assert gate.passed is False
        assert gate.remediation_attempted is True
        assert len(gate.details["remediation_attempts"]) == 2
        assert log.to_dataframe().iloc[0]["outcome"] == "FAIL (remediation failed)"

    def test_bug8_remediation_exception_recorded(self):
        lr, log = self._run(_boom)
        gate = lr.metadata["gate"]
        assert gate.remediation_attempted is True
        attempts = gate.details["remediation_attempts"]
        assert attempts[0]["status"] == "exception"
        assert "remediation exploded" in attempts[0]["error"]
        assert any("remediation exploded" in w for w in lr.warnings)

    def test_bug9_original_warnings_preserved(self):
        lr, _ = self._run(_fix)
        assert "original layer warning" in lr.warnings
        assert any("remediation succeeded" in w for w in lr.warnings)


# ---------------------------------------------------------------------------
# Bug 10: user-supplied analysis_fn honoured
# ---------------------------------------------------------------------------

class TestBug10AnalysisFn:
    def test_user_analysis_fn_used(self, survival_df, biomarker_cols):
        from cascade.core import BiomarkerScreen

        calls = []

        def my_fn(d):
            calls.append(len(d))
            return pd.DataFrame({"biomarker": biomarker_cols, "hr": [1.1] * 3})

        suite = SensitivitySuite()
        suite.add_variant("cc", lambda d: d.dropna(), category="threshold")
        p = Pipeline()
        p.add_layer("discovery", BiomarkerScreen(method="cox", min_exposed=10, min_events=3))
        p.add_layer("sensitivity", suite)
        res = p.run(data=survival_df, biomarker_cols=biomarker_cols,
                    outcome_col="OS_MONTHS", event_col="OS_STATUS", analysis_fn=my_fn)
        assert calls == [len(survival_df)]
        assert res.get("sensitivity").metadata["variant_results"]["cc"]["hr"].tolist() == [1.1] * 3


# ---------------------------------------------------------------------------
# Bug 11: report status / checklist / completion / custom gates
# ---------------------------------------------------------------------------

def _pr_with(*lrs):
    pr = PipelineResult()
    for lr in lrs:
        pr.layer_results[lr.layer_name] = lr
    return pr


class TestBug11Report:
    def test_status_reflects_failed_gate(self):
        lr = LayerResult("discovery", 1, "completed",
                         results=pd.DataFrame({"significant": [False], "p_adjusted": [0.5]}))
        lr.metadata["gate"] = GateResult("discovery", 1, False, error_message="none sig")
        report = generate_report(_pr_with(lr))
        summary = report.split("## Layer Details")[0]
        row = [ln for ln in summary.splitlines() if ln.startswith("| 1 |")][0]
        assert "FAIL" in row and "PASS" not in row

    def test_checklist_requires_evidence(self):
        lr = LayerResult("discovery", 1, "completed", results={"n_significant": 0})
        report = generate_report(_pr_with(lr))
        assert "- [x] **L1**: Multiple testing correction" not in report
        lr2 = LayerResult("discovery", 1, "completed",
                          results=pd.DataFrame({"p_adjusted": [0.01], "significant": [True]}))
        report2 = generate_report(_pr_with(lr2))
        assert "- [x] **L1**: Multiple testing correction" in report2
        # unverifiable items are never auto-ticked
        assert "- [x] **L1**: Separation problems" not in report2

    def test_checklist_unticked_when_gate_failed(self):
        lr = LayerResult("discovery", 1, "completed",
                         results=pd.DataFrame({"p_adjusted": [0.5], "significant": [False]}))
        lr.metadata["gate"] = GateResult("discovery", 1, False, error_message="x")
        assert "- [x] **L1**" not in generate_report(_pr_with(lr))

    def test_completion_never_exceeds_100(self):
        lrs = [LayerResult(f"c{i}", -1, "completed") for i in range(12)]
        lrs.append(LayerResult("discovery", 1, "completed"))
        report = generate_report(_pr_with(*lrs))
        assert "**Completion: 1/9 core layers (11%)**" in report
        assert "Custom layers (not counted above): 12/12" in report

    def test_custom_layer_gate_not_counted(self):
        ge = GateEvaluator()
        g_custom = ge.evaluate("my_custom", LayerResult("my_custom", -1, "completed", results={}))
        assert g_custom.details.get("not_gated") is True
        g_fail = GateResult("discovery", 1, False, error_message="x")
        summary = ge.summary([g_custom, g_fail])
        assert "0/1 gates passed" in summary
        assert "NOT GATED" in summary


# ---------------------------------------------------------------------------
# Bug 12: agent log override rate, fields, round-trip, timestamps, layer
# ---------------------------------------------------------------------------

class TestBug12AgentLog:
    def test_override_rate_excludes_gates_and_pitfalls(self):
        log = AgentDecisionLog()
        log.log_decision(study="S", layer=1, decision_type="method_selection", description="Cox")
        log.log_decision(study="S", layer=1, decision_type="parameter_choice", description="FDR",
                         overridden=True, original_choice="BH", override_choice="Bonferroni",
                         override_reason="PI preference")
        log.log_pitfall_prevention(study="S", pitfall_id=1, description="hex", automated=False)
        for i in range(5):
            log.log_gate_result("S", GateResult("discovery", 1, True))
        assert log.override_rate() == pytest.approx(0.5)

    def test_not_automated_is_not_override(self):
        log = AgentDecisionLog()
        log.log_decision(study="S", layer=1, decision_type="method_selection",
                         description="human chose Cox", automated=False)
        assert log.override_rate() == 0.0

    def test_gate_entries_not_parameter_choice(self):
        log = AgentDecisionLog()
        log.log_gate_result("S", GateResult("discovery", 1, True))
        assert log.to_dataframe().iloc[0]["decision_type"] == "gate_evaluation"

    def test_csv_round_trip(self, tmp_path):
        log = AgentDecisionLog()
        log.log_decision(study="S", layer=2, decision_type="method_selection", description="Cox",
                         rationale="PH holds")
        log.log_override(study="S", layer=5, description="phi", original_decision="0.1",
                         override_reason="domain", override_choice="0.2")
        log.log_pitfall_prevention(study="S", pitfall_id=3, description="collinear", layer=5)
        log.log_gate_result("S", GateResult("clinical", 6, False, error_message="ci"))
        path = str(tmp_path / "log.csv")
        log.to_csv(path)
        loaded = AgentDecisionLog.from_csv(path)
        a = log.to_dataframe().astype(object).where(lambda d: d.notna(), None)
        b = loaded.to_dataframe().astype(object).where(lambda d: d.notna(), None)
        pd.testing.assert_frame_equal(a, b, check_dtype=False)
        assert loaded.override_rate() == log.override_rate()

    def test_timestamp_timezone_aware_utc(self):
        log = AgentDecisionLog()
        log.log_decision(study="S", layer=1, decision_type="method_selection", description="x")
        ts = datetime.datetime.fromisoformat(log.to_dataframe().iloc[0]["timestamp"])
        assert ts.tzinfo is not None
        assert ts.utcoffset() == datetime.timedelta(0)

    def test_pitfall_layer_honoured(self):
        log = AgentDecisionLog()
        log.log_pitfall_prevention(study="S", pitfall_id=3, description="VIF", layer=5)
        assert log.to_dataframe().iloc[0]["layer"] == 5
