"""Regression tests: the pitfall library runs inside Layer 5 and its
warnings reach the compliance report (audit, Sep 2026)."""

import numpy as np
import pandas as pd

from cascade import Pipeline
from cascade.core import ArtifactGuard
from cascade.report import generate_report


def _censoring_frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "gene": rng.binomial(1, 0.3, n),
        "age": rng.normal(65, 8, n),
    })
    t_event = rng.exponential(24, n)
    # carriers are censored much earlier than non-carriers
    t_cens = rng.exponential(np.where(df["gene"] == 1, 8, 40))
    df["time"] = np.minimum(t_event, t_cens)
    df["event"] = (t_event <= t_cens).astype(int)
    return df


def _pitfall_inputs(df):
    return {"survival_df": df, "duration_col": "time", "event_col": "event",
            "biomarker_cols": ["gene"], "survival_covariates": ["age"]}


def test_run_all_runs_pitfall_library():
    df = _censoring_frame()
    report = ArtifactGuard().run_all(df, pitfall_inputs=_pitfall_inputs(df))
    assert "pitfall_library" in report.checks_run
    assert any(w.pitfall.id == 10 for w in report.pitfall_warnings)
    assert any("Pitfall #10" in w for w in report.warnings)


def test_run_all_without_pitfall_inputs_unchanged():
    df = _censoring_frame()
    report = ArtifactGuard().run_all(df)
    assert "pitfall_library" not in report.checks_run
    assert report.pitfall_warnings == []


def test_layer5_pitfall_warnings_reach_compliance_report():
    df = _censoring_frame()
    pipe = Pipeline()
    pipe.add_layer("artifact_guard", ArtifactGuard())
    result = pipe.run(df, pitfall_inputs=_pitfall_inputs(df))
    lr = result.layer_results["artifact_guard"]
    assert any("Pitfall #10" in w for w in lr.warnings)
    assert "Pitfall #10" in generate_report(result)
