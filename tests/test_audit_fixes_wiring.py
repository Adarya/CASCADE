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


def test_pipeline_gates_on_by_default_and_parses_cbioportal_status():
    from cascade.core import BiomarkerScreen

    rng = np.random.default_rng(1)
    n = 600
    df = pd.DataFrame({f"G{i}": rng.binomial(1, 0.25, n) for i in range(4)})
    t = rng.exponential(24 * np.exp(-0.7 * df["G0"]))
    c = rng.exponential(40, n)
    df["OS_MONTHS"] = np.minimum(t, c)
    df["OS_STATUS"] = np.where(t <= c, "1:DECEASED", "0:LIVING")
    pipe = Pipeline()
    pipe.add_layer("discovery", BiomarkerScreen(method="cox", min_exposed=20, min_events=5))
    result = pipe.run(df, biomarker_cols=[f"G{i}" for i in range(4)],
                      outcome_col="OS_MONTHS", event_col="OS_STATUS")
    lr = result.layer_results["discovery"]
    assert lr.succeeded
    assert "gate" in lr.metadata


def test_confirmation_parses_cbioportal_status_and_categorical_covariates():
    from cascade.core import BiomarkerScreen, OrthogonalConfirm

    rng = np.random.default_rng(2)
    n = 800
    df = pd.DataFrame({f"G{i}": rng.binomial(1, 0.25, n) for i in range(3)})
    df["AGE"] = rng.normal(65, 9, n)
    df["SEX"] = rng.choice(["Male", "Female"], n)
    t = rng.exponential(24 * np.exp(-0.7 * df["G0"]))
    c = rng.exponential(40, n)
    df["OS_MONTHS"] = np.minimum(t, c)
    df["OS_STATUS"] = np.where(t <= c, "1:DECEASED", "0:LIVING")
    genes = ["G0", "G1", "G2"]
    primary = BiomarkerScreen(method="cox", min_exposed=20, min_events=5).screen(
        df, genes, "OS_MONTHS", "OS_STATUS", covariates=["AGE", "SEX"])
    assert primary.loc[primary.biomarker == "G0", "p"].iloc[0] < 0.05
    conf = OrthogonalConfirm(confirm_method="logistic").confirm(
        primary, df, genes, "OS_MONTHS", event_col="OS_STATUS", covariates=["AGE", "SEX"])
    assert conf["confirm_p"].notna().all() if "confirm_p" in conf else len(conf) == 3
