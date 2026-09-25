"""Regression tests for the v0.4.0 audit fixes in cascade.stats,
cascade.pitfalls and cascade.core.clinical.

Each test reproduces an audited failure mode (bug numbers refer to the
audit list) and fails on the pre-fix code.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from cascade.stats import (
    BootstrapCI,
    CauseSpecificCox,
    InteractionCox,
    PenalizedCox,
    bootstrap_delta_c,
    cause_specific_screen,
    cv_concordance,
    independence_suite,
    stratified_effect,
    tost_equivalence,
)
from cascade.stats.equivalence import bayes_factor_independence
from cascade.pitfalls import PitfallDetector, PitfallRegistry, Severity
from cascade.pitfalls.checks.collinearity import check_collinearity
from cascade.pitfalls.checks.comment_header import check_comment_corruption
from cascade.pitfalls.checks.covariate_leakage import check_landmark_leakage
from cascade.pitfalls.checks.informative_censoring import check_informative_censoring
from cascade.pitfalls.checks.separation import check_separation_problems
from cascade.pitfalls.checks.singular_matrix import check_singular_matrix


# ----------------------------------------------------------------------
# Bug 1: empty cells of the 2x2x2 design
# ----------------------------------------------------------------------

def _three_way_frame(seed=0, n=600):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "g": rng.integers(0, 2, n),
        "t": rng.integers(0, 2, n),
        "p": rng.integers(0, 2, n),
    })
    d["T"] = rng.exponential(1, n)
    d["E"] = 1
    return d


def test_interaction_empty_cell_aliasing_not_converged():
    d = _three_way_frame()
    # cell (g=1, t=1, p=0) empty -> g:t is identical to g:t:p
    d.loc[(d.g == 1) & (d.t == 1) & (d.p == 0), "p"] = 1
    res = InteractionCox().fit_three_way(d, "g", "t", "p", "T", "E")
    assert res.converged is False
    assert "n=0" in res.error_message
    assert res.three_way_effect == {}


def test_interaction_empty_three_way_cell_not_converged():
    d = _three_way_frame()
    d.loc[(d.g == 1) & (d.t == 1), "p"] = 0  # cell (1,1,1) empty
    res = InteractionCox().fit_three_way(d, "g", "t", "p", "T", "E")
    assert res.converged is False
    assert res.error_message


def test_interaction_full_design_still_converges():
    res = InteractionCox().fit_three_way(_three_way_frame(), "g", "t", "p", "T", "E")
    assert res.converged is True
    assert "coef" in res.three_way_effect


def test_interaction_rank_check_when_min_n_zero():
    d = _three_way_frame()
    d.loc[(d.g == 1) & (d.t == 1) & (d.p == 0), "p"] = 1
    res = InteractionCox(min_subgroup_n=0, min_subgroup_events=0).fit_three_way(
        d, "g", "t", "p", "T", "E"
    )
    assert res.converged is False
    assert "rank-deficient" in res.error_message


# ----------------------------------------------------------------------
# Bug 2: min_events per exposure arm; duplicated gene covariate
# ----------------------------------------------------------------------

def _zero_event_carriers(seed=0, n=500):
    rng = np.random.default_rng(seed)
    c = pd.DataFrame({
        "gene": (rng.uniform(size=n) < 0.1).astype(int),
        "age": rng.normal(size=n),
    })
    c["T"] = rng.exponential(1, n)
    c["site"] = ((rng.uniform(size=n) < 0.3) & (c.gene == 0)).astype(int)
    return c


def test_cause_specific_screen_requires_events_in_each_arm():
    res = cause_specific_screen(_zero_event_carriers(), "gene", ["site"], "T", "E", ["age"])
    row = res.iloc[0]
    assert not row["converged"]
    assert np.isnan(row["p"])
    assert row["n_events_exposed"] == 0
    assert "events per arm" in row["skip_reason"]


def test_cause_specific_screen_gene_in_covariates_no_typeerror():
    c = _zero_event_carriers()
    c.loc[c.index[:40], "site"] = 1  # give carriers some events too
    res = cause_specific_screen(c, "gene", ["site"], "T", "E", ["gene", "age"])
    assert len(res) == 1


# ----------------------------------------------------------------------
# Bug 3: ConvergenceWarning -> converged=False
# ----------------------------------------------------------------------

def _low_variance_frame(seed=0, n=200):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"x": rng.normal(size=n), "tiny": rng.normal(size=n) * 1e-4})
    d["T"] = rng.exponential(1, n)
    d["E"] = 1
    d["site"] = (rng.uniform(size=n) < 0.5).astype(int)
    d["gene"] = (rng.uniform(size=n) < 0.4).astype(int)
    return d


def test_penalized_cox_convergence_warning_sets_not_converged():
    d = _low_variance_frame()
    m = PenalizedCox().fit(d, "T", "E", ["x", "tiny"])
    assert m.converged is False
    assert m.convergence_warnings


def test_cause_specific_convergence_warning_sets_not_converged():
    d = _low_variance_frame()
    res = cause_specific_screen(d, "gene", ["site"], "T", "E", ["tiny"])
    assert not res.iloc[0]["converged"]
    assert "ConvergenceWarning" in res.iloc[0]["skip_reason"]
    m = CauseSpecificCox().fit(d.assign(ev=d.site), "T", "ev", 1, ["gene", "tiny"])
    assert m.converged is False


def test_interaction_convergence_warning_sets_not_converged():
    d = _three_way_frame()
    d["tiny"] = np.random.default_rng(1).normal(size=len(d)) * 1e-4
    res = InteractionCox().fit_three_way(d, "g", "t", "p", "T", "E", covariates=["tiny"])
    assert res.converged is False
    assert "ConvergenceWarning" in res.error_message
    out = stratified_effect(d, "g", "t", "T", "E", covariates=["tiny"])
    assert out["converged"] is False


# ----------------------------------------------------------------------
# Bug 4: reproducible Aalen-Johansen CIF
# ----------------------------------------------------------------------

def test_cumulative_incidence_reproducible_with_ties():
    rng = np.random.default_rng(3)
    fit_df = pd.DataFrame({
        "T": np.round(rng.exponential(5, 300)),
        "ev": rng.integers(0, 3, 300),
        "a": rng.normal(size=300),
    })
    cs = CauseSpecificCox().fit(fit_df, "T", "ev", 1, ["a"])
    T = pd.Series(np.round(rng.exponential(5, 300)))
    ev = pd.Series(rng.integers(0, 3, 300))
    r1 = cs.cumulative_incidence(T, ev, times=[2, 5, 10])
    r2 = cs.cumulative_incidence(T, ev, times=[2, 5, 10])
    np.testing.assert_array_equal(r1.values, r2.values)


# ----------------------------------------------------------------------
# Bug 5: independence suite / BF / TOST alpha
# ----------------------------------------------------------------------

def test_independence_suite_constant_variable_no_crash():
    x = np.r_[np.zeros(10), np.ones(10)]
    res = independence_suite(x, np.zeros(20))
    assert np.isnan(res["bf10"])
    assert np.isnan(res["chi2_p"])
    assert res["independent"] is None
    assert "constant" in res["note"]
    # existing keys preserved
    for key in ["chi2_stat", "chi2_p", "phi", "spearman_rho", "kendall_tau",
                "cramers_v", "discordance_rate", "bf10", "n"]:
        assert key in res


def test_bayes_factor_uses_likelihood_ratio_g2():
    from scipy import stats
    tab = np.array([[40, 5], [5, 2]])
    g2 = stats.chi2_contingency(tab, correction=False, lambda_="log-likelihood")[0]
    expected = np.exp((g2 - np.log(tab.sum())) / 2)
    assert bayes_factor_independence(tab) == pytest.approx(expected)


def test_tost_alpha_parameter():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 50)
    y = rng.normal(0.1, 1.5, 60)
    r05, p1, p2 = tost_equivalence(x, y, 0.5)
    r_loose, q1, q2 = tost_equivalence(x, y, 0.5, alpha=0.5)
    assert (p1, p2) == (q1, q2)
    assert max(p1, p2) < 0.5 and r_loose is True
    assert r05 is (max(p1, p2) < 0.05)


# ----------------------------------------------------------------------
# Bug 6: bootstrap / CV failures, optimistic delta-C p-value, PH errors
# ----------------------------------------------------------------------

def test_bootstrap_delta_c_has_no_in_sample_p_value():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"x": rng.normal(size=n)})
    df["T"] = rng.exponential(np.exp(-0.5 * df.x))
    df["E"] = (rng.uniform(size=n) < 0.7).astype(int)
    noise = [f"noise{k}" for k in range(5)]
    for k in noise:
        df[k] = rng.normal(size=n)
    res = bootstrap_delta_c(df, ["x"], ["x"] + noise, "T", "E", n_bootstrap=20)
    assert len(res) == 4
    assert np.isnan(res[3])


def test_bootstrap_ci_reports_failed_resamples():
    calls = {"n": 0}

    def stat(arr):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise RuntimeError("boom")
        return float(np.mean(arr))

    boot = BootstrapCI(n_bootstrap=20, seed=0)
    with pytest.warns(RuntimeWarning, match="resamples failed"):
        boot.run(stat, np.arange(30.0))
    assert boot.n_failed_ == 10


def test_cv_concordance_constant_fold_not_scored_half():
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({"c": np.zeros(n), "T": rng.exponential(1, n), "E": 1})
    with pytest.warns(RuntimeWarning, match="folds failed"):
        mean_c, _, folds = cv_concordance(df, ["c"], "T", "E", n_folds=5)
    assert np.isnan(mean_c)
    assert all(np.isnan(folds))


def test_ph_test_failure_returns_error_marker(survival_df, monkeypatch):
    import lifelines.statistics

    model = PenalizedCox().fit(survival_df, "OS_MONTHS", "OS_STATUS", ["GENE_A", "AGE"])

    def _raise(*a, **k):
        raise ValueError("forced")

    monkeypatch.setattr(lifelines.statistics, "proportional_hazard_test", _raise)
    res = model.check_proportional_hazards()
    assert "__error__" in res and "forced" in res["__error__"]


# ----------------------------------------------------------------------
# Bug 7: DeltaC uses cv_delta_c; failed fits are NaN; qcut ties
# ----------------------------------------------------------------------

def test_deltac_uses_cv_delta_c(survival_df, monkeypatch):
    import cascade.core.clinical as clinical

    called = {}
    real = clinical.cv_delta_c

    def spy(*args, **kwargs):
        called["yes"] = True
        assert "n_bootstrap" not in kwargs
        return real(*args, **kwargs)

    monkeypatch.setattr(clinical, "cv_delta_c", spy)
    monkeypatch.setattr(
        clinical.DeltaC, "_compute_manual",
        lambda *a, **k: pytest.fail("manual fallback should not run"),
    )
    out = clinical.DeltaC(n_folds=5, n_bootstrap=50).compute(
        survival_df, ["AGE"], ["AGE", "GENE_A"], "OS_MONTHS", "OS_STATUS"
    )
    assert called.get("yes")
    for key in ["delta_c", "ci", "p_value", "base_c", "full_c", "n_failed_folds"]:
        assert key in out
    assert np.isfinite(out["delta_c"])


def test_deltac_manual_failed_fit_is_nan(survival_df, monkeypatch):
    import lifelines
    import cascade.core.clinical as clinical

    def _raise(self, *a, **k):
        raise RuntimeError("fit failed")

    monkeypatch.setattr(lifelines.CoxPHFitter, "fit", _raise)
    out = clinical.DeltaC(n_folds=5, n_bootstrap=20)._compute_manual(
        survival_df, ["AGE"], ["AGE", "GENE_A"], "OS_MONTHS", "OS_STATUS"
    )
    assert np.isnan(out["delta_c"])  # previously 0.5 - 0.5 = 0.0


def test_risk_groups_tied_scores_do_not_raise():
    from cascade.core.clinical import RiskGroupAnalysis

    df = pd.DataFrame({"score": [0.0] * 50 + list(range(1, 51))})
    with pytest.warns(RuntimeWarning, match="quantile groups"):
        out = RiskGroupAnalysis(n_groups=3).create_groups(df, "score")
    assert out["risk_group"].notna().all()
    assert out["risk_group"].nunique() == 2
    assert int(out.loc[0, "risk_group"]) == 1


# ----------------------------------------------------------------------
# Bug 8: check_all runs #10 and #11
# ----------------------------------------------------------------------

def test_check_all_runs_censoring_and_center_checks():
    rng = np.random.default_rng(0)
    n = 800
    df = pd.DataFrame({
        "gene": (rng.uniform(size=n) < 0.3).astype(int),
        "center": rng.choice(["A", "B", "C"], n),
    })
    t_event = rng.exponential(24, n)
    t_cens = rng.exponential(np.where(df["gene"] == 1, 8, 40))
    df["time"] = np.minimum(t_event, t_cens)
    df["event"] = (t_event <= t_cens).astype(int)
    # centre A has a much higher prevalence
    df.loc[df.center == "A", "gene"] = (rng.uniform(size=(df.center == "A").sum()) < 0.7).astype(int)

    found = PitfallDetector().check_all(
        survival_df=df, duration_col="time", event_col="event",
        biomarker_cols=["gene"], center_col="center",
    )
    ids = {w.pitfall.id for w in found}
    assert 10 in ids and 11 in ids


# ----------------------------------------------------------------------
# Bug 9: singular matrix -- scale-invariant, intercept-aware
# ----------------------------------------------------------------------

def _is_flagged(X):
    return any(w.severity == Severity.CRITICAL for w in check_singular_matrix(X))


def test_singular_constant_column_flagged():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50), "c": 1.0})
    assert _is_flagged(X)


def test_singular_affine_combination_flagged():
    rng = np.random.default_rng(0)
    n_sites = rng.integers(1, 8, 100).astype(float)
    X = pd.DataFrame({"N_SITES": n_sites, "ACQ_RATE": (n_sites - 1) / 6,
                      "z": rng.normal(size=100)})
    ws = check_singular_matrix(X)
    assert any("rank" in w.message for w in ws)


def test_singular_full_dummy_set_flagged():
    rng = np.random.default_rng(0)
    cat = rng.integers(0, 3, 100)
    X = pd.DataFrame({f"d{k}": (cat == k).astype(float) for k in range(3)})
    X["z"] = rng.normal(size=100)
    assert _is_flagged(X)


def test_singular_rescaled_column_not_flagged():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"days": rng.normal(size=100) * 1e6, "b": rng.normal(size=100) * 1e-5})
    assert check_singular_matrix(X) == []


# ----------------------------------------------------------------------
# Bug 10: collinearity with NaN
# ----------------------------------------------------------------------

def test_collinearity_handles_nan_rows():
    rng = np.random.default_rng(0)
    a = rng.normal(size=100)
    X = pd.DataFrame({"a": a, "b": 2 * a + 1, "c": rng.normal(size=100)})
    X.loc[::7, "c"] = np.nan
    with pytest.warns(RuntimeWarning, match="dropped 15"):
        ws = check_collinearity(X)
    assert any(w.severity == Severity.CRITICAL for w in ws)


# ----------------------------------------------------------------------
# Bug 11: covariate leakage
# ----------------------------------------------------------------------

def _leakage_frame(seed=0, n=300):
    rng = np.random.default_rng(seed)
    landmark = np.full(n, 180.0)
    ici_date = rng.uniform(0, 720, n)
    df = pd.DataFrame({
        "LANDMARK": landmark,
        "ICI_DATE": ici_date,
        "DEATH_DATE": rng.uniform(200, 1000, n),
        "AGE": rng.normal(65, 10, n),
        "SEX": rng.integers(0, 2, n),
        "ICI_PRE": (ici_date <= landmark).astype(int),   # time-locked, correct
        "EVER_ICI": np.ones(n, dtype=int),               # leaks
    })
    df["FOLLOWUP"] = rng.exponential(500, n)
    return df


def test_leakage_date_mode_only_mapped_covariates():
    df = _leakage_frame()
    ws = check_landmark_leakage(
        df, "LANDMARK", ["AGE", "ICI_PRE", "EVER_ICI"],
        {"ICI_PRE": "ICI_DATE", "EVER_ICI": "ICI_DATE"},
    )
    critical = {w.location for w in ws if w.severity == Severity.CRITICAL}
    assert critical == {"EVER_ICI"}


def test_leakage_date_mode_list_does_not_cross_test_unrelated_dates():
    df = _leakage_frame()
    ws = check_landmark_leakage(
        df, "LANDMARK", ["AGE", "ICI_PRE"], ["DEATH_DATE", "ICI_DATE"]
    )
    assert not any(w.severity == Severity.CRITICAL for w in ws)


def test_leakage_heuristic_constant_landmark_and_sex():
    df = _leakage_frame()
    ws = check_landmark_leakage(
        df, "LANDMARK", ["SEX", "EVER_ICI", "ICI_PRE"], followup_col="FOLLOWUP"
    )
    flagged = {w.location for w in ws}
    assert "EVER_ICI" in flagged
    assert "SEX" not in flagged and "ICI_PRE" not in flagged


# ----------------------------------------------------------------------
# Bug 12: comment header -- any '#', empty leading field
# ----------------------------------------------------------------------

def test_comment_any_hash_and_empty_first_field(tmp_path):
    f = tmp_path / "d.txt"
    f.write_text(
        "#meta\n"
        "PATIENT_ID\tNOTE\tECOG\n"
        "\tTumor #2\t1\n"
        "P-2\tok\t2\n"
    )
    ws = check_comment_corruption(f)
    assert len(ws) == 1
    assert "'NOTE'" in ws[0].message


# ----------------------------------------------------------------------
# Bug 13: separation on HR scale; label
# ----------------------------------------------------------------------

def test_separation_narrow_log_ci_not_flagged_and_label():
    df = pd.DataFrame({"name": ["narrow", "wide"],
                       "ci_lower": [-2.0, -8.0], "ci_upper": [-0.01, 3.0]})
    ws = check_separation_problems(df)
    names = [w.message for w in ws]
    assert not any("narrow" in m for m in names)
    assert any("wide" in m for m in names)
    assert all("Singular matrix" not in w.pitfall.name for w in ws)
    assert all("eparation" in w.pitfall.name for w in ws)


# ----------------------------------------------------------------------
# Bug 14: informative censoring -- string events, recorded fit failures
# ----------------------------------------------------------------------

def test_informative_censoring_string_events_and_fit_failures(monkeypatch):
    rng = np.random.default_rng(0)
    n = 600
    gene = (rng.uniform(size=n) < 0.3).astype(int)
    t_event = rng.exponential(24, n)
    t_cens = rng.exponential(np.where(gene == 1, 8, 40))
    df = pd.DataFrame({
        "gene": gene,
        "time": np.minimum(t_event, t_cens),
        "status": np.where(t_event <= t_cens, "1:DECEASED", "0:LIVING"),
    })
    ws = check_informative_censoring(df, "time", "status", ["gene"])
    assert len(ws) == 1 and ws[0].pitfall.id == 10

    import lifelines

    def _raise(self, *a, **k):
        raise RuntimeError("fit failed")

    monkeypatch.setattr(lifelines.CoxPHFitter, "fit", _raise)
    ws = check_informative_censoring(df, "time", "status", ["gene"])
    assert len(ws) == 1 and "NOT assessed" in ws[0].message


# ----------------------------------------------------------------------
# Bug 15: pitfall counts
# ----------------------------------------------------------------------

def test_pitfall_counts_are_eleven_and_seven():
    import cascade.pitfalls as pf
    import cascade.pitfalls.registry as reg

    r = PitfallRegistry()
    assert len(r.list_all()) == 11
    assert len(r.list_automatable(include_partial=False)) == 7
    assert len(r.list_automatable()) == 9
    assert "11 canonical" in pf.__doc__
    assert "9 canonical" not in reg.__doc__
    assert ">>> len(registry.list_all())\n    11" in reg.PitfallRegistry.__doc__
