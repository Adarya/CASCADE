"""Regression tests for audit-confirmed bugs in Layers 0-3
(cohort, discovery, confirmation, predictive).

Each test fails on the pre-fix code and passes after the fix.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from cascade.core import BiomarkerScreen, OrthogonalConfirm, PredictiveTest
from cascade.core.cohort import CohortBuilder, FeatureEngineer


# ----------------------------------------------------------------------
# Shared synthetic data
# ----------------------------------------------------------------------


@pytest.fixture
def binary_df():
    """Cross-sectional data: GENE_A raises P(HAS_SITE), GENE_B null."""
    rng = np.random.RandomState(7)
    n = 600
    a = rng.binomial(1, 0.3, n)
    b = rng.binomial(1, 0.3, n)
    lin = -1.0 + 1.2 * a
    y = rng.binomial(1, 1 / (1 + np.exp(-lin)))
    return pd.DataFrame({"GENE_A": a, "GENE_B": b, "HAS_SITE": y})


@pytest.fixture
def site_df():
    """Survival data with per-site binary event columns (CRC tropism shape)."""
    rng = np.random.RandomState(11)
    n = 800
    a = rng.binomial(1, 0.35, n)
    b = rng.binomial(1, 0.35, n)
    t = rng.exponential(30, n).clip(0.5, 120)
    liver = rng.binomial(1, 1 / (1 + np.exp(-(-1.0 + 1.3 * a))))
    lung = rng.binomial(1, 0.3, n)
    return pd.DataFrame(
        {
            "GENE_A": a,
            "GENE_B": b,
            "TIME": t,
            "EVENT": np.maximum(liver, lung),
            "LIVER": liver,
            "LUNG": lung,
        }
    )


# ----------------------------------------------------------------------
# Bug 1: confirm() with event_col=None (documented binary-outcome call)
# ----------------------------------------------------------------------


def test_bug1_confirm_binary_outcome_without_event_col(binary_df):
    primary = BiomarkerScreen(method="logistic", min_exposed=10, min_events=3).screen(
        binary_df, ["GENE_A", "GENE_B"], "HAS_SITE"
    )
    conf = OrthogonalConfirm(confirm_method="logistic").confirm(
        primary, binary_df, ["GENE_A", "GENE_B"], "HAS_SITE"
    )
    row = conf.set_index("biomarker").loc["GENE_A"]
    assert np.isfinite(row["confirm_hr"]) and row["confirm_hr"] > 1


# ----------------------------------------------------------------------
# Bug 2: confirm() accepts screen_competing_risks output (gene/site keys)
# ----------------------------------------------------------------------


def test_bug2_confirm_accepts_competing_risks_output(site_df):
    screen = BiomarkerScreen(method="competing_risks", min_exposed=10, min_events=3)
    primary = screen.screen_competing_risks(
        site_df, ["GENE_A", "GENE_B"], ["LIVER", "LUNG"], "TIME", "EVENT"
    )
    assert "biomarker" not in primary.columns
    conf = OrthogonalConfirm().confirm(
        primary, site_df, ["GENE_A", "GENE_B"], outcome_col="HAS_SITE"
    )
    assert len(conf) == len(primary)
    assert {"gene", "site", "confirm_hr", "confidence_tier"} <= set(conf.columns)
    liver_a = conf[(conf["gene"] == "GENE_A") & (conf["site"] == "LIVER")].iloc[0]
    # Each pair is confirmed against its own site column
    assert liver_a["confirm_hr"] > 1.5
    assert liver_a["concordance_direction"] == 1


# ----------------------------------------------------------------------
# Bug 3: concordance_metrics defaults; NaN-vs-NaN not discordant
# ----------------------------------------------------------------------


def test_bug3_concordance_metrics_default_columns(survival_df, biomarker_cols):
    primary = BiomarkerScreen(method="cox", min_exposed=10, min_events=3).screen(
        survival_df, biomarker_cols, "OS_MONTHS", "OS_STATUS"
    )
    confirm = BiomarkerScreen(method="logistic", min_exposed=10, min_events=3).screen(
        survival_df, biomarker_cols, "OS_STATUS"
    )
    metrics = OrthogonalConfirm.concordance_metrics(primary, confirm)
    assert 0.0 <= metrics["direction_agreement_pct"] <= 100.0


def test_bug3_nan_effects_not_counted_as_discordant(binary_df):
    df = binary_df.copy()
    df["RARE"] = 0
    df.loc[:2, "RARE"] = 1  # far below min_exposed -> NaN effect in both
    primary = BiomarkerScreen(method="logistic", min_exposed=10, min_events=3).screen(
        df, ["GENE_A", "RARE"], "HAS_SITE"
    )
    conf = OrthogonalConfirm().confirm(primary, df, ["GENE_A", "RARE"], "HAS_SITE")
    rare = conf.set_index("biomarker").loc["RARE"]
    assert np.isnan(rare["concordance_direction"])


# ----------------------------------------------------------------------
# Bug 4: load_tsv default must not corrupt '#' inside data fields
# ----------------------------------------------------------------------


def test_bug4_load_tsv_default_preserves_hash_in_data(tmp_path):
    f = tmp_path / "data_clinical.txt"
    f.write_text(
        "#Patient Identifier\tColour\tAge\n"
        "#STRING\tSTRING\tNUMBER\n"
        "PATIENT_ID\tSTYLE_COLOR\tAGE\n"
        "P-1\t#359645\t61\n"
        "P-2\tred\t55\n"
    )
    df = CohortBuilder().load_tsv(f)
    assert list(df.columns) == ["PATIENT_ID", "STYLE_COLOR", "AGE"]
    assert len(df) == 2
    assert df.loc[0, "STYLE_COLOR"] == "#359645"
    assert df.loc[0, "AGE"] == 61
    # Compatibility flag still works
    df2 = CohortBuilder().load_tsv(f, skip_comment_corruption=True)
    assert df2.loc[0, "STYLE_COLOR"] == "#359645"


# ----------------------------------------------------------------------
# Bug 5: treatment-line detection
# ----------------------------------------------------------------------


def _lines(tx, **kw):
    out = CohortBuilder.detect_treatment_lines(tx, **kw)
    return out.sort_values("START_DATE")["LINE_NUMBER"].tolist()


def test_bug5_first_stop_missing_still_splits():
    tx = pd.DataFrame(
        {
            "PATIENT_ID": ["P1"] * 3,
            "START_DATE": [0, 700, 1400],
            "STOP_DATE": [np.nan, 800, 1500],
            "AGENT": ["A", "A", "A"],
        }
    )
    assert _lines(tx) == [1, 2, 3]


def test_bug5_no_stop_column_splits_on_start_gap():
    tx = pd.DataFrame(
        {"PATIENT_ID": ["P1"] * 3, "START_DATE": [0, 700, 1400], "AGENT": ["A"] * 3}
    )
    assert _lines(tx) == [1, 2, 3]


def test_bug5_gap_below_gap_days_same_regimen_is_same_line():
    tx = pd.DataFrame(
        {
            "PATIENT_ID": ["P1"] * 2,
            "START_DATE": [0, 145],
            "STOP_DATE": [100, 200],
            "AGENT": ["FOLFOX", "FOLFOX"],
        }
    )
    assert _lines(tx, gap_days=90) == [1, 1]


def test_bug5_concurrent_and_switch_rules():
    tx = pd.DataFrame(
        {
            "PATIENT_ID": ["P1"] * 4,
            "START_DATE": [0, 10, 150, 400],
            "STOP_DATE": [120, 120, 300, 500],
            "AGENT": ["CARBOPLATIN", "PEMETREXED", "DOCETAXEL", "DOCETAXEL"],
        }
    )
    # 0/10 concurrent; 150 = new agent outside window (switch);
    # 400 = gap 100 > 90 from last stop -> new line.
    assert _lines(tx) == [1, 1, 2, 3]


# ----------------------------------------------------------------------
# Bug 6: site flags must not leak across duplicate index labels
# ----------------------------------------------------------------------


def test_bug6_site_flags_duplicate_index():
    sites = pd.DataFrame(
        {"PATIENT_ID": ["P1", "P2"], "START_DATE": [0, 0], "TUMOR_SITE": ["LIVER", "LUNG"]}
    )
    a1 = pd.DataFrame({"PATIENT_ID": ["P1"], "ANCHOR_DATE": [10]})
    a2 = pd.DataFrame({"PATIENT_ID": ["P2"], "ANCHOR_DATE": [10]})
    anchors = pd.concat([a1, a2])  # index [0, 0]
    out = CohortBuilder.compute_site_flags_at_date(sites, anchors)
    assert list(out.index) == [0, 0]
    assert out["LIVER"].tolist() == [1, 0]
    assert out["LUNG"].tolist() == [0, 1]


# ----------------------------------------------------------------------
# Bug 7: missing ONCOGENIC warns; all patients retained on request
# ----------------------------------------------------------------------


def test_bug7_missing_oncogenic_column_warns():
    muts = pd.DataFrame({"PATIENT_ID": ["P1"], "Hugo_Symbol": ["TP53"]})
    with pytest.warns(UserWarning, match="ONCOGENIC"):
        CohortBuilder.build_mutation_matrix(muts)


def test_bug7_all_patients_get_rows():
    muts = pd.DataFrame(
        {"PATIENT_ID": ["P1"], "Hugo_Symbol": ["TP53"], "ONCOGENIC": ["Oncogenic"]}
    )
    m = CohortBuilder.build_mutation_matrix(muts, all_patients=["P1", "P2", "P3"])
    assert list(m.index) == ["P1", "P2", "P3"]
    assert m.loc["P2", "TP53"] == 0 and m.loc["P1", "TP53"] == 1


# ----------------------------------------------------------------------
# Bug 8: boolean / int / string OS_STATUS in compute_ttntd
# ----------------------------------------------------------------------


@pytest.mark.parametrize("status", [True, 1, "1:DECEASED", "DECEASED"])
def test_bug8_ttntd_os_status_variants(status):
    lines = pd.DataFrame({"PATIENT_ID": ["P1"], "LINE_NUMBER": [1], "LINE_START": [0]})
    os_df = pd.DataFrame({"PATIENT_ID": ["P1"], "OS_MONTHS": [10.0], "OS_STATUS": [status]})
    out = FeatureEngineer.compute_ttntd(lines, os_df)
    assert out["TTNTD_EVENT"].iloc[0] == 1


def test_bug8_ttntd_false_is_censored():
    lines = pd.DataFrame({"PATIENT_ID": ["P1"], "LINE_NUMBER": [1], "LINE_START": [0]})
    os_df = pd.DataFrame({"PATIENT_ID": ["P1"], "OS_MONTHS": [10.0], "OS_STATUS": [False]})
    assert FeatureEngineer.compute_ttntd(lines, os_df)["TTNTD_EVENT"].iloc[0] == 0


# ----------------------------------------------------------------------
# Bug 9: continuous biomarkers; logistic uses outcome_col
# ----------------------------------------------------------------------


def test_bug9_continuous_biomarker_is_tested(survival_df):
    df = survival_df.copy()
    rng = np.random.RandomState(3)
    df["TMB"] = rng.gamma(2.0, 3.0, len(df))
    res = BiomarkerScreen(method="cox", min_exposed=20).screen(
        df, ["TMB"], "OS_MONTHS", "OS_STATUS"
    )
    row = res.iloc[0]
    assert np.isfinite(row["hr"])
    assert np.isnan(row["n_exposed"])


def test_bug9_logistic_uses_outcome_col(binary_df):
    df = binary_df.copy()
    df["OTHER"] = 0
    df.loc[df.index[::2], "OTHER"] = 1  # unrelated binary column
    res = BiomarkerScreen(method="logistic", min_exposed=10, min_events=3).screen(
        df, ["GENE_A"], "HAS_SITE", "OTHER"
    )
    import statsmodels.api as sm

    ref = sm.Logit(df["HAS_SITE"].astype(float), sm.add_constant(df[["GENE_A"]].astype(float))).fit(disp=0)
    assert res.iloc[0]["hr"] == pytest.approx(np.exp(ref.params["GENE_A"]), rel=1e-6)


# ----------------------------------------------------------------------
# Bug 10: predictive failures are not_evaluable; configurable arm size
# ----------------------------------------------------------------------


def _small_interaction_df(n_treated):
    rng = np.random.RandomState(5)
    n = 200
    treat = np.zeros(n, dtype=int)
    treat[:n_treated] = 1
    return pd.DataFrame(
        {
            "OS_MONTHS": rng.exponential(20, n),
            "OS_STATUS": rng.binomial(1, 0.7, n),
            "GENE": rng.binomial(1, 0.4, n),
            "TREATMENT": treat,
        }
    )


def test_bug10_small_arm_not_evaluable_and_configurable():
    df = _small_interaction_df(8)
    res = PredictiveTest("GENE", "TREATMENT").run(df)
    assert res.classification == "not_evaluable"
    assert "min_arm_size" in res.reason
    res2 = PredictiveTest("GENE", "TREATMENT", min_arm_size=5).run(df)
    assert res2.classification in {"predictive", "prognostic"}


def test_bug10_screen_does_not_relabel_nan_as_prognostic(interaction_df):
    df = interaction_df.copy()
    df["NOGENE"] = 0  # no variation -> untestable
    out = PredictiveTest.screen_predictive(
        df, ["GENE", "NOGENE"], "TREATMENT", "OS_MONTHS", "OS_STATUS"
    )
    out = out.set_index("gene")
    assert out.loc["NOGENE", "classification"] == "not_evaluable"
    assert out.loc["NOGENE", "reason"] != ""
    assert out.loc["GENE", "classification"] in {"predictive", "prognostic"}


def test_bug10_string_os_status_accepted(interaction_df):
    df = interaction_df.copy()
    ref = PredictiveTest("GENE", "TREATMENT").run(df)
    df["OS_STATUS"] = np.where(df["OS_STATUS"] == 1, "1:DECEASED", "0:LIVING")
    res = PredictiveTest("GENE", "TREATMENT").run(df)
    assert res.interaction_hr == pytest.approx(ref.interaction_hr)


# ----------------------------------------------------------------------
# Bug 11: multi-agent regimens get an explicit combined category
# ----------------------------------------------------------------------


def test_bug11_combination_regimen_combined_label():
    agents = pd.Series(["CARBOPLATIN+PEMBROLIZUMAB", "PEMBROLIZUMAB", "X"])
    cats = {"ICI": ["pembrolizumab"], "Platinum": ["carboplatin"]}
    out = FeatureEngineer.categorize_agents(agents, cats)
    assert out.tolist() == ["ICI + Platinum", "ICI", "Other"]
