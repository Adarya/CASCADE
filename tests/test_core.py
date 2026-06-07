"""
Tests for CASCADE core validation layers (cascade.core).

Covers: CohortBuilder, FeatureEngineer, BiomarkerScreen, SensitivitySuite,
RobustnessScorer, ArtifactGuard, LandmarkAnalysis, RiskGroupAnalysis,
ExternalValidator, and ManuscriptHelper.
"""

import numpy as np
import pandas as pd
import pytest

from cascade.core import (
    CohortBuilder,
    FeatureEngineer,
    BiomarkerScreen,
    SensitivitySuite,
    RobustnessScorer,
    ArtifactGuard,
    ArtifactReport,
    LandmarkAnalysis,
    RiskGroupAnalysis,
    ExternalValidator,
    ManuscriptHelper,
)


# ======================================================================
# Layer 0: CohortBuilder
# ======================================================================


class TestCohortBuilder:
    """Tests for CohortBuilder data loading and filtering."""

    def test_cohort_filter_stage(self):
        """filter_stage should return only rows matching the given stage values."""
        df = pd.DataFrame({
            "PATIENT_ID": [f"P-{i:07d}" for i in range(10)],
            "STAGE": ["Stage 4"] * 4 + ["Stage 1-3"] * 6,
            "OS_MONTHS": range(10),
        })

        result = CohortBuilder.filter_stage(df, stage_col="STAGE", stage_values="Stage 4")

        assert len(result) == 4, (
            f"Expected 4 Stage 4 patients, got {len(result)}."
        )
        assert all(result["STAGE"] == "Stage 4"), (
            "All filtered rows should be Stage 4."
        )

    def test_cohort_filter_stage_multiple(self):
        """filter_stage should accept multiple stage values."""
        df = pd.DataFrame({
            "STAGE": ["Stage 4", "Stage IV", "Stage 1", "Stage 4", "Stage III"],
        })

        result = CohortBuilder.filter_stage(
            df, stage_col="STAGE", stage_values=["Stage 4", "Stage IV"]
        )

        assert len(result) == 3, (
            f"Expected 3 rows matching Stage 4 or Stage IV, got {len(result)}."
        )

    def test_cohort_build_mutation_matrix(self):
        """build_mutation_matrix should produce a binary patient-by-gene matrix."""
        mutations = pd.DataFrame({
            "PATIENT_ID": ["P-001", "P-001", "P-002", "P-003", "P-003"],
            "Hugo_Symbol": ["TP53", "KRAS", "TP53", "EGFR", "TP53"],
            "ONCOGENIC": [
                "Oncogenic", "Likely Oncogenic", "Oncogenic",
                "Oncogenic", "Unknown",
            ],
        })

        matrix = CohortBuilder.build_mutation_matrix(
            mutations,
            patient_col="PATIENT_ID",
            gene_col="Hugo_Symbol",
            filter_oncogenic=True,
        )

        assert isinstance(matrix, pd.DataFrame), "Should return a DataFrame."
        assert matrix.shape == (3, 3), (
            f"Expected shape (3 patients x 3 genes), got {matrix.shape}."
        )
        # P-001 should have TP53=1 and KRAS=1
        assert matrix.loc["P-001", "TP53"] == 1, "P-001 should have TP53."
        assert matrix.loc["P-001", "KRAS"] == 1, "P-001 should have KRAS."
        # P-003 should have EGFR=1 but NOT TP53 (because TP53 for P-003 is Unknown)
        assert matrix.loc["P-003", "EGFR"] == 1, "P-003 should have EGFR."
        assert matrix.loc["P-003", "TP53"] == 0, (
            "P-003 TP53 is 'Unknown' and should not pass oncogenic filter."
        )
        # All values should be 0 or 1
        assert set(matrix.values.flatten()).issubset({0, 1}), (
            "Matrix values should be binary (0 or 1)."
        )


# ======================================================================
# Layer 0: FeatureEngineer
# ======================================================================


class TestFeatureEngineer:
    """Tests for FeatureEngineer utility methods."""

    def test_binary_from_threshold(self):
        """binary_from_threshold should correctly create binary indicators."""
        series = pd.Series([1, 2, 3, 4, 5, np.nan])
        fe = FeatureEngineer()

        result = fe.binary_from_threshold(series, threshold=3.0, above=True)

        assert result.iloc[0] == 0.0, "1 < 3, should be 0."
        assert result.iloc[2] == 1.0, "3 >= 3, should be 1."
        assert result.iloc[4] == 1.0, "5 >= 3, should be 1."
        assert np.isnan(result.iloc[5]), "NaN should be preserved."

    def test_binary_from_threshold_below(self):
        """binary_from_threshold with above=False should flag values <= threshold."""
        series = pd.Series([1, 2, 3, 4, 5])
        fe = FeatureEngineer()

        result = fe.binary_from_threshold(series, threshold=3.0, above=False)

        assert result.iloc[0] == 1.0, "1 <= 3, should be 1."
        assert result.iloc[2] == 1.0, "3 <= 3, should be 1."
        assert result.iloc[4] == 0.0, "5 > 3, should be 0."

    def test_categorize_agents(self):
        """categorize_agents should map agent names to categories."""
        agents = pd.Series([
            "PEMBROLIZUMAB", "CARBOPLATIN+PEMETREXED",
            "DOCETAXEL", "UNKNOWN_DRUG", "NIVOLUMAB",
        ])
        categories = {
            "ICI": ["pembrolizumab", "nivolumab"],
            "Platinum": ["carboplatin"],
        }

        fe = FeatureEngineer()
        result = fe.categorize_agents(agents, categories_dict=categories, default="Other")

        assert result.iloc[0] == "ICI", "PEMBROLIZUMAB should map to ICI."
        assert result.iloc[1] == "Platinum", "CARBOPLATIN+PEMETREXED should map to Platinum."
        assert result.iloc[3] == "Other", "UNKNOWN_DRUG should be Other."
        assert result.iloc[4] == "ICI", "NIVOLUMAB should map to ICI."


# ======================================================================
# Layer 1: BiomarkerScreen
# ======================================================================


class TestBiomarkerScreen:
    """Tests for BiomarkerScreen instantiation and screening."""

    def test_biomarker_screen_instantiation(self):
        """BiomarkerScreen should accept method and correction parameters."""
        screen = BiomarkerScreen(method="cox", correction="fdr_bh")
        assert screen.method == "cox", "Method should be 'cox'."
        assert screen.correction == "fdr_bh", "Correction should be 'fdr_bh'."

    def test_biomarker_screen_invalid_method(self):
        """BiomarkerScreen should reject invalid methods."""
        with pytest.raises(ValueError, match="method must be one of"):
            BiomarkerScreen(method="invalid_method")

    def test_biomarker_screen_runs(self, survival_df, biomarker_cols, covariates):
        """BiomarkerScreen.screen should produce a results DataFrame with fdr_bh."""
        screen = BiomarkerScreen(
            method="cox",
            correction="fdr_bh",
            min_exposed=10,
            min_events=3,
        )
        results = screen.screen(
            survival_df,
            biomarker_cols=biomarker_cols,
            outcome_col="OS_MONTHS",
            event_col="OS_STATUS",
            covariates=covariates,
        )

        assert isinstance(results, pd.DataFrame), "Should return a DataFrame."
        assert len(results) == len(biomarker_cols), (
            f"Should have one row per biomarker ({len(biomarker_cols)}), "
            f"got {len(results)}."
        )
        assert "biomarker" in results.columns, "Should have 'biomarker' column."
        assert "p_adjusted" in results.columns, "Should have 'p_adjusted' column."
        assert "significant" in results.columns, "Should have 'significant' column."


# ======================================================================
# Layer 4: SensitivitySuite and RobustnessScorer
# ======================================================================


class TestSensitivity:
    """Tests for SensitivitySuite and RobustnessScorer."""

    def test_sensitivity_suite_add_variant(self, survival_df):
        """add_variant and variant_names should work correctly."""
        suite = SensitivitySuite()
        suite.add_variant(
            "high_ecog",
            survival_df.loc[survival_df["ECOG"] >= 2],
            category="subgroup",
        )
        suite.add_variant(
            "young_patients",
            survival_df.loc[survival_df["AGE"] < 65],
            category="subgroup",
        )

        names = suite.variant_names
        assert len(names) == 2, f"Expected 2 variants, got {len(names)}."
        assert "high_ecog" in names, "'high_ecog' should be in variant names."
        assert "young_patients" in names, "'young_patients' should be in variant names."

    def test_sensitivity_suite_invalid_category(self, survival_df):
        """add_variant with invalid category should raise ValueError."""
        suite = SensitivitySuite()
        with pytest.raises(ValueError, match="category must be one of"):
            suite.add_variant(
                "bad_variant",
                survival_df,
                category="not_a_real_category",
            )

    def test_robustness_scorer_robust(self):
        """score should return 'robust' for concordant sensitivity results."""
        scorer = RobustnessScorer()
        result = scorer.score(
            primary_result=1.5,
            sensitivity_results=[1.3, 1.6, 1.4, 1.2],
            threshold=0.75,
        )
        assert result == "robust", (
            f"All directions concordant should be 'robust', got '{result}'."
        )

    def test_robustness_scorer_unstable(self):
        """score should return 'unstable' for direction flips."""
        scorer = RobustnessScorer()
        result = scorer.score(
            primary_result=1.5,  # HR > 1: harmful
            sensitivity_results=[1.3, 0.7, 1.4],  # 0.7 < 1: protective (flip)
            threshold=0.75,
        )
        assert result == "unstable", (
            f"Direction flip should produce 'unstable', got '{result}'."
        )

    def test_robustness_scorer_exploratory(self):
        """score should return 'exploratory' for empty sensitivity results."""
        scorer = RobustnessScorer()
        result = scorer.score(
            primary_result=1.5,
            sensitivity_results=[],
        )
        assert result == "exploratory", (
            f"Empty sensitivity results should produce 'exploratory', got '{result}'."
        )


# ======================================================================
# Layer 5: ArtifactGuard
# ======================================================================


class TestArtifactGuard:
    """Tests for ArtifactGuard statistical artifact detection."""

    def test_artifact_guard_independence(self, rng):
        """check_independence should return a dict containing 'phi'."""
        n = 300
        x = pd.Series(rng.binomial(1, 0.3, n))
        y = pd.Series(rng.binomial(1, 0.4, n))

        guard = ArtifactGuard()
        result = guard.check_independence(x, y)

        assert isinstance(result, dict), "Should return a dict."
        assert "phi" in result, "Result should contain 'phi' key."
        # For independent variables, phi should be near 0
        assert abs(result["phi"]) < 0.2, (
            f"Phi for independent variables should be near 0, got {result['phi']:.4f}."
        )

    def test_artifact_guard_confounding(self, survival_df):
        """check_baseline_confounding should detect large HR changes."""
        guard = ArtifactGuard()

        # Large difference: confounding present
        confounded = guard.check_baseline_confounding(
            survival_df,
            adjusted_hr=1.25,
            unadjusted_hr=0.84,
            threshold=0.15,
        )
        assert confounded is True, (
            "Large HR change (0.84 -> 1.25) should be detected as confounding."
        )

        # Small difference: no confounding
        not_confounded = guard.check_baseline_confounding(
            survival_df,
            adjusted_hr=1.50,
            unadjusted_hr=1.45,
            threshold=0.15,
        )
        assert not_confounded is False, (
            "Small HR change (1.45 -> 1.50) should NOT be detected as confounding."
        )

    def test_artifact_guard_floor_ceiling(self, survival_df):
        """check_floor_ceiling_effects should detect floor/ceiling at ECOG extremes."""
        guard = ArtifactGuard()
        result = guard.check_floor_ceiling_effects(
            survival_df,
            variable_col="ECOG",
            min_val=0,
            max_val=4,
        )

        assert isinstance(result, dict), "Should return a dict."
        assert "floor_pct" in result, "Should contain 'floor_pct'."
        assert "ceiling_pct" in result, "Should contain 'ceiling_pct'."
        assert result["floor_pct"] >= 0, "floor_pct should be non-negative."
        assert result["ceiling_pct"] >= 0, "ceiling_pct should be non-negative."
        # ECOG has ~20% at 0 (floor) and ~5% at 4 (ceiling) per conftest
        assert result["floor_pct"] > 5.0, (
            "ECOG floor (=0) percentage should be > 5%."
        )


# ======================================================================
# Layer 6: LandmarkAnalysis
# ======================================================================


class TestLandmarkAnalysis:
    """Tests for LandmarkAnalysis."""

    def test_landmark_restrict_alive(self, survival_df):
        """restrict_to_alive should keep only patients with OS >= landmark."""
        lm = LandmarkAnalysis()
        restricted = lm.restrict_to_alive(survival_df, os_col="OS_MONTHS", landmark_months=6.0)

        assert len(restricted) <= len(survival_df), (
            "Restricted cohort should not be larger than original."
        )
        assert all(restricted["OS_MONTHS"] >= 6.0), (
            "All patients in restricted cohort should have OS >= 6 months."
        )
        # Some patients should be excluded
        assert len(restricted) < len(survival_df), (
            "Some patients should be excluded by the 6-month landmark."
        )

    def test_landmark_residual_survival(self, survival_df):
        """compute_residual_survival should add OS_FROM_LM = OS - landmark."""
        lm = LandmarkAnalysis()
        landmark = 6.0
        restricted = lm.restrict_to_alive(survival_df, os_col="OS_MONTHS", landmark_months=landmark)
        result = lm.compute_residual_survival(restricted, os_col="OS_MONTHS", landmark_months=landmark)

        assert "OS_FROM_LM" in result.columns, "Should add OS_FROM_LM column."
        # OS_FROM_LM should equal OS_MONTHS - landmark, clipped at 0
        expected = (restricted["OS_MONTHS"] - landmark).clip(lower=0)
        pd.testing.assert_series_equal(
            result["OS_FROM_LM"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            atol=1e-6,
        )


# ======================================================================
# Layer 6: RiskGroupAnalysis
# ======================================================================


class TestRiskGroupAnalysis:
    """Tests for RiskGroupAnalysis."""

    def test_risk_group_create(self, survival_df):
        """create_groups should add a risk_group column with expected number of groups."""
        rga = RiskGroupAnalysis(n_groups=3, method="quantile")
        result = rga.create_groups(survival_df, score_col="BURDEN", group_col="risk_group")

        assert "risk_group" in result.columns, "Should add 'risk_group' column."
        # Should have 3 unique groups (some may be NaN if score is NaN)
        n_groups = result["risk_group"].dropna().nunique()
        assert n_groups <= 3, (
            f"Expected at most 3 groups, got {n_groups}."
        )
        assert n_groups >= 2, (
            f"Expected at least 2 groups, got {n_groups}."
        )


# ======================================================================
# Layer 7: ExternalValidator
# ======================================================================


class TestExternalValidator:
    """Tests for ExternalValidator feature alignment and evaluation."""

    def test_external_validator_align(self):
        """align_features should add missing columns and drop extras."""
        train = pd.DataFrame({
            "feat_a": [1, 2, 3],
            "feat_b": [4, 5, 6],
            "feat_c": [7, 8, 9],
        })
        test = pd.DataFrame({
            "feat_a": [10, 20],
            "feat_d": [30, 40],  # Extra column, not in train
            # feat_b and feat_c are missing
        })

        aligned = ExternalValidator.align_features(train, test, fill_value=0.0)

        assert list(aligned.columns) == list(train.columns), (
            "Aligned columns should match training columns exactly."
        )
        assert aligned.shape == (2, 3), (
            f"Aligned shape should be (2, 3), got {aligned.shape}."
        )
        # Missing features should be filled with 0
        assert all(aligned["feat_b"] == 0.0), "Missing feat_b should be filled with 0."
        assert all(aligned["feat_c"] == 0.0), "Missing feat_c should be filled with 0."
        # feat_d should be dropped
        assert "feat_d" not in aligned.columns, "Extra column feat_d should be dropped."

    def test_external_validator_evaluate(self, rng):
        """evaluate should return balanced_accuracy and kappa."""
        n = 100
        labels = ["A", "B", "C"]
        y_true = rng.choice(labels, n)
        # Make predictions mostly correct with some noise
        y_pred = y_true.copy()
        noise_idx = rng.choice(n, size=15, replace=False)
        y_pred[noise_idx] = rng.choice(labels, size=15)

        result = ExternalValidator.evaluate(y_true, y_pred)

        assert isinstance(result, dict), "evaluate should return a dict."
        assert "balanced_accuracy" in result, "Should contain 'balanced_accuracy'."
        assert "kappa" in result, "Should contain 'kappa'."
        assert 0.5 <= result["balanced_accuracy"] <= 1.0, (
            f"Balanced accuracy should be reasonable, got {result['balanced_accuracy']:.3f}."
        )
        assert result["kappa"] > 0, (
            f"Kappa should be positive for mostly-correct predictions, got {result['kappa']:.3f}."
        )
        assert result["n_samples"] == n, (
            f"n_samples should be {n}, got {result['n_samples']}."
        )


# ======================================================================
# Layer 8: ManuscriptHelper
# ======================================================================


class TestManuscriptHelper:
    """Tests for ManuscriptHelper quality checks."""

    def test_manuscript_word_count(self):
        """check_word_count should correctly count words."""
        text = "This is a simple test with seven words."
        result = ManuscriptHelper.check_word_count(text, max_words=100)

        assert result["word_count"] == 8, (
            f"Expected 8 words, got {result['word_count']}."
        )
        assert result["within_limit"] is True, (
            "8 words should be within 100-word limit."
        )

    def test_manuscript_word_count_exceeds(self):
        """check_word_count should flag text exceeding the limit."""
        text = " ".join(["word"] * 150)
        result = ManuscriptHelper.check_word_count(text, max_words=100)

        assert result["within_limit"] is False, (
            "150 words should exceed 100-word limit."
        )

    def test_manuscript_display_items(self):
        """check_display_items should flag when limits are exceeded."""
        result = ManuscriptHelper.check_display_items(
            figures=4, tables=2, max_total=6
        )
        assert result["within_limit"] is True, (
            "4 + 2 = 6 should be within max_total=6."
        )

        result_over = ManuscriptHelper.check_display_items(
            figures=5, tables=3, max_total=6
        )
        assert result_over["within_limit"] is False, (
            "5 + 3 = 8 should exceed max_total=6."
        )

    def test_manuscript_fabricated_refs(self):
        """flag_potential_fabricated_refs should flag references with future years."""
        refs = [
            "Smith et al. Nature 2024;636:728-736. doi:10.1038/s41586-024-08167-5",
            "Doe et al. Science 2030;400:100-110.",  # Future year
            "Anonymous. J Clin Oncol 2023.",  # No volume/pages/DOI
        ]

        warnings = ManuscriptHelper.flag_potential_fabricated_refs(
            refs, current_year=2026
        )

        assert isinstance(warnings, list), "Should return a list."
        # Should flag Ref 2 (year 2030)
        has_future_year_warning = any("future year" in w.lower() for w in warnings)
        assert has_future_year_warning, (
            "Should flag the reference with year 2030 as future year."
        )

    def test_manuscript_checklist(self):
        """generate_cascade_checklist should produce valid markdown."""
        results_dict = {
            0: {"status": "complete", "notes": "Stage IV NSCLC"},
            1: {"status": "complete", "notes": "16 FDR-significant"},
            2: True,
            3: False,
            5: {"status": "partial"},
        }

        checklist = ManuscriptHelper.generate_cascade_checklist(results_dict)

        assert isinstance(checklist, str), "Should return a string."
        assert "CASCADE Validation Checklist" in checklist, (
            "Checklist should contain the title."
        )
        assert "[x]" in checklist, (
            "Completed layers should have [x] checkboxes."
        )
        assert "[ ]" in checklist, (
            "Incomplete layers should have [ ] checkboxes."
        )
        # Layer 0 should be marked complete
        assert "Layer 0" in checklist, "Should reference Layer 0."
