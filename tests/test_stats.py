"""
Tests for CASCADE statistical primitives (cascade.stats).

Covers: PenalizedCox, InteractionCox, stratified_effect, multiple testing
corrections, equivalence / independence tests, bootstrap CI, and
cross-validated concordance.
"""

import numpy as np
import pandas as pd
import pytest

from cascade.stats import (
    PenalizedCox,
    InteractionCox,
    InteractionResult,
    stratified_effect,
    apply_fdr,
    apply_bonferroni,
    apply_holm,
    phi_coefficient,
    tost_equivalence,
    independence_suite,
    BootstrapCI,
    bootstrap_delta_c,
    cv_concordance,
    cv_delta_c,
)


# ======================================================================
# PenalizedCox
# ======================================================================


class TestPenalizedCox:
    """Tests for the PenalizedCox wrapper."""

    def test_penalized_cox_fit(self, survival_df, covariates):
        """Fit PenalizedCox on synthetic survival data and verify basic properties."""
        model = PenalizedCox(penalizer=0.01)
        model.fit(
            survival_df,
            duration_col="OS_MONTHS",
            event_col="OS_STATUS",
            covariates=["GENE_A", "GENE_B", "TREATMENT"] + covariates,
        )

        assert model.converged, "Model should converge on synthetic data."
        assert 0.4 <= model.concordance_index_ <= 0.7, (
            f"C-index {model.concordance_index_:.3f} outside expected range [0.4, 0.7]."
        )
        assert isinstance(model.summary, pd.DataFrame), (
            "Summary should be a pandas DataFrame."
        )
        assert model.summary.shape[0] > 0, "Summary should have at least one row."

    def test_penalized_cox_constant_column(self, survival_df, covariates):
        """A constant column should be automatically dropped with a warning."""
        df = survival_df.copy()
        df["CONST_COL"] = 0

        model = PenalizedCox(penalizer=0.01)
        with pytest.warns(UserWarning, match="dropped.*constant"):
            model.fit(
                df,
                duration_col="OS_MONTHS",
                event_col="OS_STATUS",
                covariates=["GENE_A", "CONST_COL"] + covariates,
            )

        assert model.converged, "Model should still converge after dropping constant column."
        assert "CONST_COL" in model.dropped_columns, (
            "CONST_COL should appear in dropped_columns."
        )
        assert "CONST_COL" not in model.summary.index, (
            "CONST_COL should not appear in model summary."
        )

    def test_penalized_cox_ph_test(self, survival_df, covariates):
        """check_proportional_hazards should return a dict of p-values."""
        model = PenalizedCox(penalizer=0.01)
        model.fit(
            survival_df,
            duration_col="OS_MONTHS",
            event_col="OS_STATUS",
            covariates=["GENE_A", "GENE_B"] + covariates,
        )

        ph_results = model.check_proportional_hazards()

        assert isinstance(ph_results, dict), "PH test should return a dict."
        # Should have a p-value for at least one covariate
        if ph_results:  # May be empty if the test cannot run
            for key, pval in ph_results.items():
                assert isinstance(key, str), "Keys should be covariate names."
                assert 0.0 <= pval <= 1.0, (
                    f"P-value for '{key}' should be in [0, 1], got {pval}."
                )


# ======================================================================
# InteractionCox
# ======================================================================


class TestInteractionCox:
    """Tests for two-way and three-way interaction models."""

    def test_interaction_two_way(self, interaction_df):
        """fit_two_way should produce a converged InteractionResult with effects."""
        model = InteractionCox(penalizer=0.01, min_subgroup_n=5, min_subgroup_events=2)
        result = model.fit_two_way(
            interaction_df,
            factor1="GENE",
            factor2="TREATMENT",
            duration_col="OS_MONTHS",
            event_col="OS_STATUS",
            covariates=["AGE", "SEX"],
        )

        assert isinstance(result, InteractionResult), (
            "Should return an InteractionResult."
        )
        assert result.converged, (
            f"Two-way model should converge. Error: {result.error_message}"
        )
        expected_key = "GENE:TREATMENT"
        assert expected_key in result.two_way_effects, (
            f"two_way_effects should contain '{expected_key}'. "
            f"Got keys: {list(result.two_way_effects.keys())}"
        )
        effect = result.two_way_effects[expected_key]
        assert "hr" in effect, "Two-way effect should contain 'hr' key."
        assert "p" in effect, "Two-way effect should contain 'p' key."

    def test_interaction_three_way(self, interaction_df):
        """fit_three_way should produce three_way_effect with hr/p/ci keys."""
        model = InteractionCox(penalizer=0.01, min_subgroup_n=5, min_subgroup_events=2)
        result = model.fit_three_way(
            interaction_df,
            factor1="GENE",
            factor2="TREATMENT",
            factor3="PHENOTYPE",
            duration_col="OS_MONTHS",
            event_col="OS_STATUS",
            covariates=["AGE", "SEX"],
        )

        assert isinstance(result, InteractionResult), (
            "Should return an InteractionResult."
        )
        assert result.converged, (
            f"Three-way model should converge. Error: {result.error_message}"
        )
        assert isinstance(result.three_way_effect, dict), (
            "three_way_effect should be a dict."
        )
        assert len(result.three_way_effect) > 0, (
            "three_way_effect should not be empty for a converged model."
        )
        for required_key in ("hr", "p", "ci_lower", "ci_upper"):
            assert required_key in result.three_way_effect, (
                f"three_way_effect should contain '{required_key}'."
            )

    def test_interaction_min_subgroup(self, rng):
        """When subgroup sizes are too small, the model should not converge."""
        n = 100
        # Create extremely skewed data so gene=1 & treatment=1 & phenotype=1
        # has very few patients.
        df = pd.DataFrame({
            "PATIENT_ID": [f"P-{i:07d}" for i in range(n)],
            "OS_MONTHS": rng.exponential(10, n).clip(0.5, 60),
            "OS_STATUS": rng.binomial(1, 0.6, n),
            "GENE": np.concatenate([np.ones(3), np.zeros(n - 3)]).astype(int),
            "TREATMENT": rng.binomial(1, 0.5, n),
            "PHENOTYPE": rng.binomial(1, 0.5, n),
            "AGE": rng.normal(65, 10, n).astype(int),
        })

        model = InteractionCox(
            penalizer=0.01,
            min_subgroup_n=20,
            min_subgroup_events=5,
        )
        result = model.fit_three_way(
            df,
            factor1="GENE",
            factor2="TREATMENT",
            factor3="PHENOTYPE",
            duration_col="OS_MONTHS",
            event_col="OS_STATUS",
        )

        assert not result.converged, (
            "Model should not converge when subgroup sizes are insufficient."
        )
        assert "insufficient" in result.error_message.lower() or "subgroup" in result.error_message.lower(), (
            f"Error message should mention subgroup issue. Got: {result.error_message}"
        )

    def test_stratified_effect(self, interaction_df):
        """stratified_effect should return hr_treated, hr_untreated, interaction_p."""
        result = stratified_effect(
            interaction_df,
            gene_col="GENE",
            treatment_col="TREATMENT",
            duration_col="OS_MONTHS",
            event_col="OS_STATUS",
            covariates=["AGE", "SEX"],
        )

        assert isinstance(result, dict), "Should return a dict."
        assert result["converged"], (
            f"Stratified effect model should converge. Error: {result.get('error_message', '')}"
        )
        assert not np.isnan(result["hr_treated"]), "hr_treated should not be NaN."
        assert not np.isnan(result["hr_untreated"]), "hr_untreated should not be NaN."
        assert not np.isnan(result["interaction_p"]), "interaction_p should not be NaN."
        assert 0.0 <= result["interaction_p"] <= 1.0, (
            f"interaction_p should be in [0, 1], got {result['interaction_p']}."
        )


# ======================================================================
# Multiple Testing Corrections
# ======================================================================


class TestMultipleTesting:
    """Tests for FDR, Bonferroni, and Holm corrections."""

    def test_apply_fdr(self):
        """FDR correction on known p-values produces sensible adjusted values."""
        raw_p = np.array([0.001, 0.01, 0.05, 0.10, 0.50])
        adjusted, significant = apply_fdr(raw_p, alpha=0.05)

        assert len(adjusted) == len(raw_p), "Adjusted array length should match input."
        assert len(significant) == len(raw_p), "Significant array length should match input."
        # Adjusted values should be >= raw values
        for i in range(len(raw_p)):
            assert adjusted[i] >= raw_p[i] - 1e-12, (
                f"FDR-adjusted p[{i}]={adjusted[i]:.4f} should be >= raw p={raw_p[i]:.4f}."
            )
        # The most significant p-value should still be significant
        assert significant[0], (
            "The smallest p-value (0.001) should remain significant after FDR."
        )

    def test_apply_bonferroni(self):
        """Bonferroni correction should multiply p by n (capped at 1)."""
        raw_p = np.array([0.01, 0.02, 0.10])
        adjusted, significant = apply_bonferroni(raw_p, alpha=0.05)

        n = len(raw_p)
        # Bonferroni: adjusted = min(p * n, 1)
        for i in range(n):
            expected = min(raw_p[i] * n, 1.0)
            assert adjusted[i] == pytest.approx(expected, abs=1e-10), (
                f"Bonferroni p[{i}]: expected {expected:.4f}, got {adjusted[i]:.4f}."
            )

    def test_apply_holm(self):
        """Holm step-down correction should produce valid adjusted values."""
        raw_p = np.array([0.001, 0.03, 0.08])
        adjusted, significant = apply_holm(raw_p, alpha=0.05)

        assert len(adjusted) == 3, "Output length should match input."
        # Holm-adjusted values should be >= raw values
        for i in range(len(raw_p)):
            assert adjusted[i] >= raw_p[i] - 1e-12, (
                f"Holm-adjusted p[{i}]={adjusted[i]:.4f} should be >= raw p={raw_p[i]:.4f}."
            )
        # Holm is less conservative than Bonferroni for the smallest p
        bonf_adjusted, _ = apply_bonferroni(raw_p, alpha=0.05)
        assert adjusted[0] <= bonf_adjusted[0] + 1e-12, (
            "Holm should be no more conservative than Bonferroni for the smallest p."
        )

    def test_fdr_nan_handling(self):
        """NaN p-values should be preserved in the output."""
        raw_p = np.array([0.01, np.nan, 0.05, np.nan, 0.10])
        adjusted, significant = apply_fdr(raw_p, alpha=0.05)

        assert len(adjusted) == 5, "Output length should match input."
        assert np.isnan(adjusted[1]), "NaN at index 1 should be preserved."
        assert np.isnan(adjusted[3]), "NaN at index 3 should be preserved."
        assert not significant[1], "NaN entries should not be marked significant."
        assert not significant[3], "NaN entries should not be marked significant."
        # Non-NaN entries should have valid adjusted values
        assert not np.isnan(adjusted[0]), "Non-NaN entry should have valid adjusted p."
        assert not np.isnan(adjusted[2]), "Non-NaN entry should have valid adjusted p."


# ======================================================================
# Equivalence and Independence Testing
# ======================================================================


class TestEquivalence:
    """Tests for phi coefficient, TOST, and independence suite."""

    def test_phi_coefficient_independent(self, rng):
        """Independent binary variables should have phi near 0."""
        n = 1000
        x = rng.binomial(1, 0.3, n)
        y = rng.binomial(1, 0.4, n)

        phi, ci_lower, ci_upper = phi_coefficient(x, y)

        assert abs(phi) < 0.15, (
            f"Phi for independent variables should be near 0, got {phi:.4f}."
        )
        # CI should contain 0 for independent variables
        assert ci_lower <= 0.0 <= ci_upper, (
            f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}] should contain 0."
        )

    def test_phi_coefficient_dependent(self, rng):
        """Dependent binary variables should have phi away from 0."""
        n = 1000
        x = rng.binomial(1, 0.5, n)
        # y is mostly a copy of x (strong dependence)
        noise = rng.binomial(1, 0.1, n)
        y = np.where(noise, 1 - x, x)

        phi, ci_lower, ci_upper = phi_coefficient(x, y)

        assert abs(phi) > 0.5, (
            f"Phi for strongly dependent variables should be > 0.5, got {phi:.4f}."
        )

    def test_tost_equivalence_similar(self, rng):
        """TOST should reject non-equivalence for very similar distributions."""
        n = 200
        x = rng.normal(5.0, 1.0, n)
        y = rng.normal(5.05, 1.0, n)

        reject, p1, p2 = tost_equivalence(x, y, equivalence_bound=0.5)

        assert reject, (
            "TOST should reject non-equivalence (declare equivalence) "
            f"for near-identical distributions. p1={p1:.4f}, p2={p2:.4f}"
        )

    def test_tost_equivalence_different(self, rng):
        """TOST should not reject non-equivalence for very different distributions."""
        n = 200
        x = rng.normal(5.0, 1.0, n)
        y = rng.normal(8.0, 1.0, n)

        reject, p1, p2 = tost_equivalence(x, y, equivalence_bound=0.5)

        assert not reject, (
            "TOST should NOT reject non-equivalence for very different means."
        )

    def test_independence_suite(self, rng):
        """independence_suite should return all expected keys."""
        n = 500
        x = rng.binomial(1, 0.3, n)
        y = rng.binomial(1, 0.4, n)

        result = independence_suite(x, y)

        expected_keys = {
            "chi2_stat", "chi2_p", "phi", "phi_ci_lower", "phi_ci_upper",
            "spearman_rho", "spearman_p", "kendall_tau", "kendall_p",
            "cramers_v", "discordance_rate", "bf10", "n",
        }
        missing_keys = expected_keys - set(result.keys())
        assert not missing_keys, (
            f"independence_suite result missing keys: {missing_keys}. "
            f"Got keys: {set(result.keys())}"
        )
        assert result["n"] == n, (
            f"Sample size should be {n}, got {result['n']}."
        )


# ======================================================================
# Bootstrap
# ======================================================================


class TestBootstrap:
    """Tests for BootstrapCI and bootstrap_delta_c."""

    def test_bootstrap_ci(self, rng):
        """Bootstrap CI for mean of normal(0,1) should contain 0."""
        data = rng.normal(0, 1, 200)

        bs = BootstrapCI(n_bootstrap=500, ci_level=0.95, seed=42)
        point, ci_lower, ci_upper = bs.run(
            statistic_fn=lambda d: float(np.mean(d)),
            data=data,
        )

        assert ci_lower <= 0.0 <= ci_upper, (
            f"95% bootstrap CI [{ci_lower:.4f}, {ci_upper:.4f}] "
            f"should contain 0 for mean of N(0,1) data."
        )
        assert abs(point) < 0.3, (
            f"Point estimate {point:.4f} should be near 0."
        )

    def test_bootstrap_delta_c(self, survival_df, covariates):
        """bootstrap_delta_c should return a 4-tuple of (delta_c, ci_lo, ci_hi, p)."""
        base_vars = covariates  # ["AGE", "SEX"]
        full_vars = covariates + ["GENE_A", "SITE_LIVER"]

        result = bootstrap_delta_c(
            survival_df,
            model_vars_base=base_vars,
            model_vars_full=full_vars,
            duration_col="OS_MONTHS",
            event_col="OS_STATUS",
            n_bootstrap=50,  # Small for speed
            penalizer=0.01,
            seed=42,
        )

        assert len(result) == 4, (
            f"bootstrap_delta_c should return 4 values, got {len(result)}."
        )
        delta_c, ci_lower, ci_upper, p_value = result
        assert isinstance(delta_c, float), "delta_c should be a float."
        assert isinstance(ci_lower, float), "ci_lower should be a float."
        assert isinstance(ci_upper, float), "ci_upper should be a float."
        assert isinstance(p_value, float), "p_value should be a float."
        assert ci_lower <= ci_upper, (
            f"CI lower ({ci_lower:.4f}) should be <= upper ({ci_upper:.4f})."
        )


# ======================================================================
# Cross-Validation
# ======================================================================


class TestCrossValidation:
    """Tests for cv_concordance and cv_delta_c."""

    def test_cv_concordance(self, survival_df, covariates):
        """cv_concordance should return (mean_c, std_c, fold_cs) with valid values."""
        mean_c, std_c, fold_cs = cv_concordance(
            survival_df,
            covariates=["GENE_A", "SITE_LIVER"] + covariates,
            duration_col="OS_MONTHS",
            event_col="OS_STATUS",
            n_folds=5,
            seed=42,
        )

        assert isinstance(mean_c, float), "mean_c should be a float."
        assert isinstance(std_c, float), "std_c should be a float."
        assert isinstance(fold_cs, list), "fold_cs should be a list."
        assert 0.3 <= mean_c <= 0.8, (
            f"Mean C-index {mean_c:.3f} should be in [0.3, 0.8] for synthetic data."
        )
        assert len(fold_cs) == 5, (
            f"Should have 5 fold C values, got {len(fold_cs)}."
        )

    def test_cv_delta_c(self, survival_df, covariates):
        """cv_delta_c should return (mean_delta, std_delta, fold_deltas)."""
        base_vars = covariates
        full_vars = covariates + ["GENE_A", "SITE_LIVER"]

        mean_d, std_d, fold_deltas = cv_delta_c(
            survival_df,
            base_vars=base_vars,
            full_vars=full_vars,
            duration_col="OS_MONTHS",
            event_col="OS_STATUS",
            n_folds=5,
            seed=42,
        )

        assert isinstance(mean_d, float), "mean_delta should be a float."
        assert isinstance(std_d, float), "std_delta should be a float."
        assert isinstance(fold_deltas, list), "fold_deltas should be a list."
        assert len(fold_deltas) == 5, (
            f"Should have 5 fold delta values, got {len(fold_deltas)}."
        )
