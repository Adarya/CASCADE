"""
Tests for CASCADE pitfall detection system (cascade.pitfalls).

Covers: PITFALL_LIBRARY enumeration, PitfallRegistry operations,
PitfallDetector checks for comment corruption, collinearity,
constant variables, separation problems, and singular matrices.
"""

import numpy as np
import pandas as pd
import pytest

from cascade.pitfalls import (
    PITFALL_LIBRARY,
    Pitfall,
    PitfallCategory,
    PitfallDetector,
    PitfallRegistry,
    PitfallWarning,
    Severity,
    Detectability,
)


# ======================================================================
# PITFALL_LIBRARY
# ======================================================================


class TestPitfallLibrary:
    """Tests for the canonical pitfall library."""

    def test_pitfall_library_count(self):
        """PITFALL_LIBRARY should contain exactly 9 items."""
        assert len(PITFALL_LIBRARY) == 9, (
            f"Expected 9 pitfalls in the library, got {len(PITFALL_LIBRARY)}."
        )

    def test_pitfall_library_ids(self):
        """Pitfall IDs should be 1 through 9."""
        ids = sorted(p.id for p in PITFALL_LIBRARY)
        assert ids == list(range(1, 10)), (
            f"Expected pitfall IDs 1-9, got {ids}."
        )

    def test_pitfall_library_types(self):
        """Every item in PITFALL_LIBRARY should be a Pitfall instance."""
        for p in PITFALL_LIBRARY:
            assert isinstance(p, Pitfall), (
                f"Item with id={p.id} should be a Pitfall instance."
            )
            assert isinstance(p.category, PitfallCategory), (
                f"Pitfall {p.id} category should be a PitfallCategory enum."
            )
            assert isinstance(p.severity, Severity), (
                f"Pitfall {p.id} severity should be a Severity enum."
            )


# ======================================================================
# PitfallRegistry
# ======================================================================


class TestPitfallRegistry:
    """Tests for the extensible pitfall registry."""

    def test_registry_preloaded(self):
        """Default registry should have 9 pitfalls pre-loaded."""
        registry = PitfallRegistry()
        assert len(registry) == 9, (
            f"Default registry should have 9 pitfalls, got {len(registry)}."
        )

    def test_registry_empty(self):
        """Registry with preload=False should be empty."""
        registry = PitfallRegistry(preload=False)
        assert len(registry) == 0, (
            "Empty registry should have 0 pitfalls."
        )

    def test_registry_register(self):
        """Registering a new pitfall should increase the count."""
        registry = PitfallRegistry()
        initial_count = len(registry)

        new_pitfall = Pitfall(
            id=100,
            name="Test pitfall",
            description="A test pitfall for unit testing.",
            category=PitfallCategory.STATISTICAL,
            severity=Severity.INFO,
            detectability=Detectability.AUTOMATED,
            detection_strategy="Check for test.",
            fix_strategy="Apply test fix.",
            source_study="Unit test",
            example="Example of test pitfall.",
        )
        registry.register(new_pitfall)

        assert len(registry) == initial_count + 1, (
            "Count should increase by 1 after registration."
        )
        assert 100 in registry, "New pitfall id should be in registry."
        assert registry.get(100).name == "Test pitfall", (
            "Retrieved pitfall name should match."
        )

    def test_registry_duplicate(self):
        """Registering a pitfall with a duplicate ID should raise ValueError."""
        registry = PitfallRegistry()

        duplicate = Pitfall(
            id=1,  # Already exists
            name="Duplicate",
            description="Duplicate pitfall.",
            category=PitfallCategory.DATA_FORMAT,
            severity=Severity.INFO,
            detectability=Detectability.MANUAL,
            detection_strategy="N/A",
            fix_strategy="N/A",
            source_study="Unit test",
            example="N/A",
        )

        with pytest.raises(ValueError, match="already registered"):
            registry.register(duplicate)

    def test_registry_list_by_category(self):
        """list_by_category should filter correctly."""
        registry = PitfallRegistry()
        statistical = registry.list_by_category(PitfallCategory.STATISTICAL)
        assert len(statistical) > 0, (
            "There should be at least one statistical pitfall."
        )
        for p in statistical:
            assert p.category == PitfallCategory.STATISTICAL, (
                f"Pitfall {p.id} should be STATISTICAL category."
            )

    def test_registry_list_automatable(self):
        """list_automatable should return AUTOMATED or PARTIAL pitfalls."""
        registry = PitfallRegistry()
        automatable = registry.list_automatable()
        assert len(automatable) > 0, (
            "There should be at least one automatable pitfall."
        )
        for p in automatable:
            assert p.detectability in (Detectability.AUTOMATED, Detectability.PARTIAL), (
                f"Pitfall {p.id} detectability should be AUTOMATED or PARTIAL."
            )


# ======================================================================
# PitfallDetector: Comment corruption
# ======================================================================


class TestCommentCorruption:
    """Tests for comment header corruption detection (Pitfall #1)."""

    def test_comment_corruption_detected(self, tsv_with_comments):
        """File with STYLE_COLOR hex values should generate a warning."""
        detector = PitfallDetector()
        warnings = detector.check_data_format(tsv_with_comments)

        assert len(warnings) > 0, (
            "Should detect comment corruption in file with hex color codes."
        )
        # Check that the warning references the STYLE_COLOR column
        has_style_color_warning = any(
            "STYLE_COLOR" in w.message for w in warnings
        )
        assert has_style_color_warning, (
            "Warning should mention STYLE_COLOR column."
        )
        # Check severity is CRITICAL
        assert any(w.severity == Severity.CRITICAL for w in warnings), (
            "At least one warning should have CRITICAL severity."
        )

    def test_comment_corruption_clean(self, tmp_path):
        """File without hex color values should produce no warnings."""
        filepath = tmp_path / "clean_data.txt"
        content = (
            "# Comment line\n"
            "PATIENT_ID\tECOG\tSTATUS\n"
            "P-0000001\t1\tAlive\n"
            "P-0000002\t2\tDeceased\n"
        )
        filepath.write_text(content)

        detector = PitfallDetector()
        warnings = detector.check_data_format(filepath)

        assert len(warnings) == 0, (
            f"Clean file should produce no warnings, got {len(warnings)}."
        )


# ======================================================================
# PitfallDetector: Collinearity
# ======================================================================


class TestCollinearity:
    """Tests for collinearity detection (Pitfall #3)."""

    def test_collinearity_detected(self, feature_matrix):
        """Feature matrix with COLLINEAR_A/B should trigger a warning."""
        detector = PitfallDetector()
        warnings = detector.check_collinearity(feature_matrix)

        assert len(warnings) > 0, (
            "Should detect collinearity between COLLINEAR_A and COLLINEAR_B."
        )
        # At least one warning should mention the collinear pair
        has_collinear_warning = any(
            "COLLINEAR_A" in w.message or "COLLINEAR_B" in w.message
            for w in warnings
        )
        assert has_collinear_warning, (
            "A warning should mention COLLINEAR_A or COLLINEAR_B."
        )


# ======================================================================
# PitfallDetector: Constant variable
# ======================================================================


class TestConstantVariable:
    """Tests for constant variable detection (Pitfall #4)."""

    def test_constant_variable_detected(self, feature_matrix):
        """Feature matrix with CONSTANT column should trigger a warning."""
        detector = PitfallDetector()
        warnings = detector.check_constant_variables(feature_matrix)

        assert len(warnings) > 0, (
            "Should detect the CONSTANT column."
        )
        has_constant_warning = any(
            "CONSTANT" in w.message for w in warnings
        )
        assert has_constant_warning, (
            "A warning should mention the CONSTANT column."
        )


# ======================================================================
# PitfallDetector: Separation
# ======================================================================


class TestSeparation:
    """Tests for separation problem detection."""

    def test_separation_detected(self):
        """Results with CI ratio > 100 should be flagged."""
        results_df = pd.DataFrame({
            "name": ["gene_A", "gene_B", "gene_C"],
            "hr": [1.5, 0.8, 50.0],
            "ci_lower": [1.1, 0.5, 0.2],
            "ci_upper": [2.0, 1.2, 500.0],
        })

        detector = PitfallDetector()
        warnings = detector.check_separation(
            results_df,
            ci_lower_col="ci_lower",
            ci_upper_col="ci_upper",
            threshold=100.0,
        )

        assert len(warnings) > 0, (
            "Should flag gene_C with CI ratio 500/0.2 = 2500."
        )
        # gene_C should be flagged
        has_gene_c_warning = any(
            "gene_C" in w.message for w in warnings
        )
        assert has_gene_c_warning, (
            "Warning should reference gene_C (extreme CI ratio)."
        )

    def test_separation_no_issues(self):
        """Results with normal CI ratios should produce no warnings."""
        results_df = pd.DataFrame({
            "name": ["gene_A", "gene_B"],
            "hr": [1.5, 0.8],
            "ci_lower": [1.1, 0.5],
            "ci_upper": [2.0, 1.2],
        })

        detector = PitfallDetector()
        warnings = detector.check_separation(
            results_df,
            ci_lower_col="ci_lower",
            ci_upper_col="ci_upper",
            threshold=100.0,
        )

        assert len(warnings) == 0, (
            f"Normal CI ratios should produce no warnings, got {len(warnings)}."
        )


# ======================================================================
# PitfallDetector: Singular matrix
# ======================================================================


class TestSingularMatrix:
    """Tests for singular matrix detection (Pitfall #7)."""

    def test_singular_matrix_detected(self, rng):
        """A rank-deficient matrix should be flagged."""
        n = 100
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        x3 = x1 + x2  # Perfect linear dependency

        df = pd.DataFrame({"X1": x1, "X2": x2, "X3": x3})

        detector = PitfallDetector()
        warnings = detector.check_singular_matrix(df)

        assert len(warnings) > 0, (
            "Should detect singularity from perfect linear dependency."
        )
        # At least one warning should mention rank deficiency or singular
        has_rank_warning = any(
            "rank" in w.message.lower() or "singular" in w.message.lower()
            or "condition" in w.message.lower()
            for w in warnings
        )
        assert has_rank_warning, (
            "Warning should mention rank deficiency, singularity, or condition number."
        )


# ======================================================================
# PitfallDetector: Orchestrator
# ======================================================================


class TestDetectorOrchestrator:
    """Tests for the PitfallDetector check_all and summary methods."""

    def test_detector_check_all(self, feature_matrix):
        """check_all should run without error and return a list of warnings."""
        detector = PitfallDetector()
        warnings = detector.check_all(feature_matrix=feature_matrix)

        assert isinstance(warnings, list), "check_all should return a list."
        for w in warnings:
            assert isinstance(w, PitfallWarning), (
                "Each item should be a PitfallWarning."
            )

    def test_detector_summary(self, feature_matrix):
        """summary should return a non-empty formatted string after check_all."""
        detector = PitfallDetector()
        detector.check_all(feature_matrix=feature_matrix)
        summary_text = detector.summary()

        assert isinstance(summary_text, str), "summary should return a string."
        assert len(summary_text) > 0, "summary should not be empty."
        # If warnings were found, the report should contain the header
        if detector.warnings:
            assert "CASCADE Pitfall Detector Report" in summary_text, (
                "Summary should contain the report header."
            )

    def test_detector_clear(self, feature_matrix):
        """clear should remove all accumulated warnings."""
        detector = PitfallDetector()
        detector.check_all(feature_matrix=feature_matrix)
        assert len(detector.warnings) > 0, (
            "Should have accumulated warnings from feature_matrix."
        )

        detector.clear()
        assert len(detector.warnings) == 0, (
            "After clear(), warnings should be empty."
        )
