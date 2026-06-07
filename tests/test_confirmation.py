"""Unit tests for Layer 2: OrthogonalConfirm (orthogonal confirmation).

Previously the OrthogonalConfirm class itself had no direct test coverage
(only its decision gate was tested with hand-built DataFrames).
"""

import numpy as np
import pandas as pd
import pytest

from cascade.core import BiomarkerScreen, OrthogonalConfirm


@pytest.fixture
def primary_results(survival_df, biomarker_cols):
    screen = BiomarkerScreen(
        method="cox", correction="fdr_bh", min_exposed=10, min_events=3
    )
    return screen.screen(
        survival_df,
        biomarker_cols=biomarker_cols,
        outcome_col="OS_MONTHS",
        event_col="OS_STATUS",
    )


def _confirm(primary_results, survival_df, biomarker_cols):
    return OrthogonalConfirm(confirm_method="logistic").confirm(
        primary_results,
        survival_df,
        biomarker_cols=biomarker_cols,
        outcome_col="OS_MONTHS",
        event_col="OS_STATUS",
    )


class TestOrthogonalConfirm:
    def test_confirm_adds_expected_columns(
        self, primary_results, survival_df, biomarker_cols
    ):
        conf = _confirm(primary_results, survival_df, biomarker_cols)
        for col in [
            "confirm_hr",
            "confirm_p",
            "confirm_significant",
            "concordance_direction",
            "confidence_tier",
        ]:
            assert col in conf.columns
        assert len(conf) == len(biomarker_cols)

    def test_confidence_tier_vocabulary(
        self, primary_results, survival_df, biomarker_cols
    ):
        conf = _confirm(primary_results, survival_df, biomarker_cols)
        valid = {"both_significant", "primary_only", "confirm_only", "neither"}
        assert set(conf["confidence_tier"]).issubset(valid)

    def test_direction_concordance_for_planted_signal(
        self, primary_results, survival_df, biomarker_cols
    ):
        """GENE_A is planted harmful (coef +0.4): primary (Cox HR) and the
        orthogonal confirm (logistic OR) should agree on direction."""
        conf = _confirm(primary_results, survival_df, biomarker_cols)
        row = conf[conf["biomarker"] == "GENE_A"].iloc[0]
        assert row["hr"] > 1.0
        assert row["confirm_hr"] > 1.0
        assert row["concordance_direction"] == 1

    def test_concordance_metrics(
        self, primary_results, survival_df, biomarker_cols
    ):
        conf = _confirm(primary_results, survival_df, biomarker_cols)
        metrics = OrthogonalConfirm.concordance_metrics(
            conf,
            conf,
            effect_col_primary="hr",
            effect_col_confirm="confirm_hr",
        )
        assert "direction_agreement_pct" in metrics
        assert "n_both_significant" in metrics
        assert 0.0 <= metrics["direction_agreement_pct"] <= 100.0
