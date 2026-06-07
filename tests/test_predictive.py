"""Unit tests for Layer 3: PredictiveTest (predictive vs prognostic).

This is the framework's marquee 'predictive vs prognostic' feature (the CBFB
story) and was previously exercised only by an un-run example.
"""

import numpy as np
import pandas as pd
import pytest

from cascade.core import PredictiveTest
from cascade.core.predictive import PredictiveResult


class TestPredictiveTest:
    """Tests for the gene x treatment interaction test."""

    def test_returns_predictive_result(self, interaction_df):
        res = PredictiveTest(gene_col="GENE", treatment_col="TREATMENT").run(
            interaction_df
        )
        assert isinstance(res, PredictiveResult)
        assert res.gene == "GENE"
        assert res.treatment == "TREATMENT"

    def test_interaction_fields_valid(self, interaction_df):
        res = PredictiveTest(gene_col="GENE", treatment_col="TREATMENT").run(
            interaction_df
        )
        assert np.isfinite(res.interaction_hr) and res.interaction_hr > 0
        assert 0.0 <= res.interaction_p <= 1.0
        lo, hi = res.interaction_ci
        assert lo <= res.interaction_hr <= hi
        assert np.isfinite(res.hr_treated) and np.isfinite(res.hr_untreated)
        assert res.n_treated > 0 and res.n_untreated > 0

    def test_classification_vocabulary(self, interaction_df):
        res = PredictiveTest(gene_col="GENE", treatment_col="TREATMENT").run(
            interaction_df
        )
        assert res.classification in {"predictive", "prognostic"}

    def test_planted_interaction_direction(self, interaction_df):
        """The fixture plants a protective gene x treatment interaction
        (coef -0.8), so the interaction HR should be < 1."""
        res = PredictiveTest(gene_col="GENE", treatment_col="TREATMENT").run(
            interaction_df
        )
        assert res.interaction_hr < 1.0

    def test_deterministic(self, interaction_df):
        a = PredictiveTest(gene_col="GENE", treatment_col="TREATMENT").run(
            interaction_df
        )
        b = PredictiveTest(gene_col="GENE", treatment_col="TREATMENT").run(
            interaction_df
        )
        assert a.interaction_hr == pytest.approx(b.interaction_hr)
        assert a.interaction_p == pytest.approx(b.interaction_p)

    def test_predictive_when_interaction_strong(self):
        """A strong, clean 2-way interaction should be flagged predictive.

        Gene is harmful only in untreated patients; treatment abolishes the
        effect (a textbook predictive biomarker).
        """
        rng = np.random.RandomState(0)
        n = 1200
        gene = rng.binomial(1, 0.5, n)
        treat = rng.binomial(1, 0.5, n)
        log_h = np.log(0.05) + 1.5 * gene - 1.5 * gene * treat
        os = rng.exponential(np.exp(-log_h)).clip(0.5, 120)
        cen = rng.exponential(80, n)
        status = (os <= cen).astype(int)
        os = np.minimum(os, cen)
        df = pd.DataFrame(
            {
                "OS_MONTHS": os,
                "OS_STATUS": status,
                "GENE": gene,
                "TREATMENT": treat,
            }
        )
        res = PredictiveTest(gene_col="GENE", treatment_col="TREATMENT").run(df)
        assert res.interaction_p < 0.05
        assert res.classification == "predictive"
        # Harmful in untreated, attenuated in treated.
        assert res.hr_untreated > res.hr_treated
