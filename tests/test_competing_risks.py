"""Unit tests for cascade.stats.competing_risks (cause-specific Cox).

The cause-specific HR is the statistical engine behind the organotropism
results (NSCLC/CRC gene -> metastatic-site associations), yet this module
previously had no test coverage.
"""

import numpy as np
import pandas as pd
import pytest

from cascade.stats.competing_risks import CauseSpecificCox, cause_specific_screen


class TestCauseSpecificScreen:
    def test_screen_shape_and_columns(self, survival_df, site_cols, covariates):
        res = cause_specific_screen(
            survival_df,
            gene_col="GENE_A",
            site_cols=site_cols,
            duration_col="OS_MONTHS",
            event_col="OS_STATUS",
            covariates=covariates,
        )
        assert len(res) == len(site_cols)  # one row per site
        for col in ["gene", "site", "hr", "p", "ci_lower", "ci_upper", "converged"]:
            assert col in res.columns
        assert (res["gene"] == "GENE_A").all()
        assert set(res["site"]) == set(site_cols)

    def test_screen_values_valid(self, survival_df, site_cols, covariates):
        res = cause_specific_screen(
            survival_df,
            gene_col="GENE_A",
            site_cols=site_cols,
            duration_col="OS_MONTHS",
            event_col="OS_STATUS",
            covariates=covariates,
        )
        ok = res[res["converged"]]
        assert len(ok) > 0
        assert (ok["hr"] > 0).all()
        assert ((ok["p"] >= 0) & (ok["p"] <= 1)).all()
        assert (ok["ci_lower"] <= ok["hr"]).all()
        assert (ok["hr"] <= ok["ci_upper"]).all()

    def test_min_exposed_filter_blocks_fit(self, survival_df, site_cols, covariates):
        """An impossibly high min_exposed should leave no converged rows."""
        res = cause_specific_screen(
            survival_df,
            gene_col="GENE_A",
            site_cols=site_cols,
            duration_col="OS_MONTHS",
            event_col="OS_STATUS",
            covariates=covariates,
            min_exposed=100000,
        )
        assert (~res["converged"]).all()


class TestCauseSpecificCox:
    @pytest.fixture
    def multi_cause_df(self, survival_df):
        """3-state event: 0 censored, 1 = death with liver, 2 = death w/o liver."""
        df = survival_df.copy()
        ev = np.zeros(len(df), dtype=int)
        dead = df["OS_STATUS"] == 1
        ev[(dead) & (df["SITE_LIVER"] == 1)] = 1
        ev[(dead) & (df["SITE_LIVER"] == 0)] = 2
        df["EVT"] = ev
        return df

    def test_fit_returns_summary(self, multi_cause_df):
        model = CauseSpecificCox(penalizer=0.05).fit(
            multi_cause_df,
            duration_col="OS_MONTHS",
            event_col="EVT",
            cause_of_interest=1,
            covariates=["GENE_A", "AGE"],
        )
        assert model.converged
        assert isinstance(model.summary, pd.DataFrame)
        assert "GENE_A" in model.summary.index
        assert "exp(coef)" in model.summary.columns
        assert 0.0 < float(model.summary.loc["GENE_A", "exp(coef)"])

    def test_cause_of_interest_changes_result(self, multi_cause_df):
        """Fitting cause 1 vs cause 2 should give different HRs (different
        events), confirming other causes are genuinely censored."""
        m1 = CauseSpecificCox(penalizer=0.05).fit(
            multi_cause_df, "OS_MONTHS", "EVT", 1, ["GENE_A", "AGE"]
        )
        m2 = CauseSpecificCox(penalizer=0.05).fit(
            multi_cause_df, "OS_MONTHS", "EVT", 2, ["GENE_A", "AGE"]
        )
        hr1 = float(m1.summary.loc["GENE_A", "exp(coef)"])
        hr2 = float(m2.summary.loc["GENE_A", "exp(coef)"])
        assert hr1 != pytest.approx(hr2)

    def test_empty_after_dropna_raises(self):
        empty = pd.DataFrame({"OS_MONTHS": [], "EVT": [], "GENE_A": []})
        with pytest.raises(Exception):
            CauseSpecificCox().fit(
                empty,
                duration_col="OS_MONTHS",
                event_col="EVT",
                cause_of_interest=1,
                covariates=["GENE_A"],
            )
