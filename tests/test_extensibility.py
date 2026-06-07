"""Tests for first-class layer extensibility (v0.3.0).

Verifies that users can define and compose NEW layers: any object with a
run() method, under any name, plugged into the pipeline, able to consume any
upstream layer's output via the ``context`` hook.
"""

import numpy as np
import pandas as pd
import pytest

from cascade import Pipeline, Layer, BaseLayer
from cascade.core import BiomarkerScreen, OrthogonalConfirm


def _discovery():
    return BiomarkerScreen(
        method="cox", correction="fdr_bh", min_exposed=10, min_events=3
    )


class _CaptureContext(BaseLayer):
    """Records the context it is handed."""

    def __init__(self):
        super().__init__()
        self.seen = None

    def run(self, df, context=None):
        self.seen = dict(context or {})
        return {"upstream_layers": sorted(self.seen.keys())}


class TestCustomLayerContext:
    def test_custom_layer_runs_with_novel_name(self, survival_df, biomarker_cols):
        p = Pipeline()
        p.add_layer("discovery", _discovery())
        p.add_layer("my_layer", _CaptureContext())  # name unknown to CASCADE
        res = p.run(
            data=survival_df,
            biomarker_cols=biomarker_cols,
            outcome_col="OS_MONTHS",
            event_col="OS_STATUS",
        )
        assert "my_layer" in res.completed_layers
        assert res.get("my_layer").layer_number == -1

    def test_context_contains_all_upstream_by_name(
        self, survival_df, biomarker_cols
    ):
        p = Pipeline()
        p.add_layer("discovery", _discovery())
        p.add_layer("confirmation", OrthogonalConfirm(confirm_method="logistic"))
        cap = _CaptureContext()
        p.add_layer("custom", cap)
        res = p.run(
            data=survival_df,
            biomarker_cols=biomarker_cols,
            outcome_col="OS_MONTHS",
            event_col="OS_STATUS",
        )
        # context carries EVERY completed upstream layer, addressable by name.
        assert res.get("custom").results["upstream_layers"] == [
            "confirmation",
            "discovery",
        ]
        assert isinstance(cap.seen["discovery"], pd.DataFrame)
        assert "confirmation" in cap.seen

    def test_custom_layer_consumes_specific_upstream(
        self, survival_df, biomarker_cols
    ):
        class CountSig(BaseLayer):
            def run(self, df, context=None):
                disc = (context or {}).get("discovery")
                return {"n_sig": int(disc["significant"].sum())}

        p = Pipeline()
        p.add_layer("discovery", _discovery())
        p.add_layer("count", CountSig())
        res = p.run(
            data=survival_df,
            biomarker_cols=biomarker_cols,
            outcome_col="OS_MONTHS",
            event_col="OS_STATUS",
        )
        assert res.get("count").results["n_sig"] >= 0


class TestLayerProtocol:
    def test_baselayer_satisfies_protocol(self):
        class L(BaseLayer):
            def run(self, df, context=None):
                return 1

        assert isinstance(L(), Layer)

    def test_plain_duck_typed_object_satisfies_protocol(self):
        class D:
            def run(self, df):
                return 1

        assert isinstance(D(), Layer)

    def test_object_without_run_is_not_a_layer(self):
        class N:
            pass

        assert not isinstance(N(), Layer)

    def test_baselayer_run_is_abstract(self):
        with pytest.raises(NotImplementedError):
            BaseLayer().run(pd.DataFrame())


class TestReportShowsCustomLayer:
    def test_custom_layer_in_report_summary(self, survival_df, biomarker_cols):
        p = Pipeline()
        p.add_layer("discovery", _discovery())
        p.add_layer("my_custom", _CaptureContext())
        p.run(
            data=survival_df,
            biomarker_cols=biomarker_cols,
            outcome_col="OS_MONTHS",
            event_col="OS_STATUS",
        )
        summary = p.report().split("## Layer Details")[0]
        assert "my_custom (custom)" in summary


class TestBackwardCompatibility:
    def test_primary_results_still_autowired(self, survival_df, biomarker_cols):
        """A layer asking for primary_results (not context) still gets it."""

        class WantsPrimary(BaseLayer):
            def run(self, df, primary_results=None):
                return {"got": primary_results is not None}

        p = Pipeline()
        p.add_layer("discovery", _discovery())
        p.add_layer("w", WantsPrimary())
        res = p.run(
            data=survival_df,
            biomarker_cols=biomarker_cols,
            outcome_col="OS_MONTHS",
            event_col="OS_STATUS",
        )
        assert res.get("w").results["got"] is True
