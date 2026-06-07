"""Tests for FeatureEngineer.compute_ttntd (time to next treatment or death).

Covers the previously-untested death-before-next-treatment path, whose two
branches used to be identical (dead code).
"""

import pandas as pd

from cascade.core.cohort import FeatureEngineer


def _lines(starts):
    return pd.DataFrame(
        {
            "PATIENT_ID": ["P1"] * len(starts),
            "LINE_NUMBER": list(range(1, len(starts) + 1)),
            "LINE_START": starts,
        }
    )


def test_next_treatment_is_event():
    lines = _lines([0, 100])
    os = pd.DataFrame({"PATIENT_ID": ["P1"], "OS_MONTHS": [10.0], "OS_STATUS": [0]})
    out = FeatureEngineer.compute_ttntd(lines, os)
    row1 = out[out["LINE_NUMBER"] == 1].iloc[0]
    assert row1["TTNTD_DAYS"] == 100
    assert row1["TTNTD_EVENT"] == 1


def test_death_before_next_treatment_uses_earlier_time():
    # Death at ~day 50 (OS_MONTHS = 50/30.44) precedes next treatment at day 100.
    lines = _lines([0, 100])
    os = pd.DataFrame(
        {"PATIENT_ID": ["P1"], "OS_MONTHS": [50 / 30.44], "OS_STATUS": [1]}
    )
    out = FeatureEngineer.compute_ttntd(lines, os)
    row1 = out[out["LINE_NUMBER"] == 1].iloc[0]
    assert row1["TTNTD_EVENT"] == 1
    assert row1["TTNTD_DAYS"] < 100
    assert abs(row1["TTNTD_DAYS"] - 50) < 1.0


def test_death_after_next_treatment_uses_next_treatment():
    # Death at ~day 300 is after the next treatment at day 100.
    lines = _lines([0, 100])
    os = pd.DataFrame(
        {"PATIENT_ID": ["P1"], "OS_MONTHS": [300 / 30.44], "OS_STATUS": [1]}
    )
    out = FeatureEngineer.compute_ttntd(lines, os)
    row1 = out[out["LINE_NUMBER"] == 1].iloc[0]
    assert row1["TTNTD_DAYS"] == 100
    assert row1["TTNTD_EVENT"] == 1


def test_last_line_censored_if_alive():
    lines = _lines([0])
    os = pd.DataFrame({"PATIENT_ID": ["P1"], "OS_MONTHS": [12.0], "OS_STATUS": [0]})
    out = FeatureEngineer.compute_ttntd(lines, os)
    row = out.iloc[0]
    assert row["TTNTD_EVENT"] == 0  # alive, no next line -> censored
