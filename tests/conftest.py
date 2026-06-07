"""
Synthetic dataset fixtures for CASCADE tests.

All fixtures generate small, reproducible datasets that exercise
CASCADE functionality without requiring real clinical data.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.RandomState(42)


@pytest.fixture
def survival_df(rng):
    """
    Synthetic survival dataset with 500 patients.

    Columns: PATIENT_ID, OS_MONTHS, OS_STATUS, AGE, SEX,
             GENE_A (binary), GENE_B (binary), GENE_C (binary),
             TREATMENT (binary), SITE_LIVER (binary), SITE_BONE (binary),
             ECOG (0-4), BURDEN (1-5)
    """
    n = 500
    gene_a = rng.binomial(1, 0.3, n)
    gene_b = rng.binomial(1, 0.2, n)
    gene_c = rng.binomial(1, 0.15, n)
    treatment = rng.binomial(1, 0.5, n)
    site_liver = rng.binomial(1, 0.25, n)
    site_bone = rng.binomial(1, 0.3, n)
    ecog = rng.choice([0, 1, 2, 3, 4], n, p=[0.2, 0.35, 0.25, 0.15, 0.05])
    burden = rng.choice([1, 2, 3, 4, 5], n, p=[0.15, 0.25, 0.3, 0.2, 0.1])
    age = rng.normal(65, 12, n).clip(30, 95).astype(int)
    sex = rng.binomial(1, 0.55, n)

    # Generate survival times with known effects
    baseline_hazard = 0.05
    log_hazard = (
        np.log(baseline_hazard)
        + 0.4 * gene_a       # Gene A increases risk
        - 0.3 * gene_b       # Gene B protective
        + 0.5 * site_liver   # Liver mets increase risk
        + 0.2 * ecog         # Higher ECOG increases risk
        + 0.01 * (age - 65)  # Age effect
        - 0.3 * gene_a * treatment  # Gene A x treatment interaction (predictive)
    )
    scale = np.exp(-log_hazard)
    os_months = rng.exponential(scale).clip(0.5, 120)

    # Censoring (~30%)
    censor_time = rng.exponential(40, n)
    os_status = (os_months <= censor_time).astype(int)
    os_months = np.minimum(os_months, censor_time)

    return pd.DataFrame({
        "PATIENT_ID": [f"P-{i:07d}" for i in range(n)],
        "OS_MONTHS": np.round(os_months, 1),
        "OS_STATUS": os_status,
        "AGE": age,
        "SEX": sex,
        "GENE_A": gene_a,
        "GENE_B": gene_b,
        "GENE_C": gene_c,
        "TREATMENT": treatment,
        "SITE_LIVER": site_liver,
        "SITE_BONE": site_bone,
        "ECOG": ecog,
        "BURDEN": burden,
    })


@pytest.fixture
def biomarker_cols():
    """List of biomarker column names."""
    return ["GENE_A", "GENE_B", "GENE_C"]


@pytest.fixture
def site_cols():
    """List of site column names."""
    return ["SITE_LIVER", "SITE_BONE"]


@pytest.fixture
def covariates():
    """Standard covariate list."""
    return ["AGE", "SEX"]


@pytest.fixture
def feature_matrix(rng):
    """
    Synthetic feature matrix for pitfall detection tests.

    Includes deliberately problematic columns:
    - CONSTANT: all zeros (constant variable)
    - COLLINEAR_A, COLLINEAR_B: perfectly correlated
    - RATE: algebraically derived from COUNT and TIME
    """
    n = 200
    count = rng.poisson(3, n).astype(float)
    time_val = rng.uniform(1, 24, n)
    rate = (count - 1) / time_val  # Algebraic identity

    return pd.DataFrame({
        "FEATURE_1": rng.normal(0, 1, n),
        "FEATURE_2": rng.normal(0, 1, n),
        "CONSTANT": np.zeros(n),
        "COLLINEAR_A": rng.normal(5, 2, n),
        "COLLINEAR_B": None,  # Will be set below
        "COUNT": count,
        "TIME": time_val,
        "RATE": rate,
    }).assign(COLLINEAR_B=lambda df: df["COLLINEAR_A"] * 2 + 0.001)


@pytest.fixture
def timeline_treatment_df(rng):
    """
    Synthetic treatment timeline for 100 patients.

    Multiple treatment lines with gaps.
    """
    records = []
    for i in range(100):
        pid = f"P-{i:07d}"
        # Line 1
        start1 = rng.randint(0, 100)
        stop1 = start1 + rng.randint(30, 180)
        agent1 = rng.choice(["CARBOPLATIN+PEMETREXED", "PEMBROLIZUMAB",
                             "DOCETAXEL", "NIVOLUMAB+IPILIMUMAB"])
        records.append({
            "PATIENT_ID": pid,
            "START_DATE": start1,
            "STOP_DATE": stop1,
            "AGENT": agent1,
        })
        # Line 2 (with gap > 90 days for ~60% of patients)
        if rng.random() > 0.4:
            gap = rng.randint(91, 200)
            start2 = stop1 + gap
            stop2 = start2 + rng.randint(30, 120)
            agent2 = rng.choice(["DOCETAXEL", "RAMUCIRUMAB+DOCETAXEL",
                                 "ERLOTINIB", "ATEZOLIZUMAB"])
            records.append({
                "PATIENT_ID": pid,
                "START_DATE": start2,
                "STOP_DATE": stop2,
                "AGENT": agent2,
            })

    return pd.DataFrame(records)


@pytest.fixture
def timeline_sites_df(rng):
    """
    Synthetic tumor site timeline for 100 patients.

    Multiple sites per patient over time.
    """
    records = []
    sites = ["Liver", "Bone", "Lung", "CNS/Brain", "Lymph Node",
             "Adrenal", "Pleura", "Intra-Abdominal", "Soft Tissue", "Skin"]

    for i in range(100):
        pid = f"P-{i:07d}"
        n_timepoints = rng.randint(2, 8)
        base_sites = rng.choice(sites, rng.randint(1, 4), replace=False)

        for t in range(n_timepoints):
            date = t * rng.randint(30, 120)
            # Base sites persist; new ones may appear
            current_sites = list(base_sites)
            if t > 0 and rng.random() > 0.7:
                new_site = rng.choice([s for s in sites if s not in current_sites])
                current_sites.append(new_site)

            for site in current_sites:
                records.append({
                    "PATIENT_ID": pid,
                    "START_DATE": date,
                    "TUMOR_SITE": site,
                })

    return pd.DataFrame(records)


@pytest.fixture
def landmark_df(rng):
    """
    Synthetic landmark analysis dataset.

    200 patients alive at 6-month landmark with pre-landmark trajectory
    features and residual survival.
    """
    n = 200
    baseline_burden = rng.choice([1, 2, 3], n, p=[0.4, 0.35, 0.25])
    burden_at_lm = baseline_burden + rng.choice([-1, 0, 1, 2], n, p=[0.1, 0.5, 0.3, 0.1])
    burden_at_lm = burden_at_lm.clip(1, 5)
    burden_change = burden_at_lm - baseline_burden

    ecog_baseline = rng.choice([0, 1, 2], n, p=[0.3, 0.5, 0.2])
    ecog_at_lm = ecog_baseline + rng.choice([-1, 0, 1], n, p=[0.15, 0.6, 0.25])
    ecog_at_lm = ecog_at_lm.clip(0, 4)
    ecog_change = ecog_at_lm - ecog_baseline

    # Residual survival from 6-month landmark
    log_hazard = (
        0.3 * burden_change
        + 0.2 * ecog_change
        + 0.1 * burden_at_lm
        + 0.05 * ecog_at_lm
    )
    scale = np.exp(-log_hazard) * 20
    os_from_lm = rng.exponential(scale).clip(0.1, 60)
    censor = rng.exponential(30, n)
    event = (os_from_lm <= censor).astype(int)
    os_from_lm = np.minimum(os_from_lm, censor)

    return pd.DataFrame({
        "PATIENT_ID": [f"P-{i:07d}" for i in range(n)],
        "OS_FROM_LM": np.round(os_from_lm, 1),
        "EVENT_FROM_LM": event,
        "BASELINE_BURDEN": baseline_burden,
        "BURDEN_AT_LM": burden_at_lm,
        "BURDEN_CHANGE": burden_change,
        "ECOG_BASELINE": ecog_baseline,
        "ECOG_AT_LM": ecog_at_lm,
        "ECOG_CHANGE": ecog_change,
        "AGE": rng.normal(65, 12, n).clip(30, 95).astype(int),
        "SEX": rng.binomial(1, 0.55, n),
        "RECEIVED_ICI": rng.binomial(1, 0.4, n),
    })


@pytest.fixture
def classification_df(rng):
    """
    Synthetic classification dataset for external validation tests.

    300 samples with features and class labels.
    """
    n = 300
    n_features = 20
    n_classes = 5

    X = pd.DataFrame(
        rng.normal(0, 1, (n, n_features)),
        columns=[f"FEAT_{i}" for i in range(n_features)],
    )

    # Generate class labels with imbalance
    y = rng.choice(
        [f"CLASS_{i}" for i in range(n_classes)],
        n,
        p=[0.3, 0.25, 0.2, 0.15, 0.1],
    )

    return X, pd.Series(y, name="CLASS")


@pytest.fixture
def tsv_with_comments(tmp_path):
    """
    Create a temporary TSV file with comment headers and STYLE_COLOR column.

    This exercises the comment header corruption pitfall (#1).
    """
    filepath = tmp_path / "test_data.txt"
    content = (
        "# This is a comment\n"
        "# Another comment\n"
        "PATIENT_ID\tECOG\tSTYLE_COLOR\n"
        "P-0000001\t1\t#FF0000\n"
        "P-0000002\t2\t#00FF00\n"
        "P-0000003\t0\t#0000FF\n"
        "P-0000004\t3\t#FFFF00\n"
    )
    filepath.write_text(content)
    return filepath


@pytest.fixture
def interaction_df(rng):
    """
    Synthetic dataset for interaction testing (Layer 3).

    Contains gene, treatment, and phenotype binary variables
    with a known interaction effect.
    """
    n = 400
    gene = rng.binomial(1, 0.25, n)
    treatment = rng.binomial(1, 0.5, n)
    phenotype = rng.binomial(1, 0.3, n)

    # Known effects
    log_hazard = (
        0.3 * gene
        - 0.2 * treatment
        + 0.4 * phenotype
        - 0.8 * gene * treatment   # Strong 2-way interaction
        + 0.1 * gene * phenotype
        - 0.05 * treatment * phenotype
        + 0.5 * gene * treatment * phenotype  # 3-way interaction
    )
    scale = np.exp(-log_hazard) * 30
    os_months = rng.exponential(scale).clip(0.5, 120)
    censor = rng.exponential(40, n)
    event = (os_months <= censor).astype(int)
    os_months = np.minimum(os_months, censor)

    return pd.DataFrame({
        "PATIENT_ID": [f"P-{i:07d}" for i in range(n)],
        "OS_MONTHS": np.round(os_months, 1),
        "OS_STATUS": event,
        "GENE": gene,
        "TREATMENT": treatment,
        "PHENOTYPE": phenotype,
        "AGE": rng.normal(65, 12, n).clip(30, 95).astype(int),
        "SEX": rng.binomial(1, 0.55, n),
    })
