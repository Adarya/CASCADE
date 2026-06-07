"""
CASCADE Example: Predictive vs Prognostic Testing (Layer 3)

Demonstrates Layer 3 — formally testing whether a biomarker is
predictive (modifies treatment effect) or merely prognostic.

This example mimics the CBFB analysis: a gene mutation that specifically
benefits patients receiving endocrine therapy.
"""

import numpy as np
import pandas as pd

# ---- Generate synthetic data ----

np.random.seed(42)
N = 600

gene = np.random.binomial(1, 0.12, N)  # 12% mutation rate (like CBFB)
treatment = np.random.binomial(1, 0.5, N)  # 50% endocrine therapy

# Survival with a TRUE interaction effect
# Gene is PREDICTIVE: benefits patients on ET but not on chemo
log_hazard = (
    0.1 * gene                    # Mild prognostic effect
    - 0.2 * treatment             # Treatment is generally beneficial
    - 0.9 * gene * treatment      # STRONG interaction: gene + treatment = benefit
    + 0.01 * np.random.normal(65, 12, N)
)
scale = np.exp(-log_hazard) * 25
os_months = np.round(np.random.exponential(scale).clip(0.5, 120), 1)
censor = np.random.exponential(40, N)
event = (os_months <= censor).astype(int)
os_months = np.round(np.minimum(os_months, censor), 1)

df = pd.DataFrame({
    "PATIENT_ID": [f"P-{i:07d}" for i in range(N)],
    "OS_MONTHS": os_months,
    "OS_STATUS": event,
    "GENE_MUT": gene,
    "IS_ET": treatment,
    "AGE": np.random.normal(65, 12, N).clip(30, 95).astype(int),
    "SEX": np.random.binomial(1, 0.55, N),
})

print(f"Dataset: {N} patients")
print(f"Gene mutation prevalence: {gene.mean():.1%}")
print(f"Treatment (ET): {treatment.mean():.1%}")
print(f"Events: {event.sum()} ({event.mean():.1%})")
print()

# ==============================================================
# Layer 3: Predictive vs Prognostic Testing
# ==============================================================
print("=" * 60)
print("LAYER 3: Predictive vs Prognostic Distinction")
print("=" * 60)

from cascade.core import PredictiveTest

test = PredictiveTest(
    gene_col="GENE_MUT",
    treatment_col="IS_ET",
    duration_col="OS_MONTHS",
    event_col="OS_STATUS",
    covariates=["AGE", "SEX"],
    penalizer=0.01,
)

result = test.run(df)

print(f"\nInteraction Test:")
print(f"  Interaction HR: {result.interaction_hr:.2f}")
print(f"  Interaction P:  {result.interaction_p:.4f}")
print(f"  Interaction CI: [{result.interaction_ci[0]:.2f}, {result.interaction_ci[1]:.2f}]")
print(f"\nStratified Effects:")
print(f"  HR in ET-treated:  {result.hr_treated:.2f} "
      f"(N={result.n_treated}, {'benefit' if result.hr_treated < 1 else 'no benefit'})")
print(f"  HR in non-ET:      {result.hr_untreated:.2f} "
      f"(N={result.n_untreated}, {'benefit' if result.hr_untreated < 1 else 'no benefit'})")
print(f"\n  Classification: {result.classification.upper()}")

if result.classification == "predictive":
    print("  → Gene mutation MODIFIES treatment effect")
    print("  → Clinical implication: test for this mutation before prescribing ET")
else:
    print("  → Gene mutation does NOT modify treatment effect")
    print("  → The gene is associated with outcome regardless of treatment")

# ==============================================================
# Visualize with KM-like summary
# ==============================================================
print("\n" + "=" * 60)
print("STRATIFIED SURVIVAL SUMMARY")
print("=" * 60)

from lifelines import KaplanMeierFitter

groups = {
    "ET + Gene+": df[(df["IS_ET"] == 1) & (df["GENE_MUT"] == 1)],
    "ET + Gene-": df[(df["IS_ET"] == 1) & (df["GENE_MUT"] == 0)],
    "Non-ET + Gene+": df[(df["IS_ET"] == 0) & (df["GENE_MUT"] == 1)],
    "Non-ET + Gene-": df[(df["IS_ET"] == 0) & (df["GENE_MUT"] == 0)],
}

print(f"\n{'Group':<20} {'N':>5} {'Events':>7} {'Median OS':>10}")
print("-" * 45)
for name, grp in groups.items():
    kmf = KaplanMeierFitter()
    kmf.fit(grp["OS_MONTHS"], grp["OS_STATUS"])
    median = kmf.median_survival_time_
    print(f"{name:<20} {len(grp):>5} {int(grp['OS_STATUS'].sum()):>7} {median:>10.1f}")

print("\nIf PREDICTIVE: ET+Gene+ should have best survival (benefit from treatment)")
print("Non-ET+Gene+ should NOT show the same benefit")

# ==============================================================
# Summary
# ==============================================================
print("\n" + "=" * 60)
print("CASCADE LAYER 3 SUMMARY")
print("=" * 60)
print(f"Classification: {result.classification.upper()}")
print(f"Interaction P: {result.interaction_p:.4f}")
print(f"Evidence: HR_treated={result.hr_treated:.2f} vs HR_untreated={result.hr_untreated:.2f}")
print(f"\nCASCADE validation: Layer 3 PASSED — biomarker classified as {result.classification}")
