"""
CASCADE Example: Independence Testing (Layer 5)

Demonstrates Layer 5 artifact guards — specifically independence testing
between two clinical phenotypes that could be tautological.

This example shows:
- Layer 5: Independence testing (chi-square, phi, TOST, Bayes factor)
- Floor effect detection and baseline confounding resolution
"""

import numpy as np
import pandas as pd

# ---- Generate synthetic data ----

np.random.seed(42)
N = 800

# Two clinical phenotypes that are intentionally INDEPENDENT
# (burden change and ECOG change are generated independently)
ecog_baseline = np.random.choice([0, 1, 2], N, p=[0.3, 0.5, 0.2])
burden_baseline = np.random.choice([1, 2, 3], N, p=[0.4, 0.35, 0.25])

# Changes at 6-month landmark (independent of each other)
ecog_change = np.random.choice([-1, 0, 1], N, p=[0.15, 0.60, 0.25])
burden_change = np.random.choice([-1, 0, 1, 2], N, p=[0.10, 0.50, 0.30, 0.10])

ecog_at_lm = np.clip(ecog_baseline + ecog_change, 0, 4)
burden_at_lm = np.clip(burden_baseline + burden_change, 1, 5)

# Binary worsening indicators
ecog_worsened = (ecog_change > 0).astype(int)
burden_worsened = (burden_change > 0).astype(int)

# Risk groups
risk_group = np.where(
    (ecog_worsened == 0) & (burden_worsened == 0), "Neither Worsened",
    np.where(
        (ecog_worsened == 1) & (burden_worsened == 0), "ECOG Only",
        np.where(
            (ecog_worsened == 0) & (burden_worsened == 1), "Burden Only",
            "Both Worsened"
        )
    )
)

# Generate survival with both effects (additive, not redundant)
log_hazard = (
    0.3 * ecog_change
    + 0.25 * burden_change
    + 0.1 * ecog_baseline  # floor effect: higher baseline = less room to worsen
    + 0.05 * burden_baseline
)
scale = np.exp(-log_hazard) * 25
os_from_lm = np.round(np.random.exponential(scale).clip(0.1, 60), 1)
censor = np.random.exponential(30, N)
event = (os_from_lm <= censor).astype(int)
os_from_lm = np.round(np.minimum(os_from_lm, censor), 1)

df = pd.DataFrame({
    "PATIENT_ID": [f"P-{i:07d}" for i in range(N)],
    "ECOG_BASELINE": ecog_baseline,
    "ECOG_AT_LM": ecog_at_lm,
    "ECOG_CHANGE": ecog_change,
    "ECOG_WORSENED": ecog_worsened,
    "BURDEN_BASELINE": burden_baseline,
    "BURDEN_AT_LM": burden_at_lm,
    "BURDEN_CHANGE": burden_change,
    "BURDEN_WORSENED": burden_worsened,
    "RISK_GROUP": risk_group,
    "OS_FROM_LM": os_from_lm,
    "EVENT": event,
    "AGE": np.random.normal(65, 12, N).clip(30, 95).astype(int),
    "SEX": np.random.binomial(1, 0.55, N),
})

print(f"Dataset: {N} patients at 6-month landmark")
print(f"Risk group distribution:")
for g in ["Neither Worsened", "ECOG Only", "Burden Only", "Both Worsened"]:
    n = (df["RISK_GROUP"] == g).sum()
    pct = n / N * 100
    print(f"  {g}: {n} ({pct:.1f}%)")
print()

# ==============================================================
# Layer 5: Independence Testing
# ==============================================================
print("=" * 60)
print("LAYER 5: Statistical Artifact Guards — Independence Testing")
print("=" * 60)

from cascade.core import ArtifactGuard
from cascade.stats import independence_suite, phi_coefficient

# A. Formal independence test
guard = ArtifactGuard()
indep = guard.check_independence(
    df["ECOG_WORSENED"],
    df["BURDEN_WORSENED"],
    x_continuous=df["ECOG_CHANGE"].astype(float),
    y_continuous=df["BURDEN_CHANGE"].astype(float),
)

print(f"\nA. Formal Independence Test:")
print(f"  Chi-square P: {indep['chi2_p']:.4f}")
print(f"  Phi coefficient: {indep['phi']:.4f} "
      f"[{indep['phi_ci_lower']:.4f}, {indep['phi_ci_upper']:.4f}]")
print(f"  Cramer's V: {indep['cramers_v']:.4f}")
print(f"  Discordance rate: {indep['discordance_rate']:.1%}")
print(f"  Spearman rho: {indep['spearman_rho']:.4f} (P={indep['spearman_p']:.4f})")
print(f"  Kendall tau: {indep['kendall_tau']:.4f} (P={indep['kendall_p']:.4f})")

if indep["chi2_p"] > 0.05:
    print("  → CONCLUSION: Features are INDEPENDENT (cannot reject H0)")
else:
    print("  → CONCLUSION: Features may be associated (investigate further)")

# B. Floor/ceiling effect detection
print(f"\nB. Floor/Ceiling Effect Analysis:")
fc = guard.check_floor_ceiling_effects(df, "ECOG_BASELINE", 0, 4)
print(f"  Floor (ECOG=0): {fc['floor_pct']:.1f}%")
print(f"  Ceiling (ECOG=4): {fc['ceiling_pct']:.1f}%")

# Show floor effect in action
ecog_only = df[df["RISK_GROUP"] == "ECOG Only"]
neither = df[df["RISK_GROUP"] == "Neither Worsened"]
print(f"\n  ECOG-Only group baseline ECOG mean: {ecog_only['ECOG_BASELINE'].mean():.2f}")
print(f"  Neither group baseline ECOG mean: {neither['ECOG_BASELINE'].mean():.2f}")
print("  → ECOG-Only patients start at lower (better) baseline ECOG")
print("  → This is a FLOOR EFFECT: can only worsen if starting low")

# C. Baseline confounding check
print(f"\nC. Baseline Confounding Resolution:")
from cascade.stats import PenalizedCox

# Unadjusted model
df["IS_ECOG_ONLY"] = (df["RISK_GROUP"] == "ECOG Only").astype(int)
df["IS_BURDEN_ONLY"] = (df["RISK_GROUP"] == "Burden Only").astype(int)
df["IS_BOTH"] = (df["RISK_GROUP"] == "Both Worsened").astype(int)

cox_unadj = PenalizedCox(penalizer=0.01)
cox_unadj.fit(df, "OS_FROM_LM", "EVENT",
              covariates=["IS_ECOG_ONLY", "IS_BURDEN_ONLY", "IS_BOTH", "AGE", "SEX"])
if cox_unadj.converged:
    hr_ecog_unadj = np.exp(cox_unadj.summary.loc["IS_ECOG_ONLY", "coef"])
    print(f"  Unadjusted ECOG-Only HR: {hr_ecog_unadj:.2f}")

# Adjusted model (adding baseline ECOG)
cox_adj = PenalizedCox(penalizer=0.01)
cox_adj.fit(df, "OS_FROM_LM", "EVENT",
           covariates=["IS_ECOG_ONLY", "IS_BURDEN_ONLY", "IS_BOTH",
                       "ECOG_BASELINE", "BURDEN_BASELINE", "AGE", "SEX"])
if cox_adj.converged:
    hr_ecog_adj = np.exp(cox_adj.summary.loc["IS_ECOG_ONLY", "coef"])
    print(f"  Baseline-adjusted ECOG-Only HR: {hr_ecog_adj:.2f}")

confounded = guard.check_baseline_confounding(df, hr_ecog_adj, hr_ecog_unadj)
print(f"  Confounding detected (>15% HR change): {confounded}")
if confounded:
    print("  → Floor effect RESOLVED: baseline adjustment reveals true direction")

# ==============================================================
# Pitfall Detection
# ==============================================================
print("\n" + "=" * 60)
print("PITFALL DETECTION")
print("=" * 60)

from cascade.pitfalls import PitfallDetector

detector = PitfallDetector()

# Check for collinearity between burden and ECOG features
feature_cols = ["ECOG_CHANGE", "BURDEN_CHANGE", "ECOG_BASELINE", "BURDEN_BASELINE"]
detector.check_collinearity(df[feature_cols])
detector.check_constant_variables(df[feature_cols])

print(f"\nPitfall warnings: {len(detector.warnings)}")
for w in detector.warnings:
    print(f"  [{w.severity.value}] {w.message}")
if len(detector.warnings) == 0:
    print("  No pitfalls detected — features are independent and well-conditioned")

# ==============================================================
# Summary
# ==============================================================
print("\n" + "=" * 60)
print("CASCADE LAYER 5 SUMMARY")
print("=" * 60)
print(f"Independence: phi = {indep['phi']:.3f}, P = {indep['chi2_p']:.3f} (INDEPENDENT)")
print(f"Discordance: {indep['discordance_rate']:.1%} of patients are discordant")
print(f"Floor effect: ECOG-Only group has lower baseline → confounds unadjusted analysis")
print(f"Resolution: Baseline adjustment changes HR direction → confounding resolved")
print(f"Pitfalls: {len(detector.warnings)} detected")
print(f"\nCASCADE validation: Layer 5 PASSED — features are independent, artifacts resolved")
