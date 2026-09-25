"""
CASCADE Example: NSCLC Tropism Analysis (Layers 1, 2, 4)

Demonstrates the discovery → confirmation → sensitivity cascade
using synthetic data that mimics the NSCLC gene-site tropism study.

This example shows:
- Layer 1: Biomarker discovery screen (Cox PH with FDR correction)
- Layer 2: Orthogonal confirmation (logistic regression + concordance)
- Layer 4: Sensitivity analysis with robustness scoring
"""

import numpy as np
import pandas as pd

# ---- Generate synthetic NSCLC-like data ----

np.random.seed(42)
N = 1000

# Genes with known effects on specific sites
genes = {
    "STK11_MUT": 0.20,  # 20% prevalence
    "KEAP1_MUT": 0.15,
    "EGFR_MUT": 0.18,
    "TP53_MUT": 0.45,
    "BRAF_MUT": 0.08,
}

# Sites
sites = ["SITE_ADRENAL", "SITE_BONE", "SITE_LIVER", "SITE_CNS", "SITE_LN"]

df = pd.DataFrame({"PATIENT_ID": [f"P-{i:07d}" for i in range(N)]})

# Generate gene mutations
for gene, prev in genes.items():
    df[gene] = np.random.binomial(1, prev, N)

# Generate site involvement with gene-dependent probabilities
# STK11 → Adrenal (strong), Bone (moderate)
# KEAP1 → Bone (strong)
# EGFR → Liver (strong)
# BRAF → CNS (protective)
# TP53 → LN (moderate)

base_rates = {"SITE_ADRENAL": 0.12, "SITE_BONE": 0.25, "SITE_LIVER": 0.20,
              "SITE_CNS": 0.10, "SITE_LN": 0.30}

for site, base in base_rates.items():
    logit = np.log(base / (1 - base)) * np.ones(N)
    if site == "SITE_ADRENAL":
        logit += 0.8 * df["STK11_MUT"] + 0.5 * df["KEAP1_MUT"]
    elif site == "SITE_BONE":
        logit += 0.5 * df["KEAP1_MUT"] + 0.3 * df["STK11_MUT"]
    elif site == "SITE_LIVER":
        logit += 0.6 * df["EGFR_MUT"]
    elif site == "SITE_CNS":
        logit += -0.5 * df["BRAF_MUT"]
    elif site == "SITE_LN":
        logit += 0.3 * df["TP53_MUT"]
    prob = 1 / (1 + np.exp(-logit))
    df[site] = np.random.binomial(1, prob, N)

# Generate survival with gene and site effects
log_hazard = (
    0.3 * df["STK11_MUT"]
    + 0.2 * df["KEAP1_MUT"]
    - 0.15 * df["EGFR_MUT"]
    + 0.4 * df["SITE_LIVER"]
    + 0.3 * df["SITE_CNS"]
    + 0.01 * np.random.normal(65, 12, N)
)
scale = np.exp(-log_hazard) * 20
df["OS_MONTHS"] = np.round(np.random.exponential(scale).clip(0.5, 120), 1)
censor = np.random.exponential(40, N)
df["OS_STATUS"] = (df["OS_MONTHS"] <= censor).astype(int)
df["OS_MONTHS"] = np.round(np.minimum(df["OS_MONTHS"], censor), 1)
df["AGE"] = np.random.normal(65, 12, N).clip(30, 95).astype(int)
df["SEX"] = np.random.binomial(1, 0.55, N)

gene_cols = list(genes.keys())
site_cols = sites
covariates = ["AGE", "SEX"]

print(f"Dataset: {N} patients, {len(gene_cols)} genes, {len(site_cols)} sites")
print(f"Events: {df['OS_STATUS'].sum()} ({df['OS_STATUS'].mean():.1%})")
print()

# ==============================================================
# Layer 1: Discovery Screen
# ==============================================================
print("=" * 60)
print("LAYER 1: Biomarker Discovery Screen")
print("=" * 60)

from cascade.core import BiomarkerScreen

screen = BiomarkerScreen(
    method="cox",
    correction="fdr_bh",
    alpha=0.05,
    min_exposed=20,
    min_events=5,
    penalizer=0.01,
)

# Screen each gene for association with each site
results_list = []
for gene in gene_cols:
    for site in site_cols:
        # For each gene-site pair, test gene → site association via logistic
        from cascade.stats import PenalizedCox
        # Use Cox to test gene effect on survival stratified by site
        subset = df[df[site] == 1]
        if subset[gene].sum() < 5 or len(subset) < 30:
            continue

        cox = PenalizedCox(penalizer=0.01)
        try:
            cox.fit(subset, "OS_MONTHS", "OS_STATUS", covariates=[gene] + covariates)
            if cox.converged:
                summary = cox.summary
                hr = np.exp(summary.loc[gene, "coef"])
                p = summary.loc[gene, "p"]
                results_list.append({
                    "gene": gene, "site": site,
                    "hr": hr, "p": p,
                    "n": len(subset), "n_exposed": int(subset[gene].sum()),
                })
        except Exception:
            pass

primary_results = pd.DataFrame(results_list)

# Apply FDR correction
from cascade.stats import apply_fdr
adj_p, sig = apply_fdr(primary_results["p"].values)
primary_results["p_adjusted"] = adj_p
primary_results["significant"] = sig

n_sig = primary_results["significant"].sum()
print(f"\nTested: {len(primary_results)} gene-site pairs")
print(f"FDR-significant: {n_sig}")
if n_sig > 0:
    print("\nTop findings:")
    top = primary_results[primary_results["significant"]].sort_values("p")
    for _, row in top.iterrows():
        direction = "risk" if row["hr"] > 1 else "protective"
        print(f"  {row['gene']} → {row['site']}: HR={row['hr']:.2f}, "
              f"P_adj={row['p_adjusted']:.4f} ({direction})")

# ==============================================================
# Layer 2: Orthogonal Confirmation (Logistic Regression)
# ==============================================================
print("\n" + "=" * 60)
print("LAYER 2: Orthogonal Confirmation (Logistic)")
print("=" * 60)

confirm_results = []
for _, row in primary_results.iterrows():
    gene, site = row["gene"], row["site"]
    try:
        import statsmodels.api as sm
        X = sm.add_constant(df[[gene] + covariates])
        y = df[site]
        model = sm.Logit(y, X).fit(disp=0)
        or_val = np.exp(model.params[gene])
        p_val = model.pvalues[gene]
        confirm_results.append({
            "gene": gene, "site": site,
            "or": or_val, "confirm_p": p_val,
        })
    except Exception:
        pass

confirm_df = pd.DataFrame(confirm_results)
adj_p2, sig2 = apply_fdr(confirm_df["confirm_p"].values)
confirm_df["confirm_p_adjusted"] = adj_p2
confirm_df["confirm_significant"] = sig2

# Merge and compute concordance
merged = primary_results.merge(confirm_df, on=["gene", "site"])
merged["direction_concordant"] = (
    ((merged["hr"] > 1) & (merged["or"] > 1))
    | ((merged["hr"] <= 1) & (merged["or"] <= 1))
)
merged["both_significant"] = merged["significant"] & merged["confirm_significant"]

concordance_pct = merged["direction_concordant"].mean() * 100
n_both = merged["both_significant"].sum()

from scipy.stats import spearmanr
rho, rho_p = spearmanr(
    -np.log10(merged["p"].clip(1e-300)),
    -np.log10(merged["confirm_p"].clip(1e-300))
)

print(f"\nDirection concordance: {concordance_pct:.1f}%")
print(f"Spearman rho (-log10 P): {rho:.3f} (P={rho_p:.2e})")
print(f"Both-significant findings: {n_both}")

# ==============================================================
# Layer 4: Sensitivity & Robustness
# ==============================================================
print("\n" + "=" * 60)
print("LAYER 4: Sensitivity & Robustness Testing")
print("=" * 60)

from cascade.core import SensitivitySuite, RobustnessScorer

suite = SensitivitySuite()
suite.add_variant("Full cohort", df, "subgroup")
suite.add_variant("Age >= 60", df[df["AGE"] >= 60], "subgroup")
suite.add_variant("Age < 60", df[df["AGE"] < 60], "subgroup")
suite.add_variant("Female", df[df["SEX"] == 1], "stratification")
suite.add_variant("Male", df[df["SEX"] == 0], "stratification")

print(f"Sensitivity variants: {len(suite.variant_names)}")
print(f"Categories covered: subgroup, stratification")

# Run sensitivity for significant findings
scorer = RobustnessScorer()
sig_findings = merged[merged["significant"]].copy()
robustness_df = pd.DataFrame(
    columns=["gene", "site", "primary_hr", "n_evaluable", "robustness_class"]
)
counts = pd.Series(dtype=int)

if len(sig_findings) > 0:
    robustness_results = []
    for _, finding in sig_findings.iterrows():
        gene, site = finding["gene"], finding["site"]
        primary_hr = finding["hr"]
        variant_hrs = []

        for vname in suite.variant_names:
            vdata = suite._variants[vname]["data"]
            subset = vdata[vdata[site] == 1]
            hr = np.nan  # a variant that cannot be estimated counts as failed
            if subset[gene].sum() >= 3 and len(subset) >= 20:
                try:
                    cox = PenalizedCox(penalizer=0.01)
                    cox.fit(subset, "OS_MONTHS", "OS_STATUS",
                            covariates=[gene] + covariates)
                    if cox.converged:
                        hr = np.exp(cox.summary.loc[gene, "coef"])
                except Exception:
                    pass
            variant_hrs.append(hr)

        detail = scorer.score_detail(primary_hr, variant_hrs)
        robustness_results.append({
            "gene": gene, "site": site,
            "primary_hr": primary_hr,
            "n_evaluable": detail["n_evaluable"],
            "robustness_class": detail["robustness_class"],
        })

    robustness_df = pd.DataFrame(robustness_results)
    counts = robustness_df["robustness_class"].value_counts()
    print(f"\nRobustness classification:")
    for cls in ["robust", "exploratory", "unstable"]:
        print(f"  {cls.upper()}: {counts.get(cls, 0)}")
else:
    print("\nNo FDR-significant findings to stress-test.")

# ==============================================================
# Decision gates
# ==============================================================
# Evaluate each layer's gate with the package's GateEvaluator so the verdict
# below reflects what actually happened, not what we hoped would happen.
from types import SimpleNamespace
from cascade.core import GateEvaluator

evaluator = GateEvaluator(study_type="tropism")
gates = {
    "discovery": evaluator.evaluate(
        "discovery", SimpleNamespace(results=primary_results)
    ),
    "confirmation": evaluator.evaluate(
        "confirmation",
        SimpleNamespace(results={
            "direction_concordance": concordance_pct / 100,
            "n_both_significant": int(n_both),
        }),
    ),
    "sensitivity": evaluator.evaluate(
        "sensitivity", SimpleNamespace(results=robustness_df)
    ),
}

# ==============================================================
# Summary
# ==============================================================
print("\n" + "=" * 60)
print("CASCADE SUMMARY")
print("=" * 60)
print(f"Layer 1 (Discovery):    {len(primary_results)} tested → {n_sig} FDR-significant")
print(f"Layer 2 (Confirmation): {concordance_pct:.0f}% concordance, {n_both} both-significant")
if len(sig_findings) > 0:
    print(f"Layer 4 (Robustness):   {counts.get('robust', 0)} robust / "
          f"{counts.get('exploratory', 0)} exploratory / "
          f"{counts.get('unstable', 0)} unstable")

print("\nDecision gates:")
for name, gate in gates.items():
    status = "PASS" if gate.passed else "FAIL"
    detail = f" -- {gate.error_message}" if gate.error_message else ""
    print(f"  Layer {gate.layer_number} ({name}): {status}{detail}")

# Validation levels (docs/cascade_checklist.md): Minimum = Layers 1 + 4;
# Recommended additionally needs Layers 2 and 5.  Layer 5 (artifact guards)
# is not run in this example, so Recommended cannot be reached here.
if gates["discovery"].passed and gates["sensitivity"].passed:
    level = "Minimum (Layers 1 + 4 passed)"
    if gates["confirmation"].passed:
        level += "; Layer 2 also passed -- run Layer 5 to reach Recommended"
    else:
        level += "; Layer 2 gate failed"
else:
    failed = [f"Layer {g.layer_number}" for g in gates.values() if not g.passed]
    level = f"NOT ACHIEVED (failed gate(s): {', '.join(failed)})"
print(f"\nCASCADE validation level: {level}")
