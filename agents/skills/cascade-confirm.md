# /cascade-confirm — CASCADE Layer 2: Orthogonal Confirmation

## Purpose
Confirm discovery screen findings using an independent statistical method, compute cross-analysis concordance, and classify findings into confidence tiers.

## Required Inputs
- **Primary results**: Output from `/cascade-screen`: either `BiomarkerScreen.screen` output (`biomarker` column) or `screen_competing_risks` output (`gene` + `site` columns), with `hr`, `p_adjusted`, `significant`
- **Data file path**: Same dataset used for discovery
- **Confirmatory method**: Independent method (e.g., if primary was `competing_risks`, use `logistic`)
- **Outcome column**: Binary outcome for `logistic` (no `event_col` needed); for gene-site results each site column of the data is used as the outcome for its pairs, and `outcome_col` is only the fallback
- **Covariates** (optional): Same or different adjustment set

## What It Does

1. **Loads** primary results and identifies significant findings
2. **Re-tests** each significant finding using the confirmatory method
3. **Computes concordance metrics**:
   - Direction agreement: % where HR>1 matches OR>1
   - Spearman rho between log effect sizes (log HR vs log OR)
4. **Classifies** findings into confidence tiers:
   - **Both-significant**: FDR-sig in BOTH primary AND confirmatory (highest confidence)
   - **Primary-only**: FDR-sig only in primary analysis
   - **Confirmatory-only**: FDR-sig only in confirmatory analysis
   - **Neither**: Not significant in either (should not exist if starting from primary hits)

## Expected Output
- CSV: `cascade_confirmation_results.csv` with primary + confirmatory columns + confidence_tier
- Concordance summary: direction agreement %, Spearman rho + P
- Count of both-significant findings

## Figure Guidelines

Any figures produced (e.g., concordance scatterplots, effect size comparisons) MUST follow these standards:
- **Dual format**: Save both PNG (300 DPI) and PDF
- **Font**: Arial, minimum 12pt axis labels, 10pt ticks
- **Panels**: Each panel as its own separate file (`fig_confirm_scatter.png`, `fig_confirm_forest.png`)
- **Layout**: Minimal whitespace, `bbox_inches='tight'`, no chart junk, spines top/right off
- **Size**: Compact figures with large text relative to plot elements
- See `CLAUDE.md > Figure Standards` for the full matplotlib rcParams block

## Post-Execution Validation
- [ ] Confirmatory method differs from primary method
- [ ] Direction concordance computed and reported
- [ ] Both-significant subset identified
- [ ] No findings upgraded without statistical justification

## Example Code

```python
from cascade.core import BiomarkerScreen, OrthogonalConfirm

genes = ["GENE_A", "GENE_B", "GENE_C"]
sites = ["SITE_LIVER", "SITE_BONE"]

# Layer 1 (tropism): cause-specific Cox per gene-site pair
screen_results = BiomarkerScreen(method="competing_risks").screen_competing_risks(
    data, genes, sites, duration_col="OS_MONTHS", event_col="OS_STATUS",
)

# Layer 2: logistic confirmation, pair-wise (each site column is the outcome)
confirm = OrthogonalConfirm(primary_method="competing_risks", confirm_method="logistic")
results = confirm.confirm(
    primary_results=screen_results,
    df=data,
    biomarker_cols=genes,
    outcome_col="SITE_LIVER",          # fallback for sites that are not columns
    covariates=["AGE", "SEX"],
)
metrics = confirm.concordance_metrics(screen_results, results)
print(f"Direction agreement: {metrics['direction_agreement_pct']:.1f}%")
print(f"Spearman rho (log effects): {metrics['spearman_rho']:.2f}")
print(f"Both-significant: {metrics['n_both_significant']}")
print(results["confidence_tier"].value_counts())
```
