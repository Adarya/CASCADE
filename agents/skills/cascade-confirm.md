# /cascade-confirm — CASCADE Layer 2: Orthogonal Confirmation

## Purpose
Confirm discovery screen findings using an independent statistical method, compute cross-analysis concordance, and classify findings into confidence tiers.

## Required Inputs
- **Primary results**: Output from `/cascade-screen` (CSV with biomarker, hr, p_adjusted, significant)
- **Data file path**: Same dataset used for discovery
- **Confirmatory method**: Independent method (e.g., if primary was `competing_risks`, use `logistic`)
- **Covariates** (optional): Same or different adjustment set

## What It Does

1. **Loads** primary results and identifies significant findings
2. **Re-tests** each significant finding using the confirmatory method
3. **Computes concordance metrics**:
   - Direction agreement: % where HR>1 matches OR>1
   - Spearman rho between -log10(P) values
   - Effect size correlation
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
from cascade.core import OrthogonalConfirm

confirm = OrthogonalConfirm(primary_method="competing_risks", confirm_method="logistic")
results = confirm.confirm(
    primary_results=screen_results,
    df=data,
    biomarker_cols=sig_genes,
    outcome_col="HAS_SITE",
    covariates=["AGE", "SEX", "HISTOLOGY"],
)
metrics = confirm.concordance_metrics(screen_results, confirm_results)
print(f"Direction agreement: {metrics['direction_agreement_pct']:.1f}%")
print(f"Both-significant: {metrics['n_both_significant']}")
```
