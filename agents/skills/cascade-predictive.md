# /cascade-predictive — CASCADE Layer 3: Predictive vs Prognostic Distinction

## Purpose
Formally test whether a biomarker is predictive (modifies treatment effect) or merely prognostic (associated with outcome regardless of treatment).

## Required Inputs
- **Data file path**: CSV/TSV with patient-level data
- **Gene column**: Binary biomarker column
- **Treatment column**: Binary treatment indicator
- **Outcome/event columns**: Survival time + event indicator
- **Covariates** (optional): Adjustment variables

## What It Does

1. **Fits interaction model**: Cox PH with gene + treatment + gene:treatment + covariates
2. **Extracts interaction term**: HR, P-value, 95% CI for gene:treatment
3. **Computes stratified effects**:
   - HR of gene among TREATED patients
   - HR of gene among UNTREATED patients
4. **Classifies**:
   - Interaction P < 0.05 → PREDICTIVE (gene modifies treatment effect)
   - Interaction P >= 0.05 → PROGNOSTIC only (gene associated with outcome regardless)
5. **Optional 3-way extension**: gene x treatment x phenotype

## Expected Output
- Interaction HR, P-value, 95% CI
- Stratified HRs (treated vs untreated)
- Classification: PREDICTIVE or PROGNOSTIC
- Forest plot data for visualization

## Figure Guidelines

Any figures produced (e.g., forest plots, stratified KM curves) MUST follow these standards:
- **Dual format**: Save both PNG (300 DPI) and PDF
- **Font**: Arial, minimum 12pt axis labels, 10pt ticks
- **Panels**: Each panel as its own separate file (`fig_pred_forest.png`, `fig_pred_km_treated.png`)
- **Layout**: Minimal whitespace, `bbox_inches='tight'`, no chart junk, spines top/right off
- **Size**: Compact figures with large text relative to plot elements
- See `CLAUDE.md > Figure Standards` for the full matplotlib rcParams block

## Post-Execution Validation
- [ ] Interaction term correctly specified (product of binary variables)
- [ ] Stratified analysis uses SEPARATE patient subsets (not overlapping)
- [ ] Classification explicitly stated
- [ ] If predictive: clinical implication noted (e.g., "test before prescribing")

## Example Code

```python
from cascade.core import PredictiveTest

test = PredictiveTest(
    gene_col="CBFB_MUT",
    treatment_col="IS_ENDOCRINE",
    duration_col="OS_MONTHS",
    event_col="OS_STATUS",
    covariates=["AGE", "STAGE", "ER_STATUS"],
)
result = test.run(df)
print(f"Classification: {result.classification}")
print(f"Interaction HR: {result.interaction_hr:.2f} (P={result.interaction_p:.4f})")
print(f"HR in treated: {result.hr_treated:.2f}")
print(f"HR in untreated: {result.hr_untreated:.2f}")
```
