# /cascade-translate — CASCADE Layer 6: Clinical Translation Assessment

## Purpose
Quantify the clinical utility of validated biomarkers through landmark analysis, cross-validated discrimination metrics, and risk group construction.

## Required Inputs
- **Data file path**: Dataset with survival outcomes
- **Biomarker/feature columns**: Variables to assess for clinical utility
- **Outcome/event columns**: Survival time + event indicator
- **Landmark timepoints**: Clinically relevant timepoints (e.g., 3, 6, 12 months)
- **Base model variables**: Standard clinical predictors (age, sex, stage)

## What It Does

1. **Landmark analysis**:
   - Restricts to patients alive at each landmark
   - Computes residual survival from landmark forward
   - Ensures no immortal time bias
2. **Cross-validated delta-C**:
   - Fits base model (clinical only) and full model (clinical + biomarker)
   - Stratified 10-fold CV: train on folds, evaluate on held-out
   - Computes delta-C = C_full - C_base (added discrimination)
3. **Uncertainty**:
   - `DeltaC.compute` uses `cv_delta_c` (stratified folds, seeded) and returns `delta_c`, `ci` (bootstrap over the CV fold deltas), `p_value` (fraction of fold-bootstrap means <= 0), `base_c`, `full_c` and `n_failed_folds`
   - `cascade.stats.bootstrap_delta_c` (whole-cohort resampling) is descriptive only. Its p-value is NaN, so use `cv_delta_c` / `DeltaC` for inference
4. **Likelihood ratio test** (not part of `DeltaC`; compute separately):
   - Nested model comparison (base vs full)
   - Chi-square test with df = number of added variables
5. **Risk group construction**:
   - Tertiles/quartiles of predicted risk score
   - KM curves per group
   - Pairwise log-rank tests with Holm correction

## Expected Output
- Delta-C at each landmark with 95% CI and P-value (the Layer 6 gate fails if delta-C or the CI is missing or non-finite)
- LRT chi-square statistic and P-value
- KM curves by risk group
- Median survival per group

## Figure Guidelines

Any figures produced (e.g., KM curves, delta-C bar charts) MUST follow these standards:
- **Dual format**: Save both PNG (300 DPI) and PDF
- **Font**: Arial, minimum 12pt axis labels, 10pt ticks
- **Panels**: Each panel as its own separate file (`fig_km_group1.png`, `fig_deltac_bar.png`)
- **Layout**: Minimal whitespace, `bbox_inches='tight'`, no chart junk, spines top/right off
- **Size**: Compact figures with large text relative to plot elements
- See `CLAUDE.md > Figure Standards` for the full matplotlib rcParams block

## Post-Execution Validation
- [ ] C-statistics are CROSS-VALIDATED (not training-set)
- [ ] Landmark restriction correctly applied (only alive-at-landmark patients)
- [ ] Delta-C > 0 (biomarker adds discrimination)
- [ ] Bootstrap CI does not cross 0 for significant claims
- [ ] Risk groups show clear separation on KM curves

## Example Code

```python
from cascade.core import LandmarkAnalysis, DeltaC, RiskGroupAnalysis

# Landmark analysis
lm = LandmarkAnalysis(landmarks_months=[3, 6, 12])
df_6mo = lm.restrict_to_alive(df, "OS_MONTHS", 6)
df_6mo = lm.compute_residual_survival(df_6mo, "OS_MONTHS", 6)

# Cross-validated delta-C
dc = DeltaC(n_folds=10, n_bootstrap=1000)
result = dc.compute(
    df=df_6mo,
    base_vars=["AGE", "SEX", "STAGE"],
    full_vars=["AGE", "SEX", "STAGE", "BIOMARKER_SCORE"],
    duration_col="OS_FROM_LM",
    event_col="OS_STATUS",
)
print(f"Delta-C: +{result['delta_c']:.3f} (95% CI: {result['ci'][0]:.3f}-{result['ci'][1]:.3f})")
print(f"Fold-bootstrap P: {result['p_value']:.3g}; failed folds: {result['n_failed_folds']}")

# Risk groups
rga = RiskGroupAnalysis(n_groups=3)
df_6mo = rga.create_groups(df_6mo, "BIOMARKER_SCORE")
km_results = rga.km_by_group(df_6mo, "OS_FROM_LM", "OS_STATUS", "risk_group")
```
