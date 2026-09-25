# /cascade-guard — CASCADE Layer 5: Statistical Artifact Guards

## Purpose
Detect and prevent statistical artifacts that could invalidate biomarker conclusions. The pitfall library has 11 pitfalls. The 7 fully automated ones (#1–#4, #7, #10, #11) are covered by 8 check functions. #5 (runtime), #6 (manual) and #8, #9 (partial) need structured audits.

## Required Inputs
- **Data file** (optional): Raw data file to check for format issues
- **Model DataFrame** (optional): Analysis-ready dataset to check for statistical issues
- **Feature matrix** (optional): Predictor matrix to check for collinearity/singularity
- **Landmark days** (optional): If landmark analysis, the landmark timepoint
- **Survival data** (optional): One-row-per-patient data with duration, event and biomarker columns (for #10), plus a centre/institution column (for #11)
- **Model results** (optional): Results table with CI columns (for the separation check)

## What It Does

### Automated Checks (7 pitfalls, 8 check functions)
1. **Comment header corruption** (#1): Scans data file for hex color values that would be corrupted by `comment='#'` parsing
2. **Covariate leakage** (#2): Verifies all covariates use only pre-landmark data. For date-based detection pass `timeline_cols` as a `{covariate: exposure_date_col}` dict (a plain list is only paired with covariates by name)
3. **Perfect collinearity** (#3): Checks pairwise correlations (|r| >= 0.99), VIF > 10 and algebraic identities
4. **Constant variables** (#4): Identifies zero-variance columns
5. **Singular matrix risk** (#7): Checks condition number and rank of the feature matrix; identifies redundant columns
6. **Separation** (reported under #7): Flags results whose CI ratio (upper/lower) exceeds 100
7. **Informative censoring** (#10): Reverse-censoring Cox model per biomarker (Bonferroni, censoring HR >= 1.25-fold)
8. **Centre / batch effect** (#11): Prevalence heterogeneity across centres and a biomarker-by-centre interaction LRT

`PitfallDetector.check_all` runs only the checks whose inputs are supplied. The same checks run inside Layer 5 via `ArtifactGuard.run_all(..., pitfall_inputs={...})`. There, CRITICAL warnings block the Layer 5 gate, and the gate also fails if no check ran.

### Structured Audit Prompts (4 pitfalls)
9. **XGBoost softprob** (#5, runtime): If using XGBoost multi-class, check if any fold or subgroup has fewer classes than `num_class`
10. **Platform package failure** (#6, manual): Note any R/Bioconductor install failures and the workaround used
11. **Fabricated citations** (#8, partial): `ManuscriptHelper.flag_potential_fabricated_refs` screens format/plausibility; every reference still needs manual verification
12. **Performance status floor effect** (#9, partial): Compare unadjusted vs baseline-adjusted HRs (`ArtifactGuard.check_baseline_confounding`) and check floor/ceiling effects of bounded variables (ECOG 0-4)

## Expected Output
- List of `PitfallWarning` objects with severity (CRITICAL/WARNING/INFO)
- Formatted summary report
- Specific fix suggestions for each detected issue

## Figure Guidelines

Any figures produced (e.g., mosaic plots, independence scatters) MUST follow these standards:
- **Dual format**: Save both PNG (300 DPI) and PDF
- **Font**: Arial, minimum 12pt axis labels, 10pt ticks
- **Panels**: Each panel as its own separate file (`fig_guard_mosaic.png`, `fig_guard_scatter.png`)
- **Layout**: Minimal whitespace, `bbox_inches='tight'`, no chart junk, spines top/right off
- **Size**: Compact figures with large text relative to plot elements
- See `CLAUDE.md > Figure Standards` for the full matplotlib rcParams block

## Post-Execution Validation
- [ ] Every automated check whose inputs are available was run (check `ArtifactReport.checks_run`; INFO warnings mark checks that could not be fitted)
- [ ] No CRITICAL warnings remain unresolved
- [ ] Audit prompts for manual checks displayed
- [ ] Fix suggestions provided for all detected issues

## Example Code

```python
from cascade.pitfalls import PitfallDetector

detector = PitfallDetector()

# Check data file for format issues (#1)
detector.check_data_format("data_timeline_performance_status.txt")

# Check model data for statistical issues (#3, #4, #7)
detector.check_collinearity(feature_matrix, threshold=0.99)
detector.check_constant_variables(feature_matrix)
detector.check_singular_matrix(feature_matrix)

# Check for covariate leakage (#2): map each covariate to its exposure date
detector.check_covariate_leakage(
    df=landmark_df,
    landmark_col="LANDMARK_DATE",
    covariate_cols=["RECEIVED_ICI", "RECEIVED_CHEMO"],
    timeline_cols={"RECEIVED_ICI": "ICI_START_DATE", "RECEIVED_CHEMO": "CHEMO_START_DATE"},
)

# Survival-data checks (#10, #11)
detector.check_informative_censoring(df, "OS_MONTHS", "OS_STATUS", ["GENE_A", "GENE_B"])
detector.check_center_effect(df, "OS_MONTHS", "OS_STATUS", ["GENE_A", "GENE_B"], center_col="CENTER")

# Print all warnings
for w in detector.warnings:
    print(f"[{w.severity.value}] #{w.pitfall.id} {w.pitfall.name}: {w.message}")
    print(f"  Fix: {w.suggestion}")
```

Running the pitfall library as part of Layer 5:

```python
from cascade.core import ArtifactGuard

report = ArtifactGuard().run_all(
    df,
    biomarker_cols=["GENE_A", "GENE_B"],
    pitfall_inputs=dict(
        feature_matrix=feature_matrix,
        survival_df=df, duration_col="OS_MONTHS", event_col="OS_STATUS",
        biomarker_cols=["GENE_A", "GENE_B"], center_col="CENTER",
    ),
)
print(report.checks_run)            # e.g. ['collinearity', 'pitfall_library']
print(report.critical_findings)     # non-empty => Layer 5 gate fails
```
