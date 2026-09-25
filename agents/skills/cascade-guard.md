# /cascade-guard — CASCADE Layer 5: Statistical Artifact Guards

## Purpose
Detect and prevent statistical artifacts that could invalidate biomarker conclusions. Run automated checks for the 6 automatable pitfalls plus structured audits for the 3 partially-automatable ones.

## Required Inputs
- **Data file** (optional): Raw data file to check for format issues
- **Model DataFrame** (optional): Analysis-ready dataset to check for statistical issues
- **Feature matrix** (optional): Predictor matrix to check for collinearity/singularity
- **Landmark days** (optional): If landmark analysis, the landmark timepoint

## What It Does

### Automated Checks (6 pitfalls)
1. **Comment header corruption** (#1): Scans data file for hex color values that would be corrupted by `comment='#'` parsing
2. **Covariate leakage** (#2): Verifies all covariates use only pre-landmark data
3. **Perfect collinearity** (#3): Checks pairwise correlations (|r| > 0.99) and VIF > 10
4. **Constant variables** (#4): Identifies zero-variance columns
5. **Separation problems** (#7 related): Checks condition number and rank of feature matrix
6. **Singular matrix risk** (#7): Identifies columns that would cause singular design matrix

### Structured Audit Prompts (3 pitfalls)
7. **XGBoost softprob** (#5): If using XGBoost multi-class, check if any subgroup has < 3 classes
8. **Fabricated citations** (#8): Flag all AI-generated references for manual verification
9. **Floor/ceiling effects** (#9): Check if bounded variables (ECOG 0-4) have extreme distributions in subgroups

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
- [ ] All 8 automated checks ran without errors
- [ ] No CRITICAL warnings remain unresolved
- [ ] Audit prompts for manual checks displayed
- [ ] Fix suggestions provided for all detected issues

## Example Code

```python
from cascade.pitfalls import PitfallDetector

detector = PitfallDetector()

# Check data file for format issues
detector.check_data_format("data_timeline_performance_status.txt")

# Check model data for statistical issues
detector.check_collinearity(feature_matrix, threshold=0.99)
detector.check_constant_variables(feature_matrix)
detector.check_singular_matrix(feature_matrix)

# Check for covariate leakage
detector.check_covariate_leakage(
    df=landmark_df,
    landmark_col="LANDMARK_DATE",
    covariate_cols=["RECEIVED_ICI", "RECEIVED_CHEMO"],
    timeline_cols=["TX_START_DATE"],
)

# Print all warnings
for w in detector.warnings:
    print(f"[{w.severity.value}] {w.pitfall.name}: {w.message}")
    print(f"  Fix: {w.suggestion}")
```
