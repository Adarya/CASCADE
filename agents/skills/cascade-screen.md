# /cascade-screen — CASCADE Layer 1: Biomarker Discovery Screen

## Purpose
Run a systematic biomarker discovery screen with proper multiple testing correction, separation problem filtering, and minimum subgroup size enforcement.

## Required Inputs
- **Data file path**: CSV/TSV with patient-level data
- **Biomarker columns**: List of binary (0/1) columns to screen
- **Outcome column**: Survival time column (e.g., OS_MONTHS)
- **Event column**: Event indicator column (e.g., OS_STATUS)
- **Method**: One of: `cox` (prognostic), `logistic` (cross-sectional), `competing_risks` (tropism)
- **Covariates** (optional): Adjustment variables

## What It Does

1. **Loads data** and validates required columns exist
2. **Enumerates** all biomarker x outcome combinations
3. **Fits models** for each combination:
   - Cox PH for prognostic questions
   - Logistic regression for cross-sectional associations
   - Cause-specific Cox for tropism (site-specific metastasis)
4. **Filters** results:
   - Removes separation problems (CI ratio > 100)
   - Removes underpowered tests (< 20 exposed, < 5 events)
5. **Applies FDR correction** (Benjamini-Hochberg, alpha=0.05)
6. **Generates** results table sorted by adjusted p-value

## Expected Output
- CSV file: `cascade_screen_results.csv` with columns: biomarker, hr/or, ci_lower, ci_upper, p, p_adjusted, significant, n_exposed, n_events
- Summary: N tested, N significant after FDR, top hits

## Figure Guidelines

Any figures produced (e.g., volcano plots, heatmaps) MUST follow these standards:
- **Dual format**: Save both PNG (300 DPI) and PDF
- **Font**: Arial, minimum 12pt axis labels, 10pt ticks
- **Panels**: Each panel as its own separate file (`fig_screen_volcano.png`, `fig_screen_heatmap.png`)
- **Layout**: Minimal whitespace, `bbox_inches='tight'`, no chart junk, spines top/right off
- **Size**: Compact figures with large text relative to plot elements
- See `CLAUDE.md > Figure Standards` for the full matplotlib rcParams block

## Post-Execution Validation
- [ ] All biomarker columns are binary (0/1)
- [ ] No separation problems in final results (CI ratio < 100)
- [ ] FDR correction applied (check p_adjusted >= p for all rows)
- [ ] Minimum subgroup sizes met
- [ ] Results saved to file

## Example Code

```python
from cascade.core import BiomarkerScreen

screen = BiomarkerScreen(method="cox", correction="fdr_bh", alpha=0.05)
results = screen.screen(
    df=data,
    biomarker_cols=["GENE_A", "GENE_B", "GENE_C"],
    outcome_col="OS_MONTHS",
    event_col="OS_STATUS",
    covariates=["AGE", "SEX"],
)
results.to_csv("cascade_screen_results.csv", index=False)
print(f"Significant: {results['significant'].sum()} / {len(results)}")
```
