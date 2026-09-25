# /cascade-sensitivity — CASCADE Layer 4: Sensitivity & Robustness Testing

## Purpose
Stress-test biomarker findings across 7 categories of variation and classify each finding as robust, exploratory, or unstable.

## Required Inputs
- **Primary results**: Output from discovery/confirmation layers
- **Data file path**: Dataset with necessary columns
- **Analysis function**: The function that produced primary results (to re-run on variants)
- **Sensitivity variants**: At least 3 of 7 categories

## The 7 Categories

1. **Feature definition**: Alternative cutoffs (e.g., burden >=2 vs >=3 sites)
2. **Stratification**: By driver mutation, treatment type, demographics
3. **Time domain**: Different landmarks (3/6/9/12 months)
4. **Statistical**: Penalizer variation, model alternatives
5. **Outcome**: OS vs TTNTD vs PFS
6. **Subgroup**: Histology-specific (adeno-only), line-specific (1L only)
7. **Threshold**: Minimum events, minimum follow-up requirements

## What It Does

1. **Defines variants** for each sensitivity category
2. **Re-runs** the primary analysis on each variant dataset
3. **Compares** each variant result to primary:
   - Direction agreement (HR > 1 in both?)
   - Effect size similarity
   - Significance preservation
4. **Classifies robustness**:
   - **ROBUST**: zero direction flips, at least `min_evaluable` (default 2) evaluable variants, AND concordance >= 75%. Concordance counts failed variants in the denominator
   - **EXPLORATORY**: otherwise (primary effect missing, too few evaluable variants, too many failed variants)
   - **UNSTABLE**: ANY direction flip in HR or OR across evaluable variants

   Variants that fail to run (`suite.failed_variants`) or give no estimate for a finding count as failed and are never dropped. The Layer 4 gate fails if the table is empty or no variant was evaluable for any finding.

## Expected Output
- CSV: `cascade_sensitivity_results.csv` with variant x finding results
- Robustness classification per finding
- Summary: N robust / N exploratory / N unstable

## Figure Guidelines

Any figures produced (e.g., robustness bar charts, HR comparison forests) MUST follow these standards:
- **Dual format**: Save both PNG (300 DPI) and PDF
- **Font**: Arial, minimum 12pt axis labels, 10pt ticks
- **Panels**: Each panel as its own separate file (`fig_sens_robustness.png`, `fig_sens_forest.png`)
- **Layout**: Minimal whitespace, `bbox_inches='tight'`, no chart junk, spines top/right off
- **Size**: Compact figures with large text relative to plot elements
- See `CLAUDE.md > Figure Standards` for the full matplotlib rcParams block

## Post-Execution Validation
- [ ] At least 3 of 7 categories covered
- [ ] Direction flips explicitly checked
- [ ] Robustness classification assigned to every finding
- [ ] Unstable findings flagged prominently

## Example Code

```python
from cascade.core import BiomarkerScreen, SensitivitySuite

genes = ["GENE_A", "GENE_B", "GENE_C"]
screen = BiomarkerScreen(method="cox")

def run_cox_screen(d):
    return screen.screen(d, genes, "OS_MONTHS", "OS_STATUS", covariates=["AGE", "SEX"])

primary_results = run_cox_screen(df)

suite = SensitivitySuite()

# Add variants from different categories (DataFrames or filter callables)
suite.add_variant(">=6mo followup", df[df.followup >= 6], "threshold")
suite.add_variant(">=12mo followup", df[df.followup >= 12], "threshold")
suite.add_variant("Adeno only", df[df.histology == "Adeno"], "subgroup")
suite.add_variant("Squamous only", df[df.histology == "Squamous"], "subgroup")
suite.add_variant("1L only", lambda d: d[d.line == 1], "stratification")

# Callable variants are applied to primary_df; **kwargs go to every analysis_fn call,
# so a statistical variant (e.g. another penalizer) needs its own analysis function.
sensitivity_results = suite.run(run_cox_screen, primary_df=df)
robustness = suite.classify_robustness(
    primary_results, sensitivity_results,
    failed_variants=list(suite.failed_variants),
)
print(robustness[["biomarker", "robustness_class", "n_evaluable", "concordance"]])
print(robustness["robustness_class"].value_counts())
```
