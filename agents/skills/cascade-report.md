# /cascade-report — CASCADE Full Compliance Report

## Purpose
Generate a comprehensive CASCADE compliance report documenting which validation layers were completed, key findings, warnings, and the validation checklist.

## Required Inputs
- **Results from any completed CASCADE layers** (CSVs, DataFrames, or Pipeline result object)
- **Study metadata**: Study name, dataset description, date

## What It Does

1. **Collects** results from all completed CASCADE layers
2. **Generates layer completion summary** (pass/fail/skipped for each of 9 layers, 0–8)
3. **Compiles warnings** from all layers (pitfall detections, convergence issues)
4. **Fills CASCADE checklist** (26 items across 9 layers, 0–8)
5. **Computes validation level**: Minimum / Recommended / Comprehensive / Gold Standard
6. **Generates pitfall audit** section
7. **Outputs** formatted markdown report

## Expected Output
- Markdown report: `cascade_compliance_report.md`
- Sections: Summary, Layer Details, Warnings, Checklist, Pitfall Audit
- Validation level classification

## How to Use

### Option 1: From Pipeline

```python
from cascade import Pipeline
# ... add layers and run ...
report = pipeline.report()
with open("cascade_compliance_report.md", "w") as f:
    f.write(report)
```

### Option 2: Standalone

```python
from cascade.report import generate_standalone_checklist

checklist = generate_standalone_checklist({
    0: True,   # Cohort assembly completed
    1: True,   # Discovery screen completed
    2: True,   # Orthogonal confirmation completed
    3: False,  # Predictive test not done
    4: True,   # Sensitivity completed
    5: True,   # Artifact guards completed
    6: True,   # Clinical translation completed
    7: False,  # External validation not done
    8: True,   # Manuscript checks completed
})
print(checklist)
```

### Option 3: Manual (Copy Checklist)

Copy the checklist from `docs/cascade_checklist.md` into your supplementary materials and fill in manually.

## Figure Guidelines

Any figures referenced or generated in the report MUST follow these standards:
- **Dual format**: Save both PNG (300 DPI) and PDF
- **Font**: Arial, minimum 12pt axis labels, 10pt ticks
- **Panels**: Each panel as its own separate file — never combine panels in code
- **Layout**: Minimal whitespace, `bbox_inches='tight'`, no chart junk, spines top/right off
- **Size**: Compact figures with large text relative to plot elements
- See `CLAUDE.md > Figure Standards` for the full matplotlib rcParams block

## Post-Execution Validation
- [ ] All completed layers reflected in report
- [ ] Warnings prominently displayed
- [ ] Validation level accurately computed
- [ ] Checklist items match actual work done
- [ ] Report suitable for supplementary material submission
