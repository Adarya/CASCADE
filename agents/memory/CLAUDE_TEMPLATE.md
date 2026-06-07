# CLAUDE.md — CASCADE-Structured Biomarker Analysis

## Project Overview

This project follows the **CASCADE** (Clinical Assessment & Systematic Cascade for Agentic Discovery & Evaluation) framework for biomarker validation. CASCADE prescribes a nine-layer validation methodology (Layers 0–8) for clinical genomics studies.

## CASCADE Framework Reference

See `docs/framework_specification.md` for full layer specifications.

### Quick Reference: 9 Layers (0–8)
0. **Cohort Assembly**: Data loading, quality guards, feature engineering
1. **Discovery Screen**: Systematic enumeration + FDR correction
2. **Orthogonal Confirmation**: Independent method confirmation
3. **Predictive vs Prognostic**: Gene x treatment interaction testing
4. **Sensitivity & Robustness**: 7-category sensitivity + 3-tier scoring
5. **Artifact Guards**: Independence, leakage, collinearity detection
6. **Clinical Translation**: Landmark analysis, cross-validated delta-C
7. **External Validation**: Independent cohort testing
8. **Manuscript & Communication**: Citation format/plausibility checks, compliance

### Validation Levels
- **Minimum**: Layers 1 + 4
- **Recommended**: Layers 1 + 2 + 4 + 5
- **Comprehensive**: Layers 1 + 2 + 4 + 5 + 6
- **Gold Standard**: All layers

## Mandatory Pitfall Checks

Before any analysis, check for these common failure modes:

1. **Data loading**: If file has columns with hex color values (e.g., STYLE_COLOR), do NOT use `comment='#'` in pd.read_csv — it will corrupt those rows
2. **Landmark models**: All covariates must use only pre-landmark data. "Ever received treatment X" variables leak future information
3. **Collinearity**: If two features are algebraically related (e.g., rate = (count-1)/time at fixed time), they cannot both be in a model
4. **Constant variables**: At short landmarks, some trajectory variables may be constant (e.g., EARLY_DISSEMINATION = 1.0 at 3 months if defined as >1 site by 3 months)
5. **Subgroup analysis**: Covariates that are constant within a subgroup (e.g., ER_STATUS in ER+ subset) will cause singular matrix errors — drop them
6. **AI citations**: Never trust AI-generated references without manual verification. Always check DOIs and publication dates.

## Analysis Workflow

### Step 1: Data Preparation (Layer 0)
```python
from cascade.core import CohortBuilder
builder = CohortBuilder()
df = builder.load_tsv("data.txt", skip_comment_corruption=True)  # if hex colors present
```

### Step 2: Run Discovery (Layer 1)
```python
from cascade.core import BiomarkerScreen
screen = BiomarkerScreen(method="cox", correction="fdr_bh")
results = screen.screen(df, biomarker_cols, "OS_MONTHS", "OS_STATUS")
```

### Step 3: Confirm (Layer 2)
```python
from cascade.core import OrthogonalConfirm
confirm = OrthogonalConfirm(confirm_method="logistic")
confirmed = confirm.confirm(results, df, biomarker_cols, outcome_col)
```

### Step 4: Sensitivity (Layer 4)
```python
from cascade.core import SensitivitySuite
suite = SensitivitySuite()
# Add at least 3 of 7 categories...
robustness = suite.classify_robustness(results, sensitivity_results)
```

### Step 5: Artifact Guards (Layer 5)
```python
from cascade.pitfalls import PitfallDetector
detector = PitfallDetector()
detector.check_all(data_file="data.txt", feature_matrix=X)
```

### Step 6: Report
```python
from cascade import Pipeline
pipeline.report()  # Generates CASCADE compliance report
```

## File Structure Convention

```
project/
├── code/           # Analysis scripts (numbered: 00_, 01_, ...)
├── results/        # Output CSVs and statistics
├── figures/        # Publication-quality figures
├── manuscript/     # Drafts and outlines
└── CLAUDE.md       # This file
```

## Figure Standards (MANDATORY)

All figures MUST follow these publication-quality standards:

### Format & Resolution
- Save BOTH **PNG (300 DPI)** and **PDF** for every figure
- Use `plt.savefig(path, dpi=300, bbox_inches='tight')` for PNG
- Use `plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')` for PDF

### Typography
- Font: **Arial** (sans-serif) exclusively — set globally with `plt.rcParams['font.family'] = 'Arial'`
- **Large text relative to figure elements** — minimum font sizes:
  - Title: 14pt
  - Axis labels: 12pt
  - Tick labels: 10pt
  - Legend: 10pt
  - Annotations: 9pt
- When in doubt, make text LARGER — small text is the #1 rejection reason

### Layout
- **Each panel in its own file** — do NOT combine panels into multi-panel figures
  - Name as `fig1a_description.png`, `fig1b_description.png`, etc.
  - Panel assembly is done in Illustrator/PowerPoint by the authors, not in code
- **Minimal whitespace and dead space** — use `bbox_inches='tight'` and remove unnecessary padding
- Keep figures compact: prefer 4-6 inch width for single-column
- Remove chart junk: no gridlines unless they aid interpretation, no unnecessary borders

### Matplotlib Setup Block (copy into every figure script)
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
```

### Color
- Use colorblind-safe palettes (e.g., `tab10`, `Set2`, or custom)
- Ensure sufficient contrast on both screen and print

### Saving Pattern
```python
for ext in ['png', 'pdf']:
    plt.savefig(f'figures/{name}.{ext}', dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close()
```

## Key Reminders
- Always cross-validate C-statistics (training-set C inflates by ~0.005-0.01)
- Use landmark analysis for any longitudinal/trajectory features
- Report balanced accuracy, not just accuracy, for classification
- FDR for exploratory screens; Bonferroni for pre-specified hypotheses
- Three-tier robustness: ROBUST (>=75% concordance, no flips), EXPLORATORY (default), UNSTABLE (any flip)
