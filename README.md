# CASCADE

**Clinical Assessment & Systematic Cascade for Agentic Discovery & Evaluation**

A multi-layer validation framework for biomarker discovery in clinical genomics, derived from human-AI collaborative research on the MSK-CHORD dataset (25,000 patients, 22 data files).

## Overview

CASCADE formalizes a nine-layer validation methodology (Layers 0–8) that emerged from three source biomarker studies and was then evaluated across seven studies in total. Unlike reporting guidelines (REMARK, TRIPOD) that prescribe *what to report*, CASCADE prescribes *what to do and in what order* to validate biomarker discoveries.

### The Nine Layers (0–8)

| Layer | Name | Purpose |
|-------|------|---------|
| 0 | Cohort Assembly | Data ingestion, quality guards, feature engineering |
| 1 | Discovery Screen | Systematic biomarker enumeration with FDR correction |
| 2 | Orthogonal Confirmation | Same hypothesis, independent statistical method |
| 3 | Predictive vs Prognostic | Formal gene x treatment interaction testing |
| 4 | Sensitivity & Robustness | 7-category sensitivity analysis + 3-tier scoring |
| 5 | Artifact Guards | Independence testing, leakage detection, collinearity |
| 6 | Clinical Translation | Landmark analysis, cross-validated delta-C, risk groups |
| 7 | External Validation | Independent cohort testing, calibrated consensus |
| 8 | Manuscript & Communication | Citation format/plausibility checks, checklist generation |

### The Pitfall Library

11 catalogued failure modes. Eight automated checks provide detection for
seven of them (#1–#4, #7, #10 and #11); the remainder are runtime, manual, or
partial:

1. Comment header corruption (`#hex` in STYLE_COLOR columns) — *automated*
2. Covariate leakage in landmark models (future treatment data) — *automated*
3. Perfect collinearity (algebraic identities) — *automated*
4. Constant variable at short landmarks — *automated*
5. XGBoost softprob failure for 2-class subgroups — *runtime*
6. Platform-specific R package compile failure (Fortran linker, macOS ARM) — *manual*
7. Singular matrix from constant covariates in subgroups — *automated*
8. Fabricated citations by AI agents — *partial* (format/plausibility screen only)
9. Performance status floor effects (baseline confounding) — *partial*
10. Informative (biomarker-dependent) censoring — *automated* (v0.4.0)
11. Centre or batch effects in multi-institutional cohorts — *automated* (v0.4.0)

## Installation

```bash
pip install git+https://github.com/Adarya/CASCADE.git
```

For ML features (XGBoost):
```bash
pip install "cascade-validate[ml] @ git+https://github.com/Adarya/CASCADE.git"
```

For development:
```bash
git clone https://github.com/Adarya/CASCADE.git
cd CASCADE
pip install -e ".[dev]"
```

## Quick Start

```python
from cascade import Pipeline
from cascade.core import (
    BiomarkerScreen, OrthogonalConfirm, SensitivitySuite, ArtifactGuard,
)

# df: one row per patient, with binary biomarker columns (gene_columns)
# and survival outcome columns OS_MONTHS / OS_STATUS.

pipeline = Pipeline()
pipeline.add_layer("discovery", BiomarkerScreen(method="cox", correction="fdr_bh"))
pipeline.add_layer("confirmation", OrthogonalConfirm(confirm_method="logistic"))

# A sensitivity analysis is only meaningful once you declare what to vary.
# The pipeline re-runs the discovery screen under each variant automatically.
sensitivity = SensitivitySuite()
sensitivity.add_variant("complete_cases", lambda d: d.dropna(), category="threshold")
pipeline.add_layer("sensitivity", sensitivity)

pipeline.add_layer("artifact_guard", ArtifactGuard())

# Inter-layer dependencies (discovery -> confirmation -> sensitivity) are wired
# automatically -- no need to declare depends_on. Pass strict=True to raise on
# any layer failure; otherwise failures are reported loudly, never swallowed.
results = pipeline.run(
    data=df,
    biomarker_cols=gene_columns,
    outcome_col="OS_MONTHS",
    event_col="OS_STATUS",
)

assert not results.failed_layers, results.failed_layers  # fail loud
print(pipeline.report())
```

### Individual Layers

Each layer is independently usable:

```python
# Sensitivity analysis with robustness scoring
from cascade.core import SensitivitySuite

suite = SensitivitySuite()
suite.add_variant(">=6mo followup", df[df.followup >= 6], category="threshold")
suite.add_variant("Adeno only", df[df.histology == "Adeno"], category="subgroup")
robustness = suite.classify_robustness(primary_results, sensitivity_results)
```

```python
# Pitfall detection
from cascade.pitfalls import PitfallDetector

detector = PitfallDetector()
detector.check_all(data_file="data.txt", model_df=landmark_df, feature_matrix=X)
for warning in detector.warnings:
    print(f"[{warning.severity.value}] {warning.message}")
```

```python
# Predictive vs prognostic testing
from cascade.core import PredictiveTest

test = PredictiveTest(gene_col="CBFB_MUT", treatment_col="IS_ET")
result = test.run(df)
print(f"Classification: {result.classification}")
print(f"Interaction HR: {result.interaction_hr:.2f}, P={result.interaction_p:.3f}")
```

## Extending CASCADE with custom layers

The nine built-in layers are not privileged. CASCADE is designed to be
explored and extended: you can run any subset of layers, in any order, reuse
each layer standalone, **and add your own layers**. A custom layer is any
object with a `run()` method — subclassing `BaseLayer` is optional.

When the pipeline executes a layer it inspects the `run` signature and injects
what it can:

| Parameter | Injected value |
|-----------|----------------|
| `df` / `data` | the input DataFrame |
| `context` / `upstream` | dict of **every** completed upstream layer's result, by name |
| `primary_results` | the most recent discovery-layer result |
| `sensitivity_results` | the most recent sensitivity-layer result |
| any other named parameter | filled from `run(**kwargs)` by name |

Inputs the pipeline can't supply (and that have no default) cause the layer to
be *skipped*, never silently failed — so a custom layer declares only the
inputs it actually needs.

```python
from cascade import Pipeline, BaseLayer
from cascade.core import BiomarkerScreen
import numpy as np

class EffectSizeFloor(BaseLayer):
    """A user-defined layer: flag discovery hits below an effect-size floor."""
    def __init__(self, min_log_hr=0.3):
        super().__init__()
        self.min_log_hr = min_log_hr

    def run(self, df, context=None):
        disc = (context or {}).get("discovery")          # any upstream layer, by name
        if disc is None:
            return {"flagged": []}
        weak = disc[np.abs(np.log(disc["hr"])) < self.min_log_hr]
        return {"n_weak": int(len(weak)), "weak": list(weak["biomarker"])}

pipeline = Pipeline()
pipeline.add_layer("discovery", BiomarkerScreen(method="cox"))
pipeline.add_layer("effect_floor", EffectSizeFloor(min_log_hr=0.3))   # novel layer
result = pipeline.run(
    data=df, biomarker_cols=gene_columns,
    outcome_col="OS_MONTHS", event_col="OS_STATUS",
)
print(result.get("effect_floor").results)
```

Custom layers appear in the compliance report and participate in fail-loud
semantics exactly like the built-in ones. This makes CASCADE a substrate for
new validation steps, not a fixed nine-step recipe.

## Case Studies

CASCADE was derived from three source studies on MSK-CHORD, then applied
without modification to three further cancer studies and one external cohort —
seven studies in total (22,694 patient-analyses across five cancer types).

**Source studies (framework derivation):**

1. **NSCLC metastatic tropism** (n = 3,368): 16 FDR-significant gene–site associations; 10 orthogonally confirmed (82% direction concordance); 13/19 classified robust.
2. **NSCLC burden/ECOG independence** (n = 1,486): metastatic burden and performance-status trajectories are statistically independent (φ = −0.033, P = 0.20); resolved the ECOG "paradox" via baseline adjustment.
3. **CBFB predictive biomarker** (breast cancer): CBFB mutation is predictive — not merely prognostic — for endocrine-therapy benefit (interaction HR = 0.37, P = 0.009).

**Generalization studies (framework applied unchanged):**

4. **CRC metastatic tropism** (n = 2,332): 18 of 104 gene–site pairs FDR-significant; 13 orthogonally confirmed (81.6% concordance, ρ = 0.748), 6 both-significant; 17/18 robust.
5. **Prostate predictive** (n = 1,546): 4 FDR-significant prognostic genes (e.g. TP53 HR = 2.02); pre-specified predictive hypotheses correctly classified as null.
6. **Pan-cancer metastatic burden** (n = 7,719 at landmark): cross-validated ΔC = +0.023; burden–ECOG independence holds pan-cancer (φ = −0.005).

**External validation:**

7. **GENIE BPC NSCLC** (n = 797, four institutions): cross-cohort tropism concordance ρ = 0.394 (65.5% direction concordance) over 55 overlapping gene–site pairs.

Agent independence: re-running the CRC study with a second, independent coding
agent reproduced effect sizes at Spearman ρ = 0.91 across the 104 overlapping
gene–site pairs.

See `examples/` for simplified, runnable versions using synthetic data.

## Claude Code Integration

CASCADE includes custom skills for AI-assisted validation:

| Skill | Layer | Description |
|-------|-------|-------------|
| `/cascade-screen` | 1 | Systematic biomarker discovery |
| `/cascade-confirm` | 2 | Orthogonal confirmation |
| `/cascade-predictive` | 3 | Predictive vs prognostic test |
| `/cascade-sensitivity` | 4 | 7-category sensitivity suite |
| `/cascade-guard` | 5 | Automated artifact checks |
| `/cascade-translate` | 6 | Landmark analysis + delta-C |
| `/cascade-report` | All | Full compliance report |

## Requirements

- Python >= 3.10
- numpy, pandas, scipy, lifelines, scikit-learn, statsmodels, matplotlib
- Optional: xgboost (for ML classification layers)

## Citation

If you use CASCADE in your research, please cite:

> CASCADE: A multi-layer validation framework for biomarker discovery in clinical genomics. (Manuscript in preparation, 2026).

## License

MIT License. See [LICENSE](LICENSE) for details.
