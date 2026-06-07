# CASCADE Validation Checklist

Analogous to REMARK for reporting and TRIPOD for prediction models, this checklist documents which CASCADE validation layers have been completed for a biomarker study.

## Instructions

For each item, mark [x] if completed, [ ] if not applicable, or add notes. Include this checklist as supplementary material with your manuscript.

---

## Layer 0: Cohort Assembly & Feature Engineering

- [ ] **0.1** Cohort defined with explicit inclusion/exclusion criteria
- [ ] **0.2** Data quality guards applied (comment headers, encoding issues)
- [ ] **0.3** Missing data patterns documented and handled
- [ ] **0.4** All time-dependent features locked to anchor dates (no future data)
- [ ] **0.5** Treatment lines detected with explicit gap/concurrent rules
- [ ] **0.6** Genomic features constructed (mutation matrix, CNA, fusions)

## Layer 1: Biomarker Discovery Screen

- [ ] **1.1** Hypothesis space systematically enumerated (not cherry-picked)
- [ ] **1.2** Analysis method matches question type (prognostic/predictive/tropism/trajectory)
- [ ] **1.3** Multiple testing correction applied (specify method: FDR-BH / Bonferroni / Holm)
- [ ] **1.4** Separation problems identified and filtered (CI ratio > 100)
- [ ] **1.5** Minimum subgroup sizes enforced (n >= ___, events >= ___)
- [ ] **1.6** Number of tests reported: ___ hypotheses tested, ___ FDR-significant

## Layer 2: Orthogonal Confirmation

- [ ] **2.1** Primary findings re-tested with independent statistical method
- [ ] **2.2** Primary method: _______________; Confirmatory method: _______________
- [ ] **2.3** Cross-analysis concordance computed (direction agreement: ___%)
- [ ] **2.4** Effect size correlation reported (Spearman rho: ___, P: ___)
- [ ] **2.5** Both-significant findings identified: ___ of ___ primary hits confirmed

## Layer 3: Predictive vs Prognostic Distinction

- [ ] **3.1** Gene x treatment interaction formally tested
- [ ] **3.2** Stratified effects reported (HR with treatment: ___; HR without: ___)
- [ ] **3.3** Classification stated: PREDICTIVE / PROGNOSTIC
- [ ] **3.4** If predictive: interaction HR = ___, P = ___

## Layer 4: Sensitivity & Robustness Testing

- [ ] **4.1** Sensitivity analyses span >= 3 of 7 categories
  - [ ] Feature definition (alternative cutoffs/thresholds)
  - [ ] Stratification (by driver mutation, demographics)
  - [ ] Time domain (different landmarks or time windows)
  - [ ] Statistical (penalizer variation, model alternatives)
  - [ ] Outcome (OS vs TTNTD vs PFS)
  - [ ] Subgroup (histology-specific, line-specific)
  - [ ] Threshold (minimum events, follow-up requirements)
- [ ] **4.2** Number of sensitivity analyses: ___
- [ ] **4.3** Robustness classification assigned to each finding
- [ ] **4.4** Robust: ___ / Exploratory: ___ / Unstable: ___
- [ ] **4.5** No direction flips in findings classified as "robust"

## Layer 5: Statistical Artifact Guards

- [ ] **5.1** Independence of combined features tested (if applicable)
  - Method: _______________ ; Result: _______________
- [ ] **5.2** Covariate leakage checked (all covariates use pre-anchor data only)
- [ ] **5.3** Immortal time bias prevented (landmark restriction applied)
- [ ] **5.4** Collinearity assessed (VIF < 10 for all predictors)
- [ ] **5.5** Baseline confounding evaluated (adjusted vs unadjusted HR comparison)
- [ ] **5.6** Floor/ceiling effects assessed for bounded variables

## Layer 6: Clinical Translation Assessment

- [ ] **6.1** Landmark analysis at clinically relevant timepoint(s): ___ months
- [ ] **6.2** Cross-validated C-statistic reported (not training-set): C = ___
- [ ] **6.3** Delta-C for added value of biomarker: +___ (95% CI: ___ to ___)
- [ ] **6.4** Likelihood ratio test for nested models: chi2 = ___, P = ___
- [ ] **6.5** Risk groups constructed with KM curves
- [ ] **6.6** Pairwise log-rank tests with multiple testing correction

## Layer 7: External Validation

- [ ] **7.1** Independent cohort used: _______________ (N = ___)
- [ ] **7.2** Feature alignment documented (missing features handled)
- [ ] **7.3** Balanced accuracy reported: ___% (not just accuracy)
- [ ] **7.4** Cohen's kappa: ___
- [ ] **7.5** Per-class metrics reported (F1, precision, recall)
- [ ] **7.6** Treatment-predictive value tested (not just prognostic)

## Layer 8: Manuscript & Communication

- [ ] **8.1** All citations manually verified (no AI fabrication)
- [ ] **8.2** Display items within journal limits: ___ figures + ___ tables = ___ total
- [ ] **8.3** Claims logically audited for tautology/circularity
- [ ] **8.4** Figures at publication quality (>= 300 DPI, sans-serif fonts)
- [ ] **8.5** Code and data availability statement included

---

## Validation Level Summary

| Level | Layers Required | Description |
|-------|----------------|-------------|
| Minimum | 1 + 4 | Discovery screen + sensitivity testing |
| Recommended | 1 + 2 + 4 + 5 | + orthogonal confirmation + artifact guards |
| Comprehensive | 1 + 2 + 4 + 5 + 6 | + clinical translation assessment |
| Gold Standard | All (0-8) | Full CASCADE validation |

**This study achieved: _______________ level validation**

---

## Pitfall Audit

Were any of the following pitfalls relevant to this study?

- [ ] P1: Comment header corruption (hex colors in data columns)
- [ ] P2: Covariate leakage (future data in landmark models)
- [ ] P3: Perfect collinearity (algebraic identities between features)
- [ ] P4: Constant variable at short landmarks
- [ ] P5: XGBoost softprob failure for binary classification
- [ ] P6: Platform-specific package failures
- [ ] P7: Singular matrix from constant covariate in subgroup
- [ ] P8: Fabricated citation by AI agent
- [ ] P9: Performance status floor effect (baseline confounding)
- [ ] Other: _______________

If yes, document how each was detected and resolved.
