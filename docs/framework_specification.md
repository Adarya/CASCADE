# CASCADE Framework Specification

**Clinical Assessment & Systematic Cascade for Agentic Discovery & Evaluation**

Specification v1.0 · package v0.4.0 · June 2026

---

## 1. Introduction

CASCADE is a multi-layer validation framework for biomarker discovery in clinical genomics. It was derived empirically from three source studies on the MSK-CHORD dataset (~25,000 patients, 22 data files, ~340 MB) — NSCLC metastatic tropism, NSCLC burden/ECOG independence, and CBFB predictive biomarker identification — and was subsequently evaluated across seven studies in total (adding CRC tropism, prostate predictive, pan-cancer burden, and GENIE BPC external validation).

CASCADE differs fundamentally from existing biomarker reporting guidelines. REMARK (Reporting Recommendations for Tumor Marker Prognostic Studies) and TRIPOD (Transparent Reporting of a Multivariable Prediction Model) prescribe **what to report** in a manuscript. CASCADE prescribes **what to do and in what order** to validate biomarker discoveries before reporting them. It is a procedural framework, not a reporting checklist.

The framework is agent-agnostic. It can be executed by an AI coding assistant (such as Claude Code with `/cascade-*` skills), by a human analyst following the specification as a protocol, or by a hybrid team. The layers, decision gates, and pitfall guards are defined independently of the execution engine.

CASCADE addresses a gap in the biomarker discovery pipeline: the space between initial statistical significance and publishable, reproducible findings. Many biomarker studies report discoveries that fail replication because they lack orthogonal confirmation, sensitivity analysis, artifact detection, or formal predictive-versus-prognostic distinction. CASCADE systematizes these validation steps into a directed acyclic graph (DAG) of nine layers (numbered 0 through 8) with explicit pass/fail decision gates.

---

## 2. Framework Overview

CASCADE is organized as a DAG of nine layers (numbered 0 through 8), where each layer produces structured outputs that feed into subsequent layers. Decision gates between layers enforce minimum quality standards before proceeding.

```
Layer 0: Cohort Assembly & Feature Engineering
    |
    v [GATE: No future data leakage]
Layer 1: Biomarker Discovery Screen
    |
    v [GATE: >= 1 FDR-significant finding]
Layer 2: Orthogonal Confirmation
    |
    v [GATE: Direction concordance >= 70%]
Layer 3: Predictive vs Prognostic Distinction
    |
    v [GATE: Classification assigned]
Layer 4: Sensitivity & Robustness Testing
    |
    v [GATE: Primary findings classified]
Layer 5: Statistical Artifact Guards
    |
    v [GATE: No critical artifacts]
Layer 6: Clinical Translation Assessment
    |
    v [GATE: delta-C > 0, CI excludes 0]
Layer 7: External Validation
    |
    v [GATE: Performance above random]
Layer 8: Manuscript & Communication
```

**Independence principle.** Each layer is independently usable. A study can apply Layer 1 alone (discovery with FDR correction), Layers 1 + 4 (discovery plus sensitivity), or any other subset. However, maximum validation strength comes from the full cascade.

**Composability.** Layers can be chained via the `Pipeline` orchestrator, which passes results between layers, tracks completion status, evaluates decision gates (on by default; `Pipeline(gate_evaluator=False)` opts out), and generates compliance reports. Each layer exposes a consistent interface: it accepts data and prior-layer results, and returns structured outputs with warnings.

**Each layer has four components:**

| Component | Description |
|-----------|-------------|
| PURPOSE | What question the layer answers |
| INPUTS | Data and prior-layer results required |
| OUTPUTS | Structured results produced |
| DECISION GATE | Pass/fail criteria for proceeding to the next layer |

---

## 3. Layer Specifications

### Layer 0: Cohort Assembly & Feature Engineering

**Purpose.** Build an analysis-ready dataset from raw clinical genomics files with quality guards that prevent data corruption and temporal leakage.

**Inputs.** Raw tab-delimited data files from a clinical genomics platform (e.g., cBioPortal exports): patient clinical data, sample data, mutations (MAF format), timeline files (treatment, tumor sites, performance status, progression, diagnosis).

**Key Operations.**

1. **Data loading with comment header handling.** Clinical data files contain metadata rows starting with `#`. Naive use of `comment='#'` in pandas corrupts files that contain hex color codes (e.g., `STYLE_COLOR = '#359645'`), because every row is truncated at the first `#`. Safe loading skips only the leading block of `#` metadata lines and never passes `comment=` to pandas, so `#` inside data fields is preserved. This is the default behaviour of `CohortBuilder.load_tsv`:

```python
# Safe loading pattern (CohortBuilder.load_tsv default)
with open(filepath, 'r') as fh:
    n_header = 0
    for line in fh:
        if not line.startswith('#'):
            break
        n_header += 1
df = pd.read_csv(filepath, sep='\t', skiprows=n_header)
```

2. **Stage filtering.** Restrict to the clinically relevant population (e.g., Stage IV for metastatic analyses) using the `STAGE_CDM_DERIVED` field in the diagnosis timeline.

3. **Mutation matrix construction.** Build a patient-by-gene binary matrix from MAF-format mutation data. Filter to OncoKB "Oncogenic" or "Likely Oncogenic" variants. Deduplicate to one entry per patient-gene pair. Output is a binary (0/1) matrix indexed by `PATIENT_ID`.

4. **Treatment line detection.** Identify treatment lines from the longitudinal treatment timeline using a gap-based algorithm:
   - Sort events by `PATIENT_ID` and `START_DATE`; the first record opens line 1
   - Agents starting within a 28-day concurrent window of the current line start are grouped into the same regimen
   - Otherwise, a gap of >90 days between the latest prior activity (the maximum earlier `STOP_DATE`, or `START_DATE` when the stop is missing) and the next `START_DATE` starts a new line
   - Within the gap limit, a record for an agent already in the current line continues that line; a new agent added outside the concurrent window starts a new line (switch or escalation)
   - Assign `LINE_NUMBER`, `LINE_START`, and `LINE_AGENTS` (comma-separated) to each record

5. **Time-dependent feature engineering.** Clinical phenotypes must be determined at the time of treatment start, not at sample collection or study end. Two core patterns:

   *Site flags at treatment start:*
   ```
   For each treatment line start date (anchor):
       Filter tumor_sites timeline to events <= anchor date
       Create binary indicators for each anatomical site
       Join back to treatment table
   ```

   *ECOG at treatment start:*
   ```
   For each treatment START_DATE (anchor):
       Filter performance_status timeline to [anchor - 30 days, anchor]
       Select most recent ECOG score within window
       Handle missing data (NaN if no score within window)
   ```

6. **Endpoint computation.** Calculate Time to Next Treatment or Death (TTNTD) from sequential treatment line dates and overall survival data. Censor at last follow-up if neither next treatment nor death is observed.

**Outputs.** Analysis-ready DataFrame with: patient identifiers, binary mutation matrix, treatment line assignments with agents, time-locked clinical phenotypes (site flags, ECOG, burden), and computed endpoints (TTNTD, OS).

**Decision Gate.** All features are time-locked to anchor dates with no future data leakage. Verification: for each time-dependent feature, confirm that all contributing events have dates on or before the anchor date. **Pitfalls guarded: #1 (comment header corruption), #2 (covariate leakage).**

---

### Layer 1: Biomarker Discovery Screen

**Purpose.** Systematically enumerate the hypothesis space with appropriate statistical models and multiple testing correction.

**Inputs.** Analysis-ready dataset from Layer 0. Specification of biomarker columns, outcome columns, and model type.

**Key Operations.**

1. **Model selection by question type:**

| Question | Model | Primary Coefficient |
|----------|-------|-------------------|
| Prognostic | Cox main effect | Gene HR |
| Predictive | Cox interaction | Gene x Treatment HR |
| Tropism | Cause-specific Cox (competing risks) | Gene cause-specific HR per site |
| Trajectory | Landmark or time-varying Cox | Longitudinal feature HR |
| Classification | ML with cross-validation | Balanced accuracy |

2. **Multiple testing correction.** Apply Benjamini-Hochberg FDR for exploratory screens (hundreds of tests). Apply Bonferroni for pre-specified hypotheses (fewer than 10 tests). NaN p-values are preserved and excluded from the correction denominator.

3. **Separation problem filtering.** When a gene-site combination has very few events in one arm, the Cox model may produce extreme hazard ratios with enormous confidence intervals. Filter results where `CI_upper / CI_lower > 100` (CI ratio check). These estimates are numerically unstable and should not be reported as findings.

4. **Minimum subgroup sizes.** Require `n >= 20` exposed patients and `>= 5` events in each analysis cell. For interaction models, require adequate patients in all 2^k subgroup cells (k = number of binary factors).

5. **Enumeration pattern.** For tropism screening, iterate over all (gene, site) combinations. For interaction screening, iterate over all (gene, treatment, phenotype) triplets. Track convergence status and exclude non-converged models.

**Outputs.** DataFrame with one row per tested hypothesis: gene, site/treatment/phenotype, n, n_events, n_exposed, coef, HR, SE, z, p, CI_lower, CI_upper, FDR-adjusted p, significance flag, convergence flag, and `fit_error` (the reason a model failed to fit, instead of a silent NaN). Event columns may be 0/1, booleans or cBioPortal status strings (`'1:DECEASED'`); non-numeric covariates are dummy-encoded.

**Decision Gate.** At least one FDR-significant finding (adjusted p < 0.05) to proceed to Layer 2. If zero findings survive FDR, the analysis stops or is reframed (e.g., broader phenotype definitions, different endpoint).

---

### Layer 2: Orthogonal Confirmation

**Purpose.** Confirm discoveries from Layer 1 using a statistically independent method. The same hypothesis is tested with a different model class to guard against model-specific artifacts.

**Inputs.** Layer 1 significant findings. Analysis-ready dataset from Layer 0.

**Key Operations.**

1. **Method pair selection:**

| Discovery Method (Layer 1) | Confirmation Method (Layer 2) |
|---------------------------|------------------------------|
| Cause-specific Cox (longitudinal) | Logistic regression (cross-sectional) |
| Cox proportional hazards | KM + log-rank test |
| XGBoost classifier | Logistic regression or nearest centroid |
| Landmark Cox | Standard Cox with binary features |

2. **Cross-analysis concordance.** Compute Spearman rank correlation (rho) between Layer 1 effect sizes and Layer 2 effect sizes across all tested hypotheses. Report the correlation coefficient and its p-value. Compute direction agreement percentage (fraction of findings where both analyses show the same direction of effect).

3. **"Both-significant" identification.** The subset of findings that reach significance (at the chosen alpha) in both the discovery analysis and the confirmation analysis constitutes the highest-confidence tier. These findings are the strongest candidates for follow-up.

**Outputs.** Confirmation results DataFrame, cross-analysis concordance statistics (rho, p, direction agreement %), list of both-significant findings, confidence tier assignments (both-significant, discovery-only, confirmation-only).

**Decision Gate.** Direction concordance >= 70% across all tested hypotheses AND at least one both-significant finding. The gate fails if concordance cannot be computed. If concordance is below 70%, the Layer 1 findings may be model-dependent and require investigation.

---

### Layer 3: Predictive vs Prognostic Distinction

**Purpose.** Formally determine whether a biomarker predicts treatment response (predictive) or predicts outcome regardless of treatment (prognostic). This distinction has direct clinical implications: predictive biomarkers guide treatment selection, while prognostic biomarkers inform prognosis but do not change treatment decisions.

**Inputs.** Significant biomarkers from Layers 1-2. Treatment exposure data. Survival outcomes.

**Key Operations.**

1. **Formal interaction test.** Fit a Cox model with the gene, treatment, and their interaction term:

```
h(t) = h0(t) * exp(beta_gene * Gene + beta_tx * Treatment
                    + beta_int * Gene:Treatment + covariates)
```

The interaction coefficient `beta_int` tests whether the gene effect differs between treated and untreated groups.

2. **Stratified effect estimation.** Estimate the gene hazard ratio separately within the treated stratum and the untreated stratum. Derive `HR_treated = exp(beta_gene + beta_int)` and `HR_untreated = exp(beta_gene)` with approximate confidence intervals via the variance-covariance matrix.

3. **Classification assignment:**
   - **PREDICTIVE**: `beta_int` p-value < 0.05 (the gene effect significantly differs by treatment)
   - **PROGNOSTIC only**: `beta_int` estimated with p-value >= 0.05 (the gene effect does not depend on treatment)
   - **NOT_EVALUABLE**: the interaction could not be tested, because a treatment arm has fewer than `min_arm_size` patients (default 10), the interaction term has no variation, or the model failed to fit. The reason is recorded. A test that could not be run is never reported as prognostic.

4. **Three-way extension.** For studies incorporating clinical phenotypes, extend to a three-way interaction model:

```
h(t) = h0(t) * exp(b1*Gene + b2*Treatment + b3*Phenotype
                    + b4*Gene:Treatment + b5*Gene:Phenotype
                    + b6*Treatment:Phenotype
                    + b7*Gene:Treatment:Phenotype + covariates)
```

The coefficient `b7` tests whether the gene-treatment relationship is modified by the clinical phenotype.

**Outputs.** For each biomarker: classification (PREDICTIVE, PROGNOSTIC or NOT_EVALUABLE), interaction HR with CI and p-value, stratified HRs (treated vs untreated), and model fit statistics (concordance, AIC).

**Decision Gate.** Every significant biomarker from Layer 2 must receive an explicit classification. The gate fails if the classification table is empty or if no biomarker was evaluable (all NOT_EVALUABLE). Report the classification alongside all downstream results. Predictive findings have higher priority for clinical translation (Layer 6).

---

### Layer 4: Sensitivity & Robustness Testing

**Purpose.** Stress-test discoveries across systematic variations in data, methods, and definitions. Assign a robustness classification to each finding.

**Inputs.** Primary findings from Layers 1-3. Analysis-ready dataset.

**Key Operations.**

1. **Seven categories of sensitivity analysis:**

| Category | Examples |
|----------|---------|
| (1) Feature definition | Binary threshold variations (e.g., oligo: 1-2 vs 1-3 sites) |
| (2) Stratification | By histology subtype, by treatment line |
| (3) Time domain | Different landmark times (3, 6, 12 months), follow-up cutoffs (>=6mo, >=12mo) |
| (4) Statistical | Alternative model (e.g., Fine-Gray vs cause-specific Cox), different penalization |
| (5) Outcome | Alternative endpoint (OS vs TTNTD vs PFS) |
| (6) Subgroup | Histology-restricted (e.g., adenocarcinoma only), sex-stratified |
| (7) Threshold | Minimum sample size variations, minimum event count variations |

2. **Concordance computation.** For each sensitivity variant, re-run the primary analysis and record whether each finding: (a) remains significant, (b) retains the same direction of effect, (c) retains a similar magnitude (HR within 2-fold of primary).

3. **Three-tier robustness classification:**

| Tier | Criteria | Interpretation |
|------|----------|---------------|
| **ROBUST** | Zero direction flips, at least `min_evaluable` (default 2) evaluable variants, AND concordance >= 75% | High confidence; report as primary finding |
| **EXPLORATORY** | Otherwise: primary effect missing, too few evaluable variants, or too many failed variants | Report with appropriate caveats |
| **UNSTABLE** | ANY direction flip across evaluable sensitivity variants | Flag prominently; consider removing from primary results |

A variant is *evaluable* for a finding if it produced an effect estimate for that finding. Concordance is the fraction of **all** variants, failed ones included, that reproduce the primary direction, so a variant that failed to run or produced no estimate lowers concordance and is never silently dropped.

**Outputs.** Sensitivity results matrix (findings x variants), concordance percentages, direction-flip flags, robustness tier for each finding.

**Decision Gate.** All primary findings receive a robustness classification. The gate fails if the table is empty or if no sensitivity variant was evaluable for any finding. Unstable findings are flagged or removed from the primary results. The manuscript must report the robustness classification for each finding. **Pitfalls guarded: #4 (constant variable at short landmark).**

---

### Layer 5: Statistical Artifact Guards

**Purpose.** Detect and prevent statistical artifacts that could invalidate conclusions. This layer runs a battery of diagnostic tests targeting specific failure modes observed in practice.

**Inputs.** Analysis-ready dataset. Model results from prior layers. Feature definitions.

**Key Operations.**

1. **Independence testing.** When combining two prognostic features (e.g., metastatic burden change and ECOG change), verify that they are not measuring the same underlying construct. The test battery includes:
   - Chi-squared test on 2x2 contingency table
   - Phi coefficient with Fisher z-transform CI
   - Spearman rank correlation on continuous versions
   - TOST (Two One-Sided Tests) equivalence procedure for formal independence demonstration
   - Bayes factor for independence (BIC approximation based on the likelihood-ratio G² statistic; BF10 < 1 supports independence)
   - Discordance rate (percentage of observations where the two features disagree)

2. **Covariate leakage detection.** For each binary covariate in landmark models, verify that the events contributing to the variable all have dates on or before the landmark date. The date-based check takes a `{covariate: exposure_date_col}` mapping. Flag any covariate computed from "ever received" logic that uses full follow-up data. Fix by recomputing from the treatment timeline filtered to `START_DATE <= landmark_date`.

3. **Immortal time bias prevention.** Verify that trajectory variables (e.g., acquisition rate, pattern of dissemination) are not computed over the entire follow-up and then used as baseline predictors. Time-varying covariates must use landmark analysis or time-varying Cox formulations.

4. **Collinearity detection.**
   - Compute the pairwise Pearson correlation matrix; flag pairs with |r| >= 0.99
   - Compute VIF (Variance Inflation Factor) for each feature; flag VIF > 10
   - Check for algebraic identities (e.g., `ACQUISITION_RATE = (N_SITES - 1) / LANDMARK_TIME` is perfectly collinear with `N_SITES` when `LANDMARK_TIME` is constant)

5. **Separation problem filtering.** Re-apply CI ratio check (`CI_upper / CI_lower > 100`) to all final results. Remove estimates with quasi-complete separation.

6. **Baseline confounding resolution.** Compare unadjusted and baseline-adjusted hazard ratios. If the direction changes (e.g., unadjusted HR < 1 but adjusted HR > 1), a floor effect or baseline confound is present. Report both estimates and explain the discrepancy.

7. **Singular matrix detection.** Before fitting subgroup models, check that no covariate is constant within the subgroup (e.g., ER_BINARY in ER-positive-only analysis). Drop constant covariates automatically and log the removal.

8. **Pitfall library.** `ArtifactGuard.run_all(..., pitfall_inputs={...})` runs `PitfallDetector.check_all` inside Layer 5 with the supplied inputs. This covers comment-header corruption, leakage, collinearity, constant variables, singular matrix and separation, informative censoring (#10) and centre effects (#11). Its warnings are added to the artifact report and the compliance report, and CRITICAL ones are blocking.

**Outputs.** Artifact detection report: list of detected artifacts, severity levels, affected findings, and applied fixes. Independence test suite results with all metrics. Collinearity matrix.

**Decision Gate.** No critical artifacts detected, or all detected artifacts are resolved with documented fixes. The gate fails closed: it fails if no artifact check actually ran (an inconclusive independence test does not count as run), and any CRITICAL finding blocks it. CRITICAL findings include an independence violation, covariate leakage, collinearity or VIF above threshold, and CRITICAL pitfall-library warnings. If a critical artifact cannot be resolved (e.g., the two combined features are highly correlated), the analysis must be reframed. **Pitfalls guarded: #2 (covariate leakage), #3 (perfect collinearity), #7 (singular matrix from constant covariate), #9 (performance status floor effect), #10 (informative censoring), #11 (centre effect).**

---

### Layer 6: Clinical Translation Assessment

**Purpose.** Quantify the clinical utility of validated biomarkers by measuring their incremental predictive value and constructing actionable risk groups.

**Inputs.** Validated biomarkers from Layers 1-5. Analysis-ready dataset with survival outcomes.

**Key Operations.**

1. **Landmark analysis.** Evaluate biomarker performance at clinically actionable timepoints (3, 6, and 12 months post-diagnosis or post-treatment start). At each landmark, restrict the cohort to patients alive and event-free at the landmark time, then use features evaluated at that landmark.

2. **Cross-validated delta-C statistic.** The primary metric is the improvement in concordance index (Harrell's C) when adding the biomarker to a base model containing standard clinical covariates:

```
delta-C = C_full - C_base
```

Critically, this must be computed via **cross-validation** (stratified k-fold, typically 10-fold), not on the training set. Training-set C inflates by approximately 0.005-0.01 relative to honest, held-out estimates.

```text
# Pattern (pseudocode): stratified k-fold CV for delta-C
for train_idx, test_idx in StratifiedKFold(n_splits=10).split(data, events):
    fit base model on train, score on test -> C_base
    fit full model on train, score on test -> C_full
    fold_delta = C_full - C_base
delta_C = mean(fold_deltas)
```

3. **Bootstrap confidence interval.** Resample the dataset 1,000 times with replacement. For each resample, compute delta-C and report the 95% bootstrap CI. The bootstrap is descriptive only: `bootstrap_delta_c` returns a NaN p-value, and inference uses the cross-validated `cv_delta_c` (stratified folds), which is what `DeltaC.compute` uses.

4. **Likelihood ratio test.** Compare nested models (base vs full) using the log-likelihood ratio statistic. This provides a parametric test of whether the biomarker significantly improves model fit.

5. **Risk group construction.** Divide the cohort into risk groups (e.g., tertiles of predicted risk score) and generate Kaplan-Meier curves. Compute pairwise log-rank p-values between groups. Report median survival per group.

**Outputs.** Delta-C at each landmark with CV and bootstrap CIs. LRT p-value. Risk group KM curves with median survival and log-rank statistics.

**Decision Gate.** Cross-validated delta-C > 0 with 95% CI excluding zero at the primary landmark timepoint. The gate fails closed if delta-C or either CI bound is missing or non-finite. If delta-C <= 0 or CI includes zero, the biomarker does not add clinically meaningful predictive value beyond standard covariates.

---

### Layer 7: External Validation

**Purpose.** Confirm generalizability of the biomarker or classifier on an independent cohort that was not used during discovery or model development.

**Inputs.** Trained model or classifier from prior layers. Independent validation cohort (e.g., TCGA for models trained on MSK-CHORD).

**Key Operations.**

1. **Feature alignment.** Map features between the training and validation datasets. Document which features are available in both datasets and which are missing. For missing features, document the imputation strategy or justify exclusion.

2. **Performance metrics.** Evaluate using metrics appropriate to the task:
   - **Classification**: Balanced accuracy (not raw accuracy, which is misleading for imbalanced classes), Cohen's kappa, per-class F1 score, confusion matrix
   - **Survival prediction**: Concordance index, calibration (predicted vs observed), Brier score
   - **Biomarker-treatment interaction**: Replicate the interaction test in the validation cohort

3. **Calibrated consensus across approaches.** When multiple classification methods are available, compute a calibrated consensus by weighting each approach proportionally to its validation accuracy:

```python
# Weighted consensus (e.g., two classifiers with different accuracy)
weight_A = 0.35  # based on validation balanced accuracy
weight_B = 0.65
consensus_prob = weight_A * prob_A + weight_B * prob_B
```

Assign confidence tiers (High, Moderate, Low) based on the maximum consensus probability.

4. **Treatment-predictive value.** Within the validation cohort, test whether the biomarker predicts treatment response. Group patients by biomarker status and treatment, then compare outcomes (TTNT, PFS, or OS) using Cox models or KM curves with log-rank tests.

**Outputs.** Validation performance metrics (balanced accuracy, kappa, per-class F1). Confusion matrix. Concordance metrics for survival models. Treatment-response analysis results. Confidence tier distribution.

**Decision Gate.** Performance above random baseline. For k-class classification, balanced accuracy must exceed 1/k (a missing balanced accuracy fails the gate). For survival prediction, concordance must exceed 0.5. For biomarker-treatment interactions, the interaction must be at least nominally significant (p < 0.05) or directionally consistent.

---

### Layer 8: Manuscript & Communication

**Purpose.** Ensure accurate, reproducible, and journal-compliant communication of validated findings.

**Inputs.** All results from Layers 0-7. Target journal formatting requirements.

**Key Operations.**

1. **Structured manuscript generation.** Produce manuscript sections (Introduction, Methods, Results, Discussion) with direct references to CASCADE layer outputs. Each finding in the Results section must trace back to a specific layer and decision gate.

2. **Anti-tautology reframing.** Perform a logical audit of all claims in the manuscript. Flag circular reasoning (e.g., using the same feature as both predictor and outcome) and ensure independence of combined features is explicitly demonstrated (cross-reference Layer 5 results).

3. **Citation verification.** Flag all references for manual verification. AI-generated references must be checked for: (a) DOI resolves to the correct paper, (b) author names match, (c) cited findings actually appear in the referenced paper. This directly addresses Pitfall #8 (fabricated citations).

4. **Figure generation.** Produce publication-quality figures at 300 DPI with sans-serif fonts. Ensure all figures have appropriate axis labels, legends, and statistical annotations. Verify that display item count is within journal-specific limits.

5. **Journal-specific formatting compliance.** Apply journal word limits, reference format, abstract structure, and display item restrictions. Generate a compliance checklist specific to the target journal.

**Outputs.** Formatted manuscript draft. Citation verification report. Figure files (PNG and PDF). CASCADE compliance checklist with layer-by-layer pass/fail status.

**Decision Gate.** All citations verified as real (no AI fabrications remain). Display item count within journal limits. CASCADE compliance checklist complete. **Pitfalls guarded: #8 (fabricated citations).**

---

## 4. The Pitfall Library

CASCADE maintains a structured catalog of empirically discovered analytical pitfalls. Pitfalls 1–9 were first identified during the source studies and generalized to apply across clinical genomics analyses; pitfalls 10–11 were added in v0.4.0 for failure modes common in multi-institutional observational cohorts.

### Pitfall Registry

| ID | Name | Category | Severity | Detectability | Layer(s) |
|----|------|----------|----------|---------------|----------|
| 1 | Comment header corruption (`#hex` in STYLE_COLOR) | Data Format | CRITICAL | Automated | 0 |
| 2 | Covariate leakage in landmark models | Statistical | CRITICAL | Automated | 0, 5 |
| 3 | Perfect collinearity (algebraic identity) | Statistical | CRITICAL | Automated | 5 |
| 4 | Constant variable at short landmark | Statistical | WARNING | Automated | 4 |
| 5 | XGBoost softprob failure for 2-class subgroups | Computational | WARNING | Runtime | 1 |
| 6 | R classifier package macOS ARM Fortran linker failure | Computational | INFO | Manual | 7 |
| 7 | Singular matrix from constant covariate in subgroup | Statistical | CRITICAL | Automated | 5 |
| 8 | Fabricated citation by AI agent | Communication | CRITICAL | Partial | 8 |
| 9 | Performance status floor effect (ECOG paradox) | Statistical | WARNING | Partial | 5 |
| 10 | Informative (biomarker-dependent) censoring | Statistical | WARNING | Automated | 5 |
| 11 | Centre or batch effect in multi-institutional cohorts | Statistical | WARNING | Automated | 5, 7 |

### Detailed Pitfall Specifications

**Pitfall #1: Comment header corruption.** Tab-delimited files with `STYLE_COLOR` columns containing `#hex` values are corrupted when `pandas.read_csv` is called with `comment='#'`, because the parser treats the hex code as a comment marker and silently drops all subsequent columns on that row. *Detection*: Scan the first 100 data rows for `#[0-9A-Fa-f]{6}` patterns. *Fix*: Strip comment lines manually before parsing.

**Pitfall #2: Covariate leakage in landmark models.** Binary covariates like "ever received immunotherapy" computed over the patient's entire follow-up incorporate post-landmark information. *Detection*: For each binary treatment covariate, verify all contributing events precede the landmark date. *Fix*: Recompute from timeline filtered to `START_DATE <= landmark_date`. Source study: Landmark MDI analysis showed delta-C changed from +0.033 to +0.038 after fixing leakage.

**Pitfall #3: Perfect collinearity.** Derived features that are algebraic functions of other model features cause singular design matrices. *Detection*: Correlation matrix (|r| >= 0.99) and VIF > 10. *Fix*: Drop the derived quantity; retain interpretable raw measures. Source: `ACQUISITION_RATE = (N_SITES - 1) / LANDMARK_TIME` with fixed landmark time.

**Pitfall #4: Constant variable at short landmark.** A feature that varies across the full cohort may become constant at a short landmark time. *Detection*: Check `nunique == 1` after landmark subsetting. *Fix*: Exclude from model at that landmark. Source: `EARLY_DISSEMINATION = 1.0` for all patients at the 3-month landmark.

**Pitfall #5: XGBoost softprob failure.** `multi:softprob` with fixed `num_class` fails when a fold or subgroup has fewer classes than specified. *Detection*: Check unique class count per fold. *Fix*: Use `binary:logistic` for 2-class subgroups.

**Pitfall #6: Platform-specific package failure.** The Bioconductor `impute` package (a dependency of some R subtype-classifier packages) fails to compile on macOS ARM due to missing Fortran runtime libraries. *Detection*: Attempt installation; observe linker error. *Fix*: Use a training-data-only companion package with direct centroid classification, bypassing the affected package.

**Pitfall #7: Singular matrix from constant covariate in subgroup.** A covariate constant within a subgroup (e.g., `ER_BINARY` in ER-positive-only analysis) creates a singular design matrix. *Detection*: Check condition number of design matrix; identify columns with `nunique <= 1`. *Fix*: Automatically drop constant covariates before fitting.

**Pitfall #8: Fabricated citation.** LLMs generating manuscript text may produce fictitious references with plausible author names, journal names, and DOIs. *Detection*: Validate every DOI; confirm author names and cited findings. *Fix*: Replace with verified references. Source: an AI-drafted manuscript section contained a fabricated "Spring et al., JCO, 2020" reference.

**Pitfall #9: Performance status floor effect.** Patients with worsened ECOG but stable burden appear paradoxically favorable in unadjusted analyses due to baseline confounding (ECOG-only patients start with ECOG = 0, the best possible score). *Detection*: Compare unadjusted and baseline-adjusted HRs; if the direction flips, a floor effect is present. *Fix*: Include baseline ECOG as a covariate; report both adjusted and unadjusted estimates.

**Pitfall #10: Informative (biomarker-dependent) censoring.** Cox models assume censoring is unrelated to the outcome given the covariates. In observational genomic cohorts, carriers and non-carriers can differ in follow-up (later panel adoption, referral patterns, loss to follow-up), so the censoring process depends on the biomarker and the HR can be biased. *Detection*: Per biomarker, fit a reverse-censoring Cox model with censoring as the event. Flag biomarkers whose censoring HR is significant after Bonferroni adjustment and at least 1.25-fold in either direction. Biomarkers whose model cannot be fitted get an INFO warning. *Fix*: Compare follow-up by biomarker status, adjust for the driver of differential follow-up (sequencing date, institution), and report an inverse-probability-of-censoring-weighted sensitivity analysis. Implemented in `checks/informative_censoring.py`.

**Pitfall #11: Centre or batch effect.** In multi-institutional data, biomarker prevalence can differ across centres (panel coverage, referral mix), which confounds the association. The biomarker effect itself can also differ across centres. *Detection*: Per biomarker, run a chi-square test and compute the absolute range of prevalence across centres (flagged when significant and range >= 0.15). Also run a likelihood-ratio test of a biomarker-by-centre interaction in a centre-stratified Cox model. Both tests use Bonferroni adjustment across biomarkers, and centres with fewer than 20 patients are excluded. *Fix*: Stratify or adjust by centre, verify panel coverage per centre, and report leave-one-centre-out and centre-specific estimates. Source: GENIE BPC external validation (four institutions). Implemented in `checks/center_effect.py`.

### Adding New Pitfalls

The pitfall library is extensible. To add a new pitfall to the community registry:

1. Assign the next available integer ID.
2. Provide all required fields: name, description, category (`DATA_FORMAT`, `STATISTICAL`, `COMPUTATIONAL`, `COMMUNICATION`), severity (`CRITICAL`, `WARNING`, `INFO`), detectability (`AUTOMATED`, `RUNTIME`, `MANUAL`, `PARTIAL`), detection strategy, fix strategy, source study, and a concrete example.
3. Identify which CASCADE layers the pitfall is relevant to.
4. If the pitfall is automatable, implement a check function in `cascade/pitfalls/checks/` that returns a `PitfallWarning` when the condition is detected.

---

## 5. Decision Gate Summary

The full cascade forms a directed graph where each gate acts as a quality checkpoint. Below is the textual flowchart with remediation paths:

```
START
  |
  v
[Layer 0] Cohort Assembly
  |
  +--> GATE: Future data leakage detected?
  |      YES --> Recompute features with time-locked anchors --> re-enter Layer 0
  |      NO  --> proceed
  v
[Layer 1] Discovery Screen
  |
  +--> GATE: >= 1 FDR-significant finding?
  |      NO  --> Broaden hypothesis space, relax filters, or terminate (null result)
  |      YES --> proceed
  v
[Layer 2] Orthogonal Confirmation
  |
  +--> GATE: Direction concordance >= 70%?
  |      NO  --> Investigate model-specific artifacts, try third method
  |      YES --> proceed
  v
[Layer 3] Predictive vs Prognostic
  |
  +--> GATE: Classification assigned to all findings?
  |      NO  --> Complete interaction testing for remaining findings
  |      YES --> proceed
  v
[Layer 4] Sensitivity & Robustness
  |
  +--> GATE: Primary findings classified (ROBUST/EXPLORATORY/UNSTABLE)?
  |      UNSTABLE present --> Flag/remove unstable findings, document rationale
  |      All classified   --> proceed
  v
[Layer 5] Artifact Guards
  |
  +--> GATE: Critical artifacts detected?
  |      YES --> Apply fixes (recompute features, drop collinear vars, adjust baseline)
  |              Re-run affected analyses --> re-enter Layer 5
  |      NO  --> proceed
  v
[Layer 6] Clinical Translation
  |
  +--> GATE: delta-C > 0 with CI excluding 0?
  |      NO  --> Biomarker lacks incremental clinical value; report as prognostic only
  |      YES --> proceed
  v
[Layer 7] External Validation
  |
  +--> GATE: Performance above random?
  |      NO  --> Findings do not generalize; report with strong caveats
  |      YES --> proceed
  v
[Layer 8] Manuscript & Communication
  |
  +--> GATE: All citations verified? Display items within limits?
  |      NO  --> Fix citations, reduce display items
  |      YES --> DONE
  v
SUBMISSION-READY
```

**Gate failures are not fatal.** A failed gate does not necessarily end the analysis. It triggers a remediation path: fix the underlying issue, document what was done, and re-enter the appropriate layer. The CASCADE compliance report records all gate outcomes, including failures that were remediated. The agent decision log records the final gate state and the remediation attempts.

**Gates fail closed.** Missing, empty or non-finite inputs never pass a gate. This covers an empty classification or robustness table, a NaN delta-C or CI, a missing balanced accuracy, and a Layer 5 report in which no check ran.

---

## 6. Comparison with Existing Guidelines

| Dimension | CASCADE | REMARK | TRIPOD |
|-----------|---------|--------|--------|
| **Purpose** | Prescribes what to DO and in what ORDER | Prescribes what to REPORT | Prescribes what to REPORT |
| **Scope** | Full validation pipeline (discovery through manuscript) | Single biomarker reporting | Prediction model reporting |
| **Prescriptive level** | Procedural (step-by-step protocol with decision gates) | Declarative (checklist of items to include) | Declarative (checklist of items to include) |
| **Orthogonal confirmation** | Explicit layer (Layer 2) requiring independent statistical method | Not addressed | Not addressed |
| **Predictive vs prognostic** | Formal interaction test with classification (Layer 3) | Recommends distinction but no protocol | Not applicable (focused on prediction models) |
| **Sensitivity framework** | 7-category systematic framework with 3-tier robustness scoring (Layer 4) | Recommends sensitivity analyses | Recommends internal validation |
| **Artifact detection** | Pitfall library with 11 documented failure modes, 7 fully automated (Layer 5) | Not addressed | Not addressed |
| **Multiple testing** | FDR/Bonferroni built into Layer 1 with separation problem filtering | Recommends addressing but no protocol | Not directly applicable |
| **Clinical translation** | Cross-validated delta-C, landmark analysis, risk groups (Layer 6) | Not addressed | Calibration and discrimination recommended |
| **External validation** | Explicit layer with balanced accuracy, consensus calibration (Layer 7) | Recommends validation | Explicit guidance (TRIPOD Type 4) |
| **Agentic implementation** | Designed for AI-assisted execution with skill commands | Human-only | Human-only |
| **Origin** | Empirical (derived from 3 source studies; evaluated across 7) | Expert consensus | Expert consensus |

CASCADE is complementary to REMARK and TRIPOD, not a replacement. A study that passes all CASCADE layers will naturally satisfy most REMARK and TRIPOD reporting requirements, but the converse is not true: a study that satisfies REMARK reporting criteria may not have performed orthogonal confirmation, formal predictive testing, systematic sensitivity analysis, or artifact detection.

---

## 7. Implementation Notes

### Software Implementation

CASCADE is implemented in the `cascade-validate` Python package, with modules organized by layer:

```
cascade/
    core/
        cohort.py          # Layer 0: CohortBuilder, FeatureEngineer
    stats/
        cox.py             # PenalizedCox wrapper (Layers 1, 3, 6)
        competing_risks.py # CauseSpecificCox, cause_specific_screen (Layer 1)
        interaction.py     # InteractionCox, stratified_effect (Layer 3)
        multiple_testing.py# FDR, Bonferroni, Holm corrections (Layer 1)
        equivalence.py     # Independence suite, phi, TOST, Bayes factor (Layer 5)
        bootstrap.py       # BootstrapCI, bootstrap_delta_c (Layer 6)
        cross_validate.py  # cv_concordance, cv_delta_c (Layer 6)
    pitfalls/
        library.py         # Pitfall dataclass and 11-pitfall catalog
        registry.py        # PitfallRegistry (in-memory pitfall lookup)
        detector.py        # PitfallDetector orchestrator
        checks/
            comment_header.py    # Pitfall #1 detection
            covariate_leakage.py # Pitfall #2 detection
            collinearity.py      # Pitfall #3 detection
            constant_variable.py # Pitfall #4 detection
            singular_matrix.py   # Pitfall #7 detection
            separation.py        # quasi-separation (relates to #7)
            informative_censoring.py # Pitfall #10 detection
            center_effect.py     # Pitfall #11 detection
    core/
        base.py            # Layer protocol + BaseLayer (custom layers)
    pipeline.py            # Pipeline orchestrator (all layers)
    report.py              # Compliance report generator
```

### Agent Skill Integration

CASCADE includes custom Claude Code skills for AI-assisted validation:

| Skill Command | Layer | Operation |
|--------------|-------|-----------|
| `/cascade-screen` | 1 | Run systematic biomarker discovery with FDR correction |
| `/cascade-confirm` | 2 | Run orthogonal confirmation on Layer 1 results |
| `/cascade-predictive` | 3 | Test predictive vs prognostic classification |
| `/cascade-sensitivity` | 4 | Run 7-category sensitivity suite |
| `/cascade-guard` | 5 | Execute automated artifact checks |
| `/cascade-translate` | 6 | Perform landmark analysis and compute delta-C |
| `/cascade-validate` | 7 | Evaluate on an independent external cohort |
| `/cascade-report` | All | Generate full CASCADE compliance report |

### Validation Tiers

Not every study requires all nine layers. The appropriate depth depends on the study's goals and the strength of evidence required (these tiers are a usage guideline, not enforced by the package):

| Tier | Layers | Use Case |
|------|--------|----------|
| **Minimum viable** | 1 + 4 | Exploratory discovery with sensitivity check |
| **Recommended** | 1 + 2 + 4 + 5 | Discovery + confirmation + sensitivity + artifact guard |
| **Comprehensive** | 0 + 1 + 2 + 3 + 4 + 5 + 6 | Full internal validation with clinical translation |
| **Gold standard** | 0 through 8 | Complete validation including external cohort and manuscript compliance |

The minimum viable tier (Layers 1 + 4) ensures that discoveries are corrected for multiple testing and stress-tested for robustness. The recommended tier adds orthogonal confirmation and artifact detection, which together catch the majority of false-positive findings. The gold standard includes all layers and is appropriate for studies intended for high-impact publication.

### Manual Use as a Checklist

For teams not using the Python package or AI agent skills, CASCADE can be followed as a manual checklist. The compliance report generator (`cascade.report.generate_standalone_checklist`) produces a 26-item summary checklist organized by layer (the more granular protocol checklist in `docs/cascade_checklist.md` expands these into per-step items):

```python
from cascade.report import generate_standalone_checklist

checklist = generate_standalone_checklist({
    0: True,   # Cohort assembled
    1: True,   # Discovery screen completed
    2: True,   # Orthogonal confirmation done
    3: False,  # Predictive testing not yet done
    4: True,   # Sensitivity analysis completed
    5: True,   # Artifact guards run
    6: False,  # Clinical translation pending
    7: False,  # External validation pending
    8: False,  # Manuscript not yet started
})
print(checklist)
```

This produces a markdown checklist that can be included in a manuscript supplement or used as an internal quality tracking tool.

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Anchor date** | A reference timepoint (e.g., treatment start date) to which time-dependent features are locked |
| **Both-significant** | A finding that reaches statistical significance in both the discovery and confirmation analyses |
| **Cause-specific HR** | Hazard ratio from a cause-specific Cox model where competing events are treated as censored |
| **CI ratio** | Upper confidence limit divided by lower confidence limit; values > 100 indicate separation problems |
| **Cross-validated delta-C** | Improvement in concordance index computed on held-out data via stratified k-fold CV |
| **Direction flip** | A finding that changes sign (protective to harmful or vice versa) across sensitivity variants |
| **FDR** | False Discovery Rate; controlled by Benjamini-Hochberg procedure in exploratory analyses |
| **Landmark analysis** | Restriction to patients alive at a fixed timepoint, using features evaluated at that time |
| **Orthogonal confirmation** | Testing the same hypothesis with a statistically independent method |
| **TOST** | Two One-Sided Tests; a procedure for demonstrating equivalence (or independence) |
| **TTNTD** | Time to Next Treatment or Death; a composite endpoint for treatment benefit |

## Appendix B: Empirical Studies

Three source studies (derivation) plus four further studies the framework was
applied to (three generalization + one external) — seven in total.

| Study | Role | Cancer Type | N Patients | CASCADE Layers | Key Findings |
|-------|------|------------|------------|----------------|--------------|
| NSCLC Tropism | Source | NSCLC | 3,368 (Stage IV) | 0, 1, 2, 4 | 16 FDR-sig cause-specific associations; 10 confirmed (82%); 13/19 robust |
| Burden/ECOG Independence | Source | NSCLC | 1,486 | 5 | φ = −0.033 (P = 0.20); ECOG paradox resolved via baseline adjustment |
| CBFB Predictive Biomarker | Source | Breast | 1,197 (metastatic ET) | 1, 3 | CBFB predictive for endocrine therapy (interaction HR = 0.37, P = 0.009) |
| CRC Tropism | Generalization | CRC | 2,332 (Stage IV) | 0, 1, 2, 4 | 18/104 FDR-sig; 13 confirmed (81.6%, ρ = 0.748), 6 both-sig; 17/18 robust |
| Prostate Predictive | Generalization | Prostate | 1,546 (Stage IV) | 1, 3 | 4 prognostic genes (TP53 HR = 2.02); predictive hypotheses null |
| Pan-Cancer Burden | Generalization | 5 types | 7,719 (landmark) | 1, 5, 6 | cross-validated ΔC = +0.023; burden–ECOG independence φ = −0.005 |
| GENIE BPC | External | NSCLC | 797 (4 institutions) | 7 | cross-cohort tropism ρ = 0.394 (65.5%) over 55 overlapping pairs |
