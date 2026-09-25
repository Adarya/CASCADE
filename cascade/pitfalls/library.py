"""
CASCADE Pitfall Library
=======================

Structured catalog of known analytical pitfalls encountered during
cancer genomics research. Each pitfall is documented with detection
strategy, fix strategy, and the study context where it was first
identified.

This library is dataset-agnostic. The pitfalls represent general
classes of errors that can occur in any clinical genomics pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class PitfallCategory(Enum):
    """Broad category of the pitfall."""
    DATA_FORMAT = "data-format"
    STATISTICAL = "statistical"
    COMPUTATIONAL = "computational"
    COMMUNICATION = "communication"


class Severity(Enum):
    """Impact severity if the pitfall goes undetected."""
    CRITICAL = "critical"    # Will produce wrong results
    WARNING = "warning"      # May produce misleading results
    INFO = "info"            # Good to know


class Detectability(Enum):
    """How the pitfall can be detected."""
    AUTOMATED = "automated"  # Can be caught by code checks
    RUNTIME = "runtime"      # Caught at runtime (errors/exceptions)
    MANUAL = "manual"        # Requires human inspection
    PARTIAL = "partial"      # Partially automatable


@dataclass
class Pitfall:
    """A documented analytical pitfall.

    Attributes:
        id: Unique integer identifier.
        name: Short descriptive name.
        description: Full explanation of what can go wrong.
        category: Broad category (data-format, statistical, etc.).
        severity: Impact if undetected.
        detectability: Whether automated checks can catch it.
        detection_strategy: How to detect this pitfall.
        fix_strategy: How to resolve it once detected.
        source_study: Which analysis context first revealed it.
        example: Concrete example of the pitfall in action.
    """
    id: int
    name: str
    description: str
    category: PitfallCategory
    severity: Severity
    detectability: Detectability
    detection_strategy: str
    fix_strategy: str
    source_study: str
    example: str


@dataclass
class PitfallWarning:
    """A warning emitted when a pitfall is detected.

    Attributes:
        pitfall: The Pitfall object that was triggered.
        message: Human-readable description of what was found.
        location: File path, column name, or variable that triggered it.
        severity: Severity level (inherited from pitfall or overridden).
        suggestion: Actionable fix suggestion.
    """
    pitfall: Pitfall
    message: str
    location: str
    severity: Severity
    suggestion: str

    def __str__(self) -> str:
        return (
            f"[{self.severity.value.upper()}] Pitfall #{self.pitfall.id}: "
            f"{self.pitfall.name}\n"
            f"  Location: {self.location}\n"
            f"  Message:  {self.message}\n"
            f"  Fix:      {self.suggestion}"
        )


# ---------------------------------------------------------------------------
# The canonical library of 11 known pitfalls
# ---------------------------------------------------------------------------

PITFALL_LIBRARY: List[Pitfall] = [
    Pitfall(
        id=1,
        name="Comment header corruption (#hex in STYLE_COLOR)",
        description=(
            "Tab-delimited clinical data files may contain columns with hex "
            "color codes (e.g., STYLE_COLOR = '#359645'). When pandas "
            "read_csv is called with comment='#', every field after the first "
            "'#' on a line is silently dropped, corrupting downstream columns "
            "such as ECOG scores or NLP probability values."
        ),
        category=PitfallCategory.DATA_FORMAT,
        severity=Severity.CRITICAL,
        detectability=Detectability.AUTOMATED,
        detection_strategy=(
            "Read the first 100 data rows without comment parsing. Check for "
            "columns whose values match the pattern #[0-9A-Fa-f]{6}. If any "
            "exist, comment='#' will corrupt the file."
        ),
        fix_strategy=(
            "Skip metadata comment lines manually (e.g., read line-by-line "
            "and discard lines starting with '#' before the header row), then "
            "parse the remainder without the comment parameter."
        ),
        source_study="MSK-CHORD performance_status / progression timelines",
        example=(
            "pd.read_csv('data_timeline_performance_status.txt', sep='\\t', "
            "comment='#') produces NaN in the ECOG column because "
            "STYLE_COLOR values like '#359645' cause mid-line truncation."
        ),
    ),
    Pitfall(
        id=2,
        name="Covariate leakage in landmark models",
        description=(
            "Binary covariates such as 'ever received immunotherapy' are "
            "computed over the patient's entire follow-up period. When used "
            "in a landmark survival model, they incorporate information from "
            "after the landmark date, introducing future-data leakage and "
            "biasing hazard ratio estimates."
        ),
        category=PitfallCategory.STATISTICAL,
        severity=Severity.CRITICAL,
        detectability=Detectability.AUTOMATED,
        detection_strategy=(
            "For each binary treatment covariate, verify that the events "
            "contributing to the variable all have dates on or before the "
            "landmark date. Flag any covariate whose derivation requires "
            "knowledge of post-landmark events."
        ),
        fix_strategy=(
            "Recompute treatment covariates using only events up to the "
            "landmark date: filter the treatment timeline to "
            "START_DATE <= landmark_date before aggregating."
        ),
        source_study="Landmark MDI analysis (NSCLC metastatic diversity)",
        example=(
            "RECEIVED_ICI = 1 for a patient who first received ICI 18 months "
            "after the 6-month landmark. This leaks future treatment "
            "assignment into the baseline model."
        ),
    ),
    Pitfall(
        id=3,
        name="Perfect collinearity (algebraic identity)",
        description=(
            "When a derived feature is an algebraic function of other "
            "features already in the model, perfect collinearity arises. "
            "This causes singular design matrices, unstable coefficient "
            "estimates, or outright model fitting failures."
        ),
        category=PitfallCategory.STATISTICAL,
        severity=Severity.CRITICAL,
        detectability=Detectability.AUTOMATED,
        detection_strategy=(
            "Compute the pairwise Pearson correlation matrix and flag pairs "
            "with |r| >= 0.99. Compute VIF for each feature and flag "
            "VIF > 10. Check for algebraic relationships: if a feature "
            "equals f(other features), it is redundant."
        ),
        fix_strategy=(
            "Drop one of the collinear features. Prefer dropping the derived "
            "quantity and retaining the interpretable raw measures."
        ),
        source_study="Landmark MDI (ACQUISITION_RATE = (n-1)/L with fixed L)",
        example=(
            "ACQUISITION_RATE = (N_SITES - 1) / LANDMARK_TIME. When "
            "LANDMARK_TIME is constant across all patients (e.g., 6 months), "
            "ACQUISITION_RATE is a perfect linear function of N_SITES, "
            "causing a singular covariance matrix in Cox regression."
        ),
    ),
    Pitfall(
        id=4,
        name="Constant variable at short landmark",
        description=(
            "A feature that varies across the full cohort may become constant "
            "when evaluated at a short landmark time. This produces a "
            "zero-variance column that cannot contribute to any model and "
            "will cause fitting errors."
        ),
        category=PitfallCategory.STATISTICAL,
        severity=Severity.WARNING,
        detectability=Detectability.AUTOMATED,
        detection_strategy=(
            "After subsetting to the landmark cohort, check each column for "
            "zero variance (std == 0 or nunique == 1). Pay special attention "
            "to binary flags that may be uniformly 0 or 1 at early landmarks."
        ),
        fix_strategy=(
            "Exclude zero-variance columns from the model at that landmark. "
            "Document which features become constant and at what landmark "
            "threshold they regain variability."
        ),
        source_study="Landmark MDI at 3-month landmark",
        example=(
            "EARLY_DISSEMINATION = 1 for all patients at the 3-month "
            "landmark because the definition threshold coincides with the "
            "landmark window, making the variable constant (= 1.0)."
        ),
    ),
    Pitfall(
        id=5,
        name="XGBoost softprob failure for 2-class subgroups",
        description=(
            "XGBoost with objective='multi:softprob' and a fixed num_class "
            "parameter fails or produces degenerate results when a fold or "
            "subgroup has fewer classes than num_class. This occurs in "
            "stratified cross-validation when rare classes are absent from "
            "some folds."
        ),
        category=PitfallCategory.COMPUTATIONAL,
        severity=Severity.WARNING,
        detectability=Detectability.RUNTIME,
        detection_strategy=(
            "Check the number of unique classes in each training fold. If "
            "any fold has fewer classes than num_class, the objective will "
            "misbehave. Also check for subgroup analyses where only 2 "
            "classes survive filtering."
        ),
        fix_strategy=(
            "For binary subgroups, switch to objective='binary:logistic'. "
            "For multi-class with rare classes, use stratified splits that "
            "guarantee all classes appear, or implement fallback logic."
        ),
        source_study="breast-cancer subtype panel classification",
        example=(
            "Fitting XGBoost with multi:softprob and num_class=10 on a "
            "subgroup that only contains two of the ten classes crashes or "
            "returns NaN probabilities for the 8 absent classes."
        ),
    ),
    Pitfall(
        id=6,
        name="R classifier package macOS ARM Fortran linker failure",
        description=(
            "The Bioconductor 'impute' package, a dependency of some R "
            "subtype-classifier packages, fails to compile on macOS ARM "
            "(Apple Silicon) due to missing Fortran runtime libraries. This "
            "blocks the classification pipeline."
        ),
        category=PitfallCategory.COMPUTATIONAL,
        severity=Severity.INFO,
        detectability=Detectability.MANUAL,
        detection_strategy=(
            "Attempt to install the package on macOS ARM. The build will "
            "fail with a Fortran linker error referencing libgfortran or "
            "libquadmath."
        ),
        fix_strategy=(
            "Use a training-data-only companion package (installs without "
            "Fortran) and implement centroid-based classification directly "
            "in R or Python, bypassing the affected package."
        ),
        source_study="a breast-cancer subtype classifier on macOS ARM",
        example=(
            "install.packages('BiocManager'); "
            "BiocManager::install('impute') fails with: "
            "'ld: library not found for -lgfortran' on M1/M2/M3 Macs."
        ),
    ),
    Pitfall(
        id=7,
        name="Singular matrix from constant covariate in subgroup",
        description=(
            "When performing survival analysis within a subgroup (e.g., "
            "ER-positive breast cancer), a covariate that varies in the full "
            "cohort may be constant within the subgroup, causing a singular "
            "design matrix and model fitting failure."
        ),
        category=PitfallCategory.STATISTICAL,
        severity=Severity.CRITICAL,
        detectability=Detectability.AUTOMATED,
        detection_strategy=(
            "Before fitting, check the condition number of the design "
            "matrix. Identify columns with zero variance or that are "
            "perfectly correlated with the intercept."
        ),
        fix_strategy=(
            "Automatically drop constant covariates from the model before "
            "fitting. Log which covariates were removed and why."
        ),
        source_study="a within-ER+ subgroup Cox model",
        example=(
            "ER_BINARY is included as a covariate in a Cox model fitted "
            "only to ER-positive patients. ER_BINARY = 1 for all rows, "
            "creating a column of ones that is collinear with the implicit "
            "intercept, causing a LinAlgError."
        ),
    ),
    Pitfall(
        id=8,
        name="Fabricated citation by AI agent",
        description=(
            "Large language models used for manuscript drafting may generate "
            "plausible-looking but completely fictitious references, "
            "including fabricated author names, journal names, DOIs, and "
            "findings. These hallucinated citations can survive multiple "
            "rounds of revision if not manually verified."
        ),
        category=PitfallCategory.COMMUNICATION,
        severity=Severity.CRITICAL,
        detectability=Detectability.PARTIAL,
        detection_strategy=(
            "For each citation in a manuscript: (1) verify the DOI resolves "
            "to the correct paper, (2) confirm author names match, (3) check "
            "that the cited findings actually appear in the paper. Automated "
            "DOI validation can catch some fabrications; full verification "
            "requires human review."
        ),
        fix_strategy=(
            "Replace fabricated references with verified ones. Implement a "
            "citation verification step in the manuscript workflow: every "
            "reference must have a validated DOI before submission."
        ),
        source_study="AI-assisted manuscript drafting",
        example=(
            "An AI agent generated 'Spring et al., J Clin Oncol, 2020' as a "
            "reference for a breast-cancer classifier's clinical utility. This "
            "paper does not exist; it was replaced with a verified reference."
        ),
    ),
    Pitfall(
        id=9,
        name="Performance status floor effect (ECOG paradox)",
        description=(
            "Patients with worsened ECOG performance status but stable "
            "metastatic burden appear to have paradoxically favorable "
            "outcomes in unadjusted analyses. This is a floor effect: "
            "patients who start with low (good) ECOG have more room to "
            "worsen, and their good baseline prognosis dominates the "
            "unadjusted hazard ratio."
        ),
        category=PitfallCategory.STATISTICAL,
        severity=Severity.WARNING,
        detectability=Detectability.PARTIAL,
        detection_strategy=(
            "Compare unadjusted and baseline-adjusted hazard ratios for "
            "ECOG worsening groups. If the unadjusted HR is protective "
            "(< 1) but the adjusted HR is harmful (> 1), a floor effect is "
            "present. Also check the distribution of baseline ECOG in each "
            "worsening category."
        ),
        fix_strategy=(
            "Always include baseline performance status as a covariate when "
            "modeling change in performance status. Report both unadjusted "
            "and adjusted estimates to make the floor effect transparent."
        ),
        source_study="NSCLC burden-ECOG independence analysis",
        example=(
            "Unadjusted HR for ECOG-only worsening = 0.84 (apparently "
            "protective). After adjusting for baseline ECOG, the HR becomes "
            "1.25 (harmful, as expected). The paradox arises because "
            "ECOG-only patients start with median ECOG = 0 and have the "
            "longest median survival (38.0 months)."
        ),
    ),
    Pitfall(
        id=10,
        name="Informative (biomarker-dependent) censoring",
        description=(
            "Cox models assume that censoring is unrelated to the outcome "
            "given the covariates. In observational genomic cohorts, "
            "carriers and non-carriers of a biomarker can differ in "
            "follow-up (later panel adoption, referral patterns, loss to "
            "follow-up), so the censoring process depends on the biomarker "
            "and the hazard ratio can be biased."
        ),
        category=PitfallCategory.STATISTICAL,
        severity=Severity.WARNING,
        detectability=Detectability.AUTOMATED,
        detection_strategy=(
            "Fit a reverse-censoring Cox model per biomarker (censoring as "
            "the event). Flag biomarkers whose censoring hazard ratio is "
            "significant after Bonferroni adjustment and at least 1.25-fold "
            "in either direction."
        ),
        fix_strategy=(
            "Compare follow-up by biomarker status, adjust for the driver "
            "of differential follow-up (sequencing date, institution), and "
            "report an inverse-probability-of-censoring-weighted "
            "sensitivity analysis."
        ),
        source_study="Added in v0.4.0 in response to peer review",
        example=(
            "A gene added to a sequencing panel in a later version is only "
            "observed in recently sequenced patients, who have shorter "
            "follow-up and are censored earlier than non-carriers."
        ),
    ),
    Pitfall(
        id=11,
        name="Centre or batch effect in multi-institutional cohorts",
        description=(
            "In multi-institutional data, biomarker prevalence can differ "
            "across centres (panel coverage, referral mix), confounding "
            "the association, and the biomarker effect itself can differ "
            "across centres. Pooling without stratification can create or "
            "mask associations."
        ),
        category=PitfallCategory.STATISTICAL,
        severity=Severity.WARNING,
        detectability=Detectability.AUTOMATED,
        detection_strategy=(
            "Per biomarker: chi-square test and absolute range of "
            "prevalence across centres; likelihood-ratio test of a "
            "biomarker-by-centre interaction in a centre-stratified Cox "
            "model. Bonferroni adjustment across biomarkers."
        ),
        fix_strategy=(
            "Stratify or adjust by centre, verify panel coverage per "
            "centre, and report leave-one-centre-out and centre-specific "
            "estimates."
        ),
        source_study="GENIE BPC external validation; formalized in v0.4.0",
        example=(
            "In the GENIE BPC NSCLC cohort, gene prevalence ranged widely "
            "across the four contributing institutions, motivating "
            "institution-stratified sensitivity analysis."
        ),
    ),
]
