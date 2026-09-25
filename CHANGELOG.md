# Changelog

## v0.4.0

### Added
- Two pitfall checks: informative (biomarker-dependent) censoring (#10,
  `checks/informative_censoring.py`) and centre/batch effect (#11,
  `checks/center_effect.py`). The library now has 11 pitfalls, 7 of them fully
  automated (8 check functions). `PitfallDetector.check_all` accepts
  `survival_df`, `duration_col`, `event_col`, `biomarker_cols`, `center_col`,
  `survival_covariates` and `followup_col`, and `event_date_cols` may be a
  `{covariate: date_col}` dict.
- Pitfall library in Layer 5: `ArtifactGuard.run_all(..., pitfall_inputs={...})`
  runs `PitfallDetector.check_all`. Its warnings reach the compliance report,
  and CRITICAL ones block the Layer 5 gate.
- `AgentDecisionLog`: explicit override fields (`overridden`, `original_choice`,
  `override_choice`, `override_reason`), `from_csv` loader (legacy CSVs
  accepted), and UTC ISO-8601 timestamps.
- Layer 3 `not_evaluable` classification with a `reason`, used when an arm is
  smaller than `min_arm_size` (default 10) or the interaction cannot be
  estimated.
- `PitfallRegistry.list_automatable(include_partial=...)`.
- `fit_error` column in discovery results: why a model failed to fit, instead of
  a silent NaN.
- `BiomarkerScreen` and `OrthogonalConfirm` parse cBioPortal event status
  strings (`'1:DECEASED'` / `'0:LIVING'`) and booleans.

### Fixed
- **Gates fail closed.** Missing, NaN or empty inputs fail the gate. This
  applies to Layers 4, 5 and 6 and to empty predictive or sensitivity tables.
  The Layer 5 gate fails if no check ran.
- **Audit log.** Gate entries (`gate_evaluation`) record the final gate state
  and the remediation attempts. The override rate is overrides / agent
  decisions, and gate and pitfall entries are excluded from the denominator.
- **Report.** Layer status reflects gate outcomes; a completed layer that
  failed its gate is no longer shown as passed.
- **Layer 2.** `OrthogonalConfirm.confirm` no longer crashes without
  `event_col`, and it accepts gene/site competing-risks output (confirmed
  pair-wise).
- **Treatment-line detection.** Gaps are measured from the latest prior stop
  (or start). A new agent outside the concurrent window starts a new line.
- **Interaction screen.** Empty subgroup cells are handled instead of failing.
- **Per-arm event minimum.** This is now enforced in the cause-specific screen.
- **Convergence.** Cox convergence warnings are recorded and reported, not
  swallowed.
- **CIF.** Aalen-Johansen cumulative incidence is seeded, so results are
  reproducible.
- **Pitfall checks.** False positives and negatives are fixed. Separation is
  labelled separately from the singular-matrix pitfall.
- **`CohortBuilder.load_tsv`.** The default now skips only the leading `#`
  metadata block, so `#` inside data fields (hex colours) is preserved.

### Changed (can alter results with default settings)
- Decision gates are on by default: `Pipeline()` creates
  `GateEvaluator('general')`. Pass `gate_evaluator=False` to opt out.
- Non-numeric covariates (e.g. `SEX='Male'/'Female'`) are dummy-encoded in
  discovery.
- `PitfallDetector.check_covariate_leakage` date mode expects `timeline_cols`
  as a `{covariate: date_col}` dict. A list is only paired with covariates by
  name.
- Treatment lines: the new detection rule above can change `LINE_NUMBER`.
- Layer 3: untestable biomarkers are `not_evaluable` rather than `prognostic`.
- Layer 4: UNSTABLE if any direction flip. ROBUST requires zero flips, at
  least `min_evaluable` (2) evaluable variants, and concordance >= 0.75, with
  failed variants counted in the denominator. Everything else is EXPLORATORY.
- The cause-specific screen skips gene-site pairs with fewer than
  `min_events` events in either arm.
- `DeltaC.compute` uses `cv_delta_c` with stratified folds.
- The `bootstrap_delta_c` p-value is NaN (descriptive only; use `cv_delta_c`
  for inference).
- The independence Bayes factor uses the likelihood-ratio G² statistic
  instead of Pearson's chi-square.
- CIF estimates are seeded (default `seed=42`).

## v0.3.0
- Initial public release.
