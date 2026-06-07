# /cascade-validate — CASCADE Layer 7: External Validation

## Purpose
Validate biomarker findings on an independent cohort from a different institution or platform. Assess generalizability with appropriate metrics.

## Required Inputs
- **Training data**: Dataset used for discovery (with features + labels)
- **External test data**: Independent dataset (different institution/platform)
- **Model or classifier**: Trained model to evaluate
- **Group mapping** (optional): For hierarchical evaluation

## What It Does

1. **Feature alignment**: Ensures test data has same columns as training
   - Adds missing features as 0
   - Drops extra features not in training
2. **Prediction**: Applies trained model to external data
3. **Evaluation metrics**:
   - Balanced accuracy (accounts for class imbalance)
   - Cohen's kappa (chance-corrected agreement)
   - Per-class F1, precision, recall
   - Confusion matrix
4. **Hierarchical evaluation** (optional):
   - Collapse classes to higher-level groups
   - Group-level accuracy and confusion matrix
5. **Calibrated consensus** (if multiple approaches):
   - Weighted combination of predictions
   - Confidence tiers (high/moderate/low)

## Expected Output
- Classification report with per-class metrics
- Confusion matrix
- Balanced accuracy and Cohen's kappa
- Hierarchical group accuracy (if applicable)

## Figure Guidelines

Any figures produced (e.g., confusion matrices, calibration plots) MUST follow these standards:
- **Dual format**: Save both PNG (300 DPI) and PDF
- **Font**: Arial, minimum 12pt axis labels, 10pt ticks
- **Panels**: Each panel as its own separate file (`fig_val_confusion.png`, `fig_val_calibration.png`)
- **Layout**: Minimal whitespace, `bbox_inches='tight'`, no chart junk, spines top/right off
- **Size**: Compact figures with large text relative to plot elements
- See `CLAUDE.md > Figure Standards` for the full matplotlib rcParams block

## Post-Execution Validation
- [ ] External cohort is truly independent (different institution or time period)
- [ ] Feature alignment documented (N missing, N extra features)
- [ ] Balanced accuracy reported (not just overall accuracy)
- [ ] Performance compared to random baseline (balanced acc > 1/k)
- [ ] Class imbalance handled (balanced weights or stratified sampling)

## Example Code

```python
from cascade.core import ExternalValidator

validator = ExternalValidator()

# Align features
X_test_aligned = validator.align_features(X_train, X_test_external)

# Get predictions
y_pred = model.predict(X_test_aligned)
y_proba = model.predict_proba(X_test_aligned)

# Evaluate
metrics = validator.evaluate(y_true=y_test_external, y_pred=y_pred, y_proba=y_proba)
print(f"Balanced accuracy: {metrics['balanced_accuracy']:.1%}")
print(f"Cohen's kappa: {metrics['kappa']:.3f}")

# Hierarchical evaluation
group_map = {"SubtypeA": "ER+", "SubtypeB": "ER+", "SubtypeC": "HER2+", "SubtypeD": "ER-/HER2-"}
group_metrics = validator.hierarchical_evaluate(y_test_external, y_pred, group_map)
print(f"Group accuracy: {group_metrics['group_accuracy']:.1%}")
```
