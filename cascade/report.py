"""
CASCADE Compliance Report Generator.

Generates markdown-formatted reports documenting which CASCADE layers
were completed, key findings, and warnings.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


LAYER_DESCRIPTIONS = {
    0: ("Cohort Assembly & Feature Engineering", "Data ingestion, quality guards, time-dependent features"),
    1: ("Biomarker Discovery Screen", "Systematic enumeration with FDR correction"),
    2: ("Orthogonal Confirmation", "Independent statistical method confirmation"),
    3: ("Predictive vs Prognostic", "Gene x treatment interaction testing"),
    4: ("Sensitivity & Robustness", "7-category sensitivity + 3-tier robustness scoring"),
    5: ("Statistical Artifact Guards", "Independence, leakage, collinearity, floor effects"),
    6: ("Clinical Translation", "Landmark analysis, cross-validated delta-C, risk groups"),
    7: ("External Validation", "Independent cohort testing"),
    8: ("Manuscript & Communication", "Citation format/plausibility checks, compliance checking"),
}


def generate_report(pipeline_result) -> str:
    """
    Generate a CASCADE compliance report.

    Parameters
    ----------
    pipeline_result : PipelineResult
        Result from Pipeline.run().

    Returns
    -------
    str
        Markdown-formatted compliance report.
    """
    lines = []
    lines.append("# CASCADE Compliance Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\nTotal duration: {pipeline_result.total_duration_seconds:.1f}s")
    lines.append("")

    # Loud banner: failures must never be missed.
    failed = pipeline_result.failed_layers
    skipped = pipeline_result.skipped_layers
    if failed:
        lines.append(
            f"> **⚠️ {len(failed)} LAYER(S) FAILED: "
            f"{', '.join(failed)}.** See 'Failed Layers' below."
        )
        lines.append("")
    if skipped:
        lines.append(
            f"> _Note: {len(skipped)} layer(s) skipped "
            f"({', '.join(skipped)}) — required inputs not supplied._"
        )
        lines.append("")

    # Summary table
    lines.append("## Layer Completion Summary")
    lines.append("")
    lines.append("| Layer | Name | Status | Duration |")
    lines.append("|-------|------|--------|----------|")

    for layer_num in range(9):
        name, desc = LAYER_DESCRIPTIONS[layer_num]
        # Find matching result
        matching = None
        for lr_name, lr in pipeline_result.layer_results.items():
            if lr.layer_number == layer_num:
                matching = lr
                break

        if matching:
            status_icon = _layer_status(matching)
            duration = f"{matching.duration_seconds:.1f}s"
            lines.append(f"| {layer_num} | {name} | {status_icon} | {duration} |")
        else:
            lines.append(f"| {layer_num} | {name} | -- | -- |")

    # Custom (non-canonical) layers added by users via add_layer().
    custom = [
        lr
        for lr in pipeline_result.layer_results.values()
        if lr.layer_number not in range(9)
    ]
    for lr in custom:
        status_icon = _layer_status(lr)
        lines.append(
            f"| + | {lr.layer_name} (custom) | {status_icon} | "
            f"{lr.duration_seconds:.1f}s |"
        )

    lines.append("")

    # Completed layers: only the 9 core layers count against /9; custom
    # layers are listed separately.
    core_completed = {
        lr.layer_number
        for lr in pipeline_result.layer_results.values()
        if lr.succeeded and lr.layer_number in range(9)
    }
    total_possible = 9
    pct = len(core_completed) / total_possible * 100
    lines.append(
        f"**Completion: {len(core_completed)}/{total_possible} core layers ({pct:.0f}%)**"
    )
    custom_completed = [lr.layer_name for lr in custom if lr.succeeded]
    if custom:
        lines.append(
            f"\nCustom layers (not counted above): "
            f"{len(custom_completed)}/{len(custom)} completed"
            + (f" ({', '.join(custom_completed)})" if custom_completed else "")
        )
    gates = [
        lr.metadata.get("gate") for lr in pipeline_result.layer_results.values()
        if lr.metadata.get("gate") is not None
    ]
    gated = [
        g for g in gates
        if not g.details.get("skipped") and not g.details.get("not_gated")
    ]
    if gated:
        n_pass = sum(1 for g in gated if g.passed)
        lines.append(f"\n**Gates: {n_pass}/{len(gated)} gates passed.**")
    lines.append("")

    # Warnings
    all_warnings = pipeline_result.all_warnings
    if all_warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in all_warnings:
            lines.append(f"- {w}")
        lines.append("")

    # Failed layers
    failed = pipeline_result.failed_layers
    if failed:
        lines.append("## Failed Layers")
        lines.append("")
        for name in failed:
            lr = pipeline_result.get(name)
            lines.append(f"### {name}")
            lines.append(f"Error: {lr.error_message}")
            lines.append("")

    # Layer details
    lines.append("## Layer Details")
    lines.append("")
    for name, lr in pipeline_result.layer_results.items():
        if not lr.succeeded:
            continue

        lines.append(f"### Layer {lr.layer_number}: {lr.layer_name}")
        lines.append("")

        results = lr.results
        if results is None:
            lines.append("No results returned.")
        elif isinstance(results, pd.DataFrame):
            lines.append(f"Results: {len(results)} rows x {len(results.columns)} columns")
            if "significant" in results.columns:
                n_sig = results["significant"].sum()
                lines.append(f"Significant findings: {n_sig}/{len(results)}")
            if "robustness_class" in results.columns:
                counts = results["robustness_class"].value_counts()
                lines.append(f"Robustness: {dict(counts)}")
            if "confidence_tier" in results.columns:
                counts = results["confidence_tier"].value_counts()
                lines.append(f"Confidence tiers: {dict(counts)}")
        elif isinstance(results, dict):
            for key, value in results.items():
                if isinstance(value, (int, float, str, bool)):
                    lines.append(f"- {key}: {value}")
                elif isinstance(value, pd.DataFrame):
                    lines.append(f"- {key}: DataFrame ({len(value)} rows)")

        lines.append("")

    # Gate results
    gate_results_found = False
    for lr in pipeline_result.layer_results.values():
        gate = lr.metadata.get("gate")
        if gate is not None:
            if not gate_results_found:
                lines.append("## Gate Evaluation Results")
                lines.append("")
                lines.append("| Layer | Name | Gate | Details |")
                lines.append("|-------|------|------|---------|")
                gate_results_found = True

            if getattr(gate, "details", {}).get("skipped"):
                status = "SKIP"
            elif getattr(gate, "details", {}).get("not_gated"):
                status = "NOT GATED"
            elif gate.passed:
                status = "PASS"
            else:
                status = "FAIL"

            if getattr(gate, "remediation_attempted", False):
                if getattr(gate, "remediation_succeeded", False):
                    status += " (remediated)"
                else:
                    status += " (rem. failed)"

            detail = getattr(gate, "error_message", "") or ""
            if not detail and hasattr(gate, "actual") and gate.actual:
                detail = ", ".join(
                    f"{k}={v}" for k, v in gate.actual.items()
                )

            lines.append(
                f"| {lr.layer_number} | {lr.layer_name} | {status} | {detail} |"
            )

    if gate_results_found:
        lines.append("")

    # CASCADE checklist
    lines.append("## CASCADE Checklist")
    lines.append("")
    lines.append(_generate_checklist(pipeline_result))

    return "\n".join(lines)


def _layer_status(lr) -> str:
    """Status shown in the layer table; reflects the gate when one ran."""
    if lr.status == "skipped":
        return "SKIP"
    if lr.status != "completed":
        return "FAIL"
    gate = lr.metadata.get("gate")
    if gate is None:
        return "DONE (not gated)"
    details = getattr(gate, "details", {}) or {}
    if details.get("skipped"):
        return "DONE (gate skipped)"
    if details.get("not_gated"):
        return "DONE (not gated)"
    if gate.passed:
        return "PASS (remediated)" if getattr(gate, "remediation_succeeded", False) else "PASS"
    return "FAIL (gate)"


def _df(results) -> Optional[pd.DataFrame]:
    return results if isinstance(results, pd.DataFrame) and not results.empty else None


def _has_cols(results, *cols) -> bool:
    df = _df(results)
    return df is not None and all(c in df.columns for c in cols)


def _has_keys(results, *keys) -> bool:
    return isinstance(results, dict) and all(
        k in results and results[k] is not None for k in keys
    )


def _finite(v) -> bool:
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def _ev_robust_no_flips(lr) -> bool:
    df = _df(lr.results)
    if df is None or not {"robustness_class", "n_evaluable", "n_concordant"} <= set(df.columns):
        return False
    robust = df[df["robustness_class"] == "robust"]
    return bool((robust["n_concordant"] == robust["n_evaluable"]).all())


def _ev_artifact(check: str):
    def _f(lr) -> bool:
        return check in (getattr(lr.results, "checks_run", None) or [])
    return _f


def _ev_independence(lr) -> bool:
    res = getattr(lr.results, "independence_results", None)
    return "independence" in (getattr(lr.results, "checks_run", None) or []) and bool(res)


# Evidence required, from the layer's own results, before a checklist item
# is ticked.  ``None`` = not verifiable from pipeline results (attest manually).
_CHECKLIST_EVIDENCE = {
    "Cohort defined with inclusion/exclusion criteria": None,
    "Data quality guards applied (comment headers, hex colors)": None,
    "Time-dependent features locked to anchor dates": None,
    "Hypothesis space systematically enumerated": lambda lr: _df(lr.results) is not None,
    "Multiple testing correction applied (FDR or Bonferroni)":
        lambda lr: _has_cols(lr.results, "p_adjusted"),
    "Separation problems filtered (CI ratio check)": None,
    "Minimum subgroup sizes enforced": None,
    "Orthogonal statistical method applied": lambda lr: _df(lr.results) is not None,
    "Cross-analysis concordance computed":
        lambda lr: _has_cols(lr.results, "direction_match")
        or _has_cols(lr.results, "concordance_direction")
        or _has_keys(lr.results, "direction_concordance"),
    "Both-significant findings identified":
        lambda lr: _has_cols(lr.results, "confidence_tier")
        or _has_cols(lr.results, "both_significant")
        or _has_keys(lr.results, "n_both_significant"),
    "Gene x treatment interaction formally tested":
        lambda lr: _has_cols(lr.results, "interaction_p"),
    "Stratified effects reported (treated vs untreated)":
        lambda lr: _has_cols(lr.results, "hr_treated", "hr_untreated"),
    "Sensitivity analyses span >= 3 categories":
        lambda lr: len(set((lr.metadata.get("variant_categories") or {}).values())) >= 3,
    "Robustness classification assigned (robust/exploratory/unstable)":
        lambda lr: _has_cols(lr.results, "robustness_class")
        and bool(lr.results["robustness_class"].notna().all()),
    "No direction flips in robust findings": _ev_robust_no_flips,
    "Independence of combined features tested": _ev_independence,
    "Covariate leakage checked": _ev_artifact("leakage"),
    "Collinearity/VIF assessed": _ev_artifact("collinearity"),
    "Baseline confounding evaluated": _ev_artifact("confounding"),
    "Landmark analysis at clinically relevant timepoints": None,
    "Cross-validated (not training-set) C-statistics reported":
        lambda lr: isinstance(lr.results, dict)
        and _finite(lr.results.get("delta_c", lr.results.get("mean_delta_c"))),
    "Risk groups with KM curves constructed": None,
    "Independent cohort validation performed": lambda lr: lr.results is not None,
    "Balanced accuracy (not just accuracy) reported":
        lambda lr: isinstance(lr.results, dict)
        and _finite(lr.results.get("balanced_accuracy")),
    "Citations screened for format/plausibility issues (manual verification still required)": None,
    "Display item count within journal limits": None,
}


def _generate_checklist(pipeline_result, attested: bool = False) -> str:
    """Generate a REMARK-style checklist for CASCADE compliance.

    An item is ticked only when the corresponding layer completed, its gate
    (if any) did not fail, and the layer's results contain evidence for that
    specific item.  Items that cannot be verified from pipeline results are
    left unticked and marked for manual attestation.  With ``attested=True``
    (standalone checklist), every item of a completed layer is ticked on the
    caller's attestation.
    """
    checklist_items = [
        (0, "Cohort defined with inclusion/exclusion criteria"),
        (0, "Data quality guards applied (comment headers, hex colors)"),
        (0, "Time-dependent features locked to anchor dates"),
        (1, "Hypothesis space systematically enumerated"),
        (1, "Multiple testing correction applied (FDR or Bonferroni)"),
        (1, "Separation problems filtered (CI ratio check)"),
        (1, "Minimum subgroup sizes enforced"),
        (2, "Orthogonal statistical method applied"),
        (2, "Cross-analysis concordance computed"),
        (2, "Both-significant findings identified"),
        (3, "Gene x treatment interaction formally tested"),
        (3, "Stratified effects reported (treated vs untreated)"),
        (4, "Sensitivity analyses span >= 3 categories"),
        (4, "Robustness classification assigned (robust/exploratory/unstable)"),
        (4, "No direction flips in robust findings"),
        (5, "Independence of combined features tested"),
        (5, "Covariate leakage checked"),
        (5, "Collinearity/VIF assessed"),
        (5, "Baseline confounding evaluated"),
        (6, "Landmark analysis at clinically relevant timepoints"),
        (6, "Cross-validated (not training-set) C-statistics reported"),
        (6, "Risk groups with KM curves constructed"),
        (7, "Independent cohort validation performed"),
        (7, "Balanced accuracy (not just accuracy) reported"),
        (8, "Citations screened for format/plausibility issues (manual verification still required)"),
        (8, "Display item count within journal limits"),
    ]

    completed: Dict[int, Any] = {}
    for lr in pipeline_result.layer_results.values():
        if lr.succeeded and lr.layer_number in range(9):
            completed.setdefault(lr.layer_number, lr)

    lines = []
    for layer_num, item in checklist_items:
        lr = completed.get(layer_num)
        note = ""
        if lr is None:
            ticked = False
        elif attested:
            ticked = True
        else:
            gate = lr.metadata.get("gate")
            gate_failed = gate is not None and not gate.passed
            evidence = _CHECKLIST_EVIDENCE.get(item)
            if evidence is None:
                ticked = False
                note = " _(not verifiable from pipeline results; attest manually)_"
            elif gate_failed:
                ticked = False
                note = " _(layer gate FAILED)_"
            else:
                try:
                    ticked = bool(evidence(lr))
                except Exception:
                    ticked = False
                if not ticked:
                    note = " _(no supporting evidence in results)_"
        box = "x" if ticked else " "
        lines.append(f"- [{box}] **L{layer_num}**: {item}{note}")

    return "\n".join(lines)


def generate_standalone_checklist(completed_layers: Dict[int, bool]) -> str:
    """
    Generate a standalone CASCADE checklist without pipeline results.

    Parameters
    ----------
    completed_layers : dict
        Mapping of layer number to completion status.

    Returns
    -------
    str
        Markdown checklist.
    """
    from cascade.pipeline import PipelineResult, LayerResult

    pr = PipelineResult()
    for layer_num, completed in completed_layers.items():
        if completed:
            lr = LayerResult(
                layer_name=f"layer_{layer_num}",
                layer_number=layer_num,
                status="completed",
            )
            pr.layer_results[f"layer_{layer_num}"] = lr

    # Standalone use: completion is attested by the caller, not evidenced.
    return _generate_checklist(pr, attested=True)
