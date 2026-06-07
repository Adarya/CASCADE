"""
CASCADE Compliance Report Generator.

Generates markdown-formatted reports documenting which CASCADE layers
were completed, key findings, and warnings.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

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
            status_icon = {
                "completed": "PASS",
                "skipped": "SKIP",
            }.get(matching.status, "FAIL")
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
        status_icon = {
            "completed": "PASS",
            "skipped": "SKIP",
        }.get(lr.status, "FAIL")
        lines.append(
            f"| + | {lr.layer_name} (custom) | {status_icon} | "
            f"{lr.duration_seconds:.1f}s |"
        )

    lines.append("")

    # Completed layers
    completed = pipeline_result.completed_layers
    total_possible = 9
    pct = len(completed) / total_possible * 100 if total_possible > 0 else 0
    lines.append(f"**Completion: {len(completed)}/{total_possible} layers ({pct:.0f}%)**")
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


def _generate_checklist(pipeline_result) -> str:
    """Generate a REMARK-style checklist for CASCADE compliance."""
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

    completed_layers = set()
    for lr in pipeline_result.layer_results.values():
        if lr.succeeded:
            completed_layers.add(lr.layer_number)

    lines = []
    for layer_num, item in checklist_items:
        if layer_num in completed_layers:
            lines.append(f"- [x] **L{layer_num}**: {item}")
        else:
            lines.append(f"- [ ] **L{layer_num}**: {item}")

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

    return _generate_checklist(pr)
