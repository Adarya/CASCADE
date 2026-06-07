"""
Agent Decision Logger
=====================

Quantitative tracking of agent behavior across CASCADE studies.
Records autonomous decisions, human overrides, gate evaluations,
and pitfall preventions for manuscript-quality reporting.

Classes
-------
AgentDecision
    A single recorded agent decision.
AgentDecisionLog
    Accumulates and summarises agent decisions across studies.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class AgentDecision:
    """A single recorded agent decision.

    Attributes
    ----------
    timestamp : str
        ISO-format timestamp of the decision.
    study : str
        Study identifier (e.g., ``'CRC_tropism'``).
    layer : int
        CASCADE layer number (0-8).
    decision_type : str
        One of: ``'method_selection'``, ``'parameter_choice'``,
        ``'pitfall_prevention'``, ``'override'``.
    description : str
        Human-readable description of the decision.
    rationale : str or None
        Why this decision was made.
    outcome : str or None
        Result of the decision (e.g., ``'gate passed'``).
    automated : bool
        True if the decision was made by the agent autonomously;
        False if it was a human override.
    """

    timestamp: str
    study: str
    layer: int
    decision_type: str
    description: str
    rationale: Optional[str] = None
    outcome: Optional[str] = None
    automated: bool = True


class AgentDecisionLog:
    """Track and quantify agent behavior across CASCADE studies.

    Records four categories of decisions:

    1. **Method selections** -- which statistical model for each layer
    2. **Parameter choices** -- thresholds, correction methods, covariates
    3. **Pitfall preventions** -- pitfalls caught before causing harm
    4. **Overrides** -- human overrides of agent decisions

    Examples
    --------
    >>> log = AgentDecisionLog()
    >>> log.log_decision(
    ...     study="CRC_tropism", layer=1,
    ...     decision_type="method_selection",
    ...     description="Selected cause-specific Cox for tropism screen",
    ...     rationale="Competing risk of death requires cause-specific approach",
    ... )
    >>> log.log_pitfall_prevention(
    ...     study="CRC_tropism", pitfall_id=1,
    ...     description="Detected hex color codes in STYLE_COLOR column",
    ... )
    >>> df = log.to_dataframe()
    """

    VALID_TYPES = {"method_selection", "parameter_choice", "pitfall_prevention", "override"}

    def __init__(self) -> None:
        self._decisions: List[AgentDecision] = []

    def _now(self) -> str:
        return datetime.datetime.now().isoformat(timespec="seconds")

    def log_decision(
        self,
        study: str,
        layer: int,
        decision_type: str,
        description: str,
        rationale: Optional[str] = None,
        outcome: Optional[str] = None,
        automated: bool = True,
    ) -> None:
        """Record an agent decision.

        Parameters
        ----------
        study : str
            Study identifier.
        layer : int
            CASCADE layer number.
        decision_type : str
            One of: ``'method_selection'``, ``'parameter_choice'``,
            ``'pitfall_prevention'``, ``'override'``.
        description : str
            What the decision was.
        rationale : str, optional
            Why it was made.
        outcome : str, optional
            What happened as a result.
        automated : bool, default True
            True if autonomous, False if human-directed.
        """
        if decision_type not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid decision_type '{decision_type}'. "
                f"Valid: {self.VALID_TYPES}"
            )
        self._decisions.append(
            AgentDecision(
                timestamp=self._now(),
                study=study,
                layer=layer,
                decision_type=decision_type,
                description=description,
                rationale=rationale,
                outcome=outcome,
                automated=automated,
            )
        )

    def log_override(
        self,
        study: str,
        layer: int,
        description: str,
        original_decision: str,
        override_reason: str,
    ) -> None:
        """Record a human override of an agent decision.

        Parameters
        ----------
        study : str
        layer : int
        description : str
            What the override changed.
        original_decision : str
            What the agent originally decided.
        override_reason : str
            Why the human overrode.
        """
        self._decisions.append(
            AgentDecision(
                timestamp=self._now(),
                study=study,
                layer=layer,
                decision_type="override",
                description=description,
                rationale=f"Original: {original_decision}. Override reason: {override_reason}",
                automated=False,
            )
        )

    def log_gate_result(
        self,
        study: str,
        gate_result: Any,
    ) -> None:
        """Record a gate evaluation result.

        Parameters
        ----------
        study : str
        gate_result : GateResult
            Result from ``GateEvaluator.evaluate()``.
        """
        outcome = "PASS" if gate_result.passed else "FAIL"
        if gate_result.remediation_succeeded:
            outcome = "PASS (after remediation)"
        elif gate_result.remediation_attempted:
            outcome = "FAIL (remediation failed)"

        self._decisions.append(
            AgentDecision(
                timestamp=self._now(),
                study=study,
                layer=gate_result.layer_number,
                decision_type="parameter_choice",
                description=f"Gate evaluation: Layer {gate_result.layer_number} ({gate_result.layer_name})",
                rationale=str(gate_result.criteria) if gate_result.criteria else None,
                outcome=outcome,
                automated=True,
            )
        )

    def log_pitfall_prevention(
        self,
        study: str,
        pitfall_id: int,
        description: str,
        automated: bool = True,
    ) -> None:
        """Record an auto-prevented pitfall.

        Parameters
        ----------
        study : str
        pitfall_id : int
            Pitfall registry ID (1-9).
        description : str
            What was prevented.
        automated : bool, default True
        """
        self._decisions.append(
            AgentDecision(
                timestamp=self._now(),
                study=study,
                layer=0,  # Pitfall prevention typically at data loading
                decision_type="pitfall_prevention",
                description=f"Pitfall #{pitfall_id}: {description}",
                rationale=f"Known pitfall from CASCADE library (ID={pitfall_id})",
                outcome="prevented",
                automated=automated,
            )
        )

    @property
    def n_decisions(self) -> int:
        """Total number of logged decisions."""
        return len(self._decisions)

    def to_dataframe(self) -> pd.DataFrame:
        """Export all decisions as a DataFrame.

        Returns
        -------
        DataFrame
            Columns: timestamp, study, layer, decision_type, description,
            rationale, outcome, automated.
        """
        if not self._decisions:
            return pd.DataFrame(
                columns=["timestamp", "study", "layer", "decision_type",
                         "description", "rationale", "outcome", "automated"]
            )
        return pd.DataFrame([
            {
                "timestamp": d.timestamp,
                "study": d.study,
                "layer": d.layer,
                "decision_type": d.decision_type,
                "description": d.description,
                "rationale": d.rationale,
                "outcome": d.outcome,
                "automated": d.automated,
            }
            for d in self._decisions
        ])

    def to_csv(self, path: str) -> None:
        """Export decisions to CSV.

        Parameters
        ----------
        path : str
            Output file path.
        """
        self.to_dataframe().to_csv(path, index=False)

    def summary_by_study(self) -> Dict[str, Dict[str, int]]:
        """Count decisions by study and type.

        Returns
        -------
        dict
            ``{study: {decision_type: count, ...}, ...}``
        """
        summary: Dict[str, Dict[str, int]] = {}
        for d in self._decisions:
            if d.study not in summary:
                summary[d.study] = {t: 0 for t in self.VALID_TYPES}
            summary[d.study][d.decision_type] = summary[d.study].get(d.decision_type, 0) + 1
        return summary

    def summary_by_layer(self) -> Dict[int, Dict[str, int]]:
        """Count decisions by layer and type.

        Returns
        -------
        dict
            ``{layer_number: {decision_type: count, ...}, ...}``
        """
        summary: Dict[int, Dict[str, int]] = {}
        for d in self._decisions:
            if d.layer not in summary:
                summary[d.layer] = {t: 0 for t in self.VALID_TYPES}
            summary[d.layer][d.decision_type] = summary[d.layer].get(d.decision_type, 0) + 1
        return summary

    def gate_heatmap_data(self) -> pd.DataFrame:
        """Return study x layer gate pass/fail matrix.

        Returns
        -------
        DataFrame
            Studies as rows, layers as columns.
            Values: 1 (pass), 0 (fail), NaN (not evaluated).
        """
        gate_decisions = [
            d for d in self._decisions
            if d.description.startswith("Gate evaluation:")
        ]

        if not gate_decisions:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        for d in gate_decisions:
            val = 1 if d.outcome and "PASS" in d.outcome else 0
            rows.append({"study": d.study, "layer": d.layer, "passed": val})

        df = pd.DataFrame(rows)
        return df.pivot_table(
            index="study", columns="layer", values="passed", aggfunc="max"
        )

    def override_rate(self) -> float:
        """Fraction of decisions that were human overrides.

        Returns
        -------
        float
            Override rate in [0, 1]. Returns 0.0 if no decisions.
        """
        if not self._decisions:
            return 0.0
        n_override = sum(1 for d in self._decisions if not d.automated)
        return n_override / len(self._decisions)

    def pitfall_prevention_count(self) -> int:
        """Number of pitfalls auto-prevented.

        Returns
        -------
        int
        """
        return sum(
            1 for d in self._decisions
            if d.decision_type == "pitfall_prevention"
        )
