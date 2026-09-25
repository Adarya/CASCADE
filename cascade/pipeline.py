"""
CASCADE Pipeline Orchestrator.

Chains validation layers together, passing results between them
and generating compliance reports.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class LayerSkipped(Exception):
    """Raised internally when a layer cannot be executed automatically because
    the pipeline cannot supply one of its required inputs.

    This is converted into a ``status="skipped"`` :class:`LayerResult` (an
    expected, benign outcome) rather than ``status="failed"`` (a genuine
    error). It lets the pipeline degrade transparently instead of silently
    swallowing a ``TypeError`` for a missing argument.
    """


@dataclass
class LayerResult:
    """Container for results from a single CASCADE layer."""

    layer_name: str
    layer_number: int
    status: str  # 'completed', 'failed', 'skipped'
    results: Any = None
    warnings: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == "completed"


@dataclass
class PipelineResult:
    """Container for results from a full CASCADE pipeline run."""

    layer_results: OrderedDict[str, LayerResult] = field(
        default_factory=OrderedDict
    )
    total_duration_seconds: float = 0.0

    @property
    def completed_layers(self) -> List[str]:
        return [
            name
            for name, r in self.layer_results.items()
            if r.status == "completed"
        ]

    @property
    def failed_layers(self) -> List[str]:
        return [
            name
            for name, r in self.layer_results.items()
            if r.status == "failed"
        ]

    @property
    def skipped_layers(self) -> List[str]:
        return [
            name
            for name, r in self.layer_results.items()
            if r.status == "skipped"
        ]

    def raise_on_failure(self) -> "PipelineResult":
        """Raise ``RuntimeError`` if any layer failed; otherwise return self."""
        if self.failed_layers:
            details = "; ".join(
                f"{n}: {self.layer_results[n].error_message}"
                for n in self.failed_layers
            )
            raise RuntimeError(f"CASCADE layer(s) failed -> {details}")
        return self

    @property
    def all_warnings(self) -> List[str]:
        warnings = []
        for r in self.layer_results.values():
            for w in r.warnings:
                warnings.append(f"[{r.layer_name}] {w}")
        return warnings

    def get(self, layer_name: str) -> Optional[LayerResult]:
        return self.layer_results.get(layer_name)


# Layer number mapping
LAYER_NUMBERS = {
    "cohort": 0,
    "discovery": 1,
    "confirmation": 2,
    "predictive": 3,
    "sensitivity": 4,
    "artifact_guard": 5,
    "clinical": 6,
    "validation": 7,
    "manuscript": 8,
}


class Pipeline:
    """
    CASCADE Pipeline Orchestrator.

    Chains validation layers together in sequence. Each layer receives
    the accumulated results from all prior layers.

    Example::

        from cascade import Pipeline
        from cascade.core import BiomarkerScreen, OrthogonalConfirm

        pipeline = Pipeline()
        pipeline.add_layer("discovery", BiomarkerScreen(method="cox"))
        pipeline.add_layer("confirmation", OrthogonalConfirm())
        results = pipeline.run(data=df, biomarker_cols=genes,
                               outcome_col="OS_MONTHS", event_col="OS_STATUS")
    """

    def __init__(
        self,
        gate_evaluator: Optional[Any] = None,
        remediation_strategies: Optional[Dict[str, Callable]] = None,
        agent_log: Optional[Any] = None,
        max_remediation_retries: int = 2,
    ) -> None:
        self._layers: OrderedDict[str, Any] = OrderedDict()
        self._result: Optional[PipelineResult] = None
        self._gate_evaluator = gate_evaluator
        self._remediation_strategies = remediation_strategies or {}
        self._agent_log = agent_log
        self._max_retries = max_remediation_retries

    def add_layer(
        self, name: str, layer_obj: Any, depends_on: Optional[List[str]] = None
    ) -> "Pipeline":
        """
        Add a validation layer to the pipeline.

        Parameters
        ----------
        name : str
            Layer identifier (e.g., 'discovery', 'confirmation').
        layer_obj : object
            Layer instance with a `run()` or `screen()` or `confirm()` method.
        depends_on : list of str, optional
            Names of layers whose results this layer needs.

        Returns
        -------
        Pipeline
            Self, for method chaining.
        """
        self._layers[name] = {
            "obj": layer_obj,
            "depends_on": depends_on or [],
        }
        return self

    def remove_layer(self, name: str) -> "Pipeline":
        """Remove a layer from the pipeline."""
        self._layers.pop(name, None)
        return self

    def list_layers(self) -> List[str]:
        """Return ordered list of layer names."""
        return list(self._layers.keys())

    def run(
        self, data: pd.DataFrame, strict: bool = False, **kwargs
    ) -> PipelineResult:
        """
        Execute all layers in sequence.

        Dependencies between layers are wired automatically: a layer that
        needs ``primary_results`` (e.g. Orthogonal Confirmation) receives the
        most recent discovery results, and a Sensitivity layer is given a
        default analysis function that re-runs the discovery screen on each
        variant -- so the documented pipeline runs end-to-end without
        requiring explicit ``depends_on`` declarations.

        A layer that raises is recorded as ``status="failed"`` and surfaced
        loudly (logged at ERROR level and headlined in :meth:`report`); it is
        never silently swallowed. A layer whose required inputs cannot be
        supplied automatically is recorded as ``status="skipped"`` (a benign,
        expected outcome) rather than failed.

        Parameters
        ----------
        data : DataFrame
            Input dataset.
        strict : bool, default False
            If True, raise ``RuntimeError`` at the end of the run if any layer
            failed. If False, failures are logged and reported but do not raise.
        **kwargs
            Additional arguments passed to each layer's execution method.
            Common kwargs: biomarker_cols, outcome_col, event_col, covariates.

        Returns
        -------
        PipelineResult
            Accumulated results from all layers.
        """
        pipeline_result = PipelineResult()
        start_time = time.time()
        # Accumulated outputs used to auto-wire dependencies between layers.
        # ``_layer_outputs`` maps every completed layer name to its result and
        # is exposed to any layer that declares a ``context``/``upstream`` param.
        auto_context: Dict[str, Any] = {"primary_df": data, "_layer_outputs": {}}

        for name, layer_info in self._layers.items():
            layer_obj = layer_info["obj"]
            layer_num = LAYER_NUMBERS.get(name, -1)

            logger.info(f"Running CASCADE Layer {layer_num}: {name}")
            layer_start = time.time()

            try:
                # Gather dependency results
                dep_results = {}
                for dep_name in layer_info["depends_on"]:
                    dep_result = pipeline_result.get(dep_name)
                    if dep_result and dep_result.succeeded:
                        dep_results[dep_name] = dep_result.results

                # Execute layer - try common method names
                result = self._execute_layer(
                    layer_obj, name, data, dep_results, auto_context, **kwargs
                )

                layer_result = LayerResult(
                    layer_name=name,
                    layer_number=layer_num,
                    status="completed",
                    results=result,
                    duration_seconds=time.time() - layer_start,
                )

                # Collect warnings from layer if available
                if hasattr(layer_obj, "warnings"):
                    layer_result.warnings = [
                        str(w) for w in layer_obj.warnings
                    ]
                # Layers that return a report object (e.g. Layer 5's
                # ArtifactReport) carry their warnings on the result.
                result_warnings = getattr(result, "warnings", None)
                if isinstance(result_warnings, list):
                    layer_result.warnings += [
                        str(w) for w in result_warnings
                        if str(w) not in layer_result.warnings
                    ]

                # Layer 4: turn raw per-variant outputs into a robustness
                # classification so the gate has something to evaluate.
                if name == "sensitivity":
                    self._classify_sensitivity(
                        layer_obj, layer_result, dep_results, auto_context
                    )

                # Gate evaluation (if evaluator provided)
                if self._gate_evaluator is not None and layer_result.succeeded:
                    layer_result = self._evaluate_and_remediate(
                        name, layer_num, layer_obj, layer_result, data,
                        dep_results, kwargs,
                    )

            except LayerSkipped as skip:
                logger.warning(f"Layer {name} skipped: {skip}")
                layer_result = LayerResult(
                    layer_name=name,
                    layer_number=layer_num,
                    status="skipped",
                    error_message=str(skip),
                    duration_seconds=time.time() - layer_start,
                )

            except Exception as e:
                logger.error(f"CASCADE Layer {layer_num} ({name}) FAILED: {e}")
                layer_result = LayerResult(
                    layer_name=name,
                    layer_number=layer_num,
                    status="failed",
                    error_message=str(e),
                    duration_seconds=time.time() - layer_start,
                )

            # Make this layer's outputs available to downstream layers.
            if layer_result.succeeded:
                self._update_auto_context(auto_context, name, layer_result.results)
                if "variant_results" in layer_result.metadata:
                    # Keep the historical `sensitivity_results` contract
                    # (variant name -> DataFrame) for downstream consumers.
                    auto_context["sensitivity_results"] = (
                        layer_result.metadata["variant_results"]
                    )

            pipeline_result.layer_results[name] = layer_result

        pipeline_result.total_duration_seconds = time.time() - start_time
        self._result = pipeline_result

        # Fail loud: never let a failed layer pass unnoticed.
        if pipeline_result.failed_layers:
            msg = (
                f"{len(pipeline_result.failed_layers)} CASCADE layer(s) FAILED: "
                f"{', '.join(pipeline_result.failed_layers)}. "
                f"See Pipeline.report() for details."
            )
            if strict:
                raise RuntimeError(msg)
            logger.error(msg)

        return pipeline_result

    def _evaluate_and_remediate(
        self,
        name: str,
        layer_num: int,
        layer_obj: Any,
        layer_result: LayerResult,
        data: pd.DataFrame,
        dep_results: Dict[str, Any],
        kwargs: Dict[str, Any],
    ) -> LayerResult:
        """Evaluate a layer's gate, attempt remediation on failure, and log
        the FINAL gate state (including every remediation attempt)."""
        gate_result = self._gate_evaluator.evaluate(name, layer_result)
        layer_result.metadata["gate"] = gate_result

        if not gate_result.passed:
            initial_error = gate_result.error_message
            logger.warning(f"Layer {name} gate FAILED: {initial_error}")
            original_warnings = list(layer_result.warnings)
            layer_result.warnings.append(f"Gate FAILED: {initial_error}")

            attempts: List[Dict[str, Any]] = []
            if name in self._remediation_strategies:
                remediation_fn = self._remediation_strategies[name]
                for attempt in range(self._max_retries):
                    logger.info(
                        f"Remediation attempt {attempt + 1}/{self._max_retries} "
                        f"for layer {name}"
                    )
                    try:
                        new_result = remediation_fn(
                            layer_result, layer_obj, data,
                            dep_results, **kwargs,
                        )
                    except Exception as rem_e:
                        logger.error(f"Remediation failed for {name}: {rem_e}")
                        attempts.append({
                            "attempt": attempt + 1,
                            "status": "exception",
                            "error": f"{type(rem_e).__name__}: {rem_e}",
                        })
                        break

                    remediated_lr = LayerResult(
                        layer_name=name,
                        layer_number=layer_num,
                        status="completed",
                        results=new_result,
                        duration_seconds=layer_result.duration_seconds,
                        metadata={"remediation_attempt": attempt + 1},
                    )
                    gate_result_2 = self._gate_evaluator.evaluate(
                        name, remediated_lr
                    )
                    attempts.append({
                        "attempt": attempt + 1,
                        "status": "passed" if gate_result_2.passed else "failed",
                        "error": gate_result_2.error_message,
                    })

                    if gate_result_2.passed:
                        gate_result_2.remediation_attempted = True
                        gate_result_2.remediation_succeeded = True
                        gate_result_2.details["initial_gate_error"] = initial_error
                        gate_result_2.details["remediation_attempts"] = attempts
                        remediated_lr.metadata["gate"] = gate_result_2
                        remediated_lr.metadata["original_results"] = layer_result.results
                        remediated_lr.warnings = original_warnings + [
                            f"Gate initially FAILED ({initial_error}); "
                            f"remediation succeeded on attempt {attempt + 1}, "
                            f"results replaced by remediated output."
                        ]
                        layer_result = remediated_lr
                        gate_result = gate_result_2
                        logger.info(f"Remediation succeeded for layer {name}")
                        break
                    logger.info(
                        f"Remediation attempt {attempt + 1} "
                        f"did not resolve gate failure"
                    )

            if attempts and not gate_result.remediation_succeeded:
                gate_result.remediation_attempted = True
                gate_result.remediation_succeeded = False
                gate_result.details["remediation_attempts"] = attempts
                summary = "; ".join(
                    f"#{a['attempt']} {a['status']}"
                    + (f" ({a['error']})" if a.get("error") else "")
                    for a in attempts
                )
                layer_result.warnings.append(
                    f"Remediation FAILED after {len(attempts)} attempt(s): {summary}"
                )

        # Log the final gate state (after any remediation).
        if self._agent_log is not None:
            self._agent_log.log_gate_result(
                study=kwargs.get("study_name", "unknown"),
                gate_result=gate_result,
            )
        return layer_result

    @staticmethod
    def _classify_sensitivity(
        layer_obj: Any,
        layer_result: LayerResult,
        dep_results: Dict[str, Any],
        auto_context: Dict[str, Any],
    ) -> None:
        """Replace raw SensitivitySuite output (variant -> DataFrame) with a
        robustness classification of the primary (discovery) findings.

        The raw variant outputs are kept in ``metadata['variant_results']``.
        If no primary results are available the raw output is left as-is and
        the Layer 4 gate will fail (no classification).
        """
        raw = layer_result.results
        if not isinstance(raw, dict) or not hasattr(layer_obj, "classify_robustness"):
            return
        if raw and not all(isinstance(v, pd.DataFrame) for v in raw.values()):
            return
        primary = dep_results.get("discovery", auto_context.get("primary_results"))
        layer_result.metadata["variant_results"] = raw
        if hasattr(layer_obj, "variant_categories"):
            layer_result.metadata["variant_categories"] = dict(
                layer_obj.variant_categories
            )
        failed = getattr(layer_obj, "failed_variants", None) or {}
        layer_result.metadata["failed_variants"] = dict(failed)
        if not isinstance(primary, pd.DataFrame):
            layer_result.warnings.append(
                "No primary (discovery) results available: robustness "
                "classification not computed."
            )
            return
        try:
            layer_result.results = layer_obj.classify_robustness(
                primary, raw, failed_variants=list(failed)
            )
        except Exception as e:
            layer_result.warnings.append(
                f"Robustness classification failed: {e}"
            )

    @staticmethod
    def _update_auto_context(
        auto_context: Dict[str, Any], name: str, result: Any
    ) -> None:
        """Record a layer's output so later layers can be auto-wired to it."""
        if result is None:
            return
        # Every completed layer's output is addressable by name via `context`.
        auto_context.setdefault("_layer_outputs", {})[name] = result
        # Canonical convenience aliases kept for backward compatibility.
        if name == "discovery":
            auto_context["primary_results"] = result
        elif name == "sensitivity":
            auto_context["sensitivity_results"] = result

    def _execute_layer(
        self,
        layer_obj: Any,
        layer_name: str,
        data: pd.DataFrame,
        dep_results: Dict[str, Any],
        auto_context: Dict[str, Any],
        **kwargs,
    ) -> Any:
        """Execute a single layer, auto-wiring its inputs from prior layers.

        Raises
        ------
        LayerSkipped
            If a required parameter cannot be supplied automatically.
        AttributeError
            If the layer object exposes no recognized execution method.
        """
        import inspect

        # Method dispatch based on layer type and available methods
        method_priority = {
            "discovery": ["screen", "run"],
            "confirmation": ["confirm", "run"],
            "predictive": ["screen_predictive", "run"],
            "sensitivity": ["run", "classify_robustness"],
            "artifact_guard": ["run_all", "check_all", "run"],
            "clinical": ["compute", "run"],
            "validation": ["evaluate", "run"],
            "manuscript": ["generate_cascade_checklist", "run"],
        }

        methods_to_try = method_priority.get(layer_name, ["run"])

        # Resolve auto-wired inputs. Explicit depends_on results take
        # precedence; otherwise fall back to context accumulated from prior
        # layers (so no depends_on declaration is required).
        primary_results = dep_results.get(
            "discovery", auto_context.get("primary_results")
        )
        sensitivity_results = dep_results.get(
            "sensitivity", auto_context.get("sensitivity_results")
        )

        for method_name in methods_to_try:
            if not hasattr(layer_obj, method_name):
                continue

            method = getattr(layer_obj, method_name)
            sig = inspect.signature(method)
            call_kwargs: Dict[str, Any] = {}

            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                if param.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                if param_name in ("df", "data", "primary_df"):
                    call_kwargs[param_name] = data
                elif param_name in ("context", "upstream"):
                    # All completed upstream layer outputs, addressable by name.
                    call_kwargs[param_name] = dict(
                        auto_context.get("_layer_outputs", {})
                    )
                elif param_name == "primary_results" and primary_results is not None:
                    call_kwargs[param_name] = primary_results
                elif (
                    param_name == "sensitivity_results"
                    and sensitivity_results is not None
                ):
                    call_kwargs[param_name] = sensitivity_results
                elif param_name == "analysis_fn":
                    # A user-supplied analysis_fn always wins over the default.
                    if kwargs.get("analysis_fn") is not None:
                        call_kwargs[param_name] = kwargs["analysis_fn"]
                    else:
                        default_fn = self._default_analysis_fn(**kwargs)
                        if default_fn is not None:
                            call_kwargs[param_name] = default_fn
                elif param_name in kwargs:
                    call_kwargs[param_name] = kwargs[param_name]
                # otherwise: rely on the parameter's default, or skip-check below

            # Identify required parameters we could not supply.
            missing = [
                p_name
                for p_name, p in sig.parameters.items()
                if p_name != "self"
                and p.kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
                and p_name not in call_kwargs
                and p.default is inspect.Parameter.empty
            ]
            if missing:
                raise LayerSkipped(
                    f"layer '{layer_name}' could not run automatically: missing "
                    f"required input(s) {missing}. Supply them via run(**kwargs), "
                    f"configure the layer object before adding it, or add an "
                    f"upstream layer that produces them."
                )

            return method(**call_kwargs)

        raise AttributeError(
            f"Layer object for '{layer_name}' has no recognized execution method. "
            f"Tried: {methods_to_try}"
        )

    def _default_analysis_fn(self, **kwargs) -> Optional[Callable]:
        """Build a default Sensitivity ``analysis_fn`` that re-runs the
        discovery layer's screen on each variant.

        Returns ``None`` if there is no discovery layer with a ``screen``
        method, or if the screen's required columns are not in ``kwargs`` --
        in which case the sensitivity layer is skipped rather than failed.
        """
        disc_info = self._layers.get("discovery")
        if disc_info is None:
            return None
        disc = disc_info["obj"]
        if not hasattr(disc, "screen"):
            return None

        biomarker_cols = kwargs.get("biomarker_cols")
        outcome_col = kwargs.get("outcome_col")
        event_col = kwargs.get("event_col")
        covariates = kwargs.get("covariates")
        if biomarker_cols is None or outcome_col is None or event_col is None:
            return None

        def _analysis(variant_df, **_ignored):
            return disc.screen(
                variant_df,
                biomarker_cols=biomarker_cols,
                outcome_col=outcome_col,
                event_col=event_col,
                covariates=covariates,
            )

        return _analysis

    def gate_summary(self) -> str:
        """
        Generate a summary of all gate evaluations from the last run.

        Returns
        -------
        str
            Markdown-formatted gate summary.
        """
        if self._result is None or self._gate_evaluator is None:
            return "No gate results available."

        gate_results = []
        for lr in self._result.layer_results.values():
            gate = lr.metadata.get("gate")
            if gate is not None:
                gate_results.append(gate)

        return self._gate_evaluator.summary(gate_results)

    def report(self) -> str:
        """
        Generate a CASCADE compliance report from the last pipeline run.

        Returns
        -------
        str
            Markdown-formatted compliance report.
        """
        if self._result is None:
            return "No pipeline results available. Run the pipeline first."

        from cascade.report import generate_report

        return generate_report(self._result)
