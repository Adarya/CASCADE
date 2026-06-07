"""Extension contract for custom CASCADE layers.

CASCADE is deliberately open: the nine built-in layers are not privileged.
Any object implementing :class:`Layer` (i.e. exposing a ``run`` method) can be
registered with :meth:`cascade.Pipeline.add_layer` under *any* name -- known or
novel -- and the pipeline will execute it, auto-wire its inputs, and surface it
in the compliance report.

When the pipeline executes a layer it inspects the ``run`` signature and fills
recognized parameters automatically:

==========================  ================================================
Parameter name              Value injected by the pipeline
==========================  ================================================
``df`` / ``data``           the input :class:`pandas.DataFrame`
``context`` / ``upstream``  ``dict`` mapping each completed upstream layer
                            name to its result (the general extension hook)
``primary_results``         the most recent discovery-layer result
``sensitivity_results``     the most recent sensitivity-layer result
any other named parameter   filled from ``Pipeline.run(**kwargs)`` by name
==========================  ================================================

Anything the pipeline cannot supply that has no default causes the layer to be
*skipped* (not failed). This means a custom layer only needs to declare the
inputs it actually wants.

Example
-------
::

    from cascade import Pipeline
    from cascade.core import BaseLayer, BiomarkerScreen

    class EffectSizeFloor(BaseLayer):
        \"\"\"Custom layer: flag discovery hits below an HR effect-size floor.\"\"\"
        def __init__(self, min_log_hr: float = 0.2):
            super().__init__()
            self.min_log_hr = min_log_hr

        def run(self, df, context=None):
            disc = (context or {}).get("discovery")
            if disc is None:
                return {"flagged": []}
            import numpy as np
            weak = disc[np.abs(np.log(disc["hr"])) < self.min_log_hr]
            return {"n_weak": int(len(weak)), "weak": list(weak["biomarker"])}

    pipeline = Pipeline()
    pipeline.add_layer("discovery", BiomarkerScreen(method="cox"))
    pipeline.add_layer("effect_floor", EffectSizeFloor(min_log_hr=0.3))
    result = pipeline.run(data=df, biomarker_cols=genes,
                          outcome_col="OS_MONTHS", event_col="OS_STATUS")
    print(result.get("effect_floor").results)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class Layer(Protocol):
    """Structural protocol for a CASCADE pipeline layer.

    Any object with a callable ``run`` satisfies this protocol -- subclassing
    is optional. See the module docstring for the parameters the pipeline
    injects automatically.
    """

    def run(self, df: Any, *args: Any, **kwargs: Any) -> Any:
        ...


class BaseLayer:
    """Optional convenience base class for custom layers.

    Subclasses implement :meth:`run`. A ``warnings`` list is provided and is
    surfaced by the pipeline in the compliance report.
    """

    def __init__(self) -> None:
        self.warnings: List[str] = []

    def run(
        self, df: Any, context: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Any:  # pragma: no cover - abstract by intent
        raise NotImplementedError("Custom layers must implement run().")
