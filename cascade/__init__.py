"""
CASCADE: Clinical Assessment & Systematic Cascade for Agentic Discovery & Evaluation.

A multi-layer validation framework for biomarker discovery in clinical genomics,
derived from human-AI collaborative research on the MSK-CHORD dataset.

Nine layers (numbered 0-8):
    0. Cohort Assembly & Feature Engineering
    1. Biomarker Discovery Screen
    2. Orthogonal Confirmation
    3. Predictive vs Prognostic Distinction
    4. Sensitivity & Robustness Testing
    5. Statistical Artifact Guards
    6. Clinical Translation Assessment
    7. External Validation
    8. Manuscript & Communication
"""

from cascade.version import __version__
from cascade.pipeline import Pipeline
from cascade.core.base import Layer, BaseLayer

__all__ = ["__version__", "Pipeline", "Layer", "BaseLayer"]
