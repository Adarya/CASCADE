"""
CASCADE Core Validation Layers
================================

Nine validation layers (0-8) that comprise the CASCADE framework for
biomarker discovery in clinical genomics.

Layers
------
0. **Cohort Assembly & Feature Engineering** (``cohort``)
1. **Biomarker Discovery Screen** (``discovery``)
2. **Orthogonal Confirmation** (``confirmation``)
3. **Predictive vs Prognostic Distinction** (``predictive``)
4. **Sensitivity & Robustness Testing** (``sensitivity``)
5. **Statistical Artifact Guards** (``artifact_guard``)
6. **Clinical Translation Assessment** (``clinical``)
7. **External Validation** (``validation``)
8. **Manuscript & Communication** (``manuscript``)
"""

# Extension contract for custom layers
from cascade.core.base import Layer, BaseLayer

# Layer 0: Cohort Assembly & Feature Engineering
from cascade.core.cohort import CohortBuilder, FeatureEngineer

# Layer 1: Biomarker Discovery Screen
from cascade.core.discovery import BiomarkerScreen

# Layer 2: Orthogonal Confirmation
from cascade.core.confirmation import OrthogonalConfirm

# Layer 3: Predictive vs Prognostic Distinction
from cascade.core.predictive import PredictiveTest, PredictiveResult

# Layer 4: Sensitivity & Robustness Testing
from cascade.core.sensitivity import SensitivitySuite, RobustnessScorer

# Layer 5: Statistical Artifact Guards
from cascade.core.artifact_guard import ArtifactGuard, ArtifactReport

# Gate Evaluation
from cascade.core.gates import GateEvaluator, GateResult

# Agent Decision Logging
from cascade.core.agent_log import AgentDecisionLog, AgentDecision

# Layer 6: Clinical Translation Assessment
from cascade.core.clinical import LandmarkAnalysis, DeltaC, RiskGroupAnalysis

# Layer 7: External Validation
from cascade.core.validation import ExternalValidator

# Layer 8: Manuscript & Communication
from cascade.core.manuscript import ManuscriptHelper

__all__ = [
    # Extension contract
    "Layer",
    "BaseLayer",
    # Layer 0
    "CohortBuilder",
    "FeatureEngineer",
    # Layer 1
    "BiomarkerScreen",
    # Layer 2
    "OrthogonalConfirm",
    # Layer 3
    "PredictiveTest",
    "PredictiveResult",
    # Layer 4
    "SensitivitySuite",
    "RobustnessScorer",
    # Layer 5
    "ArtifactGuard",
    "ArtifactReport",
    # Gate Evaluation
    "GateEvaluator",
    "GateResult",
    # Agent Decision Logging
    "AgentDecisionLog",
    "AgentDecision",
    # Layer 6
    "LandmarkAnalysis",
    "DeltaC",
    "RiskGroupAnalysis",
    # Layer 7
    "ExternalValidator",
    # Layer 8
    "ManuscriptHelper",
]
