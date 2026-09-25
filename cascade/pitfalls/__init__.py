"""
CASCADE Pitfall Detection System
=================================

A reusable framework for detecting common analytical pitfalls in
cancer genomics research pipelines. Distilled from real errors
encountered during multi-study analysis of clinical genomics data.

Main exports:
    PitfallDetector  - Orchestrator that runs all applicable checks.
    PitfallWarning   - Warning object emitted when a pitfall is detected.
    PITFALL_LIBRARY  - List of all 11 canonical pitfall definitions.
    PitfallRegistry  - Extensible registry for managing pitfalls.

Quick start:
    >>> from cascade.pitfalls import PitfallDetector, PITFALL_LIBRARY
    >>> detector = PitfallDetector()
    >>> warnings = detector.check_data_format("my_data.txt")
    >>> print(detector.summary())
"""

from .library import (
    Pitfall,
    PitfallCategory,
    PitfallWarning,
    Severity,
    Detectability,
    PITFALL_LIBRARY,
)
from .registry import PitfallRegistry
from .detector import PitfallDetector

__all__ = [
    "PitfallDetector",
    "PitfallWarning",
    "PitfallRegistry",
    "PITFALL_LIBRARY",
    "Pitfall",
    "PitfallCategory",
    "Severity",
    "Detectability",
]
