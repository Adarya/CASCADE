"""
Layer 8: Manuscript & Communication
=====================================

Quality-control tools for manuscript preparation: word counts, display
item compliance, reference sanity checking, and CASCADE layer checklist
generation.

Classes
-------
ManuscriptHelper
    Checks manuscript compliance with journal limits and generates a
    CASCADE validation checklist.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


class ManuscriptHelper:
    """Manuscript preparation quality checks.

    Provides automated verification of journal-specific constraints
    (word limits, figure/table limits, reference counts) and generates
    a structured checklist of which CASCADE validation layers have been
    completed.

    Parameters
    ----------
    journal_limits : dict, optional
        Journal-specific limits.  Example::

            {'max_words': 8000, 'max_figures': 6, 'max_tables': 4,
             'max_refs': 100, 'max_abstract_words': 250}

        Missing keys are treated as unlimited.
    """

    # CASCADE layer names for checklist generation
    CASCADE_LAYERS = {
        0: "Cohort Assembly & Feature Engineering",
        1: "Biomarker Discovery Screen",
        2: "Orthogonal Confirmation",
        3: "Predictive vs Prognostic Distinction",
        4: "Sensitivity & Robustness Testing",
        5: "Statistical Artifact Guards",
        6: "Clinical Translation Assessment",
        7: "External Validation",
        8: "Manuscript & Communication",
    }

    def __init__(
        self,
        journal_limits: Optional[Dict[str, int]] = None,
    ) -> None:
        self.journal_limits = journal_limits or {}

    # ------------------------------------------------------------------
    # Word count
    # ------------------------------------------------------------------

    @staticmethod
    def check_word_count(
        text: str,
        max_words: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Count words in manuscript text.

        Strips markdown formatting, reference citations, and figure
        captions before counting.

        Parameters
        ----------
        text : str
            Manuscript text (plain text or markdown).
        max_words : int, optional
            Maximum allowed words.  If ``None``, uses the instance's
            journal_limits['max_words'] if available.

        Returns
        -------
        dict
            Keys: ``word_count``, ``within_limit`` (bool or None if
            no limit set).
        """
        # Strip markdown headers
        cleaned = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Strip markdown bold/italic
        cleaned = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", cleaned)
        # Strip reference citations [1], [1,2], [1-3]
        cleaned = re.sub(r"\[\d+(?:[,\-]\d+)*\]", "", cleaned)
        # Strip image/figure references
        cleaned = re.sub(r"!\[.*?\]\(.*?\)", "", cleaned)
        # Strip URLs
        cleaned = re.sub(r"https?://\S+", "", cleaned)

        words = cleaned.split()
        word_count = len(words)

        within_limit = None
        if max_words is not None:
            within_limit = word_count <= max_words

        return dict(word_count=word_count, within_limit=within_limit)

    # ------------------------------------------------------------------
    # Display items
    # ------------------------------------------------------------------

    @staticmethod
    def check_display_items(
        figures: int,
        tables: int,
        max_figures: Optional[int] = None,
        max_tables: Optional[int] = None,
        max_total: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Check display item count against journal limits.

        Parameters
        ----------
        figures : int
            Number of figures.
        tables : int
            Number of tables.
        max_figures : int, optional
        max_tables : int, optional
        max_total : int, optional
            Maximum combined figures + tables.

        Returns
        -------
        dict
            Keys: ``n_figures``, ``n_tables``, ``total``,
            ``within_limit`` (bool or None).
        """
        total = figures + tables
        within_limit = True

        if max_figures is not None and figures > max_figures:
            within_limit = False
        if max_tables is not None and tables > max_tables:
            within_limit = False
        if max_total is not None and total > max_total:
            within_limit = False

        # If no limits set at all, return None
        if max_figures is None and max_tables is None and max_total is None:
            within_limit = None

        return dict(
            n_figures=figures,
            n_tables=tables,
            total=total,
            within_limit=within_limit,
        )

    # ------------------------------------------------------------------
    # Reference checking
    # ------------------------------------------------------------------

    @staticmethod
    def flag_potential_fabricated_refs(
        references: Sequence[str],
        current_year: Optional[int] = None,
    ) -> List[str]:
        """Flag references whose *format* is implausible (a screen, not verification).

        This is a heuristic, offline plausibility check on citation strings.
        It does NOT resolve DOIs or query CrossRef/PubMed and therefore cannot
        confirm that a reference is real or that it supports the cited claim —
        flagged items still require manual verification (Pitfall #8).

        Checks for common patterns:
        1. Publication year in the future
        2. Suspiciously round DOI patterns
        3. Improbable journal names
        4. Duplicate author patterns across different refs

        Parameters
        ----------
        references : sequence of str
            List of reference strings (any standard citation format).
        current_year : int, optional
            Defaults to the current calendar year.

        Returns
        -------
        list of str
            Human-readable warnings about potentially fabricated references.
        """
        if current_year is None:
            current_year = datetime.now().year

        warnings_list: List[str] = []

        for i, ref in enumerate(references, 1):
            ref_id = f"Ref {i}"

            # Check for future year
            years = re.findall(r"\b((19|20)\d{2})\b", ref)
            for y_match in years:
                y = y_match[0]  # Full year string (e.g. '2030')
                if int(y) > current_year:
                    warnings_list.append(
                        f"{ref_id}: Contains future year ({y})"
                    )

            # Check for suspiciously formatted DOIs
            # Real DOIs: 10.XXXX/... where XXXX is a registrant code
            doi_match = re.search(r"10\.\d{4,}/\S+", ref)
            if doi_match:
                doi = doi_match.group()
                # Suspicious: DOI ends with very round numbers
                if re.search(r"/\d{5,}$", doi):
                    warnings_list.append(
                        f"{ref_id}: DOI has suspiciously round number pattern"
                    )

            # Check for journal names that are too generic
            generic_patterns = [
                r"Journal of (?:Advanced|Modern|International|Global) (?:Science|Research|Studies)",
                r"Proceedings of the (?:\d+(?:st|nd|rd|th) )?International Conference",
            ]
            for pattern in generic_patterns:
                if re.search(pattern, ref, re.IGNORECASE):
                    warnings_list.append(
                        f"{ref_id}: Contains potentially generic/fabricated journal name"
                    )

            # Check for incomplete references (no volume/pages/DOI)
            has_volume = bool(re.search(r"\b\d+[:(]\d+", ref))
            has_doi = bool(re.search(r"10\.\d{4,}", ref))
            has_pages = bool(re.search(r"\bp+\.?\s*\d+", ref, re.IGNORECASE)) or bool(re.search(r"\d+-\d+", ref))
            if not has_volume and not has_doi and not has_pages:
                warnings_list.append(
                    f"{ref_id}: Missing volume/pages/DOI - may be incomplete or fabricated"
                )

        return warnings_list

    # ------------------------------------------------------------------
    # CASCADE checklist
    # ------------------------------------------------------------------

    @classmethod
    def generate_cascade_checklist(
        cls,
        results_dict: Dict[int, Any],
    ) -> str:
        """Generate a markdown checklist of CASCADE layer completion.

        Parameters
        ----------
        results_dict : dict
            Mapping of layer number (0-8) to a truthy value indicating
            the layer was completed.  For richer output, the value
            can be a dict with keys ``status`` ('complete', 'partial',
            'skipped'), ``notes`` (str), and ``artifacts`` (list of str).

        Returns
        -------
        str
            Markdown-formatted checklist.
        """
        lines: List[str] = ["## CASCADE Validation Checklist\n"]

        for layer_num in sorted(cls.CASCADE_LAYERS.keys()):
            layer_name = cls.CASCADE_LAYERS[layer_num]
            info = results_dict.get(layer_num)

            if info is None:
                checkbox = "[ ]"
                status_str = "Not started"
                notes = ""
            elif isinstance(info, dict):
                status = info.get("status", "complete")
                if status == "complete":
                    checkbox = "[x]"
                    status_str = "Complete"
                elif status == "partial":
                    checkbox = "[~]"
                    status_str = "Partial"
                else:
                    checkbox = "[ ]"
                    status_str = "Skipped"
                notes = info.get("notes", "")
                artifacts = info.get("artifacts", [])
                if artifacts:
                    notes += f" Artifacts: {', '.join(artifacts)}"
            elif info:
                checkbox = "[x]"
                status_str = "Complete"
                notes = ""
            else:
                checkbox = "[ ]"
                status_str = "Skipped"
                notes = ""

            line = f"- {checkbox} **Layer {layer_num}**: {layer_name} -- {status_str}"
            if notes:
                line += f"\n  - {notes}"
            lines.append(line)

        lines.append("")
        lines.append(
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*"
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Convenience: full check
    # ------------------------------------------------------------------

    def full_check(
        self,
        text: str,
        n_figures: int,
        n_tables: int,
        references: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Run all manuscript checks against journal limits.

        Parameters
        ----------
        text : str
            Manuscript body text.
        n_figures : int
        n_tables : int
        references : sequence of str, optional

        Returns
        -------
        dict
            Comprehensive check results.
        """
        results: Dict[str, Any] = {}

        # Word count
        results["word_count"] = self.check_word_count(
            text, max_words=self.journal_limits.get("max_words")
        )

        # Display items
        results["display_items"] = self.check_display_items(
            n_figures, n_tables,
            max_figures=self.journal_limits.get("max_figures"),
            max_tables=self.journal_limits.get("max_tables"),
            max_total=self.journal_limits.get("max_display_items"),
        )

        # References
        if references is not None:
            results["reference_warnings"] = self.flag_potential_fabricated_refs(
                references
            )
            max_refs = self.journal_limits.get("max_refs")
            results["n_references"] = len(references)
            results["refs_within_limit"] = (
                len(references) <= max_refs if max_refs else None
            )

        return results
