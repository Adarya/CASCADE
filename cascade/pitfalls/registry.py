"""
CASCADE Pitfall Registry
========================

Extensible registry for managing the pitfall catalog. Pre-populated
with the 11 canonical pitfalls from the library. Supports community
contributions via the register() method.
"""

from typing import Dict, List, Optional

from .library import (
    Pitfall,
    PitfallCategory,
    Detectability,
    PITFALL_LIBRARY,
)


class PitfallRegistry:
    """Extensible registry of known analytical pitfalls.

    Pre-populated with the canonical PITFALL_LIBRARY on instantiation.
    New pitfalls can be added at runtime via ``register()``.

    Examples
    --------
    >>> registry = PitfallRegistry()
    >>> len(registry.list_all())
    11
    >>> len(registry.list_automatable(include_partial=False))  # fully automated
    7
    >>> registry.get(1).name
    'Comment header corruption (#hex in STYLE_COLOR)'
    >>> registry.list_by_category(PitfallCategory.STATISTICAL)
    [...]
    """

    def __init__(self, preload: bool = True) -> None:
        """Initialize the registry.

        Parameters
        ----------
        preload : bool
            If True (default), populate with PITFALL_LIBRARY entries.
            Set to False to create an empty registry.
        """
        self._pitfalls: Dict[int, Pitfall] = {}
        if preload:
            for pitfall in PITFALL_LIBRARY:
                self._pitfalls[pitfall.id] = pitfall

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def register(self, pitfall: Pitfall) -> None:
        """Register a new pitfall.

        Parameters
        ----------
        pitfall : Pitfall
            The pitfall to add. Its ``id`` must not collide with an
            existing entry.

        Raises
        ------
        ValueError
            If a pitfall with the same id is already registered.
        TypeError
            If the argument is not a Pitfall instance.
        """
        if not isinstance(pitfall, Pitfall):
            raise TypeError(
                f"Expected a Pitfall instance, got {type(pitfall).__name__}"
            )
        if pitfall.id in self._pitfalls:
            raise ValueError(
                f"Pitfall with id={pitfall.id} already registered: "
                f"'{self._pitfalls[pitfall.id].name}'"
            )
        self._pitfalls[pitfall.id] = pitfall

    def get(self, pitfall_id: int) -> Optional[Pitfall]:
        """Retrieve a pitfall by id.

        Parameters
        ----------
        pitfall_id : int
            The unique identifier.

        Returns
        -------
        Pitfall or None
            The pitfall if found, else None.
        """
        return self._pitfalls.get(pitfall_id)

    def list_all(self) -> List[Pitfall]:
        """Return all registered pitfalls, sorted by id.

        Returns
        -------
        list of Pitfall
        """
        return sorted(self._pitfalls.values(), key=lambda p: p.id)

    def list_by_category(self, category: PitfallCategory) -> List[Pitfall]:
        """Return pitfalls belonging to a specific category.

        Parameters
        ----------
        category : PitfallCategory
            The category to filter by.

        Returns
        -------
        list of Pitfall
        """
        return sorted(
            [p for p in self._pitfalls.values() if p.category == category],
            key=lambda p: p.id,
        )

    def list_automatable(self, include_partial: bool = True) -> List[Pitfall]:
        """Return pitfalls with automated detection.

        Parameters
        ----------
        include_partial : bool, default True
            If True, include PARTIAL (partially automatable) pitfalls as
            well as fully AUTOMATED ones; for the canonical library this
            returns 9 (7 AUTOMATED + 2 PARTIAL).  If False, return only
            the fully automated pitfalls (7 of the 11 canonical ones --
            the count reported in the paper).

        Returns
        -------
        list of Pitfall
            Pitfalls with detectability AUTOMATED (and PARTIAL if
            ``include_partial``).
        """
        automatable = {Detectability.AUTOMATED}
        if include_partial:
            automatable.add(Detectability.PARTIAL)
        return sorted(
            [p for p in self._pitfalls.values() if p.detectability in automatable],
            key=lambda p: p.id,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._pitfalls)

    def __contains__(self, pitfall_id: int) -> bool:
        return pitfall_id in self._pitfalls

    def __repr__(self) -> str:
        return f"PitfallRegistry(n_pitfalls={len(self._pitfalls)})"
