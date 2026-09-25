"""
Check: Comment Header Corruption
=================================

Detects whether a tab-delimited data file contains data rows with the
comment character anywhere in a field -- most commonly hex color codes
(e.g., STYLE_COLOR = '#359645'), but also free text such as 'Tumor #2'
-- that would be truncated if parsed with ``comment='#'`` in pandas.

Pitfall #1 in the CASCADE library.
"""

import re
from pathlib import Path
from typing import List, Union

from ..library import PitfallWarning, Severity, PITFALL_LIBRARY

# Pre-compile the hex color pattern: # followed by 3 or 6 hex digits
_HEX_COLOR_RE = re.compile(r"#[0-9A-Fa-f]{3}(?:[0-9A-Fa-f]{3})?(?:\b|$)")

_PITFALL = PITFALL_LIBRARY[0]  # id=1, comment header corruption


def check_comment_corruption(
    filepath: Union[str, Path],
    comment_char: str = "#",
    max_lines: int = 100,
) -> List[PitfallWarning]:
    """Check whether a file has columns that would be corrupted by comment parsing.

    Reads up to ``max_lines`` non-comment data rows and flags every
    column containing ``comment_char`` (hex colours or any other text).
    Using ``comment='#'`` in pandas silently truncates such rows from
    that character onward.  Lines are split on tabs without stripping,
    so an empty leading field does not shift column indices.

    Parameters
    ----------
    filepath : str or Path
        Path to the tab-delimited data file.
    comment_char : str
        The comment character that would be used for parsing (default '#').
    max_lines : int
        Maximum number of data rows to inspect (default 100).

    Returns
    -------
    list of PitfallWarning
        One warning per column found to contain the comment character.
        Empty list if the file is safe to parse with ``comment='#'``.
    """
    filepath = Path(filepath)
    warnings: List[PitfallWarning] = []

    if not filepath.exists():
        warnings.append(
            PitfallWarning(
                pitfall=_PITFALL,
                message=f"File not found: {filepath}",
                location=str(filepath),
                severity=Severity.WARNING,
                suggestion="Verify the file path is correct.",
            )
        )
        return warnings

    # Read lines, separate comment/metadata lines from data lines
    header_row = None
    data_lines: List[str] = []

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                # Remove only the line terminator; keep leading / trailing
                # tabs so empty fields keep their column position.
                raw = line.rstrip("\r\n")
                if not raw.strip():
                    continue

                # Lines starting with comment_char are metadata
                if raw.lstrip(" ").startswith(comment_char):
                    continue

                # First non-comment line is the header
                if header_row is None:
                    header_row = raw
                    continue

                data_lines.append(raw)
                if len(data_lines) >= max_lines:
                    break
    except (IOError, OSError) as exc:
        warnings.append(
            PitfallWarning(
                pitfall=_PITFALL,
                message=f"Could not read file: {exc}",
                location=str(filepath),
                severity=Severity.WARNING,
                suggestion="Check file permissions and encoding.",
            )
        )
        return warnings

    if header_row is None or not data_lines:
        return warnings  # No data to check

    columns = header_row.split("\t")

    # Track which columns contain the comment character (and whether it
    # looks like a hex colour, for a more specific message)
    hash_columns: dict[int, str] = {}
    hex_like: dict[int, bool] = {}

    for line in data_lines:
        fields = line.split("\t")
        for idx, field_value in enumerate(fields):
            if comment_char not in field_value:
                continue
            if idx not in hash_columns:
                hash_columns[idx] = columns[idx] if idx < len(columns) else f"column_{idx}"
            if _HEX_COLOR_RE.search(field_value):
                hex_like[idx] = True

    # Emit a warning for each affected column
    for idx, col_name in sorted(hash_columns.items()):
        # Determine how many downstream columns would be affected
        downstream_count = len(columns) - idx - 1
        what = "hex color values" if hex_like.get(idx) else f"'{comment_char}' characters"

        warnings.append(
            PitfallWarning(
                pitfall=_PITFALL,
                message=(
                    f"Column '{col_name}' (index {idx}) contains {what}. "
                    f"Using comment='{comment_char}' will truncate "
                    f"this and {downstream_count} subsequent column(s) on "
                    f"affected rows."
                ),
                location=f"{filepath}:{col_name}",
                severity=Severity.CRITICAL,
                suggestion=(
                    f"Do NOT use comment='{comment_char}' with this file. "
                    f"Instead, skip comment lines manually before parsing: "
                    f"read lines, discard those starting with "
                    f"'{comment_char}', then parse the remainder."
                ),
            )
        )

    return warnings
