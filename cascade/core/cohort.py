"""
Layer 0: Cohort Assembly & Feature Engineering
===============================================

Provides utilities for loading clinical genomics data, assembling analysis
cohorts, engineering time-dependent features, and computing derived endpoints.

Classes
-------
CohortBuilder
    Handles data loading, filtering, mutation matrix construction,
    treatment line detection, and time-dependent feature engineering.
FeatureEngineer
    Derives binary features, counts, endpoints (TTNTD), and agent
    categorizations from raw clinical data.
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


def parse_event_status(value: Any) -> float:
    """Parse a survival status value to 1 (event), 0 (censored) or NaN.

    Accepts booleans (``True`` = event), numbers (1 / 0) and cBioPortal
    strings such as ``'1:DECEASED'``, ``'0:LIVING'``, ``'DECEASED'``,
    ``'LIVING'``, ``'True'`` / ``'False'``.
    """
    if value is None:
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isnan(value):
            return np.nan
        return 1.0 if value == 1 else (0.0 if value == 0 else np.nan)
    text = str(value).strip().upper()
    if not text or text in {"NA", "NAN", "NONE", "UNKNOWN"}:
        return np.nan
    if text.startswith("1") or text in {"TRUE", "YES"} or "DECEASED" in text or "DEAD" in text:
        return 1.0
    if text.startswith("0") or text in {"FALSE", "NO"} or "LIVING" in text or "ALIVE" in text:
        return 0.0
    return np.nan


class CohortBuilder:
    """Assembles analysis cohorts from tab-delimited clinical genomics files.

    Handles the common challenges of cBioPortal-style data: comment headers,
    treatment line detection from longitudinal timelines, binary mutation
    matrices, and time-dependent feature construction.

    Parameters
    ----------
    data_dir : str or Path, optional
        Default directory for data files.  If provided, ``load_tsv`` will
        resolve relative paths against this directory.
    """

    def __init__(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_tsv(
        self,
        filepath: Union[str, Path],
        comment: str = "#",
        skip_comment_corruption: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load a tab-delimited file, handling comment-header lines.

        Parameters
        ----------
        filepath : str or Path
            Path to the .txt / .tsv file.  If *data_dir* was set and
            *filepath* is relative, it is resolved against *data_dir*.
        comment : str, default '#'
            Prefix that marks metadata/comment lines.  Only the contiguous
            block of lines starting with *comment* at the top of the file
            (the cBioPortal metadata header) is skipped; '#' characters
            inside data fields (e.g. hex colour codes in STYLE_COLOR) are
            preserved.  pandas' ``comment=`` option is never used, because
            it truncates any data row at the first '#' (Pitfall #1).
            Pass ``None`` to disable comment handling.
        skip_comment_corruption : bool, default False
            Kept for backward compatibility.  If ``True``, *every* line that
            starts with *comment* is dropped (not only the leading header
            block).  Data fields containing '#' are preserved either way.
        **kwargs
            Additional keyword arguments forwarded to ``pd.read_csv``.

        Returns
        -------
        DataFrame
        """
        path = Path(filepath)
        if not path.is_absolute() and self.data_dir is not None:
            path = self.data_dir / path

        sep = kwargs.pop("sep", "\t")
        kwargs.pop("comment", None)  # never let pandas truncate at '#'

        if comment and skip_comment_corruption:
            # Read file manually, drop all comment lines, then parse
            with open(path, "r", encoding="utf-8") as fh:
                lines = [
                    line for line in fh if not line.startswith(comment)
                ]
            return pd.read_csv(io.StringIO("".join(lines)), sep=sep, **kwargs)

        n_header = 0
        if comment:
            # Count the leading metadata block only
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.startswith(comment):
                        break
                    n_header += 1
        if n_header and "skiprows" not in kwargs:
            kwargs["skiprows"] = n_header
        return pd.read_csv(path, sep=sep, **kwargs)

    # ------------------------------------------------------------------
    # Cohort filtering
    # ------------------------------------------------------------------

    @staticmethod
    def filter_stage(
        df: pd.DataFrame,
        stage_col: str,
        stage_values: Union[str, Sequence[str]],
    ) -> pd.DataFrame:
        """Filter a DataFrame to rows matching the given stage value(s).

        Parameters
        ----------
        df : DataFrame
        stage_col : str
            Column containing stage information.
        stage_values : str or sequence of str
            Allowed stage values (e.g. ``'Stage 4'`` or
            ``['Stage 4', 'Stage IV']``).

        Returns
        -------
        DataFrame
            Filtered copy.
        """
        if isinstance(stage_values, str):
            stage_values = [stage_values]
        return df.loc[df[stage_col].isin(stage_values)].copy()

    # ------------------------------------------------------------------
    # Mutation matrix
    # ------------------------------------------------------------------

    @staticmethod
    def build_mutation_matrix(
        mutations_df: pd.DataFrame,
        patient_col: str = "PATIENT_ID",
        gene_col: str = "Hugo_Symbol",
        filter_oncogenic: bool = True,
        oncogenic_col: str = "ONCOGENIC",
        oncogenic_values: Optional[Sequence[str]] = None,
        all_patients: Optional[Sequence[Any]] = None,
    ) -> pd.DataFrame:
        """Build a patient-by-gene binary mutation matrix.

        Parameters
        ----------
        mutations_df : DataFrame
            MAF-style mutation table.
        patient_col : str
            Column with patient identifiers.
        gene_col : str
            Column with gene symbols.
        filter_oncogenic : bool, default True
            If ``True``, restrict to rows where *oncogenic_col* is in
            *oncogenic_values*.
        oncogenic_col : str, default 'ONCOGENIC'
            Column indicating oncogenicity classification.
        oncogenic_values : sequence of str, optional
            Values considered oncogenic.  Defaults to
            ``['Oncogenic', 'Likely Oncogenic']``.
        all_patients : sequence, optional
            Full list of profiled patients (or samples).  If given, the
            matrix is reindexed to exactly these IDs and patients with no
            qualifying mutation get an all-zero row.  If omitted, only
            patients with at least one qualifying mutation appear -- callers
            must then treat patients absent from the matrix as wild-type.

        Returns
        -------
        DataFrame
            Binary matrix with patients as rows and genes as columns.
            Index is *patient_col* values.

        Warns
        -----
        UserWarning
            If *filter_oncogenic* is ``True`` but *oncogenic_col* is absent
            (the filter cannot be applied and all mutations are kept).
        """
        if oncogenic_values is None:
            oncogenic_values = ["Oncogenic", "Likely Oncogenic"]

        data = mutations_df.copy()
        if filter_oncogenic:
            if oncogenic_col in data.columns:
                data = data.loc[data[oncogenic_col].isin(oncogenic_values)]
            else:
                warnings.warn(
                    f"build_mutation_matrix: column '{oncogenic_col}' not found; "
                    "oncogenic filter NOT applied (all mutations kept).",
                    stacklevel=2,
                )

        if data.empty:
            warnings.warn(
                "No mutations remaining after oncogenic filter.",
                stacklevel=2,
            )
            if all_patients is not None:
                return pd.DataFrame(
                    index=pd.Index(list(dict.fromkeys(all_patients)), name=patient_col)
                )
            return pd.DataFrame()

        # Pivot to binary matrix
        data["_present"] = 1
        matrix = (
            data.drop_duplicates(subset=[patient_col, gene_col])
            .pivot_table(
                index=patient_col,
                columns=gene_col,
                values="_present",
                fill_value=0,
                aggfunc="max",
            )
            .astype(int)
        )
        if all_patients is not None:
            matrix = matrix.reindex(
                pd.Index(list(dict.fromkeys(all_patients)), name=patient_col),
                fill_value=0,
            ).astype(int)
        return matrix

    # ------------------------------------------------------------------
    # Treatment line detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_treatment_lines(
        timeline_tx: pd.DataFrame,
        patient_col: str = "PATIENT_ID",
        start_col: str = "START_DATE",
        stop_col: str = "STOP_DATE",
        agent_col: str = "AGENT",
        gap_days: int = 90,
        concurrent_window: int = 28,
    ) -> pd.DataFrame:
        """Detect treatment lines from a longitudinal treatment timeline.

        Rule (applied per patient, records sorted by start date):

        1. The first record opens line 1; its start is the line start.
        2. A record starting within *concurrent_window* days of the current
           line start joins the line (concurrent regimen components).
        3. Otherwise, the gap is measured from the latest prior activity
           date -- the maximum over earlier records of STOP_DATE, or of
           START_DATE when STOP_DATE is missing or the column is absent.
           A gap > *gap_days* starts a new line.
        4. Otherwise (gap <= *gap_days*), a record whose agent is already
           part of the current line continues that line; a *new* agent
           added outside the concurrent window starts a new line
           (switch / escalation).  Without an *agent_col*, only rule 3
           can split lines.

        Records with a missing START_DATE are assigned to the current line.

        Parameters
        ----------
        timeline_tx : DataFrame
            Treatment timeline with at minimum *patient_col*, *start_col*,
            *stop_col*, and *agent_col*.
        patient_col, start_col, stop_col, agent_col : str
            Column name overrides.
        gap_days : int, default 90
            A gap (days) strictly greater than this between the latest prior
            stop (or start, if stop is missing) and the next start indicates
            a new treatment line.
        concurrent_window : int, default 28
            Treatments starting within this many days of each other are
            grouped into the same line.

        Returns
        -------
        DataFrame
            Copy of input with added columns: ``LINE_NUMBER``,
            ``LINE_START``, ``LINE_AGENTS`` (comma-separated agent list).
        """
        df = timeline_tx.copy()
        required = [patient_col, start_col]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found.")

        df[start_col] = pd.to_numeric(df[start_col], errors="coerce")
        if stop_col in df.columns:
            df[stop_col] = pd.to_numeric(df[stop_col], errors="coerce")

        df = df.sort_values([patient_col, start_col]).reset_index(drop=True)

        line_numbers: List[int] = []
        line_starts: List[float] = []

        has_stop = stop_col in df.columns
        has_agent = agent_col in df.columns

        def _norm(agent: Any) -> Optional[str]:
            return None if pd.isna(agent) else str(agent).strip().lower()

        for _, grp in df.groupby(patient_col, sort=False):
            current_line = 1
            current_line_start = np.nan
            line_agents: set = set()
            last_activity = np.nan  # max over prior records of stop (or start)

            for _, row in grp.iterrows():
                start = row[start_col]
                stop = row[stop_col] if has_stop else np.nan
                agent = _norm(row[agent_col]) if has_agent else None

                if pd.isna(start):
                    line_numbers.append(current_line)
                    line_starts.append(current_line_start)
                    continue

                if np.isnan(current_line_start):
                    current_line_start = start  # first dated record
                elif start - current_line_start <= concurrent_window:
                    pass  # concurrent component of the current regimen
                elif not np.isnan(last_activity) and start - last_activity > gap_days:
                    current_line += 1
                    current_line_start = start
                    line_agents = set()
                elif agent is not None and agent not in line_agents:
                    # New agent added outside the concurrent window -> new line
                    current_line += 1
                    current_line_start = start
                    line_agents = set()

                if agent is not None:
                    line_agents.add(agent)

                line_numbers.append(current_line)
                line_starts.append(current_line_start)

                activity = stop if pd.notna(stop) else start
                last_activity = (
                    activity if np.isnan(last_activity) else max(last_activity, activity)
                )

        df["LINE_NUMBER"] = line_numbers
        df["LINE_START"] = line_starts

        # Build LINE_AGENTS: comma-separated list of agents in each line
        if agent_col in df.columns:
            line_agents = (
                df.groupby([patient_col, "LINE_NUMBER"])[agent_col]
                .apply(lambda x: ", ".join(sorted(x.dropna().unique())))
                .reset_index()
                .rename(columns={agent_col: "LINE_AGENTS"})
            )
            df = df.merge(line_agents, on=[patient_col, "LINE_NUMBER"], how="left")

        return df

    # ------------------------------------------------------------------
    # Time-dependent features
    # ------------------------------------------------------------------

    @staticmethod
    def compute_time_dependent_features(
        timeline_df: pd.DataFrame,
        anchor_dates_df: pd.DataFrame,
        feature_col: str,
        patient_col: str = "PATIENT_ID",
        date_col: str = "START_DATE",
        anchor_date_col: str = "ANCHOR_DATE",
        window_days: int = 30,
        method: str = "most_recent",
    ) -> pd.DataFrame:
        """Compute time-dependent feature values at anchor dates.

        For each patient and anchor date, looks back in *timeline_df* for
        values of *feature_col* and summarises them.

        Parameters
        ----------
        timeline_df : DataFrame
            Longitudinal data with *patient_col*, *date_col*, *feature_col*.
        anchor_dates_df : DataFrame
            Table with *patient_col* and *anchor_date_col* giving the
            reference dates at which to evaluate the feature.
        feature_col : str
            Column to extract from the timeline.
        patient_col : str
        date_col : str
        anchor_date_col : str
        window_days : int, default 30
            Maximum look-back window (days) before anchor.
        method : str, default 'most_recent'
            Aggregation method: ``'most_recent'`` (closest value),
            ``'mean'``, ``'max'``, ``'min'``, ``'any'`` (boolean OR).

        Returns
        -------
        DataFrame
            *anchor_dates_df* with an added column named *feature_col*
            containing the resolved values.
        """
        valid_methods = {"most_recent", "mean", "max", "min", "any"}
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

        result = anchor_dates_df.copy()
        feature_values: List[Any] = []

        timeline_grouped = timeline_df.groupby(patient_col)

        for _, row in result.iterrows():
            pid = row[patient_col]
            anchor = row[anchor_date_col]

            if pid not in timeline_grouped.groups:
                feature_values.append(np.nan)
                continue

            patient_tl = timeline_grouped.get_group(pid)
            patient_tl = patient_tl.dropna(subset=[date_col, feature_col])

            # Filter to events within window before anchor
            mask = (patient_tl[date_col] <= anchor) & (
                patient_tl[date_col] >= anchor - window_days
            )
            eligible = patient_tl.loc[mask]

            if eligible.empty:
                feature_values.append(np.nan)
                continue

            if method == "most_recent":
                # Closest to anchor date
                idx = (anchor - eligible[date_col]).abs().idxmin()
                feature_values.append(eligible.loc[idx, feature_col])
            elif method == "mean":
                feature_values.append(eligible[feature_col].mean())
            elif method == "max":
                feature_values.append(eligible[feature_col].max())
            elif method == "min":
                feature_values.append(eligible[feature_col].min())
            elif method == "any":
                feature_values.append(int(eligible[feature_col].any()))

        result[feature_col] = feature_values
        return result

    # ------------------------------------------------------------------
    # Site flags at date
    # ------------------------------------------------------------------

    @staticmethod
    def compute_site_flags_at_date(
        site_timeline: pd.DataFrame,
        anchor_dates: pd.DataFrame,
        patient_col: str = "PATIENT_ID",
        date_col: str = "START_DATE",
        site_col: str = "TUMOR_SITE",
        anchor_date_col: str = "ANCHOR_DATE",
    ) -> pd.DataFrame:
        """Create binary tumor-site flags at each anchor date.

        For each patient and anchor date, looks at all site-timeline events
        on or before the anchor and constructs binary flags for each unique
        site value.

        Parameters
        ----------
        site_timeline : DataFrame
            Longitudinal site data with *patient_col*, *date_col*, *site_col*.
        anchor_dates : DataFrame
            Table with *patient_col* and *anchor_date_col*.
        patient_col, date_col, site_col, anchor_date_col : str
            Column name overrides.

        Returns
        -------
        DataFrame
            *anchor_dates* augmented with one binary column per unique site.
        """
        all_sites = sorted(site_timeline[site_col].dropna().unique())
        # Work on a positional index internally (the caller's index may
        # contain duplicate labels, e.g. after pd.concat); restore it at the end.
        original_index = anchor_dates.index
        result = anchor_dates.reset_index(drop=True)

        # Initialise all site columns to 0
        for site in all_sites:
            result[site] = 0

        site_grouped = site_timeline.groupby(patient_col)

        for i, row in result.iterrows():
            pid = row[patient_col]
            anchor = row[anchor_date_col]

            if pid not in site_grouped.groups:
                continue

            patient_sites = site_grouped.get_group(pid)
            eligible = patient_sites.loc[patient_sites[date_col] <= anchor]

            if eligible.empty:
                continue

            present_sites = eligible[site_col].dropna().unique()
            for site in present_sites:
                if site in result.columns:
                    result.at[i, site] = 1

        result.index = original_index
        return result


class FeatureEngineer:
    """Derives analysis-ready features from raw clinical data.

    Provides common transformations: thresholding, counting, endpoint
    computation (TTNTD), and agent categorisation.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Binary from threshold
    # ------------------------------------------------------------------

    @staticmethod
    def binary_from_threshold(
        series: pd.Series,
        threshold: float,
        above: bool = True,
    ) -> pd.Series:
        """Create a binary indicator from a numeric series and threshold.

        Parameters
        ----------
        series : Series
            Numeric values.
        threshold : float
            Cutpoint.
        above : bool, default True
            If ``True``, 1 where series >= threshold; if ``False``,
            1 where series <= threshold.

        Returns
        -------
        Series
            Integer series of 0/1 values (NaN preserved).
        """
        if above:
            result = (series >= threshold).astype(float)
        else:
            result = (series <= threshold).astype(float)
        result[series.isna()] = np.nan
        return result

    # ------------------------------------------------------------------
    # Counting
    # ------------------------------------------------------------------

    @staticmethod
    def count_unique_per_group(
        df: pd.DataFrame,
        group_col: str,
        count_col: str,
        date_col: Optional[str] = None,
    ) -> pd.Series:
        """Count unique values of *count_col* per group.

        Parameters
        ----------
        df : DataFrame
        group_col : str
            Column defining groups (e.g. PATIENT_ID).
        count_col : str
            Column whose unique values to count (e.g. TUMOR_SITE).
        date_col : str, optional
            If provided, counts are computed per unique (group, date)
            combination and the maximum across dates is returned.

        Returns
        -------
        Series
            Indexed by *group_col* with integer counts.
        """
        if date_col is not None:
            per_date = (
                df.groupby([group_col, date_col])[count_col]
                .nunique()
                .reset_index(name="_count")
            )
            return per_date.groupby(group_col)["_count"].max()

        return df.groupby(group_col)[count_col].nunique()

    # ------------------------------------------------------------------
    # TTNTD
    # ------------------------------------------------------------------

    @staticmethod
    def compute_ttntd(
        treatment_lines_df: pd.DataFrame,
        os_months_df: pd.DataFrame,
        patient_col: str = "PATIENT_ID",
        line_col: str = "LINE_NUMBER",
        line_start_col: str = "LINE_START",
        os_months_col: str = "OS_MONTHS",
        os_status_col: str = "OS_STATUS",
    ) -> pd.DataFrame:
        """Compute Time to Next Treatment or Death (TTNTD).

        For each treatment line, TTNTD is the time (in days) from the
        line start to the start of the next treatment line, or death,
        whichever comes first.  If neither event occurs, the observation
        is censored at last known follow-up.

        Parameters
        ----------
        treatment_lines_df : DataFrame
            Must contain *patient_col*, *line_col*, *line_start_col*.
            Should be output of ``CohortBuilder.detect_treatment_lines``.
        os_months_df : DataFrame
            Must contain *patient_col*, *os_months_col*, *os_status_col*.
        patient_col, line_col, line_start_col : str
        os_months_col, os_status_col : str

        Returns
        -------
        DataFrame
            One row per (patient, line) with columns: *patient_col*,
            *line_col*, *line_start_col*, ``TTNTD_DAYS``, ``TTNTD_EVENT``.
        """
        # Deduplicate to one row per patient-line
        lines = (
            treatment_lines_df
            .groupby([patient_col, line_col])
            .agg({line_start_col: "min"})
            .reset_index()
            .sort_values([patient_col, line_col])
        )

        ttntd_days: List[float] = []
        ttntd_event: List[int] = []

        os_map: Dict[str, Dict[str, Any]] = {}
        for _, r in os_months_df.iterrows():
            pid = r[patient_col]
            os_map[pid] = {
                "os_days": r[os_months_col] * 30.44 if pd.notna(r[os_months_col]) else np.nan,
                "dead": 1 if parse_event_status(r.get(os_status_col, np.nan)) == 1 else 0,
            }

        for pid, grp in lines.groupby(patient_col):
            grp = grp.sort_values(line_col)
            starts = grp[line_start_col].values
            os_info = os_map.get(pid, {"os_days": np.nan, "dead": 0})

            for j in range(len(starts)):
                current_start = starts[j]

                if j + 1 < len(starts):
                    # Next treatment exists. TTNTD counts the next treatment OR
                    # death (whichever comes first) as the event.
                    time_to_next = starts[j + 1] - current_start
                    if os_info["dead"] and not np.isnan(os_info["os_days"]):
                        # If a valid death time falls before the next treatment,
                        # death is the (earlier) event; otherwise the next
                        # treatment is. The 0 <= guard avoids spurious overrides
                        # when OS and line-start timelines are not aligned.
                        time_to_death = os_info["os_days"] - current_start
                        if 0 <= time_to_death < time_to_next:
                            ttntd_days.append(time_to_death)
                        else:
                            ttntd_days.append(time_to_next)
                    else:
                        ttntd_days.append(time_to_next)
                    ttntd_event.append(1)  # next treatment or death = event
                else:
                    # Last line: censor at death or last follow-up
                    if os_info["dead"] and not np.isnan(os_info["os_days"]):
                        ttntd_days.append(os_info["os_days"] - current_start)
                        ttntd_event.append(1)
                    elif not np.isnan(os_info["os_days"]):
                        ttntd_days.append(os_info["os_days"] - current_start)
                        ttntd_event.append(0)
                    else:
                        ttntd_days.append(np.nan)
                        ttntd_event.append(0)

        lines["TTNTD_DAYS"] = ttntd_days
        lines["TTNTD_EVENT"] = ttntd_event

        # Remove negative TTNTD (data artefact)
        lines.loc[lines["TTNTD_DAYS"] < 0, "TTNTD_DAYS"] = np.nan

        return lines

    # ------------------------------------------------------------------
    # Agent categorisation
    # ------------------------------------------------------------------

    @staticmethod
    def categorize_agents(
        agent_names: pd.Series,
        categories_dict: Dict[str, List[str]],
        default: str = "Other",
    ) -> pd.Series:
        """Map free-text agent names to therapeutic categories.

        Parameters
        ----------
        agent_names : Series
            Free-text agent names (e.g. 'Pembrolizumab').
        categories_dict : dict
            Mapping of category label to list of agent name patterns
            (case-insensitive substring matching).  The dict order is the
            label order for combination regimens.  Example::

                {'ICI': ['pembrolizumab', 'nivolumab', 'atezolizumab'],
                 'Platinum': ['carboplatin', 'cisplatin']}

        default : str, default 'Other'
            Category assigned when no pattern matches.

        Returns
        -------
        Series
            Categorical labels aligned with *agent_names*.  A name matching
            patterns from several categories (e.g. a multi-agent regimen
            ``'CARBOPLATIN+PEMBROLIZUMAB'``) receives an explicit combined
            label joining the matched categories with ``' + '`` in
            *categories_dict* order (e.g. ``'ICI + Platinum'``); it is never
            silently assigned to only one of them.
        """
        lowered = agent_names.astype("string").str.lower()
        matched: List[List[str]] = [[] for _ in range(len(agent_names))]

        for category, patterns in categories_dict.items():
            mask = np.zeros(len(agent_names), dtype=bool)
            for pattern in patterns:
                mask |= lowered.str.contains(
                    pattern.lower(), regex=False, na=False
                ).to_numpy(dtype=bool)
            for pos in np.flatnonzero(mask):
                matched[pos].append(category)

        labels = [" + ".join(cats) if cats else default for cats in matched]
        return pd.Series(labels, index=agent_names.index, dtype=object)
