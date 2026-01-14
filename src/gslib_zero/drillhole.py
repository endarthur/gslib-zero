"""
Drillhole utilities for gslib-zero.

- desurvey: Minimum curvature desurvey
- composite: Length-weighted compositing
- merge_intervals: Interval table merging/intersection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Try to import pandas, but make it optional
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None  # type: ignore


def _to_arrays(data: dict | Any) -> dict[str, NDArray]:
    """Convert input data to dict of numpy arrays."""
    if HAS_PANDAS and isinstance(data, pd.DataFrame):
        return {col: data[col].to_numpy() for col in data.columns}
    elif isinstance(data, dict):
        return {k: np.asarray(v) for k, v in data.items()}
    else:
        raise TypeError(f"Expected dict or DataFrame, got {type(data)}")


def _to_output(data: dict[str, NDArray], input_was_dataframe: bool) -> dict | Any:
    """Convert dict of arrays back to original format."""
    if input_was_dataframe and HAS_PANDAS:
        return pd.DataFrame(data)
    return data


# ============================================================================
# Minimum Curvature Desurvey
# ============================================================================

def desurvey(
    collar: dict | Any,
    survey: dict | Any,
    depths: dict | Any | None = None,
    collar_cols: tuple[str, str, str, str] = ("holeid", "x", "y", "z"),
    survey_cols: tuple[str, str, str, str] = ("holeid", "depth", "azimuth", "dip"),
    depth_cols: tuple[str, str, str] | None = None,
) -> dict[str, NDArray] | Any:
    """
    Minimum curvature desurvey.

    Computes 3D coordinates along drillhole traces using the minimum
    curvature method, which assumes the drillhole follows a circular
    arc between survey stations.

    Args:
        collar: Collar data with hole ID and collar coordinates
        survey: Survey data with hole ID, depth, azimuth, and dip
        depths: Optional sample depths (holeid, from, to). If provided,
                returns coordinates at sample midpoints. If None, returns
                coordinates at survey stations.
        collar_cols: Column names for (holeid, x, y, z) in collar data
        survey_cols: Column names for (holeid, depth, azimuth, dip) in survey
        depth_cols: Column names for (holeid, from, to) in depths data.
                   Defaults to ("holeid", "from", "to")

    Returns:
        Dict or DataFrame with columns: holeid, depth, x, y, z
        (returns same type as input)

    Notes:
        - Azimuth is measured clockwise from North (0-360)
        - Dip is measured from horizontal: negative = down, positive = up
          (e.g., -90 = vertical down, 0 = horizontal, -45 = 45° below horizontal)
        - Survey at depth 0 is assumed to be at collar location
    """
    input_was_df = HAS_PANDAS and isinstance(collar, pd.DataFrame)

    if depth_cols is None:
        depth_cols = ("holeid", "from", "to")

    collar_data = _to_arrays(collar)
    survey_data = _to_arrays(survey)

    hid_col, x_col, y_col, z_col = collar_cols
    shid_col, sdepth_col, azm_col, dip_col = survey_cols

    # Get unique holes
    holes = np.unique(collar_data[hid_col])

    result_holeid = []
    result_depth = []
    result_x = []
    result_y = []
    result_z = []

    for hole in holes:
        # Get collar location
        collar_mask = collar_data[hid_col] == hole
        if not np.any(collar_mask):
            continue

        cx = float(collar_data[x_col][collar_mask][0])
        cy = float(collar_data[y_col][collar_mask][0])
        cz = float(collar_data[z_col][collar_mask][0])

        # Get surveys for this hole, sorted by depth
        survey_mask = survey_data[shid_col] == hole
        if not np.any(survey_mask):
            continue

        svey_depths = survey_data[sdepth_col][survey_mask]
        svey_azm = survey_data[azm_col][survey_mask]
        svey_dip = survey_data[dip_col][survey_mask]

        sort_idx = np.argsort(svey_depths)
        svey_depths = svey_depths[sort_idx]
        svey_azm = svey_azm[sort_idx]
        svey_dip = svey_dip[sort_idx]

        # Ensure survey at depth 0
        if svey_depths[0] > 0:
            svey_depths = np.concatenate([[0.0], svey_depths])
            svey_azm = np.concatenate([[svey_azm[0]], svey_azm])
            svey_dip = np.concatenate([[svey_dip[0]], svey_dip])

        # Compute coordinates at survey stations using minimum curvature
        n_surveys = len(svey_depths)
        station_x = np.zeros(n_surveys)
        station_y = np.zeros(n_surveys)
        station_z = np.zeros(n_surveys)
        station_x[0] = cx
        station_y[0] = cy
        station_z[0] = cz

        for i in range(1, n_surveys):
            d1, d2 = svey_depths[i - 1], svey_depths[i]
            a1, a2 = np.radians(svey_azm[i - 1]), np.radians(svey_azm[i])
            # Convert dip (from horizontal, negative=down) to inclination from vertical
            # dip = -90 (vertical down) → inclination = 0
            # dip = 0 (horizontal) → inclination = 90
            i1, i2 = np.radians(90.0 + svey_dip[i - 1]), np.radians(90.0 + svey_dip[i])

            dl = d2 - d1

            # Minimum curvature ratio factor
            cos_theta = np.cos(i2 - i1) - np.sin(i1) * np.sin(i2) * (1 - np.cos(a2 - a1))
            cos_theta = np.clip(cos_theta, -1, 1)
            theta = np.arccos(cos_theta)

            if abs(theta) < 1e-8:
                rf = 1.0
            else:
                rf = 2.0 / theta * np.tan(theta / 2.0)

            # Displacement components
            # North (positive Y)
            dn = 0.5 * dl * (np.sin(i1) * np.cos(a1) + np.sin(i2) * np.cos(a2)) * rf
            # East (positive X)
            de = 0.5 * dl * (np.sin(i1) * np.sin(a1) + np.sin(i2) * np.sin(a2)) * rf
            # Vertical (negative Z, since dip is positive downward)
            dv = 0.5 * dl * (np.cos(i1) + np.cos(i2)) * rf

            station_x[i] = station_x[i - 1] + de
            station_y[i] = station_y[i - 1] + dn
            station_z[i] = station_z[i - 1] - dv

        # Determine output depths
        if depths is not None:
            depths_data = _to_arrays(depths)
            dhid_col, dfrom_col, dto_col = depth_cols

            depth_mask = depths_data[dhid_col] == hole
            if not np.any(depth_mask):
                continue

            from_depths = depths_data[dfrom_col][depth_mask]
            to_depths = depths_data[dto_col][depth_mask]
            mid_depths = (from_depths + to_depths) / 2.0

            # Interpolate to midpoints
            out_x = np.interp(mid_depths, svey_depths, station_x)
            out_y = np.interp(mid_depths, svey_depths, station_y)
            out_z = np.interp(mid_depths, svey_depths, station_z)

            for d, x, y, z in zip(mid_depths, out_x, out_y, out_z):
                result_holeid.append(hole)
                result_depth.append(d)
                result_x.append(x)
                result_y.append(y)
                result_z.append(z)
        else:
            # Return survey station coordinates
            for d, x, y, z in zip(svey_depths, station_x, station_y, station_z):
                result_holeid.append(hole)
                result_depth.append(d)
                result_x.append(x)
                result_y.append(y)
                result_z.append(z)

    result = {
        "holeid": np.array(result_holeid),
        "depth": np.array(result_depth, dtype=np.float64),
        "x": np.array(result_x, dtype=np.float64),
        "y": np.array(result_y, dtype=np.float64),
        "z": np.array(result_z, dtype=np.float64),
    }

    return _to_output(result, input_was_df)


# ============================================================================
# Length-Weighted Compositing
# ============================================================================

@dataclass
class CompositeResult:
    """Result from compositing."""

    data: dict[str, NDArray] | Any  # Composite data
    lengths: NDArray[np.float64]  # Actual composite lengths
    coverage: NDArray[np.float64]  # Fraction of target length with data


def composite(
    data: dict | Any,
    length: float,
    min_coverage: float = 0.5,
    columns: list[str] | None = None,
    domain_column: str | None = None,
    holeid_col: str = "holeid",
    from_col: str = "from",
    to_col: str = "to",
) -> CompositeResult:
    """
    Length-weighted compositing.

    Creates fixed-length composites from interval data, aligned to hole
    start (from=0). Composites span domain boundaries are split at those
    boundaries if domain_column is specified.

    Args:
        data: Interval data with holeid, from, to, and value columns
        length: Target composite length
        min_coverage: Minimum fraction of composite length that must have
                     data (0-1). Composites below this are excluded.
        columns: Columns to composite. If None, composites all numeric columns.
        domain_column: Column defining domains. Composites break at domain
                      boundaries.
        holeid_col: Name of hole ID column
        from_col: Name of FROM column
        to_col: Name of TO column

    Returns:
        CompositeResult with data, lengths, and coverage arrays
    """
    input_was_df = HAS_PANDAS and isinstance(data, pd.DataFrame)
    arr_data = _to_arrays(data)

    holes = np.unique(arr_data[holeid_col])

    # Determine columns to composite
    if columns is None:
        columns = [
            col for col in arr_data.keys()
            if col not in (holeid_col, from_col, to_col, domain_column)
            and np.issubdtype(arr_data[col].dtype, np.number)
        ]

    # Initialize result containers
    result_holeid = []
    result_from = []
    result_to = []
    result_lengths = []
    result_coverage = []
    result_domain = [] if domain_column else None
    result_values = {col: [] for col in columns}

    for hole in holes:
        mask = arr_data[holeid_col] == hole
        hole_from = arr_data[from_col][mask]
        hole_to = arr_data[to_col][mask]

        # Sort by from
        sort_idx = np.argsort(hole_from)
        hole_from = hole_from[sort_idx]
        hole_to = hole_to[sort_idx]

        hole_values = {col: arr_data[col][mask][sort_idx] for col in columns}

        if domain_column:
            hole_domain = arr_data[domain_column][mask][sort_idx]
        else:
            hole_domain = None

        # Find hole extent
        hole_start = 0.0  # Composites aligned to hole start
        hole_end = hole_to.max()

        # Generate composite intervals
        comp_start = hole_start
        while comp_start < hole_end:
            comp_end = comp_start + length

            # If domain column, find domain at composite start
            current_domain = None
            if hole_domain is not None:
                # Find interval containing comp_start
                for i, (f, t, d) in enumerate(zip(hole_from, hole_to, hole_domain)):
                    if f <= comp_start < t:
                        current_domain = d
                        # Check for domain boundary within composite
                        for j in range(i, len(hole_from)):
                            if hole_domain[j] != current_domain and hole_from[j] < comp_end:
                                comp_end = hole_from[j]
                                break
                        break

            # Calculate composite values
            total_length = 0.0
            weighted_sums = {col: 0.0 for col in columns}

            for i in range(len(hole_from)):
                # Interval overlap with composite
                overlap_start = max(hole_from[i], comp_start)
                overlap_end = min(hole_to[i], comp_end)

                if overlap_end <= overlap_start:
                    continue

                # Domain check
                if hole_domain is not None and hole_domain[i] != current_domain:
                    continue

                overlap_length = overlap_end - overlap_start
                total_length += overlap_length

                for col in columns:
                    val = hole_values[col][i]
                    if not np.isnan(val):
                        weighted_sums[col] += val * overlap_length

            # Calculate coverage
            target_length = comp_end - comp_start
            coverage = total_length / target_length if target_length > 0 else 0.0

            if coverage >= min_coverage and total_length > 0:
                result_holeid.append(hole)
                result_from.append(comp_start)
                result_to.append(comp_end)
                result_lengths.append(total_length)
                result_coverage.append(coverage)

                if result_domain is not None:
                    result_domain.append(current_domain)

                for col in columns:
                    if total_length > 0:
                        result_values[col].append(weighted_sums[col] / total_length)
                    else:
                        result_values[col].append(np.nan)

            comp_start = comp_end

    # Build result dict
    result_data = {
        holeid_col: np.array(result_holeid),
        from_col: np.array(result_from, dtype=np.float64),
        to_col: np.array(result_to, dtype=np.float64),
    }

    if domain_column and result_domain:
        result_data[domain_column] = np.array(result_domain)

    for col in columns:
        result_data[col] = np.array(result_values[col], dtype=np.float64)

    return CompositeResult(
        data=_to_output(result_data, input_was_df),
        lengths=np.array(result_lengths, dtype=np.float64),
        coverage=np.array(result_coverage, dtype=np.float64),
    )


# ============================================================================
# Interval Merging (Table Join)
# ============================================================================

def merge_intervals(
    *tables: dict | Any,
    holeid_col: str = "holeid",
    from_col: str = "from",
    to_col: str = "to",
    tolerance: float = 0.001,
) -> dict[str, NDArray] | Any:
    """
    Merge multiple interval tables into one with combined attributes.

    Creates a new table where intervals are split at all boundaries from
    all input tables. Each output interval gets attribute values from
    whichever input table covers that interval.

    Example:
        Table 1 (Assay):  0-1 (Au=2.5), 1-3 (Au=1.2)
        Table 2 (Geology): 0-2 (rock=QZ), 2-3 (rock=GR)

        Merged:  0-1 (Au=2.5, rock=QZ)
                 1-2 (Au=1.2, rock=QZ)
                 2-3 (Au=1.2, rock=GR)

    Args:
        *tables: Two or more interval tables to merge
        holeid_col: Name of hole ID column (must be same in all tables)
        from_col: Name of FROM column
        to_col: Name of TO column
        tolerance: Tolerance for matching boundaries

    Returns:
        Merged interval table with all columns from all input tables.
        Missing data results in NaN for numeric columns, None for others.
    """
    if len(tables) < 2:
        raise ValueError("Need at least 2 tables to merge")

    input_was_df = HAS_PANDAS and isinstance(tables[0], pd.DataFrame)

    # Convert all tables to arrays
    arr_tables = [_to_arrays(t) for t in tables]

    # Get all unique holes across all tables
    all_holes = set()
    for arr in arr_tables:
        all_holes.update(np.unique(arr[holeid_col]))

    # Get all value columns (excluding holeid, from, to)
    all_columns = {}  # col_name -> (table_idx, dtype)
    for i, arr in enumerate(arr_tables):
        for col in arr.keys():
            if col not in (holeid_col, from_col, to_col):
                if col not in all_columns:
                    all_columns[col] = (i, arr[col].dtype)
                else:
                    # Column exists in multiple tables - use first occurrence
                    pass

    # Initialize result containers
    result_holeid = []
    result_from = []
    result_to = []
    result_values = {col: [] for col in all_columns.keys()}

    for hole in sorted(all_holes):
        # Collect all boundaries for this hole from all tables
        boundaries = set([0.0])

        table_intervals = []  # List of (from_arr, to_arr, values_dict) per table
        for arr in arr_tables:
            mask = arr[holeid_col] == hole
            if not np.any(mask):
                table_intervals.append((np.array([]), np.array([]), {}))
                continue

            froms = arr[from_col][mask]
            tos = arr[to_col][mask]

            boundaries.update(froms)
            boundaries.update(tos)

            values = {}
            for col in arr.keys():
                if col not in (holeid_col, from_col, to_col):
                    values[col] = arr[col][mask]

            table_intervals.append((froms, tos, values))

        # Sort boundaries
        boundaries = sorted(boundaries)

        # Create merged intervals
        for i in range(len(boundaries) - 1):
            int_from = boundaries[i]
            int_to = boundaries[i + 1]
            int_mid = (int_from + int_to) / 2.0

            # Skip very small intervals
            if int_to - int_from < tolerance:
                continue

            result_holeid.append(hole)
            result_from.append(int_from)
            result_to.append(int_to)

            # Look up values from each table
            for col, (table_idx, dtype) in all_columns.items():
                froms, tos, values = table_intervals[table_idx]

                if col not in values or len(froms) == 0:
                    # Column not in this table or no data for hole
                    if np.issubdtype(dtype, np.number):
                        result_values[col].append(np.nan)
                    else:
                        result_values[col].append(None)
                    continue

                # Find interval containing midpoint
                found = False
                for j in range(len(froms)):
                    if froms[j] <= int_mid < tos[j]:
                        result_values[col].append(values[col][j])
                        found = True
                        break

                if not found:
                    if np.issubdtype(dtype, np.number):
                        result_values[col].append(np.nan)
                    else:
                        result_values[col].append(None)

    # Build result
    result = {
        holeid_col: np.array(result_holeid),
        from_col: np.array(result_from, dtype=np.float64),
        to_col: np.array(result_to, dtype=np.float64),
    }

    for col, (_, dtype) in all_columns.items():
        if np.issubdtype(dtype, np.number):
            result[col] = np.array(result_values[col], dtype=np.float64)
        else:
            result[col] = np.array(result_values[col], dtype=object)

    return _to_output(result, input_was_df)
