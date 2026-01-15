"""
Gaussian anamorphosis transforms: nscore and backtr wrappers.

- nscore: Normal score transform (forward)
- backtr: Back-transform (reverse)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gslib_zero.core import AsciiIO, BinaryIO, GSLIBWorkspace, run_gslib
from gslib_zero.par import ParFileBuilder

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def _coerce_array(arr: "ArrayLike", dtype: type = np.float64) -> "NDArray":
    """Coerce array-like input to numpy array."""
    return np.asarray(arr, dtype=dtype)


def nscore(
    values: "ArrayLike | None" = None,
    weights: "ArrayLike | None" = None,
    tmin: float = -1.0e21,
    tmax: float = 1.0e21,
    binary: bool = False,
    *,
    data: Any | None = None,
    value_col: str = "value",
    weight_col: str | None = None,
) -> tuple["NDArray[np.float64]", "NDArray[np.float64]"]:
    """
    Apply normal score transform to data.

    Transforms data to a standard normal distribution using the quantile
    transform method.

    Two input patterns are supported:

    1. Direct arrays/Series::

        nscore(df.grade, weights=df.wt, binary=True)
        nscore(grades_array, binary=True)

    2. DataFrame with column names::

        nscore(data=df, value_col='grade', weight_col='wt', binary=True)

    Args:
        values: Input data values (1D array-like). Can be numpy array,
                pandas Series, or list. Required unless using data= parameter.
        weights: Optional declustering weights (same length as values).
                 If None, uniform weights are used.
        tmin: Minimum trimming limit (values below are excluded)
        tmax: Maximum trimming limit (values above are excluded)
        binary: If True, use binary I/O (requires gslib-zero modified binaries)
        data: DataFrame or dict containing the data columns. If provided,
              value_col specifies which column to transform.
        value_col: Column name for values when using data= parameter.
        weight_col: Column name for weights when using data= parameter.
                   If None, uniform weights are used.

    Returns:
        Tuple of (transformed_data, transform_table) where:
        - transformed_data: Normal scores (same length as input)
        - transform_table: 2-column array (original_value, normal_score)
                          for use with backtr
    """
    # Handle DataFrame vs direct array input
    if data is not None:
        values_arr = _coerce_array(data[value_col])
        if weight_col is not None:
            weights_arr = _coerce_array(data[weight_col])
        else:
            weights_arr = None
    else:
        if values is None:
            raise ValueError("values is required when not using data= parameter")
        values_arr = _coerce_array(values)
        weights_arr = _coerce_array(weights) if weights is not None else None

    values_arr = values_arr.ravel()
    n = len(values_arr)

    if weights_arr is None:
        weights_arr = np.ones(n, dtype=np.float64)
    else:
        weights_arr = weights_arr.ravel()
        if len(weights_arr) != n:
            raise ValueError(f"weights length ({len(weights_arr)}) must match values length ({n})")

    with GSLIBWorkspace() as workspace:
        # Write input data in GSLIB ASCII format
        # Use just filenames since GSLIB runs from the workspace directory
        data_file = workspace / "nscore_input.dat"
        output_file = workspace / "nscore_output.dat"
        transform_file = workspace / "nscore_transform.trn"
        par_file = workspace / "nscore.par"

        # Write data with value and weight columns
        AsciiIO.write_data(
            data_file,
            {"value": values_arr, "weight": weights_arr},
            title="nscore input data",
        )

        # Build par file matching original GSLIB nscore format
        # Use just filenames (not full paths) since GSLIB runs from workspace
        par = ParFileBuilder()
        par.comment("Parameters for NSCORE")
        par.comment("*********************")
        par.blank()
        par.line("START OF PARAMETERS:")
        par.line("nscore_input.dat", comment="file with data")
        par.line(1, 2, comment="columns for variable and weight")
        par.line(tmin, tmax, comment="trimming limits")
        par.line(0, comment="0=transform according to specified ref, 1=data")
        par.line("nscore_input.dat", comment="file with reference distribution (not used)")
        par.line(1, 2, comment="columns for variable and weight (not used)")
        par.line("nscore_output.dat", comment="file for output")
        par.line(1 if binary else 0, comment="binary output (0=ASCII, 1=binary)")
        par.line("nscore_transform.trn", comment="file for transformation table")
        par.write(par_file)

        # Run nscore
        run_gslib("nscore", par_file)

        # Read results
        if binary:
            # Binary output: header [ndim=1, n] + float32 values
            with open(output_file, "rb") as f:
                ndim = np.fromfile(f, dtype=np.int32, count=1)[0]
                shape = tuple(np.fromfile(f, dtype=np.int32, count=ndim))
                transformed = np.fromfile(f, dtype=np.float32).astype(np.float64)
        else:
            # ASCII output: GSLIB format with columns including nscore
            names, output_data = AsciiIO.read_data(output_file)

            # Find the nscore column (usually last, may have prefix like "NS:")
            ns_col = -1  # Default to last column
            for i, name in enumerate(names):
                if "nscore" in name.lower() or name.startswith("NS:"):
                    ns_col = i
                    break
            transformed = output_data[:, ns_col]

        # Read transform table (raw format: value, nscore - no header)
        # Transform table is always ASCII
        table_data = AsciiIO.read_raw(transform_file)

    return transformed, table_data


def backtr(
    values: "ArrayLike | None" = None,
    transform_table: "ArrayLike | None" = None,
    tmin: float = -1.0e21,
    tmax: float = 1.0e21,
    zmin: float | None = None,
    zmax: float | None = None,
    ltail: int = 1,
    ltpar: float = 1.0,
    utail: int = 1,
    utpar: float = 1.0,
    binary: bool = False,
    *,
    data: Any | None = None,
    value_col: str = "nscore",
) -> "NDArray[np.float64]":
    """
    Back-transform normal scores to original distribution.

    Two input patterns are supported:

    1. Direct arrays/Series::

        backtr(df.nscore, transform_table, binary=True)
        backtr(nscore_array, transform_table, binary=True)

    2. DataFrame with column names::

        backtr(data=df, value_col='nscore', transform_table=transform_table, binary=True)

    Args:
        values: Normal score values to back-transform (1D array-like).
                Can be numpy array, pandas Series, or list.
                Required unless using data= parameter.
        transform_table: Transform table from nscore (2 columns: value, nscore)
        tmin: Minimum trimming limit for input
        tmax: Maximum trimming limit for input
        zmin: Minimum allowable output value (default: min of transform table)
        zmax: Maximum allowable output value (default: max of transform table)
        ltail: Lower tail extrapolation option (1=linear, 2=power)
        ltpar: Lower tail parameter
        utail: Upper tail extrapolation option (1=linear, 2=power, 4=hyperbolic)
        utpar: Upper tail parameter
        binary: If True, use binary I/O (requires gslib-zero modified binaries)
        data: DataFrame or dict containing the data columns. If provided,
              value_col specifies which column to transform.
        value_col: Column name for values when using data= parameter.

    Returns:
        Back-transformed data in original units
    """
    # Handle DataFrame vs direct array input
    if data is not None:
        values_arr = _coerce_array(data[value_col]).ravel()
    else:
        if values is None:
            raise ValueError("values is required when not using data= parameter")
        values_arr = _coerce_array(values).ravel()

    if transform_table is None:
        raise ValueError("transform_table is required")
    transform_table_arr = np.asarray(transform_table, dtype=np.float64)

    # Get min/max from table if not specified
    if zmin is None:
        zmin = float(transform_table_arr[:, 0].min())
    if zmax is None:
        zmax = float(transform_table_arr[:, 0].max())

    with GSLIBWorkspace() as workspace:
        data_file = workspace / "backtr_input.dat"
        table_file = workspace / "backtr_table.trn"
        output_file = workspace / "backtr_output.dat"
        par_file = workspace / "backtr.par"

        # Write input data
        AsciiIO.write_column(data_file, values_arr, name="nscore", title="backtr input")

        # Write transform table (raw format - same as nscore output)
        if transform_table_arr.ndim == 1 or transform_table_arr.shape[1] < 2:
            raise ValueError("transform_table must have at least 2 columns")
        AsciiIO.write_raw(table_file, transform_table_arr[:, :2])

        # Build par file matching original GSLIB backtr format
        # See backtr.for lines 102-132 for expected format
        par = ParFileBuilder()
        par.comment("Parameters for BACKTR")
        par.comment("*********************")
        par.blank()
        par.line("START OF PARAMETERS:")
        par.line("backtr_input.dat", comment="file with data to back-transform")
        par.line(1, comment="column for variable")
        par.line(tmin, tmax, comment="trimming limits")
        par.line("backtr_output.dat", comment="file for output")
        par.line(1 if binary else 0, comment="binary output (0=ASCII, 1=binary)")
        par.line("backtr_table.trn", comment="file with transformation table")
        par.line(zmin, zmax, comment="minimum and maximum Z values")
        par.line(ltail, ltpar, comment="lower tail option and parameter")
        par.line(utail, utpar, comment="upper tail option and parameter")
        par.write(par_file)

        # Run backtr
        run_gslib("backtr", par_file)

        # Read results
        if binary:
            # Binary output: header [ndim=1, n] + float32 values
            with open(output_file, "rb") as f:
                ndim = np.fromfile(f, dtype=np.int32, count=1)[0]
                shape = tuple(np.fromfile(f, dtype=np.int32, count=ndim))
                result = np.fromfile(f, dtype=np.float32).astype(np.float64)
        else:
            # ASCII output: has 2 columns (original nscore, back-transformed)
            # We want the second column (index 1)
            result = AsciiIO.read_column(output_file, column=1)

    return result
