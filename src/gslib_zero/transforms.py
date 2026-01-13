"""
Gaussian anamorphosis transforms: nscore and backtr wrappers.

- nscore: Normal score transform (forward)
- backtr: Back-transform (reverse)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from gslib_zero.core import AsciiIO, BinaryIO, GSLIBWorkspace, run_gslib
from gslib_zero.par import ParFileBuilder

if TYPE_CHECKING:
    from numpy.typing import NDArray


def nscore(
    data: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
    tmin: float = -1.0e21,
    tmax: float = 1.0e21,
    binary: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Apply normal score transform to data.

    Transforms data to a standard normal distribution using the quantile
    transform method.

    Args:
        data: Input data values (1D array)
        weights: Optional declustering weights (same length as data).
                 If None, uniform weights are used.
        tmin: Minimum trimming limit (values below are excluded)
        tmax: Maximum trimming limit (values above are excluded)
        binary: If True, use binary I/O (requires gslib-zero modified binaries)

    Returns:
        Tuple of (transformed_data, transform_table) where:
        - transformed_data: Normal scores (same length as input)
        - transform_table: 2-column array (original_value, normal_score)
                          for use with backtr
    """
    data = np.asarray(data, dtype=np.float64).ravel()
    n = len(data)

    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).ravel()
        if len(weights) != n:
            raise ValueError(f"weights length ({len(weights)}) must match data length ({n})")

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
            {"value": data, "weight": weights},
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
    data: NDArray[np.floating],
    transform_table: NDArray[np.floating],
    tmin: float = -1.0e21,
    tmax: float = 1.0e21,
    zmin: float | None = None,
    zmax: float | None = None,
    ltail: int = 1,
    ltpar: float = 1.0,
    utail: int = 1,
    utpar: float = 1.0,
    binary: bool = False,
) -> NDArray[np.float64]:
    """
    Back-transform normal scores to original distribution.

    Args:
        data: Normal score values to back-transform (1D array)
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

    Returns:
        Back-transformed data in original units
    """
    data = np.asarray(data, dtype=np.float64).ravel()
    transform_table = np.asarray(transform_table, dtype=np.float64)

    # Get min/max from table if not specified
    if zmin is None:
        zmin = float(transform_table[:, 0].min())
    if zmax is None:
        zmax = float(transform_table[:, 0].max())

    with GSLIBWorkspace() as workspace:
        data_file = workspace / "backtr_input.dat"
        table_file = workspace / "backtr_table.trn"
        output_file = workspace / "backtr_output.dat"
        par_file = workspace / "backtr.par"

        # Write input data
        AsciiIO.write_column(data_file, data, name="nscore", title="backtr input")

        # Write transform table (raw format - same as nscore output)
        if transform_table.ndim == 1 or transform_table.shape[1] < 2:
            raise ValueError("transform_table must have at least 2 columns")
        AsciiIO.write_raw(table_file, transform_table[:, :2])

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
