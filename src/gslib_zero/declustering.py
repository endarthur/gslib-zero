"""
Cell declustering wrapper: declus.

Computes declustering weights to correct for preferential sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from gslib_zero.core import AsciiIO, BinaryIO, GSLIBWorkspace, run_gslib, _extract_points
from gslib_zero.par import ParFileBuilder

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class DeclusResult:
    """Result from declustering."""

    weights: NDArray[np.float64]
    declustered_mean: float
    optimal_cell_size: float
    summary: NDArray[np.float64]  # Cell size vs declustered mean


def declus(
    x=None,
    y=None,
    z=None,
    values=None,
    cell_min: float = None,
    cell_max: float = None,
    n_cells: int = 10,
    anisotropy_y: float = 1.0,
    anisotropy_z: float = 1.0,
    min_max: int = 0,
    n_offsets: int = 8,
    tmin: float = -1.0e21,
    tmax: float = 1.0e21,
    binary: bool = False,
    *,
    data=None,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    value_col: str | None = None,
) -> DeclusResult:
    """
    Cell declustering to compute sample weights.

    Accepts either direct arrays/Series or a DataFrame with column names.

    Args:
        x, y, z: Sample coordinates (array-like: ndarray, Series, list)
        values: Sample values (array-like)
        cell_min: Minimum cell size to test
        cell_max: Maximum cell size to test
        n_cells: Number of cell sizes to test between min and max
        anisotropy_y: Y cell anisotropy (Ysize = size * anisotropy_y)
        anisotropy_z: Z cell anisotropy (Zsize = size * anisotropy_z)
        min_max: 0=minimize declustered mean, 1=maximize declustered mean
        n_offsets: Number of origin offsets to test
        tmin: Minimum trimming limit
        tmax: Maximum trimming limit
        binary: If True, use binary I/O (requires gslib-zero modified binaries)
        data: DataFrame or dict with columns (alternative to x, y, z, values)
        x_col, y_col, z_col: Column names for coordinates (default 'x', 'y', 'z')
        value_col: Column name for values (required when using data)

    Returns:
        DeclusResult with weights, declustered mean, and optimal cell size

    Examples:
        # Using arrays or Series directly
        result = declus(df.x, df.y, df.z, df.au, cell_min=50, cell_max=200)

        # Using DataFrame with column names
        result = declus(data=df, value_col='au', cell_min=50, cell_max=200)
    """
    # Extract and validate inputs
    x, y, z, values = _extract_points(x, y, z, values, data, x_col, y_col, z_col, value_col)
    n = len(x)

    with GSLIBWorkspace() as workspace:
        data_file = workspace / "declus_input.dat"
        summary_file = workspace / "declus_summary.out"
        output_file = workspace / "declus_output.dat"
        par_file = workspace / "declus.par"

        # Write input data
        AsciiIO.write_data(
            data_file,
            {"x": x, "y": y, "z": z, "value": values},
            title="declus input data",
        )

        # Build par file - see declus.for lines 129-161
        par = ParFileBuilder()
        par.comment("Parameters for DECLUS")
        par.comment("*********************")
        par.blank()
        par.line("START OF PARAMETERS:")
        par.line("declus_input.dat", comment="file with data")
        par.line(1, 2, 3, 4, comment="columns for X, Y, Z, variable")
        par.line(tmin, tmax, comment="trimming limits")
        par.line("declus_summary.out", comment="file for summary output")
        par.line("declus_output.dat", comment="file for output with weights")
        par.line(1 if binary else 0, comment="binary output (0=ASCII, 1=binary)")
        par.line(anisotropy_y, anisotropy_z, comment="Y and Z anisotropy")
        par.line(min_max, comment="0=minimize mean, 1=maximize mean")
        par.line(n_cells, cell_min, cell_max, comment="ncell, cmin, cmax")
        par.line(n_offsets, comment="number of origin offsets")
        par.write(par_file)

        # Run declus
        run_gslib("declus", par_file)

        # Read output weights
        if binary:
            # Binary output: header [ndim=1, n] + float32 weights
            with open(output_file, "rb") as f:
                ndim = np.fromfile(f, dtype=np.int32, count=1)[0]
                shape = tuple(np.fromfile(f, dtype=np.int32, count=ndim))
                weights = np.fromfile(f, dtype=np.float32).astype(np.float64)
        else:
            # ASCII output - has columns: x, y, z, value, weight
            names, output_data = AsciiIO.read_data(output_file)

            # Find the weight column (usually last or contains 'wt')
            wt_col = -1  # Default to last
            for i, name in enumerate(names):
                if "wt" in name.lower() or "weight" in name.lower():
                    wt_col = i
                    break
            weights = output_data[:, wt_col]

        # Read summary file to get optimal cell size and declustered mean
        # Summary format: GSLIB header with columns (cell_size, declustered_mean)
        _summary_names, summary_data = AsciiIO.read_data(summary_file)

        # Find optimal (min or max mean depending on min_max flag)
        if min_max == 0:
            opt_idx = np.argmin(summary_data[:, 1])
        else:
            opt_idx = np.argmax(summary_data[:, 1])

        optimal_cell = float(summary_data[opt_idx, 0])
        declus_mean = float(summary_data[opt_idx, 1])

    return DeclusResult(
        weights=weights,
        declustered_mean=declus_mean,
        optimal_cell_size=optimal_cell,
        summary=summary_data,
    )
