"""
Kriging estimation wrappers: kt3d and ik3d.

- kt3d: 3D kriging (simple, ordinary, with trend, cokriging, external drift)
- ik3d: Indicator kriging with order relations correction
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from gslib_zero.core import AsciiIO, GSLIBWorkspace, run_gslib
from gslib_zero.par import ParFileBuilder
from gslib_zero.utils import GridSpec, VariogramModel

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class KrigingResult:
    """Result from kriging estimation."""

    estimate: NDArray[np.float64]
    variance: NDArray[np.float64]
    grid: GridSpec


@dataclass
class SearchParameters:
    """Search neighborhood parameters for kriging."""

    radius1: float  # Maximum search radius
    radius2: float  # Medium search radius
    radius3: float  # Minimum search radius
    azimuth: float = 0.0  # Azimuth of radius1
    dip: float = 0.0  # Dip of radius1
    rake: float = 0.0  # Rake (rotation in plane of radius1/radius2)
    min_samples: int = 1
    max_samples: int = 32
    max_per_octant: int = 0  # 0 = no octant search


def kt3d(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    z: NDArray[np.floating],
    values: NDArray[np.floating],
    grid: GridSpec,
    variogram: VariogramModel,
    search: SearchParameters,
    kriging_type: Literal["simple", "ordinary"] = "ordinary",
    sk_mean: float = 0.0,
    tmin: float = -1.0e21,
    tmax: float = 1.0e21,
    block_discretization: tuple[int, int, int] = (1, 1, 1),
) -> KrigingResult:
    """
    3D kriging estimation using kt3d.

    Args:
        x, y, z: Sample coordinates (1D arrays)
        values: Sample values (1D array)
        grid: Grid specification for output
        variogram: Variogram model
        search: Search neighborhood parameters
        kriging_type: 'simple' or 'ordinary' kriging
        sk_mean: Mean for simple kriging (ignored for ordinary kriging)
        tmin, tmax: Trimming limits
        block_discretization: Number of discretization points (nx, ny, nz)
                             for block kriging. (1,1,1) = point kriging.

    Returns:
        KrigingResult with estimate, variance, and grid spec
    """
    # Validate inputs
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel()

    n = len(x)
    if not (len(y) == len(z) == len(values) == n):
        raise ValueError("All input arrays must have the same length")

    # Kriging type: 0=SK, 1=OK
    ktype = 0 if kriging_type == "simple" else 1

    with GSLIBWorkspace() as workspace:
        data_file = workspace / "kt3d_input.dat"
        output_file = workspace / "kt3d_output.out"
        debug_file = workspace / "kt3d_debug.dbg"
        par_file = workspace / "kt3d.par"

        # Write input data (dhid, x, y, z, value, extvar)
        # Use dummy drillhole ID and external variable
        dhid = np.arange(1, n + 1, dtype=np.float64)
        extvar = np.zeros(n, dtype=np.float64)

        AsciiIO.write_data(
            data_file,
            {"dhid": dhid, "x": x, "y": y, "z": z, "value": values, "extvar": extvar},
            title="kt3d input data",
        )

        # Build par file - see kt3d.for lines 147-283
        par = ParFileBuilder()
        par.comment("Parameters for KT3D")
        par.comment("*******************")
        par.blank()
        par.line("START OF PARAMETERS:")
        par.line("kt3d_input.dat", comment="data file")
        par.line(1, 2, 3, 4, 5, 0, comment="columns: dhid, x, y, z, var, sec var")
        par.line(tmin, tmax, comment="trimming limits")
        par.line(0, comment="kriging option: 0=grid")
        par.line("nofile.dat", comment="jackknife file (not used)")
        par.line(0, 0, 0, 0, 0, comment="jackknife columns (not used)")
        par.line(0, comment="debugging level (0=none)")
        par.line("kt3d_debug.dbg", comment="debug file")
        par.line("kt3d_output.out", comment="output file")
        par.line(grid.nx, grid.xmin, grid.xsiz, comment="nx, xmn, xsiz")
        par.line(grid.ny, grid.ymin, grid.ysiz, comment="ny, ymn, ysiz")
        par.line(grid.nz, grid.zmin, grid.zsiz, comment="nz, zmn, zsiz")
        par.line(block_discretization[0], block_discretization[1], block_discretization[2],
                comment="block discretization")
        par.line(search.min_samples, search.max_samples, comment="min, max data")
        par.line(search.max_per_octant, comment="max per octant")
        par.line(search.radius1, search.radius2, search.radius3, comment="search radii")
        par.line(search.azimuth, search.dip, search.rake, comment="search angles")
        par.line(ktype, sk_mean, comment="ktype (0=SK, 1=OK), SK mean")
        par.line(0, 0, 0, 0, 0, 0, 0, 0, 0, comment="drift terms (all 0)")
        par.line(0, comment="trend: 0=no trend")
        par.line("nofile.dat", comment="external drift file (not used)")
        par.line(0, comment="external drift column (not used)")

        # Variogram model
        nst = len(variogram.structures)
        par.line(nst, variogram.nugget, comment="nst, nugget")

        for struct in variogram.structures:
            vtype = struct["type"]
            sill = struct["sill"]
            angles = struct.get("angles", (0.0, 0.0, 0.0))
            ranges = struct["ranges"]
            par.line(vtype, sill, angles[0], angles[1], angles[2],
                    comment="type, sill, angles")
            par.line(ranges[0], ranges[1], ranges[2], comment="ranges")

        par.write(par_file)

        # Run kt3d
        run_gslib("kt3d", par_file)

        # Read results - output is gridded GSLIB format with estimate and variance
        names, output_data, _grid_info = AsciiIO.read_gridded_data(output_file)

        # First column is estimate, second is variance
        estimate = output_data[:, 0]
        variance = output_data[:, 1] if output_data.shape[1] > 1 else np.zeros_like(estimate)

        # Reshape to grid dimensions (GSLIB uses Fortran ordering: z fastest)
        shape = (grid.nz, grid.ny, grid.nx)
        estimate = estimate.reshape(shape, order="F")
        variance = variance.reshape(shape, order="F")

    return KrigingResult(estimate=estimate, variance=variance, grid=grid)


def ik3d(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    z: NDArray[np.floating],
    values: NDArray[np.floating],
    grid: GridSpec,
    cutoffs: list[float],
    variograms: list[VariogramModel],
    search: SearchParameters,
    tmin: float = -1.0e21,
    tmax: float = 1.0e21,
) -> NDArray[np.float64]:
    """
    Indicator kriging using ik3d.

    Args:
        x, y, z: Sample coordinates (1D arrays)
        values: Sample values (1D array)
        grid: Grid specification for output
        cutoffs: List of indicator cutoff values
        variograms: List of variogram models, one per cutoff
        search: Search neighborhood parameters
        tmin, tmax: Trimming limits

    Returns:
        Array of shape (ncut, nz, ny, nx) with indicator kriging estimates
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel()

    n = len(x)
    ncut = len(cutoffs)

    if len(variograms) != ncut:
        raise ValueError(f"Need {ncut} variograms for {ncut} cutoffs")

    with GSLIBWorkspace() as workspace:
        data_file = workspace / "ik3d_input.dat"
        output_file = workspace / "ik3d_output.out"
        debug_file = workspace / "ik3d_debug.dbg"
        par_file = workspace / "ik3d.par"

        # Write input data
        AsciiIO.write_data(
            data_file,
            {"x": x, "y": y, "z": z, "value": values},
            title="ik3d input data",
        )

        # Build par file - simplified version
        par = ParFileBuilder()
        par.comment("Parameters for IK3D")
        par.comment("*******************")
        par.blank()
        par.line("START OF PARAMETERS:")
        par.line("ik3d_input.dat", comment="data file")
        par.line(1, 2, 3, 4, comment="columns: x, y, z, var")
        par.line(tmin, tmax, comment="trimming limits")
        par.line(ncut, comment="number of cutoffs")
        for cut in cutoffs:
            par.line(cut)
        par.line(0, comment="debugging level")
        par.line("ik3d_debug.dbg", comment="debug file")
        par.line("ik3d_output.out", comment="output file")
        par.line(grid.nx, grid.xmin, grid.xsiz, comment="nx, xmn, xsiz")
        par.line(grid.ny, grid.ymin, grid.ysiz, comment="ny, ymn, ysiz")
        par.line(grid.nz, grid.zmin, grid.zsiz, comment="nz, zmn, zsiz")
        par.line(1, 1, 1, comment="block discretization")
        par.line(search.min_samples, search.max_samples, comment="min, max data")
        par.line(search.max_per_octant, comment="max per octant")
        par.line(search.radius1, search.radius2, search.radius3, comment="search radii")
        par.line(search.azimuth, search.dip, search.rake, comment="search angles")

        # Variogram models for each cutoff
        for i, vario in enumerate(variograms):
            nst = len(vario.structures)
            par.line(nst, vario.nugget, comment=f"nst, nugget (cutoff {i+1})")
            for struct in vario.structures:
                vtype = struct["type"]
                sill = struct["sill"]
                angles = struct.get("angles", (0.0, 0.0, 0.0))
                ranges = struct["ranges"]
                par.line(vtype, sill, angles[0], angles[1], angles[2])
                par.line(ranges[0], ranges[1], ranges[2])

        par.write(par_file)

        # Run ik3d
        run_gslib("ik3d", par_file)

        # Read results
        names, output_data = AsciiIO.read_data(output_file)

        # Reshape: one column per cutoff
        shape = (ncut, grid.nz, grid.ny, grid.nx)
        result = np.zeros(shape, dtype=np.float64)

        for i in range(min(ncut, output_data.shape[1])):
            result[i] = output_data[:, i].reshape((grid.nz, grid.ny, grid.nx), order="F")

    return result
