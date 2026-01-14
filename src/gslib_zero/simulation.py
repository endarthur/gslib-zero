"""
Sequential simulation wrappers: sgsim and sisim.

- sgsim: Sequential Gaussian simulation
- sisim: Sequential indicator simulation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from gslib_zero.core import AsciiIO, BinaryIO, GSLIBWorkspace, run_gslib
from gslib_zero.par import ParFileBuilder
from gslib_zero.utils import GridSpec, VariogramModel

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class SimulationResult:
    """Result from simulation."""

    realizations: NDArray[np.float64]  # Shape: (nreal, nz, ny, nx)
    grid: GridSpec


def sgsim(
    x: NDArray[np.floating] | None,
    y: NDArray[np.floating] | None,
    z: NDArray[np.floating] | None,
    values: NDArray[np.floating] | None,
    grid: GridSpec,
    variogram: VariogramModel,
    search_radius: tuple[float, float, float],
    nrealizations: int = 1,
    seed: int = 69069,
    max_conditioning: int = 32,
    min_conditioning: int = 0,
    max_previously_simulated: int = 12,
    kriging_type: str = "simple",
    sk_mean: float = 0.0,
    tmin: float = -1.0e21,
    tmax: float = 1.0e21,
    search_angles: tuple[float, float, float] = (0.0, 0.0, 0.0),
    binary: bool = False,
    mask: NDArray[np.integer] | None = None,
) -> SimulationResult:
    """
    Sequential Gaussian simulation using sgsim.

    Note: Input values should already be normal scores. Use nscore() first
    to transform your data if needed.

    Args:
        x, y, z: Conditioning data coordinates (None for unconditional)
        values: Conditioning data values (should be normal scores)
        grid: Grid specification for output
        variogram: Variogram model (for normal scores)
        search_radius: Search radii (r1, r2, r3) - maximum, medium, minimum
        nrealizations: Number of realizations to generate
        seed: Random number seed
        max_conditioning: Maximum conditioning data per node
        min_conditioning: Minimum conditioning data per node
        max_previously_simulated: Maximum previously simulated nodes to use
        kriging_type: 'simple' or 'ordinary'
        sk_mean: Mean for simple kriging (usually 0 for normal scores)
        tmin, tmax: Trimming limits
        search_angles: Search anisotropy angles (azm, dip, rake)
        binary: If True, use binary I/O (faster for large grids)
        mask: Grid mask array (0=skip, 1=simulate). Shape must match grid (nz, ny, nx).
              Masked cells output UNEST (-999).

    Returns:
        SimulationResult with realizations array
    """
    # Handle unconditional case
    if x is None or values is None:
        n = 0
        x = np.array([], dtype=np.float64)
        y = np.array([], dtype=np.float64)
        z = np.array([], dtype=np.float64)
        values = np.array([], dtype=np.float64)
        has_data = False
    else:
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        z = np.asarray(z, dtype=np.float64).ravel()
        values = np.asarray(values, dtype=np.float64).ravel()
        n = len(x)
        has_data = True

    # Kriging type: 0=SK, 1=OK
    ktype = 0 if kriging_type == "simple" else 1

    with GSLIBWorkspace() as workspace:
        data_file = workspace / "sgsim_input.dat"
        output_file = workspace / "sgsim_output.out"
        debug_file = workspace / "sgsim_debug.dbg"
        par_file = workspace / "sgsim.par"
        mask_file = workspace / "sgsim_mask.bin"

        # Write mask file if provided
        if mask is not None:
            mask_arr = np.asarray(mask, dtype=np.int8)
            expected_shape = (grid.nz, grid.ny, grid.nx)
            if mask_arr.shape != expected_shape:
                raise ValueError(f"Mask shape {mask_arr.shape} doesn't match grid {expected_shape}")
            with open(mask_file, "wb") as f:
                np.array([3, grid.nz, grid.ny, grid.nx], dtype=np.int32).tofile(f)
                mask_arr.ravel(order="F").tofile(f)

        # Write input data if we have conditioning data
        if has_data:
            weights = np.ones(n, dtype=np.float64)
            secondary = np.zeros(n, dtype=np.float64)
            AsciiIO.write_data(
                data_file,
                {"x": x, "y": y, "z": z, "value": values, "weight": weights, "secondary": secondary},
                title="sgsim conditioning data",
            )

        # Build par file - see sgsim.for lines 155-304
        par = ParFileBuilder()
        par.comment("Parameters for SGSIM")
        par.comment("********************")
        par.blank()
        par.line("START OF PARAMETERS:")

        if has_data:
            par.line("sgsim_input.dat", comment="data file")
            par.line(1, 2, 3, 4, 5, 0, comment="columns: x, y, z, var, wt, sec var")
        else:
            par.line("nodata.dat", comment="data file (no conditioning)")
            par.line(0, 0, 0, 0, 0, 0, comment="columns (not used)")

        par.line(tmin, tmax, comment="trimming limits")
        par.line(0, comment="transform flag (0=no transform, data already NS)")
        par.line("notransform.trn", comment="transform file (not used)")
        par.line(0, comment="use smooth distribution (0=no)")
        par.line("nosmooth.dat", comment="smooth file (not used)")
        par.line(0, 0, comment="smooth columns (not used)")

        # Data limits for back-transform (not used since we output NS)
        par.line(-5.0, 5.0, comment="data limits for tails")
        par.line(1, -5.0, comment="lower tail option, parameter")
        par.line(1, 5.0, comment="upper tail option, parameter")

        par.line(0, comment="debugging level")
        par.line("sgsim_debug.dbg", comment="debug file")
        par.line("sgsim_output.out", comment="output file")
        par.line(1 if binary else 0, comment="binary output (0=ASCII, 1=binary)")

        # Grid mask
        if mask is not None:
            par.line(1, comment="use grid mask (0=no, 1=yes)")
            par.line("sgsim_mask.bin", comment="mask file")
        else:
            par.line(0, comment="use grid mask (0=no, 1=yes)")

        par.line(nrealizations, comment="number of realizations")
        par.line(grid.nx, grid.xmin, grid.xsiz, comment="nx, xmn, xsiz")
        par.line(grid.ny, grid.ymin, grid.ysiz, comment="ny, ymn, ysiz")
        par.line(grid.nz, grid.zmin, grid.zsiz, comment="nz, zmn, zsiz")
        par.line(seed, comment="random number seed")

        par.line(min_conditioning, max_conditioning, comment="min, max conditioning data")
        par.line(max_previously_simulated, comment="max previously simulated nodes")
        par.line(0, comment="two-part search (0=no)")
        par.line(0, 0, comment="multiple grid search (0=no)")
        par.line(0, comment="max per octant (0=no octant search)")

        par.line(search_radius[0], search_radius[1], search_radius[2], comment="search radii")
        par.line(search_angles[0], search_angles[1], search_angles[2], comment="search angles")

        # Covariance lookup table size
        cov_size = max(grid.nx // 4, 1), max(grid.ny // 4, 1), max(grid.nz // 4, 1)
        par.line(cov_size[0], cov_size[1], cov_size[2], comment="covariance lookup size")

        par.line(ktype, sk_mean, comment="ktype, SK mean")
        par.line("nolvm.dat", comment="LVM file (not used)")
        par.line(0, comment="LVM column (not used)")

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

        # Run sgsim
        run_gslib("sgsim", par_file)

        # Read results
        if binary:
            # Binary output: header is [ndim=4, nsim, nz, ny, nx]
            # Data is sequential: all cells for sim1, then sim2, etc.
            # Within each sim, cells are in Fortran order (x fastest)
            # GSLIB uses single precision (float32)
            with open(output_file, "rb") as f:
                # Read header
                ndim = np.fromfile(f, dtype=np.int32, count=1)[0]
                shape = tuple(np.fromfile(f, dtype=np.int32, count=ndim))
                nreal_actual = shape[0]  # First dim is nsim
                # Read flat data
                values_flat = np.fromfile(f, dtype=np.float32)

            ncells = grid.nx * grid.ny * grid.nz
            result = np.zeros((nreal_actual, grid.nz, grid.ny, grid.nx), dtype=np.float64)
            for i in range(nreal_actual):
                start = i * ncells
                end = start + ncells
                result[i] = values_flat[start:end].reshape(
                    (grid.nz, grid.ny, grid.nx), order="F"
                )
        else:
            # ASCII output: gridded format with all realizations concatenated
            names, output_data, _grid_info = AsciiIO.read_gridded_data(output_file)

            # Reshape to (nreal, nz, ny, nx)
            # Output data is (ncells * nrealizations, 1) - realizations are stacked sequentially
            ncells = grid.nx * grid.ny * grid.nz
            total_values = output_data.shape[0]
            nreal_actual = total_values // ncells

            result = np.zeros((nreal_actual, grid.nz, grid.ny, grid.nx), dtype=np.float64)
            values_arr = output_data[:, 0] if output_data.ndim > 1 else output_data.ravel()
            for i in range(nreal_actual):
                start = i * ncells
                end = start + ncells
                result[i] = values_arr[start:end].reshape((grid.nz, grid.ny, grid.nx), order="F")

    return SimulationResult(realizations=result, grid=grid)


def sisim(
    x: NDArray[np.floating] | None,
    y: NDArray[np.floating] | None,
    z: NDArray[np.floating] | None,
    values: NDArray[np.floating] | None,
    grid: GridSpec,
    thresholds: list[float],
    global_cdf: list[float],
    variograms: list[VariogramModel],
    search_radius: tuple[float, float, float],
    nrealizations: int = 1,
    seed: int = 69069,
    max_conditioning: int = 12,
    max_previously_simulated: int = 12,
    tmin: float = -1.0e21,
    tmax: float = 1.0e21,
    zmin: float = 0.0,
    zmax: float = 30.0,
    search_angles: tuple[float, float, float] = (0.0, 0.0, 0.0),
    kriging_type: str = "simple",
    binary: bool = False,
    mask: NDArray[np.integer] | None = None,
) -> SimulationResult:
    """
    Sequential indicator simulation using sisim.

    Args:
        x, y, z: Conditioning data coordinates (None for unconditional)
        values: Conditioning data values (continuous variable)
        grid: Grid specification for output
        thresholds: List of indicator thresholds/cutoff values
        global_cdf: Global CDF values at each threshold (cumulative probabilities)
        variograms: Variogram model for each indicator threshold
        search_radius: Search radii (r1, r2, r3) - maximum, medium, minimum
        nrealizations: Number of realizations to generate
        seed: Random number seed
        max_conditioning: Maximum conditioning data per node
        max_previously_simulated: Maximum previously simulated nodes to use
        tmin, tmax: Trimming limits
        zmin, zmax: Minimum and maximum data values (for tail extrapolation)
        search_angles: Search anisotropy angles (azm, dip, rake)
        kriging_type: 'simple' or 'ordinary'
        binary: If True, use binary I/O (faster for large grids)
        mask: Grid mask array (0=skip, 1=simulate). Shape must match grid (nz, ny, nx).
              Masked cells output UNEST (-99).

    Returns:
        SimulationResult with realizations array (simulated values)
    """
    ncut = len(thresholds)
    if len(variograms) != ncut:
        raise ValueError(f"Need {ncut} variograms for {ncut} thresholds")
    if len(global_cdf) != ncut:
        raise ValueError(f"Need {ncut} global CDF values for {ncut} thresholds")

    # Handle unconditional case
    if x is None or values is None:
        has_data = False
    else:
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        z = np.asarray(z, dtype=np.float64).ravel()
        values = np.asarray(values, dtype=np.float64).ravel()
        has_data = True

    # Kriging type: 0=SK, 1=OK
    ktype = 0 if kriging_type == "simple" else 1

    with GSLIBWorkspace() as workspace:
        data_file = workspace / "sisim_input.dat"
        output_file = workspace / "sisim_output.out"
        debug_file = workspace / "sisim_debug.dbg"
        par_file = workspace / "sisim.par"
        mask_file = workspace / "sisim_mask.bin"

        # Write mask file if provided
        if mask is not None:
            mask_arr = np.asarray(mask, dtype=np.int8)
            expected_shape = (grid.nz, grid.ny, grid.nx)
            if mask_arr.shape != expected_shape:
                raise ValueError(f"Mask shape {mask_arr.shape} doesn't match grid {expected_shape}")
            with open(mask_file, "wb") as f:
                np.array([3, grid.nz, grid.ny, grid.nx], dtype=np.int32).tofile(f)
                mask_arr.ravel(order="F").tofile(f)

        if has_data:
            AsciiIO.write_data(
                data_file,
                {"x": x, "y": y, "z": z, "value": values},
                title="sisim conditioning data",
            )

        # Build par file for sisim - matches sisim.for format
        par = ParFileBuilder()
        par.comment("Parameters for SISIM")
        par.comment("********************")
        par.blank()
        par.line("START OF PARAMETERS:")

        # Variable type: 1=continuous (cdf), 0=categorical (pdf)
        par.line(1, comment="1=continuous(cdf), 0=categorical(pdf)")
        par.line(ncut, comment="number of thresholds/categories")

        # Thresholds (all on one line)
        thresholds_str = "  ".join(f"{t}" for t in thresholds)
        par.line(thresholds_str, comment="thresholds")

        # Global CDF values (all on one line)
        cdf_str = "  ".join(f"{c}" for c in global_cdf)
        par.line(cdf_str, comment="global cdf values")

        # Data file
        if has_data:
            par.line("sisim_input.dat", comment="data file")
            par.line(1, 2, 3, 4, comment="columns: x, y, z, var")
        else:
            par.line("nodata.dat", comment="data file (no conditioning)")
            par.line(0, 0, 0, 0, comment="columns (not used)")

        # Soft indicator file (not used)
        par.line("nosoftdata.dat", comment="soft indicator file (not used)")
        par.line(0, 0, 0, *[0] * ncut, comment="soft indicator columns")
        par.line(0, comment="Markov-Bayes simulation (0=no)")
        par.line(*[0.5] * ncut, comment="calibration B(z) values (not used)")

        # Trimming and tail options
        par.line(tmin, tmax, comment="trimming limits")
        par.line(zmin, zmax, comment="minimum and maximum data value")
        par.line(1, zmin, comment="lower tail option and parameter")
        par.line(1, 1.0, comment="middle option and parameter")
        par.line(1, zmax, comment="upper tail option and parameter")

        # Tabulated values file (not used - use global cdf)
        par.line("notabfile.dat", comment="tabulated values file (not used)")
        par.line(0, 0, comment="columns for variable, weight")

        # Debug and output
        par.line(0, comment="debugging level")
        par.line("sisim_debug.dbg", comment="debug file")
        par.line("sisim_output.out", comment="output file")
        par.line(1 if binary else 0, comment="binary output (0=ASCII, 1=binary)")

        # Grid mask
        if mask is not None:
            par.line(1, comment="use grid mask (0=no, 1=yes)")
            par.line("sisim_mask.bin", comment="mask file")
        else:
            par.line(0, comment="use grid mask (0=no, 1=yes)")

        # Grid and simulation parameters
        par.line(nrealizations, comment="number of realizations")
        par.line(grid.nx, grid.xmin, grid.xsiz, comment="nx, xmn, xsiz")
        par.line(grid.ny, grid.ymin, grid.ysiz, comment="ny, ymn, ysiz")
        par.line(grid.nz, grid.zmin, grid.zsiz, comment="nz, zmn, zsiz")
        par.line(seed, comment="random number seed")
        par.line(max_conditioning, comment="max original data for each kriging")
        par.line(max_previously_simulated, comment="max previous nodes for each kriging")
        par.line(1, comment="max soft indicator nodes")
        par.line(1, comment="assign data to nodes (0=no, 1=yes)")
        par.line(0, 3, comment="multiple grid search (0=no)")
        par.line(0, comment="max per octant (0=not used)")
        par.line(search_radius[0], search_radius[1], search_radius[2], comment="search radii")
        par.line(search_angles[0], search_angles[1], search_angles[2], comment="search angles")

        # Covariance lookup table size
        cov_size = max(grid.nx // 4, 1), max(grid.ny // 4, 1), max(grid.nz // 4, 1)
        par.line(cov_size[0], cov_size[1], cov_size[2], comment="covariance lookup size")

        par.line(0, thresholds[ncut // 2] if ncut > 0 else 1.0, comment="0=full IK, 1=median approx")
        par.line(ktype, comment="0=SK, 1=OK")

        # Variograms for each cutoff
        for i, vario in enumerate(variograms):
            nst = len(vario.structures)
            par.line(nst, vario.nugget, comment=f"nst, nugget (cutoff {i + 1})")
            for struct in vario.structures:
                vtype = struct["type"]
                sill = struct["sill"]
                angles = struct.get("angles", (0.0, 0.0, 0.0))
                ranges = struct["ranges"]
                par.line(vtype, sill, angles[0], angles[1], angles[2], comment="it, cc, ang1, ang2, ang3")
                par.line(ranges[0], ranges[1], ranges[2], comment="a_hmax, a_hmin, a_vert")

        par.write(par_file)

        # Run sisim
        run_gslib("sisim", par_file)

        # Read results
        if binary:
            # Binary output: header is [ndim=4, nsim, nz, ny, nx]
            # Data is sequential: all cells for sim1, then sim2, etc.
            # Within each sim, cells are in Fortran order (x fastest)
            # GSLIB uses single precision (float32)
            with open(output_file, "rb") as f:
                # Read header
                ndim = np.fromfile(f, dtype=np.int32, count=1)[0]
                shape = tuple(np.fromfile(f, dtype=np.int32, count=ndim))
                nreal_actual = shape[0]  # First dim is nsim
                # Read flat data
                values_flat = np.fromfile(f, dtype=np.float32)

            ncells = grid.nx * grid.ny * grid.nz
            result = np.zeros((nreal_actual, grid.nz, grid.ny, grid.nx), dtype=np.float64)
            for i in range(nreal_actual):
                start = i * ncells
                end = start + ncells
                result[i] = values_flat[start:end].reshape(
                    (grid.nz, grid.ny, grid.nx), order="F"
                )
        else:
            # ASCII output: gridded format with all realizations concatenated
            names, output_data, _grid_info = AsciiIO.read_gridded_data(output_file)

            # Reshape to (nreal, nz, ny, nx)
            ncells = grid.nx * grid.ny * grid.nz
            total_values = output_data.shape[0]
            nreal_actual = total_values // ncells

            result = np.zeros((nreal_actual, grid.nz, grid.ny, grid.nx), dtype=np.float64)
            values_flat = output_data[:, 0] if output_data.ndim > 1 else output_data.ravel()
            for i in range(nreal_actual):
                start = i * ncells
                end = start + ncells
                result[i] = values_flat[start:end].reshape((grid.nz, grid.ny, grid.nx), order="F")

    return SimulationResult(realizations=result, grid=grid)
