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

from gslib_zero.core import AsciiIO, BinaryIO, GSLIBWorkspace, run_gslib
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
    binary: bool = False,
    mask: NDArray[np.integer] | None = None,
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
        binary: If True, use binary I/O (requires gslib-zero modified binaries)
        mask: Optional grid mask (same shape as output grid). 0=skip, 1=estimate.
              Masked cells will have UNEST (-999) in output.

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
        mask_file = workspace / "kt3d_mask.bin"
        par_file = workspace / "kt3d.par"

        # Write mask file if provided
        if mask is not None:
            mask_arr = np.asarray(mask, dtype=np.int8)
            expected_shape = (grid.nz, grid.ny, grid.nx)
            if mask_arr.shape != expected_shape:
                raise ValueError(f"Mask shape {mask_arr.shape} doesn't match grid {expected_shape}")
            with open(mask_file, "wb") as f:
                # Header: [ndim=3, nz, ny, nx]
                np.array([3, grid.nz, grid.ny, grid.nx], dtype=np.int32).tofile(f)
                # Data: int8 values in Fortran order
                mask_arr.ravel(order="F").tofile(f)

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
        # Binary output flag: 0=ASCII, 1=binary (gslib-zero modified binaries)
        par.line(1 if binary else 0, comment="binary output (0=ASCII, 1=binary)")
        # Grid mask: 0=no mask, 1=use mask
        if mask is not None:
            par.line(1, comment="use grid mask (0=no, 1=yes)")
            par.line("kt3d_mask.bin", comment="mask file")
        else:
            par.line(0, comment="use grid mask (0=no, 1=yes)")
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

        # Read results
        if binary:
            # Binary output: 4D array (nvars=2, nz, ny, nx) with (est, var) interleaved
            # GSLIB uses single precision (float32) for estimates
            output_data = BinaryIO.read_array(output_file, dtype=np.float32)
            # Shape is (2, nz, ny, nx) - first dim is variable (0=est, 1=var)
            estimate = output_data[0].astype(np.float64)  # Convert to float64
            variance = output_data[1].astype(np.float64)
        else:
            # ASCII output: kt3d has custom header format
            # Line 1: title
            # Line 2: nvars nx ny nz
            # Lines 3+: variable names
            # Rest: data
            with open(output_file, "r") as f:
                _title = f.readline()  # Skip title
                header = f.readline().split()
                nvars = int(header[0])
                # Skip variable names
                for _ in range(nvars):
                    f.readline()
                # Read data
                rows = []
                for line in f:
                    line = line.strip()
                    if line:
                        values = [float(x) for x in line.split()]
                        rows.append(values)

            output_data = np.array(rows, dtype=np.float64)

            # First column is estimate, second is variance
            estimate = output_data[:, 0]
            variance = output_data[:, 1] if output_data.shape[1] > 1 else np.zeros_like(estimate)

            # Reshape to grid dimensions (GSLIB uses Fortran ordering: z fastest)
            shape = (grid.nz, grid.ny, grid.nx)
            estimate = estimate.reshape(shape, order="F")
            variance = variance.reshape(shape, order="F")

    return KrigingResult(estimate=estimate, variance=variance, grid=grid)


@dataclass
class IndicatorKrigingResult:
    """Result from indicator kriging estimation."""

    probabilities: NDArray[np.float64]  # Shape (ncut, nz, ny, nx)
    cutoffs: list[float]
    grid: GridSpec


def ik3d(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    z: NDArray[np.floating],
    values: NDArray[np.floating],
    grid: GridSpec,
    cutoffs: list[float],
    global_cdf: list[float],
    variograms: list[VariogramModel],
    search: SearchParameters,
    kriging_type: Literal["simple", "ordinary"] = "ordinary",
    indicator_type: Literal["continuous", "categorical"] = "continuous",
    median_ik: bool = False,
    median_cutoff_index: int = 0,
    tmin: float = -1.0e21,
    tmax: float = 1.0e21,
    binary: bool = False,
    mask: NDArray[np.integer] | None = None,
) -> IndicatorKrigingResult:
    """
    Indicator kriging using ik3d.

    Args:
        x, y, z: Sample coordinates (1D arrays)
        values: Sample values (1D array)
        grid: Grid specification for output
        cutoffs: List of indicator cutoff values
        global_cdf: Global CDF/PDF values for each cutoff (same length as cutoffs)
        variograms: List of variogram models, one per cutoff
        search: Search neighborhood parameters
        kriging_type: 'simple' or 'ordinary' kriging
        indicator_type: 'continuous' (CDF) or 'categorical' (PDF)
        median_ik: If True, use median indicator kriging (faster)
        median_cutoff_index: Cutoff index to use for median IK (0-based)
        tmin, tmax: Trimming limits
        binary: If True, use binary I/O (requires gslib-zero modified binaries)
        mask: Optional grid mask (same shape as output grid). 0=skip, 1=estimate.
              Masked cells will have UNEST (-999) in output.

    Returns:
        IndicatorKrigingResult with probabilities array of shape (ncut, nz, ny, nx)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel()

    n = len(x)
    ncut = len(cutoffs)

    if len(variograms) != ncut:
        raise ValueError(f"Need {ncut} variograms for {ncut} cutoffs")
    if len(global_cdf) != ncut:
        raise ValueError(f"Need {ncut} global CDF values for {ncut} cutoffs")

    # GSLIB codes
    ivtype = 1 if indicator_type == "continuous" else 0
    ktype = 0 if kriging_type == "simple" else 1
    mik = 1 if median_ik else 0
    # GSLIB uses 1-based cutoff index for median IK
    cutmik = cutoffs[median_cutoff_index] if median_ik else cutoffs[0]

    with GSLIBWorkspace() as workspace:
        data_file = workspace / "ik3d_input.dat"
        output_file = workspace / "ik3d_output.out"
        debug_file = workspace / "ik3d_debug.dbg"
        mask_file = workspace / "ik3d_mask.bin"
        par_file = workspace / "ik3d.par"

        # Write mask file if provided
        if mask is not None:
            mask_arr = np.asarray(mask, dtype=np.int8)
            expected_shape = (grid.nz, grid.ny, grid.nx)
            if mask_arr.shape != expected_shape:
                raise ValueError(f"Mask shape {mask_arr.shape} doesn't match grid {expected_shape}")
            with open(mask_file, "wb") as f:
                np.array([3, grid.nz, grid.ny, grid.nx], dtype=np.int32).tofile(f)
                mask_arr.ravel(order="F").tofile(f)

        # Write input data with drillhole ID column
        dhid = np.arange(1, n + 1, dtype=np.float64)
        AsciiIO.write_data(
            data_file,
            {"dhid": dhid, "x": x, "y": y, "z": z, "value": values},
            title="ik3d input data",
        )

        # Build par file - following ik3d.for readparm order
        par = ParFileBuilder()
        par.comment("Parameters for IK3D")
        par.comment("*******************")
        par.blank()
        par.line("START OF PARAMETERS:")
        par.line(ivtype, comment="1=continuous(cdf), 0=categorical(pdf)")
        par.line(0, comment="option: 0=grid, 1=cross, 2=jackknife")
        par.line("nofile.dat", comment="jackknife file (not used)")
        par.line(1, 2, 3, 4, comment="jackknife columns (not used)")
        par.line(ncut, comment="number of thresholds/categories")
        # Thresholds on single line
        par.line(*cutoffs, comment="thresholds/categories")
        # Global CDF/PDF on single line
        par.line(*global_cdf, comment="global cdf/pdf")
        par.line("ik3d_input.dat", comment="data file")
        par.line(1, 2, 3, 4, 5, comment="columns: DH, x, y, z, var")
        par.line("nofile.dat", comment="soft indicator file (not used)")
        par.line(1, 2, 3, *range(4, 4 + ncut), comment="soft columns (not used)")
        par.line(tmin, tmax, comment="trimming limits")
        par.line(0, comment="debugging level")
        par.line("ik3d_debug.dbg", comment="debug file")
        par.line("ik3d_output.out", comment="output file")
        par.line(1 if binary else 0, comment="binary output (0=ASCII, 1=binary)")
        # Grid mask: 0=no mask, 1=use mask
        if mask is not None:
            par.line(1, comment="use grid mask (0=no, 1=yes)")
            par.line("ik3d_mask.bin", comment="mask file")
        else:
            par.line(0, comment="use grid mask (0=no, 1=yes)")
        par.line(grid.nx, grid.xmin, grid.xsiz, comment="nx, xmn, xsiz")
        par.line(grid.ny, grid.ymin, grid.ysiz, comment="ny, ymn, ysiz")
        par.line(grid.nz, grid.zmin, grid.zsiz, comment="nz, zmn, zsiz")
        par.line(search.min_samples, search.max_samples, comment="min, max data")
        par.line(search.radius1, search.radius2, search.radius3, comment="search radii")
        par.line(search.azimuth, search.dip, search.rake, comment="search angles")
        par.line(search.max_per_octant, comment="max per octant")
        par.line(mik, cutmik, comment="0=full IK, 1=median IK (cutoff value)")
        par.line(ktype, comment="0=SK, 1=OK")

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
        if binary:
            # Binary output: 4D array (ncut, nz, ny, nx)
            with open(output_file, "rb") as f:
                ndim = np.fromfile(f, dtype=np.int32, count=1)[0]
                shape = tuple(np.fromfile(f, dtype=np.int32, count=ndim))
                ncut_actual = shape[0]
                values_flat = np.fromfile(f, dtype=np.float32)

            # Reshape with F-order to match other GSLIB programs (kt3d, sgsim, etc.)
            # Data is ncut values per cell, so reshape to (ncut, nz, ny, nx)
            result = values_flat.reshape(
                (ncut_actual, grid.nz, grid.ny, grid.nx), order="F"
            ).astype(np.float64)
        else:
            # ASCII output: custom header format for ik3d
            with open(output_file, "r") as f:
                _title = f.readline()  # Skip title
                _header = f.readline()  # Skip header line
                # Skip cutoff labels
                for _ in range(ncut):
                    f.readline()
                # Read data
                rows = []
                for line in f:
                    line = line.strip()
                    if line:
                        row_values = [float(val) for val in line.split()]
                        rows.append(row_values)

            output_data = np.array(rows, dtype=np.float64)

            # Reshape with F-order to match other GSLIB programs
            result = np.zeros((ncut, grid.nz, grid.ny, grid.nx), dtype=np.float64)
            for i in range(min(ncut, output_data.shape[1])):
                result[i] = output_data[:, i].reshape(
                    (grid.nz, grid.ny, grid.nx), order="F"
                )

    return IndicatorKrigingResult(
        probabilities=result, cutoffs=list(cutoffs), grid=grid
    )
