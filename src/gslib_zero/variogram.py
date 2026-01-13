"""
Variogram calculation: gamv wrapper.

gamv computes experimental variograms, covariograms, correlograms,
and other spatial statistics from sample data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from gslib_zero.core import AsciiIO, BinaryIO, GSLIBWorkspace, run_gslib
from gslib_zero.par import ParFileBuilder

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class VariogramResult:
    """Result from variogram calculation for one direction/variogram combo."""

    lag_distances: NDArray[np.float64]
    gamma: NDArray[np.float64]
    num_pairs: NDArray[np.int64]
    tail_mean: NDArray[np.float64]
    head_mean: NDArray[np.float64]
    direction: tuple[float, float]  # (azimuth, dip)
    variogram_type: int


@dataclass
class GamvDirection:
    """
    Variogram calculation direction specification.

    Angles follow GSLIB/Deutsch convention:
    - azimuth: Clockwise from north (0-360)
    - dip: Down from horizontal (-90 to 90)
    """

    azimuth: float = 0.0
    azimuth_tolerance: float = 90.0
    bandwidth_horizontal: float = 1.0e21
    dip: float = 0.0
    dip_tolerance: float = 90.0
    bandwidth_vertical: float = 1.0e21


# Variogram type codes
VARIOGRAM_TYPES = {
    "semivariogram": 1,
    "cross_semivariogram": 2,
    "covariance": 3,
    "correlogram": 4,
    "general_relative": 5,
    "pairwise_relative": 6,
    "log_semivariogram": 7,
    "semimadogram": 8,
    "indicator_continuous": 9,
    "indicator_categorical": 10,
}


def gamv(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    z: NDArray[np.floating],
    values: NDArray[np.floating],
    nlag: int,
    lag_distance: float,
    lag_tolerance: float | None = None,
    directions: list[GamvDirection] | None = None,
    variogram_type: int | str = 1,
    standardize_sill: bool = False,
    tmin: float = -1.0e21,
    tmax: float = 1.0e21,
    binary: bool = False,
) -> list[VariogramResult]:
    """
    Calculate experimental variogram from sample data.

    Args:
        x, y, z: Sample coordinates (1D arrays)
        values: Sample values (1D array)
        nlag: Number of lags
        lag_distance: Lag distance/spacing
        lag_tolerance: Lag tolerance (default: half lag distance)
        directions: List of GamvDirection objects. Default is omnidirectional.
        variogram_type: Type of variogram to compute:
            1 or "semivariogram" = traditional semivariogram (default)
            2 or "cross_semivariogram" = cross-semivariogram
            3 or "covariance" = covariance
            4 or "correlogram" = correlogram
            5 or "general_relative" = general relative semivariogram
            6 or "pairwise_relative" = pairwise relative semivariogram
            7 or "log_semivariogram" = semivariogram of logarithms
            8 or "semimadogram" = semimadogram
        standardize_sill: If True, standardize sill to 1.0
        tmin, tmax: Trimming limits
        binary: If True, use binary I/O (requires gslib-zero modified binaries)

    Returns:
        List of VariogramResult objects, one per direction
    """
    # Validate and prepare inputs
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel()

    n = len(x)
    if not (len(y) == len(z) == len(values) == n):
        raise ValueError("All input arrays must have the same length")

    if lag_tolerance is None:
        lag_tolerance = lag_distance / 2.0

    if directions is None:
        # Default: omnidirectional
        directions = [GamvDirection()]

    # Convert string variogram type to int
    if isinstance(variogram_type, str):
        variogram_type = VARIOGRAM_TYPES.get(variogram_type.lower(), 1)

    ndir = len(directions)

    with GSLIBWorkspace() as workspace:
        data_file = workspace / "gamv_input.dat"
        output_file = workspace / "gamv_output.out"
        par_file = workspace / "gamv.par"

        # Write input data
        AsciiIO.write_data(
            data_file,
            {"x": x, "y": y, "z": z, "value": values},
            title="gamv input data",
        )

        # Build par file - see gamv.for lines 169-306
        par = ParFileBuilder()
        par.comment("Parameters for GAMV")
        par.comment("*******************")
        par.blank()
        par.line("START OF PARAMETERS:")
        par.line("gamv_input.dat", comment="file with data")
        par.line(1, 2, 3, comment="columns for X, Y, Z")
        par.line(1, 4, comment="number of variables, column numbers")
        par.line(tmin, tmax, comment="trimming limits")
        par.line("gamv_output.out", comment="output file")
        par.line(1 if binary else 0, comment="binary output (0=ASCII, 1=binary)")
        par.line(nlag, comment="number of lags")
        par.line(lag_distance, comment="lag distance")
        par.line(lag_tolerance, comment="lag tolerance")
        par.line(ndir, comment="number of directions")

        for d in directions:
            par.line(
                d.azimuth, d.azimuth_tolerance, d.bandwidth_horizontal,
                d.dip, d.dip_tolerance, d.bandwidth_vertical,
                comment="azm, atol, bandwh, dip, dtol, bandwd"
            )

        par.line(1 if standardize_sill else 0, comment="standardize sill (0=no, 1=yes)")
        par.line(1, comment="number of variograms")
        par.line(1, 1, variogram_type, comment="tail, head, variogram type")
        par.write(par_file)

        # Run gamv
        run_gslib("gamv", par_file)

        # Read results
        if binary:
            # Binary output: Header [ndim=4, nfields=5, nvarg, ndir, nlag+2]
            # Data: 5 float32 values per lag (dis, gam, np, hm, tm)
            with open(output_file, "rb") as f:
                ndim = np.fromfile(f, dtype=np.int32, count=1)[0]
                shape = tuple(np.fromfile(f, dtype=np.int32, count=ndim))
                # shape = (nfields=5, nvarg, ndir, nlag+2)
                nfields, nvarg_out, ndir_out, nlags_out = shape
                data = np.fromfile(f, dtype=np.float32)

            # Reshape data: (nfields, nvarg, ndir, nlag+2)
            data = data.reshape(shape, order="F").astype(np.float64)
            # data[0] = dis, data[1] = gam, data[2] = np, data[3] = hm, data[4] = tm

            results = []
            for i, direction in enumerate(directions):
                # Extract data for this direction (variogram 0, direction i)
                # Use first nlag lags (skip lag 0 which is at index 0, use indices 1:nlag+1)
                # Actually gamv writes nlag+2 lags, we typically want lags 1 to nlag
                lag_start = 1  # Skip lag 0
                lag_end = nlag + 1

                results.append(
                    VariogramResult(
                        lag_distances=data[0, 0, i, lag_start:lag_end].copy(),
                        gamma=data[1, 0, i, lag_start:lag_end].copy(),
                        num_pairs=data[2, 0, i, lag_start:lag_end].astype(np.int64),
                        tail_mean=data[3, 0, i, lag_start:lag_end].copy(),
                        head_mean=data[4, 0, i, lag_start:lag_end].copy(),
                        direction=(direction.azimuth, direction.dip),
                        variogram_type=variogram_type,
                    )
                )
        else:
            # ASCII output format:
            # For each variogram/direction combination:
            #   Title line
            #   nlag+2 lines: lag, distance, gamma, npairs, tail_mean, head_mean
            output_lines = []
            with open(output_file, "r") as f:
                for line in f:
                    line = line.strip()
                    # Skip header lines and empty lines
                    if line and not line.startswith("G") and not line.startswith("v"):
                        try:
                            values_row = [float(v) for v in line.split()]
                            if len(values_row) >= 4:  # At least lag, dist, gamma, npairs
                                output_lines.append(values_row)
                        except ValueError:
                            continue

            # Parse results into VariogramResult objects
            results = []
            # nlag+2 lines per direction in ASCII output
            lines_per_dir = nlag + 2

            for i, direction in enumerate(directions):
                start_idx = i * lines_per_dir
                end_idx = start_idx + lines_per_dir

                if end_idx <= len(output_lines):
                    dir_data = np.array(output_lines[start_idx:end_idx])
                    # Use lags 1 to nlag (indices 1:nlag+1, skip lag 0)
                    lag_start = 1
                    lag_end = nlag + 1

                    results.append(
                        VariogramResult(
                            lag_distances=dir_data[lag_start:lag_end, 1].copy() if dir_data.shape[1] > 1 else dir_data[lag_start:lag_end, 0].copy(),
                            gamma=dir_data[lag_start:lag_end, 2].copy() if dir_data.shape[1] > 2 else np.zeros(nlag),
                            num_pairs=dir_data[lag_start:lag_end, 3].astype(np.int64) if dir_data.shape[1] > 3 else np.zeros(nlag, dtype=np.int64),
                            tail_mean=dir_data[lag_start:lag_end, 4].copy() if dir_data.shape[1] > 4 else np.zeros(nlag),
                            head_mean=dir_data[lag_start:lag_end, 5].copy() if dir_data.shape[1] > 5 else np.zeros(nlag),
                            direction=(direction.azimuth, direction.dip),
                            variogram_type=variogram_type,
                        )
                    )

    return results
