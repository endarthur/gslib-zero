"""
gslib-zero: Python wrapper for Stanford's GSLIB Fortran 90 geostatistics programs.

This package provides high-performance geostatistical analysis through binary I/O
and grid mask support, wrapping battle-tested GSLIB algorithms with a modern Python API.

Key Features:
    - Binary I/O for 10-100x faster data exchange than ASCII
    - Grid masks to skip inactive cells during estimation/simulation
    - Pre-built Fortran binaries for Windows, Linux, and macOS
    - Optional double precision (f64) builds for higher numerical accuracy

Programs Wrapped:
    - :func:`declus`: Cell declustering for preferential sampling correction
    - :func:`nscore`: Normal score transform (Gaussian anamorphosis)
    - :func:`backtr`: Back-transform from normal scores to original distribution
    - :func:`gamv`: Experimental variogram/covariogram calculation
    - :func:`kt3d`: 3D kriging (ordinary, simple, with trend, cokriging)
    - :func:`ik3d`: Indicator kriging with order relations correction
    - :func:`sgsim`: Sequential Gaussian simulation
    - :func:`sisim`: Sequential indicator simulation

Python Utilities:
    - Variogram plotting and model export
    - Drillhole desurvey, compositing, and interval merging
    - Rotation convention conversions (GSLIB, Leapfrog, Datamine, Vulcan)

Example:
    >>> import numpy as np
    >>> from gslib_zero import kt3d, GridSpec, VariogramModel, SearchParameters
    >>>
    >>> # Sample data
    >>> x = np.array([10, 20, 30, 40])
    >>> y = np.array([10, 20, 30, 40])
    >>> z = np.array([0, 0, 0, 0])
    >>> values = np.array([1.5, 2.0, 2.5, 3.0])
    >>>
    >>> # Define grid and variogram
    >>> grid = GridSpec(nx=10, ny=10, nz=1, xmin=5, ymin=5, zmin=0,
    ...                 xsiz=5, ysiz=5, zsiz=1)
    >>> variogram = VariogramModel.spherical(sill=1.0, ranges=(50, 50, 10), nugget=0.1)
    >>> search = SearchParameters(radius1=100, radius2=100, radius3=10)
    >>>
    >>> # Run kriging
    >>> result = kt3d(x, y, z, values, grid, variogram, search, binary=True)
    >>> print(result.estimate.shape)
    (1, 10, 10)
"""

__version__ = "1.0.0"

# Core I/O and execution
from gslib_zero.core import (
    run_gslib,
    BinaryIO,
    AsciiIO,
    ExperimentalWarning,
)

# Grid and variogram specifications
from gslib_zero.utils import (
    GridSpec,
    VariogramModel,
    VariogramType,
    evaluate_variogram,
    # GSLIB sentinel values
    UNEST,
    UNEST_IK,
    is_unestimated,
    mask_unestimated,
    # Rotation conversions
    deutsch_to_math,
    math_to_deutsch,
    leapfrog_to_deutsch,
    deutsch_to_leapfrog,
    dipdirection_to_deutsch,
    deutsch_to_dipdirection,
    datamine_to_deutsch,
    deutsch_to_datamine,
    vulcan_to_deutsch,
    deutsch_to_vulcan,
    rotation_matrix_deutsch,
)

# GSLIB program wrappers
from gslib_zero.transforms import nscore, backtr
from gslib_zero.variogram import gamv, VariogramResult, GamvDirection
from gslib_zero.estimation import (
    kt3d,
    ik3d,
    SearchParameters,
    KrigingResult,
    IndicatorKrigingResult,
)
from gslib_zero.simulation import sgsim, sisim, SimulationResult
from gslib_zero.declustering import declus, DeclusResult

# Plotting utilities
from gslib_zero.plotting import (
    plot_experimental,
    plot_model,
    plot_variogram,
    export_variogram_par,
)

# Drillhole utilities
from gslib_zero.drillhole import (
    desurvey,
    composite,
    merge_intervals,
    CompositeResult,
)

__all__ = [
    # Version
    "__version__",
    # Core I/O
    "run_gslib",
    "BinaryIO",
    "AsciiIO",
    "ExperimentalWarning",
    # Grid and variogram specs
    "GridSpec",
    "VariogramModel",
    "VariogramType",
    "SearchParameters",
    "evaluate_variogram",
    # GSLIB sentinel values
    "UNEST",
    "UNEST_IK",
    "is_unestimated",
    "mask_unestimated",
    # GSLIB program wrappers
    "nscore",
    "backtr",
    "declus",
    "gamv",
    "kt3d",
    "ik3d",
    "sgsim",
    "sisim",
    # Result classes
    "VariogramResult",
    "GamvDirection",
    "KrigingResult",
    "IndicatorKrigingResult",
    "SimulationResult",
    "DeclusResult",
    "CompositeResult",
    # Plotting
    "plot_experimental",
    "plot_model",
    "plot_variogram",
    "export_variogram_par",
    # Drillhole utilities
    "desurvey",
    "composite",
    "merge_intervals",
    # Rotation conversions
    "deutsch_to_math",
    "math_to_deutsch",
    "leapfrog_to_deutsch",
    "deutsch_to_leapfrog",
    "dipdirection_to_deutsch",
    "deutsch_to_dipdirection",
    "datamine_to_deutsch",
    "deutsch_to_datamine",
    "vulcan_to_deutsch",
    "deutsch_to_vulcan",
    "rotation_matrix_deutsch",
]
