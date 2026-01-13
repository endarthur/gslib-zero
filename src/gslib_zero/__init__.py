"""
gslib-zero: Python wrapper for Stanford's GSLIB Fortran 90 geostatistics programs.

This package provides binary I/O and grid mask support for improved performance
over traditional ASCII-based GSLIB workflows.

Programs wrapped:
- declus: Cell declustering
- nscore: Normal score transform
- backtr: Back-transform
- gamv: Variogram calculation
- kt3d: 3D kriging (OK, SK, KT, cokriging, external drift)
- ik3d: Indicator kriging
- sgsim: Sequential Gaussian simulation
- sisim: Sequential indicator simulation
"""

__version__ = "0.1.0"

from gslib_zero.core import run_gslib, BinaryIO, AsciiIO
from gslib_zero.utils import VariogramModel, GridSpec

__all__ = [
    "__version__",
    "run_gslib",
    "BinaryIO",
    "AsciiIO",
    "VariogramModel",
    "GridSpec",
]
