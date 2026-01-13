"""
Utility classes and functions for gslib-zero.

- GridSpec: Grid definition
- VariogramModel: Variogram model specification
- Rotation conversions between conventions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class VariogramType(IntEnum):
    """GSLIB variogram model types."""

    SPHERICAL = 1
    EXPONENTIAL = 2
    GAUSSIAN = 3
    POWER = 4
    HOLE_EFFECT = 5


@dataclass
class GridSpec:
    """
    3D grid specification.

    GSLIB convention: grids are indexed (nz, ny, nx) internally
    and iterate z-fastest (column-major/Fortran order).

    Attributes:
        nx, ny, nz: Number of cells in each direction
        xmin, ymin, zmin: Minimum coordinates (cell centers)
        xsiz, ysiz, zsiz: Cell sizes
    """

    nx: int
    ny: int
    nz: int
    xmin: float
    ymin: float
    zmin: float
    xsiz: float
    ysiz: float
    zsiz: float

    @property
    def shape(self) -> tuple[int, int, int]:
        """Grid shape as (nz, ny, nx) - GSLIB convention."""
        return (self.nz, self.ny, self.nx)

    @property
    def ncells(self) -> int:
        """Total number of cells."""
        return self.nx * self.ny * self.nz

    @property
    def xmax(self) -> float:
        """Maximum x coordinate (cell center)."""
        return self.xmin + (self.nx - 1) * self.xsiz

    @property
    def ymax(self) -> float:
        """Maximum y coordinate (cell center)."""
        return self.ymin + (self.ny - 1) * self.ysiz

    @property
    def zmax(self) -> float:
        """Maximum z coordinate (cell center)."""
        return self.zmin + (self.nz - 1) * self.zsiz

    def cell_centers(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Get cell center coordinates.

        Returns:
            Tuple of (x, y, z) 1D arrays of cell centers
        """
        x = self.xmin + np.arange(self.nx) * self.xsiz
        y = self.ymin + np.arange(self.ny) * self.ysiz
        z = self.zmin + np.arange(self.nz) * self.zsiz
        return x, y, z

    def meshgrid(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Get 3D meshgrid of cell centers.

        Returns:
            Tuple of (X, Y, Z) 3D arrays with shape (nz, ny, nx)
        """
        x, y, z = self.cell_centers()
        # Note: indexing='ij' gives (nz, ny, nx) shape
        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
        return X, Y, Z

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """Check if a point falls within the grid extent."""
        half_x, half_y, half_z = self.xsiz / 2, self.ysiz / 2, self.zsiz / 2
        return (
            self.xmin - half_x <= x <= self.xmax + half_x and
            self.ymin - half_y <= y <= self.ymax + half_y and
            self.zmin - half_z <= z <= self.zmax + half_z
        )

    def point_to_index(
        self, x: float, y: float, z: float
    ) -> tuple[int, int, int] | None:
        """
        Convert point coordinates to grid indices.

        Args:
            x, y, z: Point coordinates

        Returns:
            Tuple of (iz, iy, ix) indices, or None if outside grid
        """
        if not self.contains_point(x, y, z):
            return None

        ix = int(round((x - self.xmin) / self.xsiz))
        iy = int(round((y - self.ymin) / self.ysiz))
        iz = int(round((z - self.zmin) / self.zsiz))

        # Clamp to valid range
        ix = max(0, min(ix, self.nx - 1))
        iy = max(0, min(iy, self.ny - 1))
        iz = max(0, min(iz, self.nz - 1))

        return (iz, iy, ix)


@dataclass
class VariogramStructure:
    """Single variogram structure (nested structure)."""

    type: int | VariogramType
    sill: float
    ranges: tuple[float, float, float]
    angles: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> dict:
        """Convert to dict for par file generation."""
        return {
            "type": int(self.type),
            "sill": self.sill,
            "ranges": self.ranges,
            "angles": self.angles,
        }


@dataclass
class VariogramModel:
    """
    Complete variogram model specification.

    A variogram model consists of a nugget effect plus one or more
    nested structures (spherical, exponential, Gaussian, etc.).

    Angles follow GSLIB/Deutsch convention:
    - azimuth: Clockwise from north (0-360 degrees)
    - dip: Down from horizontal (-90 to 90 degrees)
    - rake: Rotation about the dip vector (-90 to 90 degrees)
    """

    nugget: float = 0.0
    structures: list[dict] = field(default_factory=list)

    @classmethod
    def spherical(
        cls,
        sill: float,
        ranges: tuple[float, float, float],
        nugget: float = 0.0,
        angles: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "VariogramModel":
        """Create a simple spherical variogram model."""
        return cls(
            nugget=nugget,
            structures=[{
                "type": VariogramType.SPHERICAL,
                "sill": sill,
                "ranges": ranges,
                "angles": angles,
            }]
        )

    @classmethod
    def exponential(
        cls,
        sill: float,
        ranges: tuple[float, float, float],
        nugget: float = 0.0,
        angles: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "VariogramModel":
        """Create a simple exponential variogram model."""
        return cls(
            nugget=nugget,
            structures=[{
                "type": VariogramType.EXPONENTIAL,
                "sill": sill,
                "ranges": ranges,
                "angles": angles,
            }]
        )

    @classmethod
    def gaussian(
        cls,
        sill: float,
        ranges: tuple[float, float, float],
        nugget: float = 0.0,
        angles: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "VariogramModel":
        """Create a simple Gaussian variogram model."""
        return cls(
            nugget=nugget,
            structures=[{
                "type": VariogramType.GAUSSIAN,
                "sill": sill,
                "ranges": ranges,
                "angles": angles,
            }]
        )

    def add_structure(
        self,
        vtype: int | VariogramType,
        sill: float,
        ranges: tuple[float, float, float],
        angles: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "VariogramModel":
        """Add a nested structure to the model."""
        self.structures.append({
            "type": int(vtype),
            "sill": sill,
            "ranges": ranges,
            "angles": angles,
        })
        return self

    @property
    def total_sill(self) -> float:
        """Total sill (nugget + all structure contributions)."""
        return self.nugget + sum(s["sill"] for s in self.structures)


# ============================================================================
# Rotation Convention Conversions
# ============================================================================

def deutsch_to_math(
    azimuth: float, dip: float, rake: float
) -> tuple[float, float, float]:
    """
    Convert Deutsch convention angles to mathematical convention.

    Deutsch (GSLIB):
    - azimuth: Clockwise from north (0-360)
    - dip: Down from horizontal (-90 to 90)
    - rake: Rotation about dip vector

    Mathematical:
    - alpha: Counter-clockwise from east
    - beta: Up from horizontal
    - gamma: Rotation about normal
    """
    alpha = 90.0 - azimuth  # CW from N -> CCW from E
    beta = -dip  # Down -> Up
    gamma = rake
    return (alpha, beta, gamma)


def math_to_deutsch(
    alpha: float, beta: float, gamma: float
) -> tuple[float, float, float]:
    """
    Convert mathematical convention to Deutsch (GSLIB) convention.

    See deutsch_to_math for convention definitions.
    """
    azimuth = 90.0 - alpha
    dip = -beta
    rake = gamma
    return (azimuth, dip, rake)


def leapfrog_to_deutsch(
    dip: float, dip_direction: float, pitch: float = 0.0
) -> tuple[float, float, float]:
    """
    Convert Leapfrog convention to Deutsch (GSLIB) convention.

    Leapfrog:
    - dip: Angle of steepest descent (0-90)
    - dip_direction: Direction of steepest descent, CW from N (0-360)
    - pitch: Rotation within the plane

    Note: Leapfrog dip is always positive (unsigned).
    """
    # Azimuth is perpendicular to dip direction (strike direction)
    azimuth = (dip_direction - 90.0) % 360.0
    # GSLIB dip convention
    gslib_dip = dip
    rake = pitch
    return (azimuth, gslib_dip, rake)


def deutsch_to_leapfrog(
    azimuth: float, dip: float, rake: float
) -> tuple[float, float, float]:
    """
    Convert Deutsch (GSLIB) convention to Leapfrog convention.

    See leapfrog_to_deutsch for convention definitions.
    """
    dip_direction = (azimuth + 90.0) % 360.0
    lf_dip = abs(dip)
    pitch = rake
    return (lf_dip, dip_direction, pitch)


def rotation_matrix_deutsch(
    azimuth: float, dip: float, rake: float
) -> NDArray[np.float64]:
    """
    Compute 3D rotation matrix from Deutsch convention angles.

    Args:
        azimuth: Clockwise from north (degrees)
        dip: Down from horizontal (degrees)
        rake: Rotation about dip vector (degrees)

    Returns:
        3x3 rotation matrix
    """
    # Convert to radians
    az = np.radians(azimuth)
    dp = np.radians(dip)
    rk = np.radians(rake)

    # Rotation matrices for each angle
    # R = Rz(azimuth) @ Rx(dip) @ Ry(rake)

    cos_az, sin_az = np.cos(az), np.sin(az)
    cos_dp, sin_dp = np.cos(dp), np.sin(dp)
    cos_rk, sin_rk = np.cos(rk), np.sin(rk)

    # Build combined rotation matrix
    r11 = cos_az * cos_rk - sin_az * sin_dp * sin_rk
    r12 = -sin_az * cos_dp
    r13 = cos_az * sin_rk + sin_az * sin_dp * cos_rk

    r21 = sin_az * cos_rk + cos_az * sin_dp * sin_rk
    r22 = cos_az * cos_dp
    r23 = sin_az * sin_rk - cos_az * sin_dp * cos_rk

    r31 = -cos_dp * sin_rk
    r32 = sin_dp
    r33 = cos_dp * cos_rk

    return np.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33],
    ], dtype=np.float64)
