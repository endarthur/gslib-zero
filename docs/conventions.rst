GSLIB Conventions
=================

This document explains the coordinate systems, rotation conventions, and data
formats used by GSLIB and gslib-zero. Understanding these conventions is
essential for correct interpretation of results.

Grid Indexing
-------------

GSLIB uses Fortran-style column-major ordering where the **Z coordinate varies
fastest**, followed by Y, then X. This is the opposite of typical Python/C
row-major ordering.

.. code-block:: text

    GSLIB grid order: Z varies fastest
    Array shape: (nz, ny, nx)

    Index mapping:
    (iz, iy, ix) → position = ix*ny*nz + iy*nz + iz

When working with gslib-zero, all grid arrays have shape ``(nz, ny, nx)``:

.. code-block:: python

    from gslib_zero import GridSpec

    grid = GridSpec(nx=10, ny=20, nz=5, ...)

    # Result arrays have shape (nz, ny, nx)
    result = kt3d(...)
    assert result.estimate.shape == (5, 20, 10)

The :meth:`GridSpec.meshgrid` method returns coordinates in this order:

.. code-block:: python

    X, Y, Z = grid.meshgrid()
    # All arrays have shape (nz, ny, nx)
    # X[iz, iy, ix] gives the x-coordinate at that grid position

Grid Origin
-----------

GSLIB grid parameters define the **center of the first cell**, not the corner:

.. code-block:: text

    xmin = center of cell (0, :, :)
    ymin = center of cell (:, 0, :)
    zmin = center of cell (:, :, 0)

The cell extends from ``xmin - xsiz/2`` to ``xmin + xsiz/2``.

.. code-block:: python

    grid = GridSpec(
        nx=10, xmin=5.0, xsiz=10.0,  # Cells at x = 5, 15, 25, ..., 95
        ny=10, ymin=5.0, ysiz=10.0,
        nz=1, zmin=0.0, zsiz=1.0
    )

    # Cell boundaries
    # First cell: x ∈ [0, 10], center at x=5
    # Last cell:  x ∈ [90, 100], center at x=95

Rotation Conventions
--------------------

GSLIB uses the **Deutsch convention** for specifying anisotropy orientations.
This is common in geostatistics but differs from conventions used by mining
software.

**Deutsch Convention (GSLIB)**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Angle
     - Definition
   * - Azimuth
     - Clockwise rotation from North (Y-axis), 0-360°
   * - Dip
     - Rotation down from horizontal, -90° to +90°
   * - Rake
     - Rotation within the dipping plane, -90° to +90°

The rotation is applied in order: Azimuth → Dip → Rake.

**Converting from Other Conventions**

gslib-zero provides conversion functions for common mining software conventions:

.. code-block:: python

    from gslib_zero import (
        dipdirection_to_deutsch,  # Dip Direction / Dip / Rake
        datamine_to_deutsch,      # Datamine convention
        vulcan_to_deutsch,        # Vulcan (Strike/Dip)
        leapfrog_to_deutsch,      # Leapfrog convention
    )

    # Dip Direction / Dip (most intuitive for geologists)
    # Dip direction = direction of steepest descent
    azm, dip, rake = dipdirection_to_deutsch(
        dip_direction=135.0,  # Dipping toward SE
        dip=45.0,             # 45° from horizontal
        rake=0.0
    )

    # Vulcan Strike/Dip (right-hand rule)
    # Strike = horizontal line on plane, dip is 90° clockwise
    azm, dip, rake = vulcan_to_deutsch(
        strike=45.0,
        dip=60.0,
        rake=0.0
    )

**Convention Reference Table**

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Convention
     - Angle 1
     - Angle 2
     - Notes
   * - Deutsch (GSLIB)
     - Azimuth (CW from N)
     - Dip (down from horiz.)
     - Standard for geostatistics
   * - Dip Direction/Dip
     - Dip Direction
     - Dip (always positive)
     - Strike = DipDir - 90°
   * - Vulcan
     - Strike
     - Dip
     - Right-hand rule
   * - Leapfrog
     - Dip
     - Dip Direction
     - Note reversed order
   * - Datamine
     - Dip Direction
     - Dip
     - Similar to DipDir/Dip

Variogram Model Types
---------------------

GSLIB uses integer codes for variogram model types:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Code
     - Type
     - Formula
   * - 1
     - Spherical
     - γ(h) = c·[1.5(h/a) - 0.5(h/a)³] for h<a, else c
   * - 2
     - Exponential
     - γ(h) = c·[1 - exp(-3h/a)]
   * - 3
     - Gaussian
     - γ(h) = c·[1 - exp(-3(h/a)²)]
   * - 4
     - Power
     - γ(h) = c·h^ω
   * - 5
     - Hole Effect
     - γ(h) = c·[1 - cos(πh/a)]

Where:
- h = lag distance
- a = range parameter (practical range for exponential/Gaussian)
- c = sill (variance contribution)
- ω = power exponent (for power model)

gslib-zero provides an enum for clarity:

.. code-block:: python

    from gslib_zero import VariogramType

    model = VariogramModel(nugget=0.1)
    model.add_structure(
        VariogramType.SPHERICAL,  # or just 1
        sill=0.9,
        ranges=(100, 50, 10)
    )

Binary I/O Format
-----------------

gslib-zero's binary format is designed for efficient numpy interoperability:

**Header**

.. code-block:: text

    [ndim: int32] [shape: int32 × ndim]

**Data**

.. code-block:: text

    [values: float32 × prod(shape)]  # or float64 for f64 builds

Data is stored in **Fortran (column-major) order** for direct compatibility
with GSLIB's internal array layout.

**Example: Reading Binary Output**

.. code-block:: python

    from gslib_zero import BinaryIO
    import numpy as np

    # Read kriging output (shape: nvars, nz, ny, nx)
    data = BinaryIO.read_array("kt3d_output.bin", dtype=np.float32)
    estimate = data[0]  # First variable (estimate)
    variance = data[1]  # Second variable (variance)

**Mask Format**

Grid masks use int8 values (0=inactive, 1=active):

.. code-block:: text

    Header: [ndim=3: int32] [nz: int32] [ny: int32] [nx: int32]
    Data:   [mask_values: int8 × (nz*ny*nx)]

UNEST Values
------------

GSLIB uses special values to indicate unestimated cells:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Program
     - UNEST Value
     - When Used
   * - kt3d
     - -999.0
     - Cells with insufficient data or masked
   * - ik3d
     - -9.9999
     - Cells with insufficient data or masked
   * - sgsim
     - -999.0
     - Masked cells
   * - sisim
     - -99.0
     - Masked cells

When processing results, filter out UNEST values:

.. code-block:: python

    estimate = result.estimate
    valid_mask = estimate > -900  # or != UNEST value
    valid_values = estimate[valid_mask]

Coordinate Systems
------------------

GSLIB makes no assumptions about coordinate units or projections. All coordinates
are treated as numeric values in a Cartesian system.

**Best Practices:**

1. Use projected coordinates (meters/feet), not lat/lon
2. Keep coordinates in a reasonable numeric range (avoid very large values)
3. Use consistent units for coordinates and variogram ranges
4. Z-axis is typically elevation (positive up) or depth (positive down)

.. code-block:: python

    # Good: Projected coordinates in meters
    x = np.array([500000, 500100, 500200])  # Easting
    y = np.array([4500000, 4500100, 4500200])  # Northing
    z = np.array([100, 105, 98])  # Elevation

    # Also good: Local coordinates
    x_local = x - x.min()
    y_local = y - y.min()
