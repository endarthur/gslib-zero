Rotation Conventions
====================

Convert between different rotation conventions used in geostatistics and
mining software.

GSLIB uses the Deutsch convention (azimuth, dip, rake), but other software
packages use different conventions. These functions enable seamless conversion.

Convention Summary
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Convention
     - Angle 1
     - Angle 2
     - Angle 3
   * - Deutsch (GSLIB)
     - Azimuth (CW from N)
     - Dip (down)
     - Rake
   * - Dip Direction/Dip
     - Dip Direction
     - Dip (positive)
     - Rake
   * - Vulcan
     - Strike
     - Dip
     - Rake
   * - Leapfrog
     - Dip
     - Dip Direction
     - Pitch
   * - Datamine
     - Dip Direction
     - Dip
     - Plunge

Dip Direction / Dip
-------------------

The most intuitive convention for geologists. Dip direction points in the
direction of steepest descent; dip is the angle from horizontal.

.. autofunction:: gslib_zero.dipdirection_to_deutsch

.. autofunction:: gslib_zero.deutsch_to_dipdirection

Vulcan (Strike/Dip)
-------------------

Uses the right-hand rule: dip direction is 90° clockwise from strike.

.. autofunction:: gslib_zero.vulcan_to_deutsch

.. autofunction:: gslib_zero.deutsch_to_vulcan

Leapfrog
--------

Note the reversed order: (dip, dip_direction, pitch).

.. autofunction:: gslib_zero.leapfrog_to_deutsch

.. autofunction:: gslib_zero.deutsch_to_leapfrog

Datamine
--------

Similar to dip direction/dip with different naming.

.. autofunction:: gslib_zero.datamine_to_deutsch

.. autofunction:: gslib_zero.deutsch_to_datamine

Mathematical Convention
-----------------------

Counter-clockwise from East, up from horizontal.

.. autofunction:: gslib_zero.deutsch_to_math

.. autofunction:: gslib_zero.math_to_deutsch

Rotation Matrix
---------------

.. autofunction:: gslib_zero.rotation_matrix_deutsch

Example: Converting Structural Data
-----------------------------------

.. code-block:: python

    from gslib_zero import (
        dipdirection_to_deutsch,
        vulcan_to_deutsch,
        rotation_matrix_deutsch
    )
    import numpy as np

    # Geologist's measurement: plane dipping 45° toward SE (135°)
    dip_direction = 135.0  # Direction of steepest descent
    dip = 45.0

    # Convert to GSLIB convention for variogram modeling
    azimuth, gslib_dip, rake = dipdirection_to_deutsch(dip_direction, dip)
    print(f"GSLIB angles: azimuth={azimuth:.0f}°, dip={gslib_dip:.0f}°")

    # Use in variogram model
    from gslib_zero import VariogramModel
    variogram = VariogramModel.spherical(
        sill=1.0,
        ranges=(200, 100, 50),  # Major, minor, vertical ranges
        nugget=0.1,
        angles=(azimuth, gslib_dip, rake)
    )

Example: Vulcan to GSLIB
------------------------

.. code-block:: python

    from gslib_zero import vulcan_to_deutsch

    # Vulcan structural data: N45E striking plane, dipping 60° SE
    strike = 45.0
    dip = 60.0

    azimuth, gslib_dip, rake = vulcan_to_deutsch(strike, dip)
    print(f"Strike {strike}° with {dip}° dip")
    print(f"→ GSLIB: azimuth={azimuth:.0f}°, dip={gslib_dip:.0f}°")

Example: Building Rotation Matrix
---------------------------------

.. code-block:: python

    from gslib_zero import rotation_matrix_deutsch
    import numpy as np

    # Anisotropy axes: major axis trends N45E, dipping 30°
    R = rotation_matrix_deutsch(azimuth=45.0, dip=30.0, rake=0.0)

    print("Rotation matrix:")
    print(R)

    # Verify orthogonality
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)

    # Verify determinant = 1 (proper rotation, no reflection)
    assert np.isclose(np.linalg.det(R), 1.0)

    # Transform coordinates
    point = np.array([100, 50, 20])
    rotated = R @ point
    print(f"Original: {point}")
    print(f"Rotated: {rotated}")
