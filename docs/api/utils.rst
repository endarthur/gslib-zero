Utilities
=========

Core data structures and I/O utilities.

Grid Specification
------------------

.. autoclass:: gslib_zero.GridSpec
   :members:
   :undoc-members:

Variogram Model
---------------

.. autoclass:: gslib_zero.VariogramModel
   :members:
   :undoc-members:

.. autoclass:: gslib_zero.VariogramType
   :members:
   :undoc-members:

Binary I/O
----------

.. autoclass:: gslib_zero.BinaryIO
   :members:
   :undoc-members:

ASCII I/O
---------

.. autoclass:: gslib_zero.AsciiIO
   :members:
   :undoc-members:

Running GSLIB Programs
----------------------

.. autofunction:: gslib_zero.run_gslib

Experimental Warning
--------------------

.. autoclass:: gslib_zero.ExperimentalWarning
   :members:
   :undoc-members:

Example: Working with GridSpec
------------------------------

.. code-block:: python

    from gslib_zero import GridSpec
    import numpy as np

    # Define a 3D grid
    grid = GridSpec(
        nx=100, ny=100, nz=10,
        xmin=5.0, ymin=5.0, zmin=0.5,
        xsiz=10.0, ysiz=10.0, zsiz=1.0
    )

    print(f"Total cells: {grid.ncells}")
    print(f"Shape (nz, ny, nx): {grid.shape}")
    print(f"X extent: {grid.xmin} to {grid.xmax}")

    # Get cell centers
    x_centers, y_centers, z_centers = grid.cell_centers()
    print(f"First X center: {x_centers[0]}, Last: {x_centers[-1]}")

    # Get 3D meshgrid
    X, Y, Z = grid.meshgrid()
    print(f"Meshgrid shape: {X.shape}")

    # Check if point is in grid
    print(f"Point (500, 500, 5) in grid: {grid.contains_point(500, 500, 5)}")

    # Get grid indices for a point
    indices = grid.point_to_index(505, 505, 5.5)
    print(f"Indices (iz, iy, ix): {indices}")

Example: Building Variogram Models
----------------------------------

.. code-block:: python

    from gslib_zero import VariogramModel, VariogramType

    # Simple spherical model
    simple = VariogramModel.spherical(
        sill=1.0,
        ranges=(100, 50, 10),  # Major, minor, vertical
        nugget=0.1
    )
    print(f"Total sill: {simple.total_sill}")

    # Nested model with two structures
    nested = VariogramModel(nugget=0.05)
    nested.add_structure(
        VariogramType.SPHERICAL,
        sill=0.4,
        ranges=(30, 30, 5),
        angles=(0, 0, 0)
    )
    nested.add_structure(
        VariogramType.EXPONENTIAL,
        sill=0.55,
        ranges=(150, 100, 20),
        angles=(45, 0, 0)  # Rotated anisotropy
    )
    print(f"Nested structures: {len(nested.structures)}")
    print(f"Total sill: {nested.total_sill}")

Example: Binary I/O
-------------------

.. code-block:: python

    import numpy as np
    from gslib_zero import BinaryIO

    # Write a 3D array
    data = np.random.randn(5, 20, 30).astype(np.float32)
    BinaryIO.write_array(data, 'output.bin', fortran_order=True)

    # Read it back
    loaded = BinaryIO.read_array('output.bin', dtype=np.float32)
    assert np.allclose(data, loaded)

    # Write a mask
    mask = np.ones((5, 20, 30), dtype=np.int8)
    mask[2:4, 10:15, 15:25] = 0  # Inactive region
    BinaryIO.write_mask(mask, 'mask.bin')

    # Read mask
    loaded_mask = BinaryIO.read_mask('mask.bin')
    assert np.array_equal(mask, loaded_mask)

Example: ASCII I/O
------------------

.. code-block:: python

    import numpy as np
    from gslib_zero import AsciiIO

    # Write GSLIB-format data
    data = {
        'x': np.array([1, 2, 3, 4, 5]),
        'y': np.array([10, 20, 30, 40, 50]),
        'value': np.array([1.5, 2.3, 0.8, 3.1, 2.7]),
    }
    AsciiIO.write_data('samples.dat', data, title='Sample Data')

    # Read it back
    names, array = AsciiIO.read_data('samples.dat')
    print(f"Columns: {names}")
    print(f"Shape: {array.shape}")

    # Read single column by name
    values = AsciiIO.read_column('samples.dat', column='value')
    print(f"Values: {values}")
