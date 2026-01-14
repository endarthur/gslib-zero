Estimation
==========

Kriging estimation methods: ordinary kriging, simple kriging, and indicator kriging.

Ordinary/Simple Kriging
-----------------------

.. autofunction:: gslib_zero.kt3d

Indicator Kriging
-----------------

.. autofunction:: gslib_zero.ik3d

Data Structures
---------------

.. autoclass:: gslib_zero.SearchParameters
   :members:
   :undoc-members:

.. autoclass:: gslib_zero.KrigingResult
   :members:
   :undoc-members:

.. autoclass:: gslib_zero.IndicatorKrigingResult
   :members:
   :undoc-members:

Example: Simple vs Ordinary Kriging
-----------------------------------

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gslib_zero import kt3d, GridSpec, VariogramModel, SearchParameters

    # Sample data as DataFrame
    np.random.seed(42)
    df = pd.DataFrame({
        'x': np.random.uniform(0, 100, 30),
        'y': np.random.uniform(0, 100, 30),
        'z': np.zeros(30),
        'grade': np.random.normal(10, 2, 30),
    })

    # Grid and model
    grid = GridSpec(nx=20, ny=20, nz=1, xmin=2.5, ymin=2.5, zmin=0,
                    xsiz=5, ysiz=5, zsiz=1)
    variogram = VariogramModel.spherical(sill=4.0, ranges=(50, 50, 1), nugget=0.5)
    search = SearchParameters(radius1=100, radius2=100, radius3=10,
                              min_samples=4, max_samples=12)

    # Ordinary kriging - pass Series directly
    ok_result = kt3d(df.x, df.y, df.z, df.grade, grid, variogram, search,
                     kriging_type="ordinary", binary=True)

    # Simple kriging - using DataFrame pattern
    sk_result = kt3d(data=df, value_col='grade', grid=grid, variogram=variogram,
                     search=search, kriging_type="simple", sk_mean=df.grade.mean(),
                     binary=True)

    print(f"OK mean: {ok_result.estimate.mean():.2f}")
    print(f"SK mean: {sk_result.estimate.mean():.2f}")
    print(f"Data mean: {df.grade.mean():.2f}")

Example: Kriging with Mask
--------------------------

.. code-block:: python

    import numpy as np
    from gslib_zero import kt3d, GridSpec, VariogramModel, SearchParameters

    # Create mask for irregular domain
    grid = GridSpec(nx=50, ny=50, nz=1, xmin=5, ymin=5, zmin=0,
                    xsiz=10, ysiz=10, zsiz=1)
    X, Y, Z = grid.meshgrid()

    # Only estimate within a polygon or distance threshold
    mask = np.zeros((grid.nz, grid.ny, grid.nx), dtype=np.int8)
    center = (grid.xmax + grid.xmin) / 2, (grid.ymax + grid.ymin) / 2
    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask[dist < 200] = 1  # Active cells within radius

    # Run kriging with mask - using DataFrame pattern
    result = kt3d(data=df, value_col='grade', grid=grid, variogram=variogram,
                  search=search, mask=mask, binary=True)
