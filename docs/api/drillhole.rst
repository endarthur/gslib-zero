Drillhole Utilities
===================

Python utilities for drillhole data processing: desurvey, compositing, and
interval merging.

These functions accept both dict-of-arrays and pandas DataFrames, returning
the same type as the input.

Minimum Curvature Desurvey
--------------------------

.. autofunction:: gslib_zero.desurvey

The minimum curvature method is the industry standard for calculating 3D
coordinates from downhole survey data. It provides smooth interpolation
between survey stations while honoring both endpoints.

Length-Weighted Compositing
---------------------------

.. autofunction:: gslib_zero.composite

Compositing aggregates samples into regular-length intervals, weighting by
the length of each contributing sample. This is essential for:

- Regularizing sample support for variogram analysis
- Preparing data for resource estimation
- Handling variable sample lengths

Result Classes
--------------

.. autoclass:: gslib_zero.CompositeResult
   :members:
   :undoc-members:

Interval Merging
----------------

.. autofunction:: gslib_zero.merge_intervals

Interval merging combines multiple FROM/TO tables with different interval
boundaries into a single table containing all attributes. This is the
classic "interval intersection" problem in drillhole databases.

Example: Complete Drillhole Workflow
------------------------------------

.. code-block:: python

    import numpy as np
    from gslib_zero import desurvey, composite, merge_intervals

    # 1. Define collar data
    collar = {
        'holeid': np.array(['DH001', 'DH002']),
        'x': np.array([1000.0, 1050.0]),
        'y': np.array([2000.0, 2000.0]),
        'z': np.array([500.0, 505.0]),
    }

    # 2. Define survey data (downhole azimuth and dip)
    survey = {
        'holeid': np.array(['DH001', 'DH001', 'DH001',
                           'DH002', 'DH002', 'DH002']),
        'depth': np.array([0.0, 50.0, 100.0, 0.0, 50.0, 100.0]),
        'azimuth': np.array([45.0, 47.0, 50.0, 135.0, 133.0, 130.0]),
        'dip': np.array([-60.0, -58.0, -55.0, -70.0, -68.0, -65.0]),
    }

    # 3. Define assay intervals
    assay = {
        'holeid': np.array(['DH001']*5 + ['DH002']*5),
        'from': np.array([0, 10, 20, 30, 40, 0, 15, 30, 45, 60]),
        'to': np.array([10, 20, 30, 40, 50, 15, 30, 45, 60, 75]),
        'au': np.array([1.2, 2.5, 3.1, 1.8, 0.9, 0.5, 1.1, 2.8, 4.2, 1.5]),
    }

    # 4. Desurvey to get 3D coordinates at sample midpoints
    coords = desurvey(collar, survey, assay)
    print("Desurveyed coordinates:")
    for key in ['holeid', 'x', 'y', 'z', 'depth']:
        print(f"  {key}: {coords[key][:3]}...")

    # 5. Composite to 5m intervals
    composited = composite(assay, length=5.0, min_coverage=0.5)
    print(f"\nComposites: {len(composited.data['from'])} intervals")
    print(f"Coverage range: {composited.coverage.min():.0%} - {composited.coverage.max():.0%}")

Example: Merging Assay and Geology Tables
-----------------------------------------

.. code-block:: python

    import numpy as np
    from gslib_zero import merge_intervals

    # Assay intervals (variable lengths)
    assay = {
        'holeid': np.array(['DH001', 'DH001', 'DH001']),
        'from': np.array([0.0, 1.0, 3.0]),
        'to': np.array([1.0, 3.0, 5.0]),
        'au': np.array([2.5, 1.2, 0.8]),
    }

    # Geology intervals (different boundaries)
    geology = {
        'holeid': np.array(['DH001', 'DH001']),
        'from': np.array([0.0, 2.0]),
        'to': np.array([2.0, 5.0]),
        'rock': np.array(['OXIDE', 'FRESH']),
    }

    # Merge creates intervals at all boundaries
    merged = merge_intervals(assay, geology, tolerance=0.001)

    print("Merged intervals:")
    for i in range(len(merged['from'])):
        print(f"  {merged['from'][i]:.1f}-{merged['to'][i]:.1f}: "
              f"Au={merged['au'][i]:.2f}, Rock={merged['rock'][i]}")

    # Output:
    # 0.0-1.0: Au=2.50, Rock=OXIDE
    # 1.0-2.0: Au=1.20, Rock=OXIDE
    # 2.0-3.0: Au=1.20, Rock=FRESH
    # 3.0-5.0: Au=0.80, Rock=FRESH

Using with Pandas
-----------------

When pandas is available, you can use DataFrames directly:

.. code-block:: python

    import pandas as pd
    from gslib_zero import desurvey, composite

    # Load data as DataFrames
    collar_df = pd.read_csv('collar.csv')
    survey_df = pd.read_csv('survey.csv')
    assay_df = pd.read_csv('assay.csv')

    # Functions accept DataFrames and return DataFrames
    coords_df = desurvey(collar_df, survey_df, assay_df)
    composited = composite(assay_df, length=2.0)

    # Results are pandas DataFrames
    print(coords_df.head())
    print(composited.data.head())
