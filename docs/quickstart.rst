Quick Start Guide
=================

This guide walks through a complete geostatistical workflow using gslib-zero:
from raw data to kriged estimates with variogram analysis.

Sample Dataset
--------------

Let's create a synthetic dataset that mimics a typical mineral deposit:

.. code-block:: python

    import numpy as np

    # Set seed for reproducibility
    np.random.seed(42)

    # Generate 100 sample locations
    n_samples = 100
    x = np.random.uniform(0, 1000, n_samples)
    y = np.random.uniform(0, 1000, n_samples)
    z = np.zeros(n_samples)  # 2D example (single level)

    # Generate spatially correlated values using a simple trend + noise
    # This simulates a grade distribution with spatial structure
    trend = 0.005 * x + 0.003 * y
    noise = np.random.normal(0, 1.5, n_samples)
    values = 5.0 + trend + noise

    print(f"Samples: {n_samples}")
    print(f"Grade range: {values.min():.2f} - {values.max():.2f}")
    print(f"Mean: {values.mean():.2f}, Std: {values.std():.2f}")

Step 1: Exploratory Data Analysis
---------------------------------

Before geostatistical analysis, understand your data distribution:

.. code-block:: python

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    axes[0].hist(values, bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Grade')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Grade Distribution')

    # Sample locations colored by value
    scatter = axes[1].scatter(x, y, c=values, cmap='viridis', s=50)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_title('Sample Locations')
    plt.colorbar(scatter, ax=axes[1], label='Grade')

    plt.tight_layout()
    plt.show()

Step 2: Normal Score Transform
------------------------------

Many geostatistical methods assume Gaussian distributions. Transform your data:

.. code-block:: python

    from gslib_zero import nscore

    # Transform to normal scores
    nscore_values, transform_table = nscore(values, binary=True)

    print(f"Original mean: {values.mean():.3f}")
    print(f"Normal score mean: {nscore_values.mean():.6f}")  # Should be ~0
    print(f"Normal score std: {nscore_values.std():.6f}")   # Should be ~1

Step 3: Variogram Analysis
--------------------------

Compute experimental variograms to understand spatial correlation:

.. code-block:: python

    from gslib_zero import gamv, plot_variogram, VariogramModel

    # Compute omnidirectional variogram
    variogram_result = gamv(
        x, y, z, nscore_values,
        lag_distance=50.0,     # Lag spacing
        lag_tolerance=25.0,    # Lag tolerance
        n_lags=15,             # Number of lags
        azimuth=0.0,           # Direction (0 = omnidirectional when atol=90)
        azimuth_tolerance=90.0,
        binary=True
    )

    print(f"Number of lags computed: {len(variogram_result.gamma)}")
    print(f"Max lag distance: {variogram_result.lag_distances.max():.0f}")

Step 4: Variogram Model Fitting
-------------------------------

Fit a theoretical model to the experimental variogram. gslib-zero does not
perform automatic fitting (which often produces poor results). Instead,
manually select model parameters based on the experimental points:

.. code-block:: python

    # Create a spherical variogram model
    # Parameters should be chosen by examining the experimental variogram plot
    variogram_model = VariogramModel.spherical(
        sill=0.85,              # Sill (total variance contribution)
        ranges=(300, 300, 1),   # Ranges (major, minor, vertical)
        nugget=0.15             # Nugget effect (short-scale variability)
    )

    # Plot experimental variogram with fitted model
    ax = plot_variogram(
        experimental=variogram_result,
        model=variogram_model,
        title="Variogram: Experimental vs Model"
    )
    plt.show()

    # Verify model parameters
    print(f"Total sill: {variogram_model.total_sill}")
    print(f"Nugget: {variogram_model.nugget}")

Step 5: Define Estimation Grid
------------------------------

Create a grid covering the area of interest:

.. code-block:: python

    from gslib_zero import GridSpec

    # Define a 50x50 cell grid covering the sample area
    grid = GridSpec(
        nx=50, ny=50, nz=1,       # Grid dimensions
        xmin=10, ymin=10, zmin=0,  # Grid origin (cell center)
        xsiz=20, ysiz=20, zsiz=1   # Cell sizes
    )

    print(f"Grid cells: {grid.ncells}")
    print(f"Grid extent: X({grid.xmin}-{grid.xmax}), Y({grid.ymin}-{grid.ymax})")

Step 6: Kriging
---------------

Run ordinary kriging to estimate values at grid nodes:

.. code-block:: python

    from gslib_zero import kt3d, SearchParameters

    # Define search neighborhood
    search = SearchParameters(
        radius1=400,       # Maximum search radius
        radius2=400,       # Medium search radius
        radius3=10,        # Minimum search radius (vertical)
        min_samples=4,     # Minimum samples required
        max_samples=16,    # Maximum samples to use
    )

    # Run kriging on normal scores
    result = kt3d(
        x, y, z, nscore_values,
        grid=grid,
        variogram=variogram_model,
        search=search,
        kriging_type="ordinary",
        binary=True
    )

    print(f"Estimate shape: {result.estimate.shape}")
    print(f"Kriged mean: {result.estimate.mean():.4f}")

Step 7: Back-Transform
----------------------

Transform kriged normal scores back to original units:

.. code-block:: python

    from gslib_zero import backtr

    # Back-transform the estimates
    estimates_original = backtr(
        result.estimate.ravel(),
        transform_table,
        tmin=values.min() * 0.9,  # Extrapolation limits
        tmax=values.max() * 1.1,
        binary=True
    )

    # Reshape to grid dimensions
    estimates_original = estimates_original.reshape(result.estimate.shape)

    print(f"Back-transformed mean: {estimates_original.mean():.2f}")
    print(f"Original data mean: {values.mean():.2f}")

Step 8: Visualize Results
-------------------------

Display the kriged estimates:

.. code-block:: python

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Kriged estimates
    im1 = axes[0].imshow(
        estimates_original[0],  # First (only) z-level
        extent=[grid.xmin, grid.xmax, grid.ymin, grid.ymax],
        origin='lower',
        cmap='viridis'
    )
    axes[0].scatter(x, y, c='red', s=20, alpha=0.5, label='Samples')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('Kriged Estimates')
    axes[0].legend()
    plt.colorbar(im1, ax=axes[0], label='Grade')

    # Kriging variance (uncertainty)
    im2 = axes[1].imshow(
        result.variance[0],
        extent=[grid.xmin, grid.xmax, grid.ymin, grid.ymax],
        origin='lower',
        cmap='Reds'
    )
    axes[1].scatter(x, y, c='blue', s=20, alpha=0.5, label='Samples')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_title('Kriging Variance')
    axes[1].legend()
    plt.colorbar(im2, ax=axes[1], label='Variance')

    plt.tight_layout()
    plt.show()

Using Binary I/O for Performance
--------------------------------

For large datasets, binary I/O significantly improves performance:

.. code-block:: python

    import time

    # Time comparison for a larger grid
    large_grid = GridSpec(
        nx=200, ny=200, nz=1,
        xmin=10, ymin=10, zmin=0,
        xsiz=5, ysiz=5, zsiz=1
    )

    # ASCII mode
    start = time.time()
    result_ascii = kt3d(x, y, z, nscore_values, large_grid,
                        variogram_model, search, binary=False)
    ascii_time = time.time() - start

    # Binary mode
    start = time.time()
    result_binary = kt3d(x, y, z, nscore_values, large_grid,
                         variogram_model, search, binary=True)
    binary_time = time.time() - start

    print(f"ASCII time: {ascii_time:.2f}s")
    print(f"Binary time: {binary_time:.2f}s")
    print(f"Speedup: {ascii_time/binary_time:.1f}x")

Using Grid Masks
----------------

Skip inactive cells to save computation time:

.. code-block:: python

    # Create a circular mask (estimate only within a radius)
    X, Y, Z = grid.meshgrid()
    center_x, center_y = 500, 500
    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    mask = (distance < 400).astype(np.int8)

    print(f"Active cells: {mask.sum()} / {grid.ncells}")

    # Run kriging with mask
    result_masked = kt3d(
        x, y, z, nscore_values,
        grid=grid,
        variogram=variogram_model,
        search=search,
        mask=mask,
        binary=True
    )

    # Masked cells will have value -999 (UNEST)
    valid_estimates = result_masked.estimate[mask == 1]
    print(f"Mean of valid estimates: {valid_estimates.mean():.4f}")

Next Steps
----------

- See :doc:`conventions` for details on GSLIB's coordinate and rotation conventions
- See :doc:`api/index` for complete API reference
- Explore :func:`~gslib_zero.sgsim` for stochastic simulation
- Use :func:`~gslib_zero.gamv` with multiple directions for anisotropy analysis
