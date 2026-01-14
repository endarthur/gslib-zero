Simulation
==========

Sequential simulation methods for generating equally probable realizations
of spatial variables.

Sequential Gaussian Simulation
------------------------------

.. autofunction:: gslib_zero.sgsim

Sequential Indicator Simulation
-------------------------------

.. autofunction:: gslib_zero.sisim

Result Classes
--------------

.. autoclass:: gslib_zero.SimulationResult
   :members:
   :undoc-members:

Example: Conditional Simulation
-------------------------------

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from gslib_zero import sgsim, nscore, backtr, GridSpec, VariogramModel

    # Conditioning data (should be normal scores for sgsim)
    np.random.seed(42)
    x = np.random.uniform(0, 100, 20)
    y = np.random.uniform(0, 100, 20)
    z = np.zeros(20)
    raw_values = np.random.lognormal(2, 0.5, 20)

    # Transform to normal scores
    ns_values, table = nscore(raw_values, binary=True)

    # Grid and model
    grid = GridSpec(nx=50, ny=50, nz=1, xmin=1, ymin=1, zmin=0,
                    xsiz=2, ysiz=2, zsiz=1)
    variogram = VariogramModel.spherical(sill=0.9, ranges=(30, 30, 1), nugget=0.1)

    # Generate 10 realizations
    result = sgsim(
        x, y, z, ns_values,
        grid=grid,
        variogram=variogram,
        search_radius=(50, 50, 1),
        nrealizations=10,
        seed=42,
        binary=True
    )

    print(f"Realizations shape: {result.realizations.shape}")

    # Back-transform each realization
    realizations_bt = np.zeros_like(result.realizations)
    for i in range(10):
        realizations_bt[i] = backtr(
            result.realizations[i].ravel(),
            table,
            tmin=raw_values.min() * 0.5,
            tmax=raw_values.max() * 1.5,
            binary=True
        ).reshape(grid.shape)

    # Plot first 4 realizations
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(realizations_bt[i, 0], origin='lower', cmap='viridis')
        ax.scatter(x/2, y/2, c='red', s=20)  # Conditioning points
        ax.set_title(f'Realization {i+1}')
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

Example: Unconditional Simulation
---------------------------------

.. code-block:: python

    from gslib_zero import sgsim, GridSpec, VariogramModel

    # No conditioning data
    result = sgsim(
        None, None, None, None,  # No conditioning
        grid=GridSpec(nx=100, ny=100, nz=1, xmin=0.5, ymin=0.5, zmin=0,
                      xsiz=1, ysiz=1, zsiz=1),
        variogram=VariogramModel.spherical(sill=1.0, ranges=(20, 20, 1)),
        search_radius=(50, 50, 1),
        nrealizations=5,
        seed=12345,
        binary=True
    )

    # Results are standard normal (mean=0, var=1)
    print(f"Mean: {result.realizations.mean():.4f}")
    print(f"Variance: {result.realizations.var():.4f}")
