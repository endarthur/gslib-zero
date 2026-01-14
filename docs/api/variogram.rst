Variogram Analysis
==================

Compute and visualize experimental variograms and fit theoretical models.

Experimental Variogram
----------------------

.. autofunction:: gslib_zero.gamv

Result Classes
--------------

.. autoclass:: gslib_zero.VariogramResult
   :members:
   :undoc-members:

Variogram Plotting
------------------

.. autofunction:: gslib_zero.plot_experimental

.. autofunction:: gslib_zero.plot_model

.. autofunction:: gslib_zero.plot_variogram

Model Evaluation
----------------

.. autofunction:: gslib_zero.evaluate_variogram

Export
------

.. autofunction:: gslib_zero.export_variogram_par

Example: Directional Variograms
-------------------------------

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from gslib_zero import gamv, plot_experimental, VariogramModel

    # Sample data with anisotropy
    np.random.seed(42)
    n = 200
    x = np.random.uniform(0, 1000, n)
    y = np.random.uniform(0, 1000, n)
    z = np.zeros(n)
    values = np.random.normal(0, 1, n)

    # Compute variograms in multiple directions
    directions = [0, 45, 90, 135]  # N, NE, E, SE
    results = []

    for azm in directions:
        result = gamv(
            x, y, z, values,
            lag_distance=50,
            lag_tolerance=25,
            n_lags=15,
            azimuth=azm,
            azimuth_tolerance=22.5,
            bandwidth=100,
            binary=True
        )
        results.append(result)

    # Plot all directions
    labels = ['N-S', 'NE-SW', 'E-W', 'SE-NW']
    ax = plot_experimental(results, labels=labels, show_pairs=True)
    ax.set_title('Directional Variograms')
    plt.show()
