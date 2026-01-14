Declustering
============

Cell declustering for correcting preferential sampling bias.

When samples are clustered (common in mineral exploration where high-grade
zones are drilled more intensively), the arithmetic mean of samples is
biased. Declustering assigns weights to samples to produce an unbiased
estimate of the population mean.

Cell Declustering
-----------------

.. autofunction:: gslib_zero.declus

Result Classes
--------------

.. autoclass:: gslib_zero.DeclusResult
   :members:
   :undoc-members:

Example: Declustering Clustered Samples
---------------------------------------

.. code-block:: python

    import numpy as np
    from gslib_zero import declus

    # Simulate clustered sampling (high-grade zone more densely sampled)
    np.random.seed(42)

    # Background samples (sparse)
    n_background = 50
    x_bg = np.random.uniform(0, 1000, n_background)
    y_bg = np.random.uniform(0, 1000, n_background)
    values_bg = np.random.normal(5, 1, n_background)

    # High-grade cluster (dense sampling)
    n_cluster = 100
    x_cl = np.random.normal(500, 50, n_cluster)
    y_cl = np.random.normal(500, 50, n_cluster)
    values_cl = np.random.normal(15, 2, n_cluster)

    # Combine
    x = np.concatenate([x_bg, x_cl])
    y = np.concatenate([y_bg, y_cl])
    z = np.zeros(len(x))
    values = np.concatenate([values_bg, values_cl])

    # Naive mean (biased toward cluster)
    naive_mean = values.mean()
    print(f"Naive mean: {naive_mean:.2f}")

    # Declustered mean
    result = declus(
        x, y, z, values,
        cell_size_min=50,
        cell_size_max=200,
        n_cell_sizes=10,
        binary=True
    )

    # Weighted mean
    declustered_mean = np.average(values, weights=result.weights)
    print(f"Declustered mean: {declustered_mean:.2f}")

    # The true population mean should be closer to:
    # (5 * 50 + 15 * 100) / 150 ≈ 11.67 if equal populations
    # But we have more cluster samples, so naive is biased high

Example: Selecting Optimal Cell Size
------------------------------------

.. code-block:: python

    import matplotlib.pyplot as plt
    from gslib_zero import declus

    # Run declustering with many cell sizes
    result = declus(
        x, y, z, values,
        cell_size_min=20,
        cell_size_max=300,
        n_cell_sizes=30,
        binary=True
    )

    # The summary contains mean estimates for each cell size
    cell_sizes = result.summary['cell_sizes']
    means = result.summary['means']

    # Plot mean vs cell size
    plt.figure(figsize=(10, 5))
    plt.plot(cell_sizes, means, 'o-')
    plt.axhline(y=result.declustered_mean, color='r', linestyle='--',
                label=f'Optimal mean: {result.declustered_mean:.2f}')
    plt.xlabel('Cell Size')
    plt.ylabel('Declustered Mean')
    plt.title('Declustering: Mean vs Cell Size')
    plt.legend()
    plt.show()

    # Optimal cell size (minimum variance)
    print(f"Optimal cell size: {result.optimal_cell_size:.1f}")
