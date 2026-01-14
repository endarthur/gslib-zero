Transforms
==========

Normal score transformation and back-transformation for Gaussian-based
geostatistical methods.

Normal Score Transform
----------------------

.. autofunction:: gslib_zero.nscore

Back-Transform
--------------

.. autofunction:: gslib_zero.backtr

Example: Round-Trip Transformation
----------------------------------

.. code-block:: python

    import numpy as np
    from gslib_zero import nscore, backtr

    # Original data (lognormal distribution)
    np.random.seed(42)
    original = np.exp(np.random.normal(2, 0.5, 1000))
    print(f"Original: mean={original.mean():.2f}, std={original.std():.2f}")

    # Transform to normal scores
    ns_values, transform_table = nscore(original, binary=True)
    print(f"Normal scores: mean={ns_values.mean():.4f}, std={ns_values.std():.4f}")

    # Back-transform
    recovered = backtr(
        ns_values,
        transform_table,
        tmin=original.min() * 0.5,
        tmax=original.max() * 1.5,
        binary=True
    )
    print(f"Recovered: mean={recovered.mean():.2f}, std={recovered.std():.2f}")

    # Check correlation
    correlation = np.corrcoef(original, recovered)[0, 1]
    print(f"Correlation: {correlation:.6f}")
