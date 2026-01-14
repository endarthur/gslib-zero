gslib-zero
==========

**High-performance geostatistics with battle-tested GSLIB algorithms**

gslib-zero is a Python wrapper for Stanford's GSLIB Fortran 90 programs, providing
a modern API while preserving the reliability of algorithms that have been trusted
by the geostatistics community for decades.

Why gslib-zero?
---------------

GSLIB (Geostatistical Software Library) remains the gold standard for geostatistical
algorithms. The Fortran implementations are numerically stable, well-documented, and
extensively validated through decades of academic and industrial use. However, the
traditional GSLIB workflow—manual parameter file editing and ASCII data exchange—creates
friction in modern data science pipelines.

gslib-zero solves this by:

**Eliminating the I/O bottleneck**
    ASCII parsing is typically the slowest part of a GSLIB workflow. gslib-zero's
    binary I/O mode provides 10-100x faster data exchange for large datasets.

**Providing a Pythonic API**
    Define grids, variograms, and search parameters as Python objects. No more
    counting lines in parameter files or remembering column positions.

**Supporting grid masks**
    Skip inactive cells during estimation and simulation. Re-estimate only updated
    domains without re-running entire grids.

**Pre-built binaries**
    CI-built executables for Windows, Linux, and macOS. No Fortran compiler required.

What's Wrapped
--------------

gslib-zero provides Python wrappers for eight core GSLIB programs:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Program
     - Purpose
   * - :func:`~gslib_zero.declus`
     - Cell declustering for correcting preferential sampling bias
   * - :func:`~gslib_zero.nscore`
     - Normal score transform (Gaussian anamorphosis forward)
   * - :func:`~gslib_zero.backtr`
     - Back-transform from normal scores to original distribution
   * - :func:`~gslib_zero.gamv`
     - Experimental variogram and covariogram calculation
   * - :func:`~gslib_zero.kt3d`
     - 3D kriging: ordinary, simple, with trend, cokriging, external drift
   * - :func:`~gslib_zero.ik3d`
     - Indicator kriging with order relations correction
   * - :func:`~gslib_zero.sgsim`
     - Sequential Gaussian simulation
   * - :func:`~gslib_zero.sisim`
     - Sequential indicator simulation

Additionally, gslib-zero includes Python utilities for:

- **Variogram visualization**: Plot experimental variograms and model overlays
- **Drillhole processing**: Minimum curvature desurvey, compositing, interval merging
- **Rotation conventions**: Convert between GSLIB (Deutsch), Leapfrog, Datamine, and Vulcan

Quick Example
-------------

.. code-block:: python

    import numpy as np
    from gslib_zero import kt3d, GridSpec, VariogramModel, SearchParameters

    # Sample data
    x = np.random.uniform(0, 100, 50)
    y = np.random.uniform(0, 100, 50)
    z = np.zeros(50)
    values = np.random.normal(10, 2, 50)

    # Define estimation grid
    grid = GridSpec(
        nx=20, ny=20, nz=1,
        xmin=2.5, ymin=2.5, zmin=0,
        xsiz=5, ysiz=5, zsiz=1
    )

    # Define variogram model
    variogram = VariogramModel.spherical(
        sill=4.0,           # Variance contribution
        ranges=(50, 50, 1), # Correlation ranges (major, minor, vertical)
        nugget=0.5          # Nugget effect
    )

    # Define search neighborhood
    search = SearchParameters(
        radius1=100, radius2=100, radius3=10,
        min_samples=4, max_samples=16
    )

    # Run ordinary kriging with binary I/O
    result = kt3d(x, y, z, values, grid, variogram, search, binary=True)

    print(f"Estimate shape: {result.estimate.shape}")
    print(f"Mean estimate: {result.estimate.mean():.2f}")

Installation
------------

.. code-block:: bash

    pip install gslib-zero

For drillhole utilities with pandas support:

.. code-block:: bash

    pip install gslib-zero[drillhole]

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   conventions
   api/index
   changelog

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
