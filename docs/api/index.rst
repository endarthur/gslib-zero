API Reference
=============

gslib-zero is organized into modules by functionality. The main entry point
exports all commonly used functions and classes, allowing simple imports:

.. code-block:: python

    from gslib_zero import kt3d, GridSpec, VariogramModel

Module Organization
-------------------

.. code-block:: text

    gslib_zero/
    ├── Core
    │   ├── run_gslib()        - Execute GSLIB programs
    │   ├── BinaryIO           - Binary file I/O
    │   └── AsciiIO            - ASCII file I/O
    │
    ├── Data Structures
    │   ├── GridSpec           - Grid definition
    │   ├── VariogramModel     - Variogram specification
    │   └── SearchParameters   - Kriging search neighborhood
    │
    ├── GSLIB Programs
    │   ├── Transforms
    │   │   ├── nscore()       - Normal score transform
    │   │   └── backtr()       - Back-transform
    │   │
    │   ├── Variogram
    │   │   └── gamv()         - Experimental variogram
    │   │
    │   ├── Estimation
    │   │   ├── kt3d()         - 3D kriging
    │   │   └── ik3d()         - Indicator kriging
    │   │
    │   ├── Simulation
    │   │   ├── sgsim()        - Sequential Gaussian simulation
    │   │   └── sisim()        - Sequential indicator simulation
    │   │
    │   └── Declustering
    │       └── declus()       - Cell declustering
    │
    ├── Python Utilities
    │   ├── Plotting
    │   │   ├── plot_experimental()
    │   │   ├── plot_model()
    │   │   └── plot_variogram()
    │   │
    │   ├── Drillhole
    │   │   ├── desurvey()     - Minimum curvature desurvey
    │   │   ├── composite()    - Length-weighted compositing
    │   │   └── merge_intervals()
    │   │
    │   └── Rotations
    │       ├── dipdirection_to_deutsch()
    │       ├── vulcan_to_deutsch()
    │       └── ...

Common Workflows
----------------

**Standard Kriging Workflow**

.. code-block:: python

    from gslib_zero import (
        nscore, backtr,
        gamv, plot_variogram,
        kt3d,
        GridSpec, VariogramModel, SearchParameters
    )

    # 1. Transform data
    ns_values, table = nscore(values)

    # 2. Analyze variogram
    vario = gamv(x, y, z, ns_values, ...)
    model = VariogramModel.spherical(...)
    plot_variogram(vario, model)

    # 3. Estimate
    grid = GridSpec(...)
    search = SearchParameters(...)
    result = kt3d(x, y, z, ns_values, grid, model, search)

    # 4. Back-transform
    estimates = backtr(result.estimate.ravel(), table)

**Simulation Workflow**

.. code-block:: python

    from gslib_zero import sgsim, nscore, backtr

    # Transform to normal scores
    ns_values, table = nscore(values)

    # Simulate
    result = sgsim(x, y, z, ns_values, grid, variogram, search,
                   nrealizations=100)

    # Back-transform each realization
    for i in range(100):
        realization = backtr(result.realizations[i].ravel(), table)

Module Reference
----------------

.. toctree::
   :maxdepth: 2

   transforms
   variogram
   estimation
   simulation
   declustering
   drillhole
   rotations
   utils
