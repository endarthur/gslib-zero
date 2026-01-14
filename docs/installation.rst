Installation
============

gslib-zero requires Python 3.9 or later and includes pre-built Fortran binaries
for Windows, Linux, and macOS.

Standard Installation
---------------------

Install from PyPI:

.. code-block:: bash

    pip install gslib-zero

This installs the core package with NumPy and Matplotlib dependencies.

Optional Dependencies
---------------------

**Drillhole utilities with pandas**

For desurvey, compositing, and interval merging with DataFrame support:

.. code-block:: bash

    pip install gslib-zero[drillhole]

**Development tools**

For running tests and contributing:

.. code-block:: bash

    pip install gslib-zero[dev]

**Documentation build**

For building documentation locally:

.. code-block:: bash

    pip install gslib-zero[docs]

Development Installation
------------------------

To install from source for development:

.. code-block:: bash

    git clone https://github.com/endarthur/gslib-zero.git
    cd gslib-zero
    pip install -e ".[dev]"

This installs the package in editable mode with development dependencies.

Binary Location
---------------

gslib-zero looks for GSLIB executables in the following locations:

1. The ``GSLIB_BIN_DIR`` environment variable (if set)
2. The ``bin/`` directory in the package installation

Pre-built binaries are included in the package distribution. If you need to use
custom binaries (e.g., for specialized hardware or debugging), set the environment
variable:

.. code-block:: bash

    export GSLIB_BIN_DIR=/path/to/your/binaries

Platform Notes
--------------

**Windows**

Executables have the ``.exe`` extension. The package includes statically linked
binaries that work without additional runtime dependencies.

**Linux**

Binaries are built with gfortran and statically linked. They should work on most
distributions without additional dependencies.

**macOS**

Binaries are built for both Intel and Apple Silicon. If you encounter issues,
ensure you're using the correct architecture or set ``GSLIB_BIN_DIR`` to point
to compatible binaries.

Verifying Installation
----------------------

After installation, verify that gslib-zero can find the executables:

.. code-block:: python

    from gslib_zero import run_gslib
    from gslib_zero.core import get_executable

    # Check that kt3d executable exists
    kt3d_path = get_executable("kt3d")
    print(f"kt3d found at: {kt3d_path}")

You can also run the test suite:

.. code-block:: bash

    pytest tests/ -v

Troubleshooting
---------------

**"GSLIB executable not found" error**

1. Verify the binaries exist in the expected location
2. Check file permissions (executables need execute permission on Unix)
3. Set ``GSLIB_BIN_DIR`` to the correct path

**Import errors**

Ensure you have the required dependencies:

.. code-block:: bash

    pip install numpy matplotlib

**Tests failing**

Some tests require the GSLIB binaries. If tests are skipped or failing:

1. Verify binaries are accessible
2. Check that the correct architecture binaries are being used
3. Run with verbose output: ``pytest -v --tb=long``
