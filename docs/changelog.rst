Changelog
=========

Version 1.0.0 (2024)
--------------------

**First stable release of gslib-zero.**

This release marks the completion of the core feature set: Python wrappers for
all eight GSLIB programs with binary I/O support, grid masks, and comprehensive
Python utilities.

Features
^^^^^^^^

**GSLIB Program Wrappers**

- :func:`~gslib_zero.nscore`: Normal score transform with binary I/O
- :func:`~gslib_zero.backtr`: Back-transform with binary I/O
- :func:`~gslib_zero.declus`: Cell declustering with binary I/O
- :func:`~gslib_zero.gamv`: Experimental variogram calculation with binary I/O
- :func:`~gslib_zero.kt3d`: 3D kriging (ordinary, simple) with binary I/O and grid masks
- :func:`~gslib_zero.ik3d`: Indicator kriging with binary I/O and grid masks
- :func:`~gslib_zero.sgsim`: Sequential Gaussian simulation with binary I/O and grid masks
- :func:`~gslib_zero.sisim`: Sequential indicator simulation with binary I/O and grid masks

**Binary I/O**

- 10-100x faster data exchange compared to ASCII for large datasets
- Direct numpy array serialization/deserialization
- Fortran-compatible column-major ordering

**Grid Masks**

- Skip inactive cells during estimation and simulation
- Re-estimate only updated domains without full grid computation
- Significant performance improvement for sparse domains

**Flexible Input API**

All GSLIB wrappers accept flexible input formats:

- Pass pandas Series directly: ``kt3d(df.x, df.y, df.z, df.grade, ...)``
- Use DataFrame with column names: ``kt3d(data=df, value_col='grade', ...)``
- NumPy arrays and lists work without conversion

No ``.values`` needed when working with pandas—gslib-zero handles the conversion
automatically.

**Python Utilities**

- Variogram plotting: :func:`~gslib_zero.plot_experimental`, :func:`~gslib_zero.plot_model`,
  :func:`~gslib_zero.plot_variogram`
- Variogram export: :func:`~gslib_zero.export_variogram_par`
- Drillhole desurvey: :func:`~gslib_zero.desurvey` (minimum curvature method)
- Compositing: :func:`~gslib_zero.composite` (length-weighted with domain breaks)
- Interval merging: :func:`~gslib_zero.merge_intervals`
- Rotation conversions for Leapfrog, Datamine, Vulcan, and dip direction/dip conventions

**Example Notebooks**

- ``01_quickstart_kriging.ipynb`` - Complete kriging workflow
- ``02_variogram_analysis.ipynb`` - Directional variograms and model fitting
- ``03_drillhole_workflow.ipynb`` - Desurvey, composite, merge, and estimate

**Double Precision (Experimental)**

- Optional f64 builds for kt3d, ik3d, and sisim
- Higher numerical precision for sensitive calculations
- Marked as experimental; use with validation

Breaking Changes from Pre-1.0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Package renamed to ``gslib-zero`` (was ``gslib_zero`` in development)
- Version bumped to 1.0.0
- API is now considered stable; breaking changes will follow semantic versioning

Known Issues
^^^^^^^^^^^^

- ``sgsim`` with ``precision='f64'`` crashes (SIGSEGV) due to memory layout issues
  in the Fortran code. Use ``precision='f32'`` (default) for now.
- Some GSLIB programs may produce slightly different results between platforms due
  to floating-point arithmetic differences in the Fortran runtime.

Dependencies
^^^^^^^^^^^^

- Python >= 3.9
- NumPy >= 1.20
- Matplotlib >= 3.5
- Optional: pandas >= 1.4 (for drillhole utilities with DataFrame support)

Pre-1.0 Development History
---------------------------

**v0.5** - Python Utilities

- Added variogram plotting and export
- Added drillhole utilities (desurvey, composite, merge_intervals)
- Added rotation convention conversions

**v0.4** - Double Precision Builds

- Added f64 variants of executables
- Identified sgsim_f64 crash issue

**v0.3** - Grid Mask Support

- Added mask parameter to kt3d, ik3d, sgsim, sisim
- Implemented binary mask file format

**v0.2** - Binary I/O

- Implemented binary I/O for all eight programs
- Added BinaryIO class for numpy array serialization

**v0.1** - Initial Release

- Python wrappers for eight GSLIB programs
- ASCII I/O with original GSLIB binaries
- Parameter file generation from Python
