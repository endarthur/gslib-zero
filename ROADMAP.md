# gslib-zero Roadmap

## Current Status (v0.1)

### Completed
- [x] Python wrappers for 8 GSLIB programs (nscore, backtr, declus, gamv, kt3d, ik3d, sgsim, sisim)
- [x] Par file generation from Python
- [x] ASCII I/O with original GSLIB binaries
- [x] CI/CD for multi-platform Fortran builds (Linux, macOS, Windows)
- [x] gfortran compatibility fixes (msflib removal, static linking)
- [x] Binary I/O for kt3d (gslib-zero modified binary)
- [x] Binary vs ASCII cross-validation test

---

## Phase 1: Binary I/O for All Programs

Extend binary I/O support to remaining programs. Pattern established with kt3d.

| Program | Binary Output | Binary Input | Status |
|---------|--------------|--------------|--------|
| kt3d    | ✅ Done      | N/A          | Complete |
| ik3d    | ✅ Done      | N/A          | Complete |
| sgsim   | ✅ Done      | ⬜ Todo      | Complete |
| sisim   | ✅ Done      | ⬜ Todo      | Complete |
| gamv    | ✅ Done      | N/A          | Complete |
| nscore  | ✅ Done      | N/A          | Complete |
| backtr  | ✅ Done      | N/A          | Complete |
| declus  | ✅ Done      | N/A          | Complete |

### Binary Format Specification
```
Header: [ndim: int32] [shape: int32 × ndim]
Data:   [values: float32 × prod(shape)]
```

For grid outputs (kt3d, sgsim, etc.): shape = (nvars, nz, ny, nx)

### Fortran Source Changes Summary

All changes are **minimal and limited to I/O**. No algorithm changes.

#### kt3d.for / kt3d.inc
- **kt3d.inc**: Added `ibinary` to `/datcom/` common block
- **readparm**: Added `read(lin,*) ibinary` after output filename (line ~237)
- **File open**: Conditional `access='STREAM',form='UNFORMATTED'` for binary mode
- **Header write**: `write(lout) 4, 2, nz, ny, nx` (ndim, nvars, dimensions)
- **Data write**: `write(lout) est, estv` instead of formatted write
- **makepar**: Added ibinary parameter line

#### sgsim.for / sgsim.inc
- **sgsim.inc**: Added `ibinary` to `/generl/` common block
- **readparm**: Added `read(lin,*) ibinary` after output filename
- **File open**: Conditional stream/unformatted for binary mode
- **Header write**: `write(lout) 4, nsim, nz, ny, nx`
- **Data write**: `write(lout) simval` per cell (sequential by realization)
- **makepar**: Added ibinary parameter line

#### sisim.for / sisim.inc
- **sisim.inc**: Added `ibinary` to `/simula/` common block
- **readparm**: Added `read(lin,*) ibinary` after output filename
- **File open**: Conditional stream/unformatted for binary mode
- **Header write**: `write(lout) 4, nsim, nz, ny, nx`
- **Data write**: `write(lout) sim(ind)` per cell
- **makepar**: Added ibinary parameter line

#### ik3d.for / ik3d.inc
- **ik3d.inc**: Added `ibinary` to `/datcom/` common block
- **readparm**: Added `read(lin,*) ibinary` after output filename
- **File open**: Conditional stream/unformatted for binary mode
- **Header write**: `write(lout) 4, ncut, nz, ny, nx`
- **Data write**: `write(lout) (ccdfo(i),i=1,ncut)` per cell
- **makepar**: Added ibinary parameter line

#### gamv.for (uses module instead of .inc)
- **geostat module**: Added `ibinary` to module variables
- **readparm**: Added `read(lin,*) ibinary` after output filename
- **File open**: Conditional stream/unformatted for binary mode in `writeout`
- **Header write**: `write(lout) 4, 5, nvarg, ndir, nlag+2` (ndim, nfields, variogram dims)
- **Data write**: `write(lout) real(dis),real(gam),real(np),real(hm),real(tm)` per lag
- **makepar**: Added ibinary parameter line (format 150)
- **Note**: gamv has different output structure (variograms, not grids)

#### nscore.for (no .inc file, variables in main program)
- **main program**: Added `ibinary` integer variable
- **readparm**: Added `read(lin,*) ibinary` after output filename
- **File open**: Conditional stream/unformatted for binary mode
- **Header write**: `write(lout) 1, nout` (ndim=1, number of values)
- **Data write**: `write(lout) real(vrg)` per value
- **makepar**: Added ibinary parameter line (format 170)
- **Note**: nscore outputs 1D array of normal scores, not echoing input in binary mode

#### backtr.for (no .inc file, variables in main program)
- **main program**: Added `ibinary` integer variable
- **readparm**: Added `read(lin,*) ibinary` after output filename
- **File open**: Conditional stream/unformatted for binary mode
- **Header write**: `write(lout) 1, nout` (ndim=1, number of values)
- **Data write**: `write(lout) bac` per value
- **makepar**: Added ibinary parameter line (format 140)
- **Note**: backtr outputs 1D array of back-transformed values, not echoing input in binary mode

#### declus.for (no .inc file, variables in main program)
- **main program**: Added `ibinary` integer variable
- **readparm**: Added `read(lin,*) ibinary` after output filename
- **File open**: Moved file open after counting data, conditional stream/unformatted for binary mode
- **Header write**: `write(lout) 1, maxdat` (ndim=1, number of values)
- **Data write**: `write(lout) thewt` per value (weight only in binary mode)
- **makepar**: Added ibinary parameter line (format 150)
- **Note**: declus outputs 1D array of declustering weights, not echoing input in binary mode

#### Pattern for each program
1. Add `ibinary` integer to appropriate common block in `.inc` file
2. Read `ibinary` from par file after output filename
3. Open output file with `access='STREAM',form='UNFORMATTED'` if binary
4. Write binary header: `[ndim, dim1, dim2, ...]` as int32
5. Write data as float32 (native Fortran `real`)
6. Update `makepar` to include ibinary in example par file

---

## Phase 2: Grid Mask Support

Skip inactive cells during estimation/simulation. Significant speedup for sparse domains.

### Implementation
- **Fortran side**: Read int8 mask array, skip cells where mask=0
- **Python side**: Pass mask as binary file, reshape output accordingly

| Program | Mask Support | Status |
|---------|-------------|--------|
| kt3d    | ✅ Done     | Complete |
| ik3d    | ✅ Done     | Complete |
| sgsim   | ✅ Done     | Complete |
| sisim   | ✅ Done     | Complete |

### Mask File Format
```
Header: [ndim=3: int32] [nz: int32] [ny: int32] [nx: int32]
Data:   [mask_values: int8 × (nx*ny*nz)]
```
- Mask value 0 = inactive (skip), 1 = active (estimate/simulate)
- Order: Fortran column-major (z varies fastest)
- Masked cells output placeholder value (UNEST: -999.0 for kt3d/sgsim/sisim, -9.9999 for ik3d)

### Fortran Source Changes Summary

#### kt3d.for / kt3d.inc
- **kt3d.inc**: Added `imask` to `/datcom/` common block, added `maskfl` character variable with `/maskcom/`
- **geostat module**: Added `integer*1,allocatable :: gridmask(:)`
- **readparm**: Added `read(lin,*) imask` and conditional `read(lin,'(a512)') maskfl` after ibinary
- **Mask read**: After `close(lin)`, allocate and read mask file if `imask=1`
- **Main loop**: Skip masked cells with `go to 1` after setting `est=UNEST, estv=UNEST`
- **makepar**: Added imask/maskfl parameter lines

#### ik3d.for / ik3d.inc
- **ik3d.inc**: Added `imask` to `/datcom/`, added `maskfl` and `/maskcom/`
- **geostat module**: Added `integer*1,allocatable :: gridmask(:)`
- **readparm**: Added imask/maskfl reading after ibinary
- **Mask read**: After `close(lin)`, allocate and read mask file
- **Main loop**: Skip masked cells, set all `ccdfo(ic) = UNEST` and `go to 1`
- **makepar**: Added imask/maskfl parameter lines

#### sgsim.for / sgsim.inc
- **sgsim.inc**: Added `imask` to `/generl/`, added `maskfl` and `/maskcom/`
- **geostat module**: Added `integer*1,allocatable :: gridmask(:)`
- **readparm**: Added imask/maskfl reading after ibinary
- **Mask read**: After `close(lin)`, allocate and read mask file
- **Main loop**: After `index = order(in)`, check mask and set `sim(index) = UNEST`, `go to 5`
- **makepar**: Added imask/maskfl parameter lines

#### sisim.for / sisim.inc
- **sisim.inc**: Added `imask` to `/simula/`, added `maskfl` and `/maskcom/`
- **geostat module**: Added `integer*1,allocatable :: gridmask(:)`
- **readparm**: Added imask/maskfl reading after ibinary
- **Mask read**: After `close(lin)`, allocate and read mask file
- **Main loop**: After `index = int(order(in)+0.5)`, check mask and set `sim(index) = UNEST`, `go to 20`
- **makepar**: Added imask/maskfl parameter lines

### Python API

All four grid programs (kt3d, ik3d, sgsim, sisim) now accept an optional `mask` parameter:
```python
result = kt3d(
    x, y, z, values,
    grid=grid,
    variogram=variogram,
    search=search,
    mask=mask_array,  # np.ndarray of shape (nz, ny, nx), dtype=int8
)
```
- Mask must match grid dimensions exactly
- Masked cells (mask=0) have placeholder values in output

---

## Phase 3: Double Precision Builds

Provide float64 builds for applications requiring higher numerical precision.

### Build Variants
```
bin/
├── kt3d.exe           # Standard (float32 internal, float32 binary output)
├── kt3d_f64.exe       # Double precision (float64 internal, float64 binary output)
```

### Current Status

| Program | f64 Binary | f64 Works | Notes |
|---------|-----------|-----------|-------|
| kt3d    | ✅ Built  | ✅ Yes    | Passes all tests |
| ik3d    | ✅ Built  | ✅ Yes    | Passes all tests |
| sgsim   | ✅ Built  | ❌ No     | SIGSEGV at runtime - needs investigation |
| sisim   | ✅ Built  | ✅ Yes    | Passes all tests |
| gamv    | ✅ Built  | ⬜ TBD    | Not yet tested |
| nscore  | ✅ Built  | ⬜ TBD    | Not yet tested |
| backtr  | ✅ Built  | ⬜ TBD    | Not yet tested |
| declus  | ✅ Built  | ⬜ TBD    | Not yet tested |

### Implementation

Used `-freal-4-real-8` compile flag which promotes all 4-byte reals to 8-byte.
This is less invasive than `-fdefault-real-8` which also affects literal constants.

```makefile
FFLAGS_F64 = -O2 -std=legacy -w -freal-4-real-8
```

### Python API

Wrappers (kt3d, ik3d, sgsim, sisim) now accept `precision` parameter:
```python
result = kt3d(
    x, y, z, values,
    grid=grid,
    variogram=variogram,
    search=search,
    precision="f64",  # Use double precision binary
)
```

### Known Issues

**sgsim_f64 crashes with SIGSEGV** during simulation loop (around node 50).
The `-freal-4-real-8` flag changes array sizes and memory layout, which may
cause issues with:
- Hardcoded array dimensions in sgsim
- Covariance lookup table calculations
- Memory allocation that doesn't account for doubled data size

This requires deeper Fortran investigation to fix.

### Naming Convention
- Standard builds: `program.exe` / `program`
- Float64 builds: `program_f64.exe` / `program_f64`

---

## Phase 4: Additional Features

### 4.1 Variogram Fitting (Python-side)
- Automatic model fitting from experimental variogram
- Interactive fitting with matplotlib
- Export to GSLIB par file format

### 4.2 Drillhole Utilities (Python-side)
- Minimum curvature desurvey
- Length-weighted compositing
- Domain-aware interval merging

### 4.3 Visualization Helpers
- Quick 3D grid visualization (pyvista integration)
- Variogram plotting
- Cross-validation plots

---

## Version Milestones

| Version | Features | Status |
|---------|----------|--------|
| v0.1    | Initial release - ASCII I/O, all 8 programs | ✅ Complete |
| v0.2    | Binary I/O for all programs | ✅ Complete |
| v0.3    | Grid mask support | ✅ Complete |
| v0.4    | Double precision builds | 🔄 Partial (sgsim needs fix) |
| v1.0    | Stable API, full documentation | ⬜ Planned |

---

## Contributing

Priority areas for contribution:
1. Test coverage for edge cases
2. Documentation and examples
3. Performance benchmarks (ASCII vs binary vs masked)
4. Double precision build implementation
