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
| ik3d    | ⬜ Todo      | N/A          | - |
| sgsim   | ⬜ Todo      | ⬜ Todo      | - |
| sisim   | ⬜ Todo      | ⬜ Todo      | - |
| gamv    | ⬜ Todo      | N/A          | - |
| nscore  | ⬜ Todo      | N/A          | - |
| backtr  | ⬜ Todo      | ⬜ Todo      | - |
| declus  | ⬜ Todo      | N/A          | - |

### Binary Format Specification
```
Header: [ndim: int32] [shape: int32 × ndim]
Data:   [values: float32 × prod(shape)]
```

For grid outputs (kt3d, sgsim, etc.): shape = (nvars, nz, ny, nx)

---

## Phase 2: Grid Mask Support

Skip inactive cells during estimation/simulation. Significant speedup for sparse domains.

### Implementation
- **Fortran side**: Read int8 mask array, skip cells where mask=0
- **Python side**: Pass mask as binary file, reshape output accordingly

| Program | Mask Support | Status |
|---------|-------------|--------|
| kt3d    | ⬜ Todo     | - |
| ik3d    | ⬜ Todo     | - |
| sgsim   | ⬜ Todo     | - |
| sisim   | ⬜ Todo     | - |

---

## Phase 3: Double Precision Builds

Provide float64 builds for applications requiring higher numerical precision.

### Build Variants
```
bin/
├── kt3d.exe           # Standard (float32 internal, float32 binary output)
├── kt3d_f64.exe       # Double precision (float64 internal, float64 binary output)
```

### Implementation Options
1. **Compile-time flag**: `-fdefault-real-8` makes all `real` become `real*8`
2. **Separate source**: Maintain f32/f64 variants of critical subroutines
3. **Runtime flag**: Single binary with precision parameter (more complex)

### Naming Convention
- Standard builds: `program.exe` / `program`
- Float64 builds: `program_f64.exe` / `program_f64`

Python wrapper detects available binaries and exposes `precision='f32'|'f64'` parameter.

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

| Version | Features |
|---------|----------|
| v0.1    | Initial release - ASCII I/O, all 8 programs |
| v0.2    | Binary I/O for all programs |
| v0.3    | Grid mask support |
| v0.4    | Double precision builds |
| v1.0    | Stable API, full documentation |

---

## Contributing

Priority areas for contribution:
1. Binary I/O implementation for remaining programs
2. Test coverage for edge cases
3. Documentation and examples
4. Performance benchmarks (ASCII vs binary)
