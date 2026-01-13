# CLAUDE.md — gslib-zero

> **Name status**: The project name is provisional. `gslib-zero` and `gslib-ng` are both candidates. Use whichever appears in the repo name.

## What This Is

A Python wrapper for Stanford's GSLIB Fortran 90 geostatistics programs. The architecture follows GeostatsPy's approach (shell out to executables, generate par files, serialize/deserialize data) but with key improvements:

- **Binary I/O** instead of ASCII — dramatically faster for large datasets
- **Grid mask support** — skip inactive cells, re-estimate only updated domains
- **Modernized Fortran** — same algorithms, cleaner I/O interface
- **CI-built binaries** — no user compilation required

This is NOT a Python reimplementation of geostatistics (that's GGR/wabisabi). This wraps battle-tested Fortran for production use cases where speed matters.

## Who It's For

- Geologists/geostatisticians who want GSLIB's reliability with a modern Python API
- Production workflows where ASCII I/O is the bottleneck
- Validation reference for pure-Python implementations (GGR)

## Architecture

```
User Python code
    ↓
gslib-zero API (par file generation, input validation, rotation handling)
    ↓
Binary serialization (numpy arrays → binary files)
    ↓
GSLIB Fortran executables (modified for binary I/O + masks)
    ↓
Binary deserialization (binary files → numpy arrays)
    ↓
Results back to Python
```

Key principle: **Fortran stays Fortran**. We modify the I/O layer, not the algorithms. The hard math remains unchanged and battle-tested.

## Scope — Eight Programs

| Program | Purpose | Notes |
|---------|---------|-------|
| `declus` | Cell declustering | Weights for preferential sampling |
| `nscore` | Normal score transform | Gaussian anamorphosis forward |
| `backtr` | Back-transform | Gaussian anamorphosis reverse |
| `gamv` | Variogram calculation | Experimental variograms/covariograms |
| `kt3d` | 3D kriging | OK, SK, KT, cokriging, external drift |
| `ik3d` | Indicator kriging | Multiple cutoffs + order relations correction |
| `sgsim` | Sequential Gaussian simulation | Conditional simulation |
| `sisim` | Sequential indicator simulation | Categorical/indicator simulation |

All from Stanford's f90 release (not CCG — licensing is cleaner).

## Key Design Decisions

### Binary I/O
GSLIB's bottleneck is ASCII parsing. Binary I/O means:
- Direct numpy array read/write
- No whitespace parsing ambiguity
- 10-100x faster for large datasets

### Masks
- **Grid mask**: Implemented in Fortran. Int8 array matching grid dimensions. Skip cells where mask=0.
- **Sample mask**: Python-side filtering before calling Fortran. Just subset the numpy array.

This keeps Fortran modifications minimal while providing both capabilities.

### Rotation Conventions
GSLIB uses Deutsch convention (azimuth, dip, rake). Be explicit about this in the API. Consider providing conversion utilities for other conventions (Leapfrog, Datamine, etc.) but don't change GSLIB's internal handling.

### Par File Generation
Generate par files programmatically from Python function arguments. Validate inputs before writing. Include sensible defaults but expose all GSLIB options for power users.

## Technical Gotchas

### Column-Major vs Row-Major
Fortran is column-major, NumPy is row-major by default. When passing 3D grids:
```python
# Ensure Fortran-contiguous before serialization
grid_f = np.asfortranarray(grid)
```

### GSLIB Grid Indexing
GSLIB indexes grids as (nz, ny, nx) internally and iterates z-fastest. Be consistent about this in the API or document the convention clearly.

### Variogram Model Codes
GSLIB uses integer codes for variogram models:
- 1 = Spherical
- 2 = Exponential  
- 3 = Gaussian
- 4 = Power
- 5 = Hole effect

Provide an enum or constants rather than expecting users to remember these.

### kt3d Is Deceptively Powerful
kt3d handles: simple kriging, ordinary kriging, kriging with trend (polynomial drift), simple/ordinary cokriging, colocated cokriging, kriging with external drift. The par file is complex but all options are there.

### ik3d vs kt3d on Indicators
ik3d exists mainly for order relations correction (ensuring CDF monotonicity). Could theoretically be implemented as kt3d + Python post-processing, but keeping ik3d is simpler.

## Project Structure (Suggested)

```
gslib-zero/
├── CLAUDE.md                 # This file
├── pyproject.toml
├── src/
│   └── gslib_zero/
│       ├── __init__.py
│       ├── core.py           # Subprocess handling, binary I/O
│       ├── par.py            # Par file generation
│       ├── transforms.py     # nscore, backtr wrappers
│       ├── variogram.py      # gamv wrapper
│       ├── estimation.py     # kt3d, ik3d wrappers
│       ├── simulation.py     # sgsim, sisim wrappers
│       ├── declustering.py   # declus wrapper
│       └── utils.py          # Rotation conversions, grid helpers
├── fortran/
│   └── src/                  # Modified GSLIB f90 sources
├── bin/                      # Pre-built binaries (or downloaded via CI)
└── tests/
    ├── test_transforms.py
    ├── test_variogram.py
    ├── test_estimation.py
    └── fixtures/             # Test data, expected outputs
```

## Coding Preferences

Arthur's style:
- Vanilla Python, minimal dependencies (numpy, maybe scipy for validation)
- Type hints appreciated but not religious about it
- Tests that actually test something (not just "it runs")
- Convention-over-configuration where sensible
- Clear error messages when inputs are invalid
- No excessive abstraction — this is a thin wrapper, keep it thin

## Future Scope (Not v1)

These are explicitly out of scope for initial release but worth keeping in mind:

- **Desurvey**: Minimum curvature drillhole desurvey (Python-side, low complexity)
- **Compositing**: Length-weighted composites with domain boundaries (Python-side, medium complexity)
- **Interval merging**: FROM/TO table joins (Python-side, many edge cases)
- **Variogram fitting**: Automatic/interactive model fitting (probably a separate concern)

The long-term vision is a minimal drillhole-to-estimate pipeline: CSVs in, numpy arrays out. But v1 is just the GSLIB wrappers.

## Validation Strategy

Compare outputs against:
1. Original Stanford GSLIB (ASCII mode) — exact match expected
2. GeostatsPy (uses same GSLIB binaries) — should match
3. GGR (pure Python) — should be within numerical tolerance

## Commands

```bash
# Run tests
pytest

# Build Fortran (requires gfortran)
cd fortran && make

# Download pre-built binaries
python -m gslib_zero.download_binaries
```

## Questions to Ask Arthur

If you're unsure about something:
- "Should this match GSLIB behavior exactly, or is deviation acceptable?"
- "Is this a v1 requirement or future scope?"
- "ASCII fallback or binary-only?"
