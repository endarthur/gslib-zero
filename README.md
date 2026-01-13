# gslib-zero

Python wrapper for Stanford's GSLIB Fortran 90 geostatistics programs with binary I/O support.

## Features

- **Binary I/O**: 10-100x faster than ASCII for large datasets
- **Grid mask support**: Skip inactive cells, re-estimate only updated domains
- **Modern Python API**: Type hints, dataclasses, sensible defaults
- **Battle-tested algorithms**: Same GSLIB Fortran code, cleaner interface

## Programs

| Program | Function |
|---------|----------|
| `declus` | Cell declustering weights |
| `nscore` | Normal score transform |
| `backtr` | Back-transform from normal scores |
| `gamv` | Experimental variograms |
| `kt3d` | 3D kriging (OK, SK, KT, cokriging, external drift) |
| `ik3d` | Indicator kriging |
| `sgsim` | Sequential Gaussian simulation |
| `sisim` | Sequential indicator simulation |

## Installation

```bash
pip install gslib-zero
```

## Quick Start

```python
import numpy as np
from gslib_zero import GridSpec, VariogramModel
from gslib_zero.estimation import kt3d, SearchParameters

# Sample data
x = np.array([10, 30, 50, 70, 90])
y = np.array([10, 30, 50, 70, 90])
z = np.array([5, 5, 5, 5, 5])
values = np.array([2.1, 3.5, 2.8, 4.2, 3.1])

# Define grid
grid = GridSpec(
    nx=10, ny=10, nz=1,
    xmin=5, ymin=5, zmin=5,
    xsiz=10, ysiz=10, zsiz=1,
)

# Variogram model
variogram = VariogramModel.spherical(
    sill=1.0,
    ranges=(50.0, 50.0, 10.0),
    nugget=0.1,
)

# Search parameters
search = SearchParameters(
    radius1=100, radius2=100, radius3=10,
    min_samples=1, max_samples=16,
)

# Run kriging
result = kt3d(x, y, z, values, grid, variogram, search)
print(result.estimate.shape)  # (1, 10, 10)
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=gslib_zero
```

## License

MIT
