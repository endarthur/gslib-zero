"""
Pytest configuration and shared fixtures for gslib-zero tests.
"""

import numpy as np
import pytest

from gslib_zero.utils import GridSpec, VariogramModel


@pytest.fixture
def sample_data():
    """Generate simple sample data for testing."""
    rng = np.random.default_rng(42)
    n = 100

    x = rng.uniform(0, 100, n)
    y = rng.uniform(0, 100, n)
    z = rng.uniform(0, 10, n)
    values = rng.normal(10, 2, n)

    return {"x": x, "y": y, "z": z, "values": values}


@pytest.fixture
def simple_grid():
    """Create a simple test grid."""
    return GridSpec(
        nx=10, ny=10, nz=5,
        xmin=0.5, ymin=0.5, zmin=0.5,
        xsiz=1.0, ysiz=1.0, zsiz=1.0,
    )


@pytest.fixture
def spherical_variogram():
    """Create a simple spherical variogram model."""
    return VariogramModel.spherical(
        sill=1.0,
        ranges=(50.0, 50.0, 10.0),
        nugget=0.1,
    )


@pytest.fixture
def irregular_grid():
    """Create an irregular (non-cubic) test grid with odd dimensions."""
    return GridSpec(
        nx=17, ny=23, nz=7,  # Odd, non-equal dimensions
        xmin=2.5, ymin=5.0, zmin=0.25,  # Non-standard origins
        xsiz=2.5, ysiz=1.5, zsiz=3.0,  # Non-equal cell sizes
    )


@pytest.fixture
def clustered_data():
    """Generate clustered sample data for declustering tests."""
    rng = np.random.default_rng(123)

    # Create two clusters
    n1, n2 = 80, 20

    # Dense cluster (oversampled high values)
    x1 = rng.normal(25, 5, n1)
    y1 = rng.normal(25, 5, n1)
    z1 = rng.normal(5, 1, n1)
    v1 = rng.normal(15, 1, n1)  # Higher values

    # Sparse samples (undersampled low values)
    x2 = rng.uniform(0, 100, n2)
    y2 = rng.uniform(0, 100, n2)
    z2 = rng.uniform(0, 10, n2)
    v2 = rng.normal(8, 1, n2)  # Lower values

    return {
        "x": np.concatenate([x1, x2]),
        "y": np.concatenate([y1, y2]),
        "z": np.concatenate([z1, z2]),
        "values": np.concatenate([v1, v2]),
    }
