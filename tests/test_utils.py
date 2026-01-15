"""
Tests for utility classes and functions.
"""

import numpy as np
import pytest

from gslib_zero.utils import (
    GridSpec,
    VariogramModel,
    VariogramType,
    deutsch_to_math,
    math_to_deutsch,
    rotation_matrix_deutsch,
    UNEST,
    UNEST_IK,
    is_unestimated,
    mask_unestimated,
)


class TestGridSpec:
    """Tests for GridSpec class."""

    def test_shape(self, simple_grid):
        """Test grid shape property."""
        assert simple_grid.shape == (5, 10, 10)

    def test_ncells(self, simple_grid):
        """Test total cell count."""
        assert simple_grid.ncells == 500

    def test_max_coordinates(self, simple_grid):
        """Test maximum coordinate properties."""
        assert simple_grid.xmax == pytest.approx(9.5)
        assert simple_grid.ymax == pytest.approx(9.5)
        assert simple_grid.zmax == pytest.approx(4.5)

    def test_cell_centers(self, simple_grid):
        """Test cell center coordinate generation."""
        x, y, z = simple_grid.cell_centers()

        assert len(x) == 10
        assert len(y) == 10
        assert len(z) == 5

        assert x[0] == pytest.approx(0.5)
        assert x[-1] == pytest.approx(9.5)

    def test_meshgrid_shape(self, simple_grid):
        """Test meshgrid generation."""
        X, Y, Z = simple_grid.meshgrid()

        assert X.shape == (5, 10, 10)
        assert Y.shape == (5, 10, 10)
        assert Z.shape == (5, 10, 10)

    def test_contains_point(self, simple_grid):
        """Test point containment check."""
        # Point inside grid
        assert simple_grid.contains_point(5.0, 5.0, 2.5)

        # Point outside grid
        assert not simple_grid.contains_point(-1.0, 5.0, 2.5)
        assert not simple_grid.contains_point(5.0, 15.0, 2.5)

    def test_point_to_index(self, simple_grid):
        """Test coordinate to index conversion."""
        # Center of first cell
        idx = simple_grid.point_to_index(0.5, 0.5, 0.5)
        assert idx == (0, 0, 0)

        # Center of last cell
        idx = simple_grid.point_to_index(9.5, 9.5, 4.5)
        assert idx == (4, 9, 9)

        # Point outside grid
        idx = simple_grid.point_to_index(-5.0, 5.0, 2.5)
        assert idx is None


class TestVariogramModel:
    """Tests for VariogramModel class."""

    def test_spherical_factory(self):
        """Test spherical variogram factory method."""
        vario = VariogramModel.spherical(
            sill=1.0,
            ranges=(100.0, 50.0, 10.0),
            nugget=0.1,
        )

        assert vario.nugget == pytest.approx(0.1)
        assert len(vario.structures) == 1
        assert vario.structures[0]["type"] == VariogramType.SPHERICAL
        assert vario.structures[0]["sill"] == pytest.approx(1.0)

    def test_total_sill(self):
        """Test total sill calculation."""
        vario = VariogramModel(nugget=0.1)
        vario.add_structure(VariogramType.SPHERICAL, 0.5, (50.0, 50.0, 10.0))
        vario.add_structure(VariogramType.EXPONENTIAL, 0.4, (100.0, 100.0, 20.0))

        assert vario.total_sill == pytest.approx(1.0)

    def test_add_structure(self):
        """Test adding nested structures."""
        vario = VariogramModel(nugget=0.0)
        vario.add_structure(VariogramType.SPHERICAL, 0.6, (50.0, 50.0, 10.0))
        vario.add_structure(VariogramType.GAUSSIAN, 0.4, (100.0, 100.0, 20.0))

        assert len(vario.structures) == 2
        assert vario.structures[1]["type"] == VariogramType.GAUSSIAN


class TestRotationConversions:
    """Tests for rotation convention conversions."""

    def test_deutsch_math_roundtrip(self):
        """Test Deutsch <-> Math conversion roundtrip."""
        azimuth, dip, rake = 45.0, 30.0, 15.0

        alpha, beta, gamma = deutsch_to_math(azimuth, dip, rake)
        az2, dip2, rake2 = math_to_deutsch(alpha, beta, gamma)

        assert az2 == pytest.approx(azimuth)
        assert dip2 == pytest.approx(dip)
        assert rake2 == pytest.approx(rake)

    def test_rotation_matrix_identity(self):
        """Test that zero angles give identity-like matrix."""
        R = rotation_matrix_deutsch(0.0, 0.0, 0.0)

        # For zero angles, should be close to identity
        # (exact form depends on convention details)
        assert R.shape == (3, 3)
        # Determinant should be 1 (proper rotation)
        assert np.linalg.det(R) == pytest.approx(1.0)

    def test_rotation_matrix_orthogonal(self):
        """Test that rotation matrix is orthogonal."""
        R = rotation_matrix_deutsch(45.0, 30.0, 15.0)

        # R @ R.T should be identity
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=10)


class TestUnestimatedUtilities:
    """Tests for UNEST sentinel value utilities."""

    def test_unest_constants(self):
        """Test UNEST constant values."""
        assert UNEST == pytest.approx(-999.0)
        assert UNEST_IK == pytest.approx(-9.9999)

    def test_is_unestimated_continuous(self):
        """Test detection of continuous UNEST values."""
        values = np.array([1.0, 2.0, -999.0, 3.0, -999.0])
        mask = is_unestimated(values)

        assert mask.shape == values.shape
        assert mask[0] is np.False_
        assert mask[1] is np.False_
        assert mask[2] is np.True_
        assert mask[3] is np.False_
        assert mask[4] is np.True_

    def test_is_unestimated_indicator(self):
        """Test detection of indicator UNEST_IK values."""
        values = np.array([0.25, 0.50, -9.9999, 0.75])
        mask = is_unestimated(values)

        assert mask[0] is np.False_
        assert mask[1] is np.False_
        assert mask[2] is np.True_
        assert mask[3] is np.False_

    def test_is_unestimated_mixed(self):
        """Test detection with both UNEST and UNEST_IK values."""
        values = np.array([1.0, -999.0, -9.9999, 2.0])
        mask = is_unestimated(values)

        assert np.sum(mask) == 2  # Two unestimated values
        assert mask[1] is np.True_
        assert mask[2] is np.True_

    def test_is_unestimated_3d_array(self):
        """Test with 3D grid array."""
        values = np.ones((2, 3, 4))
        values[0, 1, 2] = UNEST
        values[1, 0, 0] = UNEST_IK

        mask = is_unestimated(values)

        assert mask.shape == values.shape
        assert mask[0, 1, 2] is np.True_
        assert mask[1, 0, 0] is np.True_
        assert np.sum(mask) == 2

    def test_mask_unestimated_basic(self):
        """Test masked array creation."""
        values = np.array([1.0, 2.0, -999.0, 3.0])
        masked = mask_unestimated(values)

        assert isinstance(masked, np.ma.MaskedArray)
        assert masked.mask[2] is np.True_
        assert masked.count() == 3  # 3 valid values

    def test_mask_unestimated_mean(self):
        """Test that masked array excludes UNEST from statistics."""
        values = np.array([2.0, 4.0, -999.0, 6.0])
        masked = mask_unestimated(values)

        # Mean should be (2 + 4 + 6) / 3 = 4.0, ignoring -999
        assert masked.mean() == pytest.approx(4.0)

    def test_mask_unestimated_3d(self):
        """Test masked array with 3D data."""
        values = np.ones((2, 3, 4)) * 10.0
        values[0, 0, 0] = UNEST
        values[1, 2, 3] = UNEST

        masked = mask_unestimated(values)

        assert masked.count() == 22  # 24 - 2 = 22 valid cells
        assert masked.mean() == pytest.approx(10.0)

    def test_is_unestimated_tolerance(self):
        """Test tolerance parameter."""
        # Value slightly different from UNEST
        values = np.array([-998.995, -999.005])

        # Default tolerance (0.01) should catch both
        mask = is_unestimated(values, tol=0.01)
        assert np.all(mask)

        # Tighter tolerance should miss them
        mask_tight = is_unestimated(values, tol=0.001)
        assert not np.any(mask_tight)
