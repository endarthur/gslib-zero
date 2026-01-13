"""
Tests for GSLIB wrapper functions with original GSLIB binaries.
"""

import numpy as np
import pytest

from gslib_zero.transforms import nscore, backtr
from gslib_zero.declustering import declus
from gslib_zero.variogram import gamv, GamvDirection
from gslib_zero.estimation import kt3d, SearchParameters
from gslib_zero.simulation import sgsim
from gslib_zero.utils import GridSpec, VariogramModel


class TestNscoreBacktr:
    """Test nscore and backtr round-trip."""

    def test_nscore_basic(self, sample_data):
        """Test basic nscore transform."""
        values = sample_data["values"]

        transformed, table = nscore(values)

        # Should have same length
        assert len(transformed) == len(values)
        # Transform table should have 2 columns
        assert table.shape[1] == 2
        # Transformed should be roughly standard normal
        assert abs(np.mean(transformed)) < 0.2
        assert abs(np.std(transformed) - 1.0) < 0.2

    def test_nscore_backtr_roundtrip(self, sample_data):
        """Test that backtr reverses nscore."""
        values = sample_data["values"]

        transformed, table = nscore(values)
        recovered = backtr(transformed, table)

        # Should recover original values closely
        max_error = np.max(np.abs(values - recovered))
        assert max_error < 0.01, f"Max round-trip error: {max_error}"


class TestDeclus:
    """Test cell declustering."""

    def test_declus_basic(self, clustered_data):
        """Test basic declustering."""
        x = clustered_data["x"]
        y = clustered_data["y"]
        z = clustered_data["z"]
        values = clustered_data["values"]

        result = declus(
            x, y, z, values,
            cell_min=5.0,
            cell_max=50.0,
            n_cells=5,
        )

        # Should have weights for each sample
        assert len(result.weights) == len(x)
        # Weights should be positive
        assert np.all(result.weights > 0)
        # Optimal cell size should be in range
        assert 5.0 <= result.optimal_cell_size <= 50.0
        # Declustered mean should differ from naive mean (data is clustered)
        naive_mean = np.mean(values)
        # Declustered mean should be lower (high values are oversampled)
        assert result.declustered_mean < naive_mean


class TestGamv:
    """Test experimental variogram calculation."""

    def test_gamv_omnidirectional(self, sample_data):
        """Test omnidirectional variogram."""
        x = sample_data["x"]
        y = sample_data["y"]
        z = sample_data["z"]
        values = sample_data["values"]

        results = gamv(
            x, y, z, values,
            nlag=10,
            lag_distance=10.0,
        )

        # Should return one result for default omnidirectional
        assert len(results) == 1
        result = results[0]

        # Should have correct number of lags
        assert len(result.gamma) == 10
        assert len(result.lag_distances) == 10
        assert len(result.num_pairs) == 10

    def test_gamv_directional(self, sample_data):
        """Test directional variograms."""
        x = sample_data["x"]
        y = sample_data["y"]
        z = sample_data["z"]
        values = sample_data["values"]

        directions = [
            GamvDirection(azimuth=0.0, azimuth_tolerance=22.5),
            GamvDirection(azimuth=90.0, azimuth_tolerance=22.5),
        ]

        results = gamv(
            x, y, z, values,
            nlag=10,
            lag_distance=10.0,
            directions=directions,
        )

        # Should return one result per direction
        assert len(results) == 2


class TestKt3d:
    """Test kriging estimation."""

    def test_kt3d_ordinary(self, sample_data, simple_grid, spherical_variogram):
        """Test ordinary kriging."""
        x = sample_data["x"]
        y = sample_data["y"]
        z = sample_data["z"]
        values = sample_data["values"]

        # Use smaller grid for faster test
        small_grid = GridSpec(
            nx=5, ny=5, nz=2,
            xmin=25.0, ymin=25.0, zmin=2.5,
            xsiz=10.0, ysiz=10.0, zsiz=2.5,
        )

        search = SearchParameters(
            radius1=50.0, radius2=50.0, radius3=10.0,
            min_samples=1, max_samples=16,
        )

        result = kt3d(
            x, y, z, values,
            grid=small_grid,
            variogram=spherical_variogram,
            search=search,
            kriging_type="ordinary",
        )

        # Check output shape
        assert result.estimate.shape == (small_grid.nz, small_grid.ny, small_grid.nx)
        assert result.variance.shape == (small_grid.nz, small_grid.ny, small_grid.nx)
        # Estimates should be in reasonable range
        assert np.nanmin(result.estimate) > 0
        assert np.nanmax(result.estimate) < 20
        # Variance should be non-negative
        assert np.all(result.variance >= 0)

    def test_kt3d_simple(self, sample_data, spherical_variogram):
        """Test simple kriging."""
        x = sample_data["x"]
        y = sample_data["y"]
        z = sample_data["z"]
        values = sample_data["values"]

        small_grid = GridSpec(
            nx=5, ny=5, nz=2,
            xmin=25.0, ymin=25.0, zmin=2.5,
            xsiz=10.0, ysiz=10.0, zsiz=2.5,
        )

        search = SearchParameters(
            radius1=50.0, radius2=50.0, radius3=10.0,
            min_samples=1, max_samples=16,
        )

        result = kt3d(
            x, y, z, values,
            grid=small_grid,
            variogram=spherical_variogram,
            search=search,
            kriging_type="simple",
            sk_mean=np.mean(values),
        )

        assert result.estimate.shape == (small_grid.nz, small_grid.ny, small_grid.nx)

    def test_kt3d_binary_matches_ascii(self, sample_data, spherical_variogram):
        """Test that binary mode produces same results as ASCII mode."""
        x = sample_data["x"]
        y = sample_data["y"]
        z = sample_data["z"]
        values = sample_data["values"]

        small_grid = GridSpec(
            nx=5, ny=5, nz=2,
            xmin=25.0, ymin=25.0, zmin=2.5,
            xsiz=10.0, ysiz=10.0, zsiz=2.5,
        )

        search = SearchParameters(
            radius1=50.0, radius2=50.0, radius3=10.0,
            min_samples=1, max_samples=16,
        )

        # Run in ASCII mode
        result_ascii = kt3d(
            x, y, z, values,
            grid=small_grid,
            variogram=spherical_variogram,
            search=search,
            kriging_type="ordinary",
            binary=False,
        )

        # Run in binary mode
        result_binary = kt3d(
            x, y, z, values,
            grid=small_grid,
            variogram=spherical_variogram,
            search=search,
            kriging_type="ordinary",
            binary=True,
        )

        # Results should match within float32 precision (~1e-5 relative error)
        assert result_binary.estimate.shape == result_ascii.estimate.shape
        assert result_binary.variance.shape == result_ascii.variance.shape

        # Check estimates match
        est_diff = np.abs(result_ascii.estimate - result_binary.estimate)
        assert est_diff.max() < 1e-4, f"Max estimate diff: {est_diff.max()}"

        # Check variances match
        var_diff = np.abs(result_ascii.variance - result_binary.variance)
        assert var_diff.max() < 1e-4, f"Max variance diff: {var_diff.max()}"


class TestSgsim:
    """Test Sequential Gaussian simulation."""

    def test_sgsim_unconditional(self, simple_grid, spherical_variogram):
        """Test unconditional simulation."""
        result = sgsim(
            x=None, y=None, z=None, values=None,
            grid=simple_grid,
            variogram=spherical_variogram,
            search_radius=(50.0, 50.0, 10.0),
            nrealizations=2,
            seed=12345,
        )

        # Check output shape
        assert result.realizations.shape == (2, simple_grid.nz, simple_grid.ny, simple_grid.nx)
        # Values should be roughly standard normal (since no transform)
        assert np.abs(np.mean(result.realizations)) < 1.0
        # Different realizations should be different
        assert not np.allclose(result.realizations[0], result.realizations[1])

    def test_sgsim_conditional(self, sample_data, spherical_variogram):
        """Test conditional simulation."""
        x = sample_data["x"]
        y = sample_data["y"]
        z = sample_data["z"]
        values = sample_data["values"]

        # Transform to normal scores first
        ns_values, table = nscore(values)

        # Use a grid that covers the sample data extent
        cond_grid = GridSpec(
            nx=10, ny=10, nz=2,
            xmin=5.0, ymin=5.0, zmin=0.5,
            xsiz=10.0, ysiz=10.0, zsiz=5.0,
        )

        result = sgsim(
            x=x, y=y, z=z, values=ns_values,
            grid=cond_grid,
            variogram=spherical_variogram,
            search_radius=(50.0, 50.0, 10.0),
            nrealizations=1,
            seed=12345,
        )

        # Check output shape
        assert result.realizations.shape == (1, cond_grid.nz, cond_grid.ny, cond_grid.nx)


class TestSisim:
    """Test Sequential Indicator simulation."""

    def test_sisim_unconditional(self, simple_grid):
        """Test unconditional indicator simulation."""
        from gslib_zero.simulation import sisim

        # Create indicator variograms for 3 thresholds
        variograms = [
            VariogramModel.spherical(sill=0.15, ranges=(10.0, 10.0, 5.0), nugget=0.1),
            VariogramModel.spherical(sill=0.20, ranges=(10.0, 10.0, 5.0), nugget=0.05),
            VariogramModel.spherical(sill=0.15, ranges=(10.0, 10.0, 5.0), nugget=0.1),
        ]

        result = sisim(
            x=None, y=None, z=None, values=None,
            grid=simple_grid,
            thresholds=[5.0, 10.0, 15.0],
            global_cdf=[0.25, 0.50, 0.75],
            variograms=variograms,
            search_radius=(20.0, 20.0, 10.0),
            nrealizations=1,
            seed=12345,
            zmin=0.0,
            zmax=20.0,
        )

        # Check output shape
        assert result.realizations.shape == (1, simple_grid.nz, simple_grid.ny, simple_grid.nx)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
