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

    def test_nscore_binary_matches_ascii(self, sample_data):
        """Test that binary mode produces same results as ASCII mode."""
        values = sample_data["values"]

        # Run in ASCII mode
        transformed_ascii, table_ascii = nscore(values, binary=False)

        # Run in binary mode
        transformed_binary, table_binary = nscore(values, binary=True)

        # Transformed values should match closely
        ns_diff = np.abs(transformed_ascii - transformed_binary)
        assert ns_diff.max() < 1e-4, f"Max nscore diff: {ns_diff.max()}"

        # Transform tables should match (both are ASCII)
        table_diff = np.abs(table_ascii - table_binary)
        assert table_diff.max() < 1e-6, f"Max table diff: {table_diff.max()}"

    def test_backtr_binary_matches_ascii(self, sample_data):
        """Test that backtr binary mode produces same results as ASCII mode."""
        values = sample_data["values"]

        # First get transform table
        transformed, table = nscore(values)

        # Run backtr in ASCII mode
        recovered_ascii = backtr(transformed, table, binary=False)

        # Run backtr in binary mode
        recovered_binary = backtr(transformed, table, binary=True)

        # Results should match closely
        backtr_diff = np.abs(recovered_ascii - recovered_binary)
        assert backtr_diff.max() < 1e-4, f"Max backtr diff: {backtr_diff.max()}"


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

    def test_declus_binary_matches_ascii(self, clustered_data):
        """Test that binary mode produces same results as ASCII mode."""
        x = clustered_data["x"]
        y = clustered_data["y"]
        z = clustered_data["z"]
        values = clustered_data["values"]

        # Run in ASCII mode
        result_ascii = declus(
            x, y, z, values,
            cell_min=5.0,
            cell_max=50.0,
            n_cells=5,
            binary=False,
        )

        # Run in binary mode
        result_binary = declus(
            x, y, z, values,
            cell_min=5.0,
            cell_max=50.0,
            n_cells=5,
            binary=True,
        )

        # Weights should match closely (within float32 precision)
        wt_diff = np.abs(result_ascii.weights - result_binary.weights)
        assert wt_diff.max() < 1e-4, f"Max weight diff: {wt_diff.max()}"

        # Optimal cell size and declustered mean should match (from summary, always ASCII)
        assert result_ascii.optimal_cell_size == result_binary.optimal_cell_size
        assert result_ascii.declustered_mean == result_binary.declustered_mean


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

    def test_gamv_binary_matches_ascii(self, sample_data):
        """Test that binary mode produces same results as ASCII mode."""
        x = sample_data["x"]
        y = sample_data["y"]
        z = sample_data["z"]
        values = sample_data["values"]

        # Run in ASCII mode
        results_ascii = gamv(
            x, y, z, values,
            nlag=10,
            lag_distance=10.0,
            binary=False,
        )

        # Run in binary mode
        results_binary = gamv(
            x, y, z, values,
            nlag=10,
            lag_distance=10.0,
            binary=True,
        )

        # Should have same number of results
        assert len(results_binary) == len(results_ascii)

        for ascii_res, binary_res in zip(results_ascii, results_binary):
            # Distances should match closely (ASCII has limited precision in output format)
            dist_diff = np.abs(ascii_res.lag_distances - binary_res.lag_distances)
            assert dist_diff.max() < 1e-3, f"Max distance diff: {dist_diff.max()}"

            # Gamma values should match closely
            gamma_diff = np.abs(ascii_res.gamma - binary_res.gamma)
            assert gamma_diff.max() < 1e-4, f"Max gamma diff: {gamma_diff.max()}"

            # Number of pairs should match
            assert np.array_equal(ascii_res.num_pairs, binary_res.num_pairs)

            # Means should match closely
            tail_diff = np.abs(ascii_res.tail_mean - binary_res.tail_mean)
            assert tail_diff.max() < 1e-4, f"Max tail mean diff: {tail_diff.max()}"

            head_diff = np.abs(ascii_res.head_mean - binary_res.head_mean)
            assert head_diff.max() < 1e-4, f"Max head mean diff: {head_diff.max()}"


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

    def test_sgsim_binary_matches_ascii(self, simple_grid, spherical_variogram):
        """Test that binary mode produces same results as ASCII mode."""
        # Run in ASCII mode
        result_ascii = sgsim(
            x=None, y=None, z=None, values=None,
            grid=simple_grid,
            variogram=spherical_variogram,
            search_radius=(50.0, 50.0, 10.0),
            nrealizations=2,
            seed=12345,
            binary=False,
        )

        # Run in binary mode
        result_binary = sgsim(
            x=None, y=None, z=None, values=None,
            grid=simple_grid,
            variogram=spherical_variogram,
            search_radius=(50.0, 50.0, 10.0),
            nrealizations=2,
            seed=12345,
            binary=True,
        )

        # Results should match within float32 precision (~1e-5 relative error)
        assert result_binary.realizations.shape == result_ascii.realizations.shape

        # Check values match
        diff = np.abs(result_ascii.realizations - result_binary.realizations)
        assert diff.max() < 1e-4, f"Max diff: {diff.max()}"


class TestIk3d:
    """Test Indicator kriging."""

    def test_ik3d_basic(self, sample_data, simple_grid):
        """Test basic indicator kriging."""
        from gslib_zero.estimation import ik3d

        x = sample_data["x"]
        y = sample_data["y"]
        z = sample_data["z"]
        values = sample_data["values"]

        # Create cutoffs and global CDF
        cutoffs = [8.0, 10.0, 12.0]
        global_cdf = [0.25, 0.50, 0.75]

        # Create indicator variograms for each cutoff
        variograms = [
            VariogramModel.spherical(sill=0.15, ranges=(50.0, 50.0, 10.0), nugget=0.1),
            VariogramModel.spherical(sill=0.20, ranges=(50.0, 50.0, 10.0), nugget=0.05),
            VariogramModel.spherical(sill=0.15, ranges=(50.0, 50.0, 10.0), nugget=0.1),
        ]

        search = SearchParameters(
            radius1=50.0, radius2=50.0, radius3=10.0,
            min_samples=1, max_samples=16,
        )

        result = ik3d(
            x, y, z, values,
            grid=simple_grid,
            cutoffs=cutoffs,
            global_cdf=global_cdf,
            variograms=variograms,
            search=search,
        )

        # Check output shape
        assert result.probabilities.shape == (3, simple_grid.nz, simple_grid.ny, simple_grid.nx)
        # Probabilities should be between 0 and 1
        assert np.all(result.probabilities >= 0)
        assert np.all(result.probabilities <= 1)
        # CDFs should be monotonically increasing (after order relations correction)
        # Check at a random cell
        for iz in range(simple_grid.nz):
            for iy in range(simple_grid.ny):
                for ix in range(simple_grid.nx):
                    probs = result.probabilities[:, iz, iy, ix]
                    # Order relations: p1 <= p2 <= p3
                    assert probs[0] <= probs[1] + 1e-4
                    assert probs[1] <= probs[2] + 1e-4

    def test_ik3d_binary_matches_ascii(self, sample_data, simple_grid):
        """Test that binary mode produces same results as ASCII mode."""
        from gslib_zero.estimation import ik3d

        x = sample_data["x"]
        y = sample_data["y"]
        z = sample_data["z"]
        values = sample_data["values"]

        # Create cutoffs and global CDF
        cutoffs = [8.0, 10.0, 12.0]
        global_cdf = [0.25, 0.50, 0.75]

        # Create indicator variograms for each cutoff
        variograms = [
            VariogramModel.spherical(sill=0.15, ranges=(50.0, 50.0, 10.0), nugget=0.1),
            VariogramModel.spherical(sill=0.20, ranges=(50.0, 50.0, 10.0), nugget=0.05),
            VariogramModel.spherical(sill=0.15, ranges=(50.0, 50.0, 10.0), nugget=0.1),
        ]

        search = SearchParameters(
            radius1=50.0, radius2=50.0, radius3=10.0,
            min_samples=1, max_samples=16,
        )

        # Run in ASCII mode
        result_ascii = ik3d(
            x, y, z, values,
            grid=simple_grid,
            cutoffs=cutoffs,
            global_cdf=global_cdf,
            variograms=variograms,
            search=search,
            binary=False,
        )

        # Run in binary mode
        result_binary = ik3d(
            x, y, z, values,
            grid=simple_grid,
            cutoffs=cutoffs,
            global_cdf=global_cdf,
            variograms=variograms,
            search=search,
            binary=True,
        )

        # Results should match within float32 precision
        assert result_binary.probabilities.shape == result_ascii.probabilities.shape

        diff = np.abs(result_ascii.probabilities - result_binary.probabilities)
        assert diff.max() < 1e-4, f"Max diff: {diff.max()}"


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

    def test_sisim_binary_matches_ascii(self, simple_grid):
        """Test that binary mode produces same results as ASCII mode."""
        from gslib_zero.simulation import sisim

        # Create indicator variograms for 3 thresholds
        variograms = [
            VariogramModel.spherical(sill=0.15, ranges=(10.0, 10.0, 5.0), nugget=0.1),
            VariogramModel.spherical(sill=0.20, ranges=(10.0, 10.0, 5.0), nugget=0.05),
            VariogramModel.spherical(sill=0.15, ranges=(10.0, 10.0, 5.0), nugget=0.1),
        ]

        # Run in ASCII mode
        result_ascii = sisim(
            x=None, y=None, z=None, values=None,
            grid=simple_grid,
            thresholds=[5.0, 10.0, 15.0],
            global_cdf=[0.25, 0.50, 0.75],
            variograms=variograms,
            search_radius=(20.0, 20.0, 10.0),
            nrealizations=2,
            seed=12345,
            zmin=0.0,
            zmax=20.0,
            binary=False,
        )

        # Run in binary mode
        result_binary = sisim(
            x=None, y=None, z=None, values=None,
            grid=simple_grid,
            thresholds=[5.0, 10.0, 15.0],
            global_cdf=[0.25, 0.50, 0.75],
            variograms=variograms,
            search_radius=(20.0, 20.0, 10.0),
            nrealizations=2,
            seed=12345,
            zmin=0.0,
            zmax=20.0,
            binary=True,
        )

        # Results should match within float32 precision (~1e-5 relative error)
        assert result_binary.realizations.shape == result_ascii.realizations.shape

        # Check values match
        diff = np.abs(result_ascii.realizations - result_binary.realizations)
        assert diff.max() < 1e-4, f"Max diff: {diff.max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
