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

    def test_kt3d_with_mask(self, sample_data, spherical_variogram):
        """Test kriging with grid mask - masked cells should have UNEST values."""
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

        # Create mask: 1s for active cells, 0s for masked cells
        # Mask out half the cells in a checkerboard pattern
        mask = np.ones((small_grid.nz, small_grid.ny, small_grid.nx), dtype=np.int8)
        mask[0, ::2, ::2] = 0  # Mask some cells in first layer
        mask[1, 1::2, 1::2] = 0  # Mask different cells in second layer

        result = kt3d(
            x, y, z, values,
            grid=small_grid,
            variogram=spherical_variogram,
            search=search,
            kriging_type="ordinary",
            binary=True,
            mask=mask,
        )

        # Check output shape
        assert result.estimate.shape == (small_grid.nz, small_grid.ny, small_grid.nx)

        # Masked cells should have UNEST value (-999)
        UNEST = -999.0
        masked_estimates = result.estimate[mask == 0]
        assert np.allclose(masked_estimates, UNEST), "Masked cells should have UNEST value"

        # Active cells should have valid estimates (not UNEST)
        active_estimates = result.estimate[mask == 1]
        assert np.all(active_estimates != UNEST), "Active cells should have valid estimates"
        assert np.all(active_estimates > 0), "Active cells should have positive estimates"

    def test_kt3d_mask_irregular_grid(self, sample_data, spherical_variogram, irregular_grid):
        """Test kriging with irregular grid (17x23x7) and realistic mask pattern."""
        x = sample_data["x"]
        y = sample_data["y"]
        z = sample_data["z"]
        values = sample_data["values"]

        # Create a realistic irregular mask - simulate domain boundary
        mask = np.zeros((irregular_grid.nz, irregular_grid.ny, irregular_grid.nx), dtype=np.int8)

        # Active region: ellipsoidal shape with irregular boundary
        cx, cy, cz = irregular_grid.nx / 2, irregular_grid.ny / 2, irregular_grid.nz / 2
        for iz in range(irregular_grid.nz):
            for iy in range(irregular_grid.ny):
                for ix in range(irregular_grid.nx):
                    # Ellipsoidal distance
                    dist = ((ix - cx) / 7) ** 2 + ((iy - cy) / 9) ** 2 + ((iz - cz) / 3) ** 2
                    # Irregular boundary with trigonometric perturbation
                    threshold = 1.0 + 0.2 * np.sin(ix * 0.7) * np.cos(iy * 0.5) * np.sin(iz * 0.9)
                    if dist < threshold:
                        mask[iz, iy, ix] = 1

        search = SearchParameters(
            radius1=80.0, radius2=80.0, radius3=30.0,
            min_samples=1, max_samples=16,
        )

        result = kt3d(
            x, y, z, values,
            grid=irregular_grid,
            variogram=spherical_variogram,
            search=search,
            kriging_type="ordinary",
            mask=mask,
        )

        assert result.estimate.shape == (irregular_grid.nz, irregular_grid.ny, irregular_grid.nx)

        UNEST = -999.0
        masked_estimates = result.estimate[mask == 0]
        active_estimates = result.estimate[mask == 1]

        assert np.allclose(masked_estimates, UNEST), f"Masked cells should have UNEST, got: {np.unique(masked_estimates)}"
        assert not np.any(np.isclose(active_estimates, UNEST)), "Active cells should not have UNEST"

    def test_kt3d_mask_mostly_masked(self, sample_data, spherical_variogram):
        """Test kriging where most cells are masked (sparse domain)."""
        x = sample_data["x"]
        y = sample_data["y"]
        z = sample_data["z"]
        values = sample_data["values"]

        # Grid with odd dimensions
        sparse_grid = GridSpec(nx=19, ny=11, nz=3, xmin=0.5, ymin=0.5, zmin=0.5, xsiz=5.0, ysiz=9.0, zsiz=3.0)

        # Only 10% of cells active (scattered pattern)
        rng = np.random.default_rng(42)
        mask = (rng.random((sparse_grid.nz, sparse_grid.ny, sparse_grid.nx)) < 0.1).astype(np.int8)

        # Ensure at least some cells are active
        if np.sum(mask) < 10:
            mask[0, 0, 0] = 1
            mask[1, 5, 9] = 1
            mask[2, 10, 18] = 1

        search = SearchParameters(
            radius1=100.0, radius2=100.0, radius3=20.0,
            min_samples=1, max_samples=8,
        )

        result = kt3d(
            x, y, z, values,
            grid=sparse_grid,
            variogram=spherical_variogram,
            search=search,
            kriging_type="simple",
            sk_mean=np.mean(values),
            mask=mask,
        )

        UNEST = -999.0
        masked_estimates = result.estimate[mask == 0]
        active_estimates = result.estimate[mask == 1]

        assert np.allclose(masked_estimates, UNEST)
        # With SK, all active cells should get valid estimates
        assert not np.any(np.isclose(active_estimates, UNEST))


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

    def test_sgsim_with_mask(self, simple_grid, spherical_variogram):
        """Test simulation with grid mask - masked cells should have UNEST values."""
        # Create a mask - mask out some cells
        mask = np.ones((simple_grid.nz, simple_grid.ny, simple_grid.nx), dtype=np.int8)
        mask[0, ::2, ::2] = 0  # Mask some cells in z=0 layer

        result = sgsim(
            x=None, y=None, z=None, values=None,
            grid=simple_grid,
            variogram=spherical_variogram,
            search_radius=(50.0, 50.0, 10.0),
            nrealizations=1,
            seed=12345,
            mask=mask,
        )

        assert result.realizations.shape == (1, simple_grid.nz, simple_grid.ny, simple_grid.nx)

        # Check that masked cells have UNEST value (-99.0 for sgsim)
        UNEST = -99.0
        realization = result.realizations[0]
        masked_values = realization[mask == 0]
        assert np.allclose(masked_values, UNEST), f"Masked cells should have UNEST={UNEST}, got {np.unique(masked_values)}"

        # Check that non-masked cells do NOT have UNEST value (should be simulated)
        active_values = realization[mask == 1]
        assert not np.any(np.isclose(active_values, UNEST)), "Active cells should not have UNEST value"

    def test_sgsim_mask_irregular_grid(self, spherical_variogram, irregular_grid):
        """Test simulation with irregular grid (17x23x7) and realistic mask pattern."""
        # Create a more realistic mask - simulate an ore body boundary
        # Mask out ~60% of cells, keeping a central irregular region
        mask = np.zeros((irregular_grid.nz, irregular_grid.ny, irregular_grid.nx), dtype=np.int8)

        # Create an irregular active region (not just a box)
        for iz in range(irregular_grid.nz):
            for iy in range(irregular_grid.ny):
                for ix in range(irregular_grid.nx):
                    # Distance from center of grid
                    cx, cy, cz = irregular_grid.nx // 2, irregular_grid.ny // 2, irregular_grid.nz // 2
                    dist = ((ix - cx) / 8) ** 2 + ((iy - cy) / 10) ** 2 + ((iz - cz) / 3) ** 2
                    # Add some noise to make it irregular
                    if dist < 1.0 + 0.3 * np.sin(ix * 0.5) * np.cos(iy * 0.3):
                        mask[iz, iy, ix] = 1

        n_active = np.sum(mask)
        n_total = mask.size

        result = sgsim(
            x=None, y=None, z=None, values=None,
            grid=irregular_grid,
            variogram=spherical_variogram,
            search_radius=(50.0, 50.0, 20.0),
            nrealizations=1,
            seed=54321,
            mask=mask,
        )

        assert result.realizations.shape == (1, irregular_grid.nz, irregular_grid.ny, irregular_grid.nx)

        UNEST = -99.0
        realization = result.realizations[0]

        # Verify masked cells have UNEST
        masked_values = realization[mask == 0]
        assert np.allclose(masked_values, UNEST), f"Masked cells should have UNEST, got unique values: {np.unique(masked_values)}"

        # Verify active cells are simulated (not UNEST)
        active_values = realization[mask == 1]
        assert not np.any(np.isclose(active_values, UNEST)), "Active cells should not have UNEST"
        assert n_active == len(active_values), f"Expected {n_active} active cells"

    def test_sgsim_mask_edge_cases(self, spherical_variogram):
        """Test mask with edge cases: first/last rows and sparse active cells."""
        # Small grid with specific edge case testing
        edge_grid = GridSpec(nx=11, ny=13, nz=5, xmin=0.5, ymin=0.5, zmin=0.5, xsiz=1.0, ysiz=1.0, zsiz=1.0)

        # Start with all masked
        mask = np.zeros((edge_grid.nz, edge_grid.ny, edge_grid.nx), dtype=np.int8)

        # Only activate specific cells:
        # - First and last cells in x direction
        mask[:, :, 0] = 1   # First x column
        mask[:, :, -1] = 1  # Last x column
        # - A diagonal stripe
        for i in range(min(edge_grid.ny, edge_grid.nx)):
            if i < edge_grid.nx and i < edge_grid.ny:
                mask[edge_grid.nz // 2, i, i] = 1

        result = sgsim(
            x=None, y=None, z=None, values=None,
            grid=edge_grid,
            variogram=spherical_variogram,
            search_radius=(50.0, 50.0, 20.0),
            nrealizations=1,
            seed=99999,
            mask=mask,
        )

        UNEST = -99.0
        realization = result.realizations[0]

        masked_values = realization[mask == 0]
        active_values = realization[mask == 1]

        assert np.allclose(masked_values, UNEST)
        assert not np.any(np.isclose(active_values, UNEST))


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

    def test_ik3d_with_mask(self, sample_data, simple_grid):
        """Test indicator kriging with grid mask - masked cells should have UNEST values."""
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

        # Create a mask - mask out some cells
        mask = np.ones((simple_grid.nz, simple_grid.ny, simple_grid.nx), dtype=np.int8)
        mask[0, ::2, ::2] = 0  # Mask some cells in z=0 layer

        result = ik3d(
            x, y, z, values,
            grid=simple_grid,
            cutoffs=cutoffs,
            global_cdf=global_cdf,
            variograms=variograms,
            search=search,
            mask=mask,
        )

        assert result.probabilities.shape == (len(cutoffs), simple_grid.nz, simple_grid.ny, simple_grid.nx)

        # Check that masked cells have UNEST value (ik3d UNEST = -9.9999)
        UNEST = -9.9999
        # Check first cutoff
        probs = result.probabilities[0]
        masked_values = probs[mask == 0]
        assert np.allclose(masked_values, UNEST), f"Masked cells should have UNEST={UNEST}, got {np.unique(masked_values)}"


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

    def test_sisim_with_mask(self, simple_grid):
        """Test indicator simulation with grid mask - masked cells should have UNEST values."""
        from gslib_zero.simulation import sisim

        # Create indicator variograms for 3 thresholds
        variograms = [
            VariogramModel.spherical(sill=0.15, ranges=(10.0, 10.0, 5.0), nugget=0.1),
            VariogramModel.spherical(sill=0.20, ranges=(10.0, 10.0, 5.0), nugget=0.05),
            VariogramModel.spherical(sill=0.15, ranges=(10.0, 10.0, 5.0), nugget=0.1),
        ]

        # Create a mask - mask out some cells
        mask = np.ones((simple_grid.nz, simple_grid.ny, simple_grid.nx), dtype=np.int8)
        mask[0, ::2, ::2] = 0  # Mask some cells in z=0 layer

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
            mask=mask,
        )

        assert result.realizations.shape == (1, simple_grid.nz, simple_grid.ny, simple_grid.nx)

        # Check that masked cells have UNEST value (-99.0 for sisim)
        UNEST = -99.0
        realization = result.realizations[0]
        masked_values = realization[mask == 0]
        assert np.allclose(masked_values, UNEST), f"Masked cells should have UNEST={UNEST}, got {np.unique(masked_values)}"

        # Check that non-masked cells do NOT have UNEST value (should be simulated)
        active_values = realization[mask == 1]
        assert not np.any(np.isclose(active_values, UNEST)), "Active cells should not have UNEST value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
