"""
Tests for drillhole utilities (desurvey, composite, merge_intervals).
"""

import numpy as np
import pytest

from gslib_zero.drillhole import desurvey, composite, merge_intervals


class TestDesurvey:
    """Test minimum curvature desurvey."""

    def test_vertical_hole(self):
        """Test a perfectly vertical hole."""
        collar = {
            "holeid": np.array(["DH001"]),
            "x": np.array([1000.0]),
            "y": np.array([2000.0]),
            "z": np.array([500.0]),
        }

        survey = {
            "holeid": np.array(["DH001", "DH001", "DH001"]),
            "depth": np.array([0.0, 50.0, 100.0]),
            "azimuth": np.array([0.0, 0.0, 0.0]),
            "dip": np.array([-90.0, -90.0, -90.0]),
        }

        result = desurvey(collar, survey)

        # For vertical hole, x and y should stay constant
        assert np.allclose(result["x"], 1000.0, atol=0.01)
        assert np.allclose(result["y"], 2000.0, atol=0.01)

        # z should decrease by depth
        expected_z = np.array([500.0, 450.0, 400.0])
        assert np.allclose(result["z"], expected_z, atol=0.01)

    def test_inclined_north(self):
        """Test a hole dipping north at 45 degrees."""
        collar = {
            "holeid": np.array(["DH001"]),
            "x": np.array([0.0]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
        }

        survey = {
            "holeid": np.array(["DH001", "DH001"]),
            "depth": np.array([0.0, 100.0]),
            "azimuth": np.array([0.0, 0.0]),  # North
            "dip": np.array([-45.0, -45.0]),  # 45 degrees down
        }

        result = desurvey(collar, survey)

        # At 45 degrees, horizontal and vertical components are equal
        # After 100m at 45 degrees: ~70.7m horizontal, ~70.7m vertical
        expected_horiz = 100.0 * np.sin(np.radians(45))
        expected_vert = 100.0 * np.cos(np.radians(45))

        assert result["y"][1] == pytest.approx(expected_horiz, rel=0.01)
        assert result["z"][1] == pytest.approx(-expected_vert, rel=0.01)
        assert result["x"][1] == pytest.approx(0.0, abs=0.01)

    def test_desurvey_at_depths(self):
        """Test desurvey at specified sample depths."""
        collar = {
            "holeid": np.array(["DH001"]),
            "x": np.array([0.0]),
            "y": np.array([0.0]),
            "z": np.array([100.0]),
        }

        survey = {
            "holeid": np.array(["DH001", "DH001"]),
            "depth": np.array([0.0, 100.0]),
            "azimuth": np.array([0.0, 0.0]),
            "dip": np.array([-90.0, -90.0]),
        }

        depths = {
            "holeid": np.array(["DH001", "DH001", "DH001"]),
            "from": np.array([0.0, 20.0, 50.0]),
            "to": np.array([20.0, 50.0, 80.0]),
        }

        result = desurvey(collar, survey, depths)

        # Should have coordinates at midpoints: 10, 35, 65
        assert len(result["depth"]) == 3
        assert result["depth"][0] == pytest.approx(10.0, abs=0.01)
        assert result["depth"][1] == pytest.approx(35.0, abs=0.01)
        assert result["depth"][2] == pytest.approx(65.0, abs=0.01)

    def test_multiple_holes(self):
        """Test desurvey with multiple holes."""
        collar = {
            "holeid": np.array(["DH001", "DH002"]),
            "x": np.array([0.0, 100.0]),
            "y": np.array([0.0, 0.0]),
            "z": np.array([0.0, 0.0]),
        }

        survey = {
            "holeid": np.array(["DH001", "DH001", "DH002", "DH002"]),
            "depth": np.array([0.0, 50.0, 0.0, 50.0]),
            "azimuth": np.array([0.0, 0.0, 90.0, 90.0]),
            "dip": np.array([-90.0, -90.0, -90.0, -90.0]),
        }

        result = desurvey(collar, survey)

        # Should have 4 points total
        assert len(result["holeid"]) == 4

        # First hole should be at x=0
        dh001_mask = result["holeid"] == "DH001"
        assert np.allclose(result["x"][dh001_mask], 0.0, atol=0.01)

        # Second hole should be at x=100
        dh002_mask = result["holeid"] == "DH002"
        assert np.allclose(result["x"][dh002_mask], 100.0, atol=0.01)


class TestComposite:
    """Test length-weighted compositing."""

    def test_simple_composite(self):
        """Test simple compositing with equal intervals."""
        data = {
            "holeid": np.array(["DH001", "DH001", "DH001", "DH001"]),
            "from": np.array([0.0, 1.0, 2.0, 3.0]),
            "to": np.array([1.0, 2.0, 3.0, 4.0]),
            "grade": np.array([1.0, 2.0, 3.0, 4.0]),
        }

        result = composite(data, length=2.0)

        # Should have 2 composites: 0-2 and 2-4
        assert len(result.data["from"]) == 2

        # First composite: (1*1 + 2*1) / 2 = 1.5
        assert result.data["grade"][0] == pytest.approx(1.5, abs=0.01)

        # Second composite: (3*1 + 4*1) / 2 = 3.5
        assert result.data["grade"][1] == pytest.approx(3.5, abs=0.01)

    def test_composite_partial_coverage(self):
        """Test compositing with gaps in data."""
        data = {
            "holeid": np.array(["DH001", "DH001"]),
            "from": np.array([0.0, 3.0]),
            "to": np.array([1.0, 4.0]),
            "grade": np.array([2.0, 4.0]),
        }

        result = composite(data, length=2.0, min_coverage=0.4)

        # Composite 0-2: only 1m of data (50% coverage)
        # Composite 2-4: only 1m of data (50% coverage)
        assert result.coverage[0] == pytest.approx(0.5, abs=0.01)
        assert result.coverage[1] == pytest.approx(0.5, abs=0.01)

    def test_composite_below_min_coverage(self):
        """Test that composites below min_coverage are excluded."""
        data = {
            "holeid": np.array(["DH001", "DH001"]),
            "from": np.array([0.0, 8.0]),
            "to": np.array([1.0, 10.0]),
            "grade": np.array([2.0, 4.0]),
        }

        result = composite(data, length=5.0, min_coverage=0.5)

        # Composite 0-5: 1m of data = 20% coverage (below threshold)
        # Composite 5-10: 2m of data = 40% coverage (below threshold)
        # Both should be excluded
        assert len(result.data["from"]) == 0

    def test_composite_multiple_columns(self):
        """Test compositing multiple value columns."""
        data = {
            "holeid": np.array(["DH001", "DH001"]),
            "from": np.array([0.0, 1.0]),
            "to": np.array([1.0, 2.0]),
            "au": np.array([1.0, 3.0]),
            "cu": np.array([0.5, 1.5]),
        }

        result = composite(data, length=2.0, columns=["au", "cu"])

        # Both columns should be composited
        assert "au" in result.data
        assert "cu" in result.data
        assert result.data["au"][0] == pytest.approx(2.0, abs=0.01)
        assert result.data["cu"][0] == pytest.approx(1.0, abs=0.01)

    def test_composite_domain_boundaries(self):
        """Test that domain boundaries break composites."""
        data = {
            "holeid": np.array(["DH001", "DH001", "DH001", "DH001"]),
            "from": np.array([0.0, 1.0, 2.0, 3.0]),
            "to": np.array([1.0, 2.0, 3.0, 4.0]),
            "grade": np.array([1.0, 2.0, 10.0, 11.0]),
            "domain": np.array([1, 1, 2, 2]),
        }

        result = composite(data, length=2.0, domain_column="domain")

        # Should have 2 composites, split by domain
        # Domain 1: 0-2, Domain 2: 2-4
        assert len(result.data["from"]) == 2
        assert result.data["grade"][0] == pytest.approx(1.5, abs=0.01)  # avg of 1, 2
        assert result.data["grade"][1] == pytest.approx(10.5, abs=0.01)  # avg of 10, 11

    def test_composite_multiple_holes(self):
        """Test compositing multiple holes."""
        data = {
            "holeid": np.array(["DH001", "DH001", "DH002", "DH002"]),
            "from": np.array([0.0, 1.0, 0.0, 1.0]),
            "to": np.array([1.0, 2.0, 1.0, 2.0]),
            "grade": np.array([1.0, 2.0, 5.0, 6.0]),
        }

        result = composite(data, length=2.0)

        # Should have 2 composites, one per hole
        assert len(result.data["from"]) == 2

        # Each hole composited separately
        dh001_mask = result.data["holeid"] == "DH001"
        dh002_mask = result.data["holeid"] == "DH002"

        assert result.data["grade"][dh001_mask][0] == pytest.approx(1.5, abs=0.01)
        assert result.data["grade"][dh002_mask][0] == pytest.approx(5.5, abs=0.01)


class TestMergeIntervals:
    """Test interval merging (intersection)."""

    def test_simple_merge(self):
        """Test simple merge of two tables with different intervals."""
        assay = {
            "holeid": np.array(["DH001", "DH001"]),
            "from": np.array([0.0, 1.0]),
            "to": np.array([1.0, 3.0]),
            "au": np.array([2.5, 1.2]),
        }

        geology = {
            "holeid": np.array(["DH001", "DH001"]),
            "from": np.array([0.0, 2.0]),
            "to": np.array([2.0, 3.0]),
            "rock": np.array(["QZ", "GR"]),
        }

        result = merge_intervals(assay, geology)

        # Should have 3 intervals: 0-1, 1-2, 2-3
        assert len(result["from"]) == 3

        # Check boundaries
        assert list(result["from"]) == [0.0, 1.0, 2.0]
        assert list(result["to"]) == [1.0, 2.0, 3.0]

        # Check values
        # 0-1: au=2.5, rock=QZ
        # 1-2: au=1.2, rock=QZ
        # 2-3: au=1.2, rock=GR
        assert result["au"][0] == pytest.approx(2.5, abs=0.01)
        assert result["au"][1] == pytest.approx(1.2, abs=0.01)
        assert result["au"][2] == pytest.approx(1.2, abs=0.01)
        assert result["rock"][0] == "QZ"
        assert result["rock"][1] == "QZ"
        assert result["rock"][2] == "GR"

    def test_merge_with_gaps(self):
        """Test merging where one table has gaps."""
        assay = {
            "holeid": np.array(["DH001", "DH001"]),
            "from": np.array([0.0, 2.0]),
            "to": np.array([1.0, 3.0]),
            "au": np.array([2.5, 1.2]),
        }

        geology = {
            "holeid": np.array(["DH001"]),
            "from": np.array([0.0]),
            "to": np.array([3.0]),
            "rock": np.array(["QZ"]),
        }

        result = merge_intervals(assay, geology)

        # Should have 3 intervals: 0-1, 1-2, 2-3
        assert len(result["from"]) == 3

        # Gap region (1-2) should have NaN for au
        assert np.isnan(result["au"][1])
        assert result["rock"][1] == "QZ"

    def test_merge_multiple_holes(self):
        """Test merging with multiple holes."""
        assay = {
            "holeid": np.array(["DH001", "DH001", "DH002"]),
            "from": np.array([0.0, 1.0, 0.0]),
            "to": np.array([1.0, 2.0, 1.0]),
            "au": np.array([1.0, 2.0, 5.0]),
        }

        geology = {
            "holeid": np.array(["DH001", "DH002"]),
            "from": np.array([0.0, 0.0]),
            "to": np.array([2.0, 1.0]),
            "rock": np.array(["QZ", "GR"]),
        }

        result = merge_intervals(assay, geology)

        # Each hole should be processed separately
        dh001_mask = result["holeid"] == "DH001"
        dh002_mask = result["holeid"] == "DH002"

        assert np.sum(dh001_mask) == 2  # DH001: 0-1, 1-2
        assert np.sum(dh002_mask) == 1  # DH002: 0-1

    def test_merge_three_tables(self):
        """Test merging three tables."""
        assay = {
            "holeid": np.array(["DH001"]),
            "from": np.array([0.0]),
            "to": np.array([3.0]),
            "au": np.array([2.0]),
        }

        geology = {
            "holeid": np.array(["DH001", "DH001"]),
            "from": np.array([0.0, 1.5]),
            "to": np.array([1.5, 3.0]),
            "rock": np.array(["QZ", "GR"]),
        }

        density = {
            "holeid": np.array(["DH001", "DH001", "DH001"]),
            "from": np.array([0.0, 1.0, 2.0]),
            "to": np.array([1.0, 2.0, 3.0]),
            "sg": np.array([2.7, 2.8, 2.9]),
        }

        result = merge_intervals(assay, geology, density)

        # Should have intervals at: 0, 1, 1.5, 2, 3
        # So intervals: 0-1, 1-1.5, 1.5-2, 2-3
        assert len(result["from"]) == 4

        # Check all columns present
        assert "au" in result
        assert "rock" in result
        assert "sg" in result

    def test_merge_tolerance(self):
        """Test that close boundaries are merged within tolerance."""
        assay = {
            "holeid": np.array(["DH001"]),
            "from": np.array([0.0]),
            "to": np.array([1.0]),
            "au": np.array([2.0]),
        }

        geology = {
            "holeid": np.array(["DH001"]),
            "from": np.array([0.0]),
            "to": np.array([1.001]),  # Slightly different due to floating point
            "rock": np.array(["QZ"]),
        }

        result = merge_intervals(assay, geology, tolerance=0.01)

        # Should treat 1.0 and 1.001 as the same boundary
        assert len(result["from"]) == 1
        assert result["to"][0] == pytest.approx(1.0, abs=0.01)

