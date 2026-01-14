"""
Tests for rotation convention conversions.
"""

import numpy as np
import pytest

from gslib_zero.utils import (
    deutsch_to_dipdirection,
    deutsch_to_datamine,
    deutsch_to_leapfrog,
    deutsch_to_vulcan,
    dipdirection_to_deutsch,
    datamine_to_deutsch,
    leapfrog_to_deutsch,
    vulcan_to_deutsch,
    rotation_matrix_deutsch,
)


class TestDipDirectionConversion:
    """Test Dip Direction / Dip convention conversions."""

    def test_dipdirection_to_deutsch_basic(self):
        """Test basic dip direction to deutsch conversion."""
        # Dip direction 90 (East) -> Strike is 0 (North)
        azm, dip, rake = dipdirection_to_deutsch(90.0, 45.0, 0.0)
        assert azm == pytest.approx(0.0, abs=0.01)
        assert dip == pytest.approx(45.0, abs=0.01)
        assert rake == pytest.approx(0.0, abs=0.01)

    def test_dipdirection_to_deutsch_north(self):
        """Test dip direction pointing north."""
        # Dip direction 0 (North) -> Strike is 270 (West)
        azm, dip, rake = dipdirection_to_deutsch(0.0, 30.0, 0.0)
        assert azm == pytest.approx(270.0, abs=0.01)
        assert dip == pytest.approx(30.0, abs=0.01)

    def test_roundtrip_dipdirection(self):
        """Test round-trip conversion."""
        original = (135.0, 60.0, 10.0)
        deutsch = dipdirection_to_deutsch(*original)
        recovered = deutsch_to_dipdirection(*deutsch)

        assert recovered[0] == pytest.approx(original[0], abs=0.01)
        assert recovered[1] == pytest.approx(original[1], abs=0.01)
        assert recovered[2] == pytest.approx(original[2], abs=0.01)


class TestDatamineConversion:
    """Test Datamine convention conversions."""

    def test_datamine_to_deutsch_basic(self):
        """Test basic datamine to deutsch conversion."""
        # Datamine is essentially same as dip direction
        azm, dip, rake = datamine_to_deutsch(90.0, 45.0, 0.0)
        assert azm == pytest.approx(0.0, abs=0.01)
        assert dip == pytest.approx(45.0, abs=0.01)

    def test_roundtrip_datamine(self):
        """Test round-trip conversion."""
        original = (180.0, 30.0, 5.0)
        deutsch = datamine_to_deutsch(*original)
        recovered = deutsch_to_datamine(*deutsch)

        assert recovered[0] == pytest.approx(original[0], abs=0.01)
        assert recovered[1] == pytest.approx(original[1], abs=0.01)
        assert recovered[2] == pytest.approx(original[2], abs=0.01)


class TestVulcanConversion:
    """Test Vulcan (Strike/Dip) convention conversions."""

    def test_vulcan_to_deutsch_basic(self):
        """Test basic vulcan to deutsch conversion."""
        # Strike 0 (North), dip 45 -> Deutsch azimuth = 0, dip = 45
        azm, dip, rake = vulcan_to_deutsch(0.0, 45.0, 0.0)
        assert azm == pytest.approx(0.0, abs=0.01)
        assert dip == pytest.approx(45.0, abs=0.01)

    def test_vulcan_to_deutsch_east_strike(self):
        """Test vulcan with east-striking plane."""
        # Strike 90 (East), dip 60 -> Deutsch azimuth = 90
        azm, dip, rake = vulcan_to_deutsch(90.0, 60.0, 0.0)
        assert azm == pytest.approx(90.0, abs=0.01)
        assert dip == pytest.approx(60.0, abs=0.01)

    def test_roundtrip_vulcan(self):
        """Test round-trip conversion."""
        original = (45.0, 75.0, 0.0)
        deutsch = vulcan_to_deutsch(*original)
        recovered = deutsch_to_vulcan(*deutsch)

        assert recovered[0] == pytest.approx(original[0], abs=0.01)
        assert recovered[1] == pytest.approx(original[1], abs=0.01)
        assert recovered[2] == pytest.approx(original[2], abs=0.01)


class TestLeapfrogConversion:
    """Test Leapfrog convention conversions (already implemented)."""

    def test_roundtrip_leapfrog(self):
        """Test round-trip conversion."""
        original = (45.0, 135.0, 10.0)  # dip, dip_direction, pitch
        deutsch = leapfrog_to_deutsch(*original)
        recovered = deutsch_to_leapfrog(*deutsch)

        assert recovered[0] == pytest.approx(original[0], abs=0.01)
        assert recovered[1] == pytest.approx(original[1], abs=0.01)
        assert recovered[2] == pytest.approx(original[2], abs=0.01)


class TestRotationMatrix:
    """Test rotation matrix generation."""

    def test_identity_at_zero(self):
        """Test that zero angles give near-identity matrix."""
        R = rotation_matrix_deutsch(0.0, 0.0, 0.0)
        # Should be close to identity (within numerical precision)
        assert R.shape == (3, 3)
        # Determinant should be 1 (proper rotation)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)

    def test_orthogonal(self):
        """Test that rotation matrix is orthogonal."""
        R = rotation_matrix_deutsch(45.0, 30.0, 15.0)
        # R @ R.T should be identity
        RRT = R @ R.T
        I = np.eye(3)
        assert np.allclose(RRT, I, atol=1e-10)

    def test_determinant_one(self):
        """Test that rotation matrix has determinant 1."""
        for azm in [0, 45, 90, 180, 270]:
            for dip in [-45, 0, 45]:
                for rake in [-30, 0, 30]:
                    R = rotation_matrix_deutsch(azm, dip, rake)
                    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)


class TestEdgeCases:
    """Test edge cases for all conventions."""

    @pytest.mark.parametrize("convention,converter,inverse", [
        ("dipdirection", dipdirection_to_deutsch, deutsch_to_dipdirection),
        ("datamine", datamine_to_deutsch, deutsch_to_datamine),
        ("vulcan", vulcan_to_deutsch, deutsch_to_vulcan),
    ])
    def test_zero_angles(self, convention, converter, inverse):
        """Test conversion with zero angles."""
        deutsch = converter(0.0, 0.0, 0.0)
        recovered = inverse(*deutsch)
        # All angles should be non-negative after conversion
        assert all(a >= 0 for a in recovered[:2])

    @pytest.mark.parametrize("convention,converter,inverse", [
        ("dipdirection", dipdirection_to_deutsch, deutsch_to_dipdirection),
        ("datamine", datamine_to_deutsch, deutsch_to_datamine),
        ("vulcan", vulcan_to_deutsch, deutsch_to_vulcan),
    ])
    def test_90_dip(self, convention, converter, inverse):
        """Test conversion with 90 degree dip (vertical)."""
        deutsch = converter(0.0, 90.0, 0.0)
        assert deutsch[1] == pytest.approx(90.0, abs=0.01)

    @pytest.mark.parametrize("convention,converter,inverse", [
        ("dipdirection", dipdirection_to_deutsch, deutsch_to_dipdirection),
        ("datamine", datamine_to_deutsch, deutsch_to_datamine),
        ("vulcan", vulcan_to_deutsch, deutsch_to_vulcan),
    ])
    def test_360_wrap(self, convention, converter, inverse):
        """Test that 360 degree direction wraps to 0."""
        deutsch1 = converter(0.0, 45.0, 0.0)
        deutsch2 = converter(360.0, 45.0, 0.0)
        # Should be equivalent
        assert deutsch1[0] % 360 == pytest.approx(deutsch2[0] % 360, abs=0.01)
