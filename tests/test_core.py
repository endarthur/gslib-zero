"""
Tests for core module: binary I/O and subprocess handling.
"""

import numpy as np
import pytest
from pathlib import Path

from gslib_zero.core import BinaryIO, GSLIBWorkspace


class TestBinaryIO:
    """Tests for BinaryIO class."""

    def test_write_read_1d_array(self, tmp_path):
        """Test round-trip for 1D array."""
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        filepath = tmp_path / "test_1d.bin"

        BinaryIO.write_array(original, filepath)
        result = BinaryIO.read_array(filepath)

        np.testing.assert_array_almost_equal(result, original)

    def test_write_read_2d_array(self, tmp_path):
        """Test round-trip for 2D Fortran-order array."""
        # Create F-contiguous array - the format GSLIB expects
        original = np.asfortranarray(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        )
        filepath = tmp_path / "test_2d.bin"

        BinaryIO.write_array(original, filepath)
        result = BinaryIO.read_array(filepath)

        np.testing.assert_array_almost_equal(result, original)
        assert result.shape == original.shape

    def test_write_read_3d_array(self, tmp_path):
        """Test round-trip for 3D array with Fortran ordering."""
        # Create F-contiguous array directly
        original = np.asfortranarray(np.arange(24, dtype=np.float64).reshape((2, 3, 4)))
        filepath = tmp_path / "test_3d.bin"

        BinaryIO.write_array(original, filepath)
        result = BinaryIO.read_array(filepath)

        np.testing.assert_array_almost_equal(result, original)
        assert result.flags["F_CONTIGUOUS"]

    def test_write_read_mask(self, tmp_path):
        """Test round-trip for mask array."""
        original = np.array([[[1, 0], [1, 1]], [[0, 1], [1, 0]]], dtype=np.int8)
        filepath = tmp_path / "test_mask.bin"

        BinaryIO.write_mask(original, filepath)
        result = BinaryIO.read_mask(filepath)

        np.testing.assert_array_equal(result, original)

    def test_fortran_ordering_preserved(self, tmp_path):
        """Test that Fortran ordering is preserved through write/read."""
        # Start with F-contiguous array
        f_array = np.asfortranarray(np.arange(24, dtype=np.float64).reshape((2, 3, 4)))
        assert f_array.flags["F_CONTIGUOUS"]

        filepath = tmp_path / "test_order.bin"
        BinaryIO.write_array(f_array, filepath, fortran_order=True)
        result = BinaryIO.read_array(filepath, fortran_order=True)

        # Result should be F-contiguous
        assert result.flags["F_CONTIGUOUS"]
        # Shape and values should match
        assert result.shape == f_array.shape
        np.testing.assert_array_almost_equal(result, f_array)


class TestGSLIBWorkspace:
    """Tests for GSLIBWorkspace context manager."""

    def test_workspace_creates_directory(self):
        """Test that workspace creates a temporary directory."""
        with GSLIBWorkspace() as path:
            assert path.exists()
            assert path.is_dir()

    def test_workspace_cleanup(self):
        """Test that workspace cleans up after exit."""
        with GSLIBWorkspace() as path:
            temp_path = path
            # Create a file in the workspace
            (path / "test.txt").write_text("test")

        # Directory should be cleaned up
        assert not temp_path.exists()

    def test_workspace_preserve(self, tmp_path):
        """Test that preserve=True keeps the directory."""
        import shutil

        with GSLIBWorkspace(preserve=True) as path:
            temp_path = path
            test_file = path / "test.txt"
            test_file.write_text("test")

        # Directory should still exist after exiting context
        assert temp_path.exists()
        assert test_file.exists()

        # Clean up manually
        shutil.rmtree(temp_path, ignore_errors=True)
