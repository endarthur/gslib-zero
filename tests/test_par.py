"""
Tests for par file generation utilities.
"""

import pytest
from pathlib import Path

from gslib_zero.par import ParFileBuilder, validate_positive, validate_range


class TestParFileBuilder:
    """Tests for ParFileBuilder class."""

    def test_simple_lines(self):
        """Test adding simple parameter lines."""
        par = ParFileBuilder()
        par.line(10, 5.0, 2.5)

        content = par.build()
        assert "10 5.0 2.5" in content

    def test_comment(self):
        """Test adding comments."""
        par = ParFileBuilder()
        par.comment("This is a test")

        content = par.build()
        assert "# This is a test" in content

    def test_inline_comment(self):
        """Test inline comments on parameter lines."""
        par = ParFileBuilder()
        par.line(10, 20, comment="nx, ny")

        content = par.build()
        assert "10 20  # nx, ny" in content

    def test_path_line(self):
        """Test path parameter lines."""
        par = ParFileBuilder()
        par.path("/path/to/file.dat", comment="input file")

        content = par.build()
        assert "/path/to/file.dat" in content

    def test_grid_definition(self):
        """Test grid definition helper."""
        par = ParFileBuilder()
        par.grid(10, 0.5, 1.0, 20, 0.5, 1.0, 5, 0.5, 2.0)

        content = par.build()
        lines = content.strip().split("\n")

        assert "10 0.5 1.0" in lines[0]
        assert "20 0.5 1.0" in lines[1]
        assert "5 0.5 2.0" in lines[2]

    def test_search_ellipse(self):
        """Test search ellipse helper."""
        par = ParFileBuilder()
        par.search_ellipse(100.0, 50.0, 10.0, 45.0, 30.0, 0.0)

        content = par.build()
        assert "100.0 50.0 10.0" in content
        assert "45.0 30.0 0.0" in content

    def test_variogram_model(self):
        """Test variogram model definition."""
        par = ParFileBuilder()
        structures = [
            {"type": 1, "sill": 0.8, "ranges": (50.0, 50.0, 10.0), "angles": (0.0, 0.0, 0.0)},
            {"type": 2, "sill": 0.2, "ranges": (100.0, 100.0, 20.0)},
        ]
        par.variogram_model(nugget=0.1, structures=structures)

        content = par.build()
        assert "2 0.1" in content  # nst, nugget
        assert "1 0.8" in content  # type, sill for first structure

    def test_write_file(self, tmp_path):
        """Test writing par file to disk."""
        par = ParFileBuilder()
        par.comment("Test par file")
        par.line(1, 2, 3)

        filepath = tmp_path / "test.par"
        result_path = par.write(filepath)

        assert result_path == filepath
        assert filepath.exists()
        assert "Test par file" in filepath.read_text()


class TestValidation:
    """Tests for validation functions."""

    def test_validate_positive_valid(self):
        """Test validate_positive with valid input."""
        validate_positive(1.0, "test")
        validate_positive(0.001, "test")

    def test_validate_positive_invalid(self):
        """Test validate_positive with invalid input."""
        with pytest.raises(ValueError):
            validate_positive(0.0, "test")
        with pytest.raises(ValueError):
            validate_positive(-1.0, "test")

    def test_validate_range_valid(self):
        """Test validate_range with valid input."""
        validate_range(5.0, 0.0, 10.0, "test")
        validate_range(0.0, 0.0, 10.0, "test")
        validate_range(10.0, 0.0, 10.0, "test")

    def test_validate_range_invalid(self):
        """Test validate_range with invalid input."""
        with pytest.raises(ValueError):
            validate_range(-1.0, 0.0, 10.0, "test")
        with pytest.raises(ValueError):
            validate_range(11.0, 0.0, 10.0, "test")
