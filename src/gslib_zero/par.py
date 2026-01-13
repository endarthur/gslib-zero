"""
Par file generation utilities for GSLIB programs.

GSLIB programs are controlled by parameter files. This module provides
utilities for generating these files programmatically with validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO


@dataclass
class ParFileBuilder:
    """
    Builder for GSLIB parameter files.

    Provides a fluent interface for constructing par files with proper
    formatting and line tracking.
    """

    lines: list[str] = field(default_factory=list)
    comments: dict[int, str] = field(default_factory=dict)

    def comment(self, text: str) -> "ParFileBuilder":
        """Add a comment line."""
        self.lines.append(f"# {text}")
        return self

    def blank(self) -> "ParFileBuilder":
        """Add a blank line."""
        self.lines.append("")
        return self

    def line(self, *values: Any, comment: str | None = None) -> "ParFileBuilder":
        """
        Add a parameter line with optional inline comment.

        Args:
            *values: Values to write, space-separated
            comment: Optional comment to append after values
        """
        line_str = " ".join(str(v) for v in values)
        if comment:
            line_str = f"{line_str}  # {comment}"
        self.lines.append(line_str)
        return self

    def path(self, filepath: Path | str, comment: str | None = None) -> "ParFileBuilder":
        """Add a file path line."""
        return self.line(str(filepath), comment=comment)

    def grid(
        self,
        nx: int, xmin: float, xsiz: float,
        ny: int, ymin: float, ysiz: float,
        nz: int, zmin: float, zsiz: float,
    ) -> "ParFileBuilder":
        """Add standard GSLIB grid definition (3 lines)."""
        self.line(nx, xmin, xsiz, comment="nx, xmn, xsiz")
        self.line(ny, ymin, ysiz, comment="ny, ymn, ysiz")
        self.line(nz, zmin, zsiz, comment="nz, zmn, zsiz")
        return self

    def search_ellipse(
        self,
        radius1: float, radius2: float, radius3: float,
        azimuth: float = 0.0, dip: float = 0.0, rake: float = 0.0,
    ) -> "ParFileBuilder":
        """Add search ellipse definition (2 lines)."""
        self.line(radius1, radius2, radius3, comment="search radii")
        self.line(azimuth, dip, rake, comment="angles (azimuth, dip, rake)")
        return self

    def variogram_model(
        self,
        nugget: float,
        structures: list[dict],
    ) -> "ParFileBuilder":
        """
        Add variogram model definition.

        Args:
            nugget: Nugget effect (C0)
            structures: List of structure dicts with keys:
                - type: int (1=sph, 2=exp, 3=gau, 4=pow, 5=hole)
                - sill: float (contribution)
                - ranges: tuple of (a1, a2, a3)
                - angles: tuple of (azimuth, dip, rake), optional
        """
        self.line(len(structures), nugget, comment="nst, nugget")

        for struct in structures:
            vtype = struct["type"]
            sill = struct["sill"]
            angles = struct.get("angles", (0.0, 0.0, 0.0))
            ranges = struct["ranges"]

            self.line(vtype, sill, angles[0], angles[1], angles[2],
                     comment="type, sill, angles")
            self.line(ranges[0], ranges[1], ranges[2],
                     comment="ranges")

        return self

    def build(self) -> str:
        """Build the complete par file content."""
        return "\n".join(self.lines) + "\n"

    def write(self, filepath: Path | str) -> Path:
        """Write par file to disk."""
        filepath = Path(filepath)
        filepath.write_text(self.build())
        return filepath


def write_par_file(filepath: Path | str, content: str) -> Path:
    """
    Write a parameter file to disk.

    Args:
        filepath: Output path
        content: Par file content string

    Returns:
        Path to written file
    """
    filepath = Path(filepath)
    filepath.write_text(content)
    return filepath


def validate_positive(value: float, name: str) -> None:
    """Validate that a value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_negative(value: float, name: str) -> None:
    """Validate that a value is non-negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def validate_range(value: float, min_val: float, max_val: float, name: str) -> None:
    """Validate that a value is within a range."""
    if not min_val <= value <= max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
