"""
Core functionality for gslib-zero: subprocess handling and binary I/O.

This module provides:
- Binary serialization/deserialization of numpy arrays
- Subprocess execution of GSLIB Fortran executables
- Temporary file management
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_bin_dir() -> Path:
    """Get the directory containing GSLIB executables."""
    # First check for environment variable override
    if env_path := os.environ.get("GSLIB_BIN_DIR"):
        return Path(env_path)

    # Default locations relative to package
    package_dir = Path(__file__).parent
    repo_root = package_dir.parent.parent

    # Check for modified binaries first, then fall back to original Gslib90
    candidates = [
        repo_root / "bin",
        repo_root / "bin" / "Gslib90",
    ]

    for candidate in candidates:
        if candidate.exists() and any(candidate.iterdir()):
            # Check if there are actual executables (not just .gitkeep)
            exes = list(candidate.glob("*.exe")) + [
                f for f in candidate.iterdir()
                if f.is_file() and not f.suffix and f.stat().st_mode & 0o111
            ]
            if exes:
                return candidate

    # Default to bin/ even if empty (will error on get_executable)
    return repo_root / "bin"


def get_executable(name: str, precision: str = "f32") -> Path:
    """
    Get the path to a GSLIB executable.

    Args:
        name: Name of the executable (e.g., 'kt3d', 'sgsim')
        precision: 'f32' for single precision (default), 'f64' for double precision

    Returns:
        Path to the executable

    Raises:
        FileNotFoundError: If executable doesn't exist
        ValueError: If precision is invalid
    """
    if precision not in ("f32", "f64"):
        raise ValueError(f"precision must be 'f32' or 'f64', got '{precision}'")

    bin_dir = get_bin_dir()

    # For f64, use program_f64 suffix
    exe_name = f"{name}_f64" if precision == "f64" else name

    # Handle Windows .exe extension
    if os.name == "nt":
        exe_path = bin_dir / f"{exe_name}.exe"
    else:
        exe_path = bin_dir / exe_name

    if not exe_path.exists():
        raise FileNotFoundError(
            f"GSLIB executable '{exe_name}' not found at {exe_path}. "
            f"Set GSLIB_BIN_DIR environment variable or run: "
            f"python -m gslib_zero.download_binaries"
        )

    return exe_path


class BinaryIO:
    """
    Binary I/O utilities for GSLIB data exchange.

    GSLIB's bottleneck is ASCII parsing. Binary I/O provides:
    - Direct numpy array read/write
    - No whitespace parsing ambiguity
    - 10-100x faster for large datasets
    """

    @staticmethod
    def write_array(
        array: NDArray[np.floating],
        filepath: Path | str,
        fortran_order: bool = True,
    ) -> None:
        """
        Write a numpy array to binary file for GSLIB consumption.

        Args:
            array: Numpy array to write
            filepath: Output file path
            fortran_order: If True, ensure Fortran (column-major) ordering.
                          GSLIB Fortran code expects column-major arrays.
        """
        filepath = Path(filepath)

        # Ensure correct dtype and memory layout
        arr = np.asarray(array, dtype=np.float64)
        if fortran_order:
            arr = np.asfortranarray(arr)

        # Write header: ndim, then shape
        with open(filepath, "wb") as f:
            # Write number of dimensions
            np.array([arr.ndim], dtype=np.int32).tofile(f)
            # Write shape
            np.array(arr.shape, dtype=np.int32).tofile(f)
            # Write data in correct order
            # Note: tofile() always writes C-order, so we must flatten explicitly
            write_order = "F" if fortran_order else "C"
            arr.flatten(order=write_order).tofile(f)

    @staticmethod
    def read_array(
        filepath: Path | str,
        fortran_order: bool = True,
        dtype: type = np.float64,
    ) -> NDArray[np.floating]:
        """
        Read a numpy array from binary file produced by GSLIB.

        Args:
            filepath: Input file path
            fortran_order: If True, reshape with Fortran ordering
            dtype: Data type of the array elements. Use np.float32 for GSLIB
                   single-precision output (default np.float64 for general use)

        Returns:
            Numpy array with data from file
        """
        filepath = Path(filepath)

        with open(filepath, "rb") as f:
            # Read header
            ndim = np.fromfile(f, dtype=np.int32, count=1)[0]
            shape = tuple(np.fromfile(f, dtype=np.int32, count=ndim))

            # Read data
            data = np.fromfile(f, dtype=dtype)

        # Reshape with correct ordering
        order = "F" if fortran_order else "C"
        return data.reshape(shape, order=order)

    @staticmethod
    def write_mask(
        mask: NDArray[np.integer],
        filepath: Path | str,
    ) -> None:
        """
        Write a grid mask to binary file.

        Args:
            mask: Int8 array where 1=active, 0=inactive
            filepath: Output file path
        """
        filepath = Path(filepath)

        arr = np.asarray(mask, dtype=np.int8)
        arr = np.asfortranarray(arr)

        with open(filepath, "wb") as f:
            np.array([arr.ndim], dtype=np.int32).tofile(f)
            np.array(arr.shape, dtype=np.int32).tofile(f)
            # Write in Fortran order for GSLIB compatibility
            arr.flatten(order="F").tofile(f)

    @staticmethod
    def read_mask(filepath: Path | str) -> NDArray[np.int8]:
        """
        Read a grid mask from binary file.

        Args:
            filepath: Input file path

        Returns:
            Int8 array mask
        """
        filepath = Path(filepath)

        with open(filepath, "rb") as f:
            ndim = np.fromfile(f, dtype=np.int32, count=1)[0]
            shape = tuple(np.fromfile(f, dtype=np.int32, count=ndim))
            data = np.fromfile(f, dtype=np.int8)

        return data.reshape(shape, order="F")


def run_gslib(
    program: str,
    par_file: Path | str,
    capture_output: bool = True,
    timeout: float | None = None,
    precision: str = "f32",
) -> subprocess.CompletedProcess:
    """
    Run a GSLIB executable with the given parameter file.

    Args:
        program: Name of the GSLIB program (e.g., 'kt3d')
        par_file: Path to the parameter file
        capture_output: If True, capture stdout/stderr
        timeout: Timeout in seconds (None for no timeout)
        precision: 'f32' for single precision, 'f64' for double precision

    Returns:
        CompletedProcess with return code and captured output

    Raises:
        FileNotFoundError: If executable not found
        subprocess.TimeoutExpired: If timeout exceeded
        subprocess.CalledProcessError: If program returns non-zero exit code
    """
    exe_path = get_executable(program, precision=precision)
    par_file = Path(par_file)

    if not par_file.exists():
        raise FileNotFoundError(f"Parameter file not found: {par_file}")

    # Run the program
    result = subprocess.run(
        [str(exe_path)],
        input=str(par_file) + "\n",
        capture_output=capture_output,
        text=True,
        timeout=timeout,
        cwd=par_file.parent,
    )

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            exe_path,
            result.stdout,
            result.stderr,
        )

    return result


class AsciiIO:
    """
    ASCII I/O utilities for classic GSLIB file format.

    GSLIB ASCII format:
    - Line 1: Title/description
    - Line 2: Number of variables (nvars)
    - Lines 3 to 3+nvars: Variable names
    - Remaining lines: Data (space-separated values)
    """

    @staticmethod
    def write_data(
        filepath: Path | str,
        data: dict[str, NDArray[np.floating]],
        title: str = "gslib-zero output",
    ) -> None:
        """
        Write data to GSLIB ASCII format.

        Args:
            filepath: Output file path
            data: Dictionary of {column_name: values_array}
            title: Title line for the file
        """
        filepath = Path(filepath)
        names = list(data.keys())
        arrays = [np.asarray(data[n], dtype=np.float64).ravel() for n in names]

        # Verify all arrays have same length
        n = len(arrays[0])
        if not all(len(a) == n for a in arrays):
            raise ValueError("All arrays must have the same length")

        with open(filepath, "w") as f:
            f.write(f"{title}\n")
            f.write(f"{len(names)}\n")
            for name in names:
                f.write(f"{name}\n")
            for i in range(n):
                row = " ".join(f"{a[i]:.10g}" for a in arrays)
                f.write(f"{row}\n")

    @staticmethod
    def read_data(filepath: Path | str) -> tuple[list[str], NDArray[np.float64]]:
        """
        Read data from GSLIB ASCII format.

        Args:
            filepath: Input file path

        Returns:
            Tuple of (column_names, data_array) where data_array has shape (n, nvars)
        """
        filepath = Path(filepath)

        with open(filepath, "r") as f:
            _title = f.readline().strip()
            nvars = int(f.readline().strip())
            names = [f.readline().strip() for _ in range(nvars)]

            rows = []
            for line in f:
                line = line.strip()
                if line:
                    values = [float(x) for x in line.split()]
                    rows.append(values)

        data = np.array(rows, dtype=np.float64)
        return names, data

    @staticmethod
    def write_column(
        filepath: Path | str,
        values: NDArray[np.floating],
        name: str = "value",
        title: str = "gslib-zero output",
    ) -> None:
        """Convenience method to write a single column."""
        AsciiIO.write_data(filepath, {name: values}, title)

    @staticmethod
    def read_column(
        filepath: Path | str,
        column: int | str = 0,
    ) -> NDArray[np.float64]:
        """
        Read a single column from GSLIB file.

        Args:
            filepath: Input file path
            column: Column index (0-based) or column name

        Returns:
            1D array of values
        """
        names, data = AsciiIO.read_data(filepath)

        if isinstance(column, str):
            if column not in names:
                raise ValueError(f"Column '{column}' not found. Available: {names}")
            col_idx = names.index(column)
        else:
            col_idx = column

        return data[:, col_idx]

    @staticmethod
    def read_raw(filepath: Path | str) -> NDArray[np.float64]:
        """
        Read raw numeric data (no GSLIB header).

        Some GSLIB outputs (like transformation tables) are just
        whitespace-separated numbers with no header.

        Args:
            filepath: Input file path

        Returns:
            2D array of values
        """
        filepath = Path(filepath)
        return np.loadtxt(filepath, dtype=np.float64)

    @staticmethod
    def read_gridded_data(
        filepath: Path | str,
    ) -> tuple[list[str], NDArray[np.float64], dict]:
        """
        Read gridded GSLIB output format (kt3d, sgsim, etc.).

        Format:
        - Line 1: Title
        - Line 2: nvars nx ny nz xmin ymin zmin xsiz ysiz zsiz [extras]
        - Lines 3+: Column names (nvars of them)
        - Rest: Data values

        Args:
            filepath: Input file path

        Returns:
            Tuple of (column_names, data_array, grid_info) where:
            - column_names: List of variable names
            - data_array: 2D array of shape (ncells, nvars)
            - grid_info: Dict with nx, ny, nz, xmin, ymin, zmin, xsiz, ysiz, zsiz
        """
        filepath = Path(filepath)

        with open(filepath, "r") as f:
            _title = f.readline().strip()

            # Parse grid info line
            grid_line = f.readline().strip().split()
            nvars = int(grid_line[0])
            nx = int(grid_line[1])
            ny = int(grid_line[2])
            nz = int(grid_line[3])
            xmin = float(grid_line[4])
            ymin = float(grid_line[5])
            zmin = float(grid_line[6])
            xsiz = float(grid_line[7])
            ysiz = float(grid_line[8])
            zsiz = float(grid_line[9])

            grid_info = {
                "nx": nx, "ny": ny, "nz": nz,
                "xmin": xmin, "ymin": ymin, "zmin": zmin,
                "xsiz": xsiz, "ysiz": ysiz, "zsiz": zsiz,
            }

            # Read column names
            names = [f.readline().strip() for _ in range(nvars)]

            # Read data
            rows = []
            for line in f:
                line = line.strip()
                if line:
                    values = [float(x) for x in line.split()]
                    rows.append(values)

        data = np.array(rows, dtype=np.float64)
        return names, data, grid_info

    @staticmethod
    def write_raw(
        filepath: Path | str,
        data: NDArray[np.floating],
        fmt: str = "%.10g",
    ) -> None:
        """
        Write raw numeric data (no GSLIB header).

        Args:
            filepath: Output file path
            data: 2D array of values
            fmt: Format string for values
        """
        filepath = Path(filepath)
        np.savetxt(filepath, data, fmt=fmt)


class GSLIBWorkspace:
    """
    Context manager for temporary GSLIB working directory.

    Creates a temporary directory for par files, input data, and output data.
    Cleans up automatically on exit unless preserve=True.
    """

    def __init__(self, preserve: bool = False, prefix: str = "gslib_"):
        """
        Args:
            preserve: If True, don't delete the workspace on exit
            prefix: Prefix for the temporary directory name
        """
        self.preserve = preserve
        self.prefix = prefix
        self._path: Path | None = None

    def __enter__(self) -> Path:
        # Use mkdtemp directly so we control cleanup
        self._path = Path(tempfile.mkdtemp(prefix=self.prefix))
        return self._path

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._path is not None and not self.preserve:
            import shutil
            shutil.rmtree(self._path, ignore_errors=True)
        return None

    @property
    def path(self) -> Path:
        if self._path is None:
            raise RuntimeError("Workspace not initialized. Use as context manager.")
        return self._path
