# Binary I/O Implementation Guide

This document describes how binary I/O was implemented in gslib-zero, using kt3d as the reference implementation. Use this guide when adding binary I/O support to other GSLIB programs.

## Overview

### Why Binary I/O?

GSLIB's primary bottleneck is ASCII parsing. For large grids (millions of cells), reading/writing text files dominates runtime. Binary I/O provides:

- **10-100x faster** I/O for large datasets
- **No parsing ambiguity** (whitespace, precision)
- **Exact value preservation** (within float32 precision)

### Design Principles

1. **Minimal Fortran changes** - Only modify I/O, not algorithms
2. **Backward compatible** - ASCII mode still works with original binaries
3. **Stream writes** - Write directly in loops, no buffering needed
4. **Simple format** - Header + raw data, easy to read in any language

---

## Binary Format Specification

### File Structure

```
┌─────────────────────────────────────┐
│  Header (variable length)           │
├─────────────────────────────────────┤
│  Data (float32 array)               │
└─────────────────────────────────────┘
```

### Header Format

```
[ndim: int32] [dim1: int32] [dim2: int32] ... [dimN: int32]
```

- `ndim`: Number of dimensions (e.g., 4 for grid output with multiple variables)
- `dim1..dimN`: Size of each dimension

### Example: kt3d Output

kt3d outputs estimate and variance for each grid cell:

```
Header: [4] [2] [nz] [ny] [nx]
        │   │   └─────────────── grid dimensions
        │   └───────────────────  nvars (estimate + variance)
        └─────────────────────── ndim = 4

Data: [est_1, var_1, est_2, var_2, ..., est_N, var_N] as float32
      (interleaved by cell, Fortran column-major order)
```

Total file size: `(1 + ndim) * 4 + nvars * nz * ny * nx * 4` bytes

### Data Type

- **float32 (real\*4)** - GSLIB uses single precision internally
- Future: float64 builds planned (see ROADMAP.md)

### Byte Order

- **Little-endian** (native on x86/x64)
- Fortran stream I/O uses native byte order

---

## Fortran Implementation

### Step 1: Add ibinary to Common Block

In the include file (e.g., `kt3d.inc`), add `ibinary` to the common block:

```fortran
c kt3d.inc - add ibinary to datcom common block
      common /datcom/ nd,tmin,tmax,nx,ny,nz,xmn,ymn,zmn,test,
     +                xsiz,ysiz,zsiz,ndmax,ndmin,radius,noct,nxdis,
     +                nydis,nzdis,idrif,itrend,ktype,skmean,koption,
     +                idbg,ldbg,iout,lout,iext,lext,iextve,ljack,
     +                idhlj,ixlj,iylj,izlj,ivrlj,iextvj,nvarij,
     +                ibinary    ! <-- ADD THIS
```

### Step 2: Read ibinary from Par File

In the `readparm` subroutine, read the binary flag after the output filename:

```fortran
c Read output file name
      read(lin,'(a512)',err=98) outfl
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl(1:40)

c Read binary output flag (gslib-zero extension)
      read(lin,*,err=98) ibinary
      write(*,*) ' binary output = ',ibinary
```

### Step 3: Open File with Correct Mode

Replace the standard `open` with conditional binary/ASCII:

```fortran
c Open output file
      if(ibinary.eq.1) then
c           Binary mode: stream access, unformatted
            open(lout,file=outfl,status='UNKNOWN',
     +           access='STREAM',form='UNFORMATTED')
c           Write header: ndim, nvars, nz, ny, nx
            write(lout) 4, 2, nz, ny, nx
      else
c           ASCII mode: standard formatted output
            open(lout,file=outfl,status='UNKNOWN')
            write(lout,'(a80)') title
            write(lout,'(4i5)') 2, nx, ny, nz
            write(lout,'(a20)') 'Estimate'
            write(lout,'(a20)') 'EstimationVariance'
      end if
```

Key points:
- `access='STREAM'` - Enables direct binary writes without record markers
- `form='UNFORMATTED'` - Binary mode
- Header written immediately after opening

### Step 4: Write Data in Main Loop

In the estimation/simulation loop, write binary or ASCII:

```fortran
c Write results for this cell
      if(ibinary.eq.1) then
c           Binary: write estimate and variance directly
            write(lout) est, estv
      else
c           ASCII: formatted output
            write(lout,'(g14.8,1x,g14.8)') est, estv
      end if
```

Key points:
- Binary writes are simpler - no format strings needed
- Data order must match header (estimate first, then variance)
- Fortran writes in column-major order naturally

### Step 5: Update makepar Subroutine

Add the ibinary parameter to the example par file generator:

```fortran
      subroutine makepar
c-----------------------------------------------------------------------
c     Write default parameter file
c-----------------------------------------------------------------------
      ...
      write(lun,'(a)') 'kt3d.out                    -output file'
      write(lun,'(a)') '0                           -binary output (0=no, 1=yes)'
      ...
      return
      end
```

---

## Python Implementation

### Step 1: Add binary Parameter to Wrapper

```python
def kt3d(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    z: NDArray[np.floating],
    values: NDArray[np.floating],
    grid: GridSpec,
    variogram: VariogramModel,
    search: SearchParameters,
    kriging_type: Literal["simple", "ordinary"] = "ordinary",
    sk_mean: float = 0.0,
    tmin: float = -1.0e21,
    tmax: float = 1.0e21,
    block_discretization: tuple[int, int, int] = (1, 1, 1),
    binary: bool = False,  # <-- ADD THIS
) -> KrigingResult:
```

### Step 2: Always Include ibinary in Par File

```python
# Write ibinary flag (required by gslib-zero binaries)
# 0 = ASCII output, 1 = binary output
par.line(1 if binary else 0, comment="binary output (0=ASCII, 1=binary)")
```

**Important**: Always include the ibinary line, even for ASCII mode. The modified Fortran expects to read this parameter.

### Step 3: Read Output Based on Mode

```python
if binary:
    # Binary output: 4D array (nvars, nz, ny, nx)
    # GSLIB uses single precision (float32)
    output_data = BinaryIO.read_array(output_file, dtype=np.float32)
    # Shape is (2, nz, ny, nx) - first dim is variable (0=est, 1=var)
    estimate = output_data[0].astype(np.float64)  # Convert to float64
    variance = output_data[1].astype(np.float64)
else:
    # ASCII output: custom header format for kt3d
    # Line 1: title
    # Line 2: nvars nx ny nz
    # Lines 3+: variable names
    # Rest: data
    with open(output_file, "r") as f:
        _title = f.readline()  # Skip title
        header = f.readline().split()
        nvars = int(header[0])
        # Skip variable names
        for _ in range(nvars):
            f.readline()
        # Read data
        rows = []
        for line in f:
            line = line.strip()
            if line:
                values = [float(x) for x in line.split()]
                rows.append(values)

    output_data = np.array(rows, dtype=np.float64)
    estimate = output_data[:, 0].reshape((grid.nz, grid.ny, grid.nx), order="F")
    variance = output_data[:, 1].reshape((grid.nz, grid.ny, grid.nx), order="F")
```

### BinaryIO.read_array Method

```python
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
        dtype: Data type (use np.float32 for GSLIB output)
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
```

---

## Testing

### Cross-Validation Test

Always add a test that verifies binary and ASCII modes produce identical results:

```python
def test_kt3d_binary_matches_ascii(self, sample_data, spherical_variogram):
    """Test that binary mode produces same results as ASCII mode."""
    # ... setup code ...

    # Run in ASCII mode
    result_ascii = kt3d(..., binary=False)

    # Run in binary mode
    result_binary = kt3d(..., binary=True)

    # Results should match within float32 precision (~1e-5)
    assert result_binary.estimate.shape == result_ascii.estimate.shape

    est_diff = np.abs(result_ascii.estimate - result_binary.estimate)
    assert est_diff.max() < 1e-4, f"Max estimate diff: {est_diff.max()}"

    var_diff = np.abs(result_ascii.variance - result_binary.variance)
    assert var_diff.max() < 1e-4, f"Max variance diff: {var_diff.max()}"
```

### Expected Precision

- **Max difference**: ~1e-5 (float32 precision limit)
- **Cause**: ASCII formatting loses precision, binary preserves exact float32

---

## Checklist for New Programs

When adding binary I/O to a new GSLIB program:

### Fortran Side
- [ ] Add `ibinary` to the program's common block (`.inc` file)
- [ ] Add `read(lin,*) ibinary` after output filename in `readparm`
- [ ] Add conditional file open (stream/unformatted for binary)
- [ ] Write binary header with correct dimensions
- [ ] Modify main loop to write binary or ASCII
- [ ] Update `makepar` to include ibinary parameter

### Python Side
- [ ] Add `binary: bool = False` parameter to wrapper function
- [ ] Always include ibinary in par file generation
- [ ] Add conditional output reading (BinaryIO vs ASCII parsing)
- [ ] Convert float32 to float64 when reading binary

### Testing
- [ ] Add `test_<program>_binary_matches_ascii` test
- [ ] Verify shapes match
- [ ] Verify values match within 1e-4

### Documentation
- [ ] Update ROADMAP.md status table
- [ ] Add any program-specific notes to this guide

---

## Program-Specific Notes

### kt3d
- Output: estimate + variance per cell
- Header: `[4, 2, nz, ny, nx]`
- Data interleaved: `[est1, var1, est2, var2, ...]`

### sgsim
- Output: one value per cell per realization
- Header: `[4, nsim, nz, ny, nx]`
- Data: all cells for sim 1, then all cells for sim 2, etc.
- **Important**: Data is written sequentially (sim1 cells, sim2 cells, ...), not interleaved.
  Python must read flat data and reshape each realization separately with `order="F"`.

### sisim
- Output: one simulated value per cell per realization
- Header: `[4, nsim, nz, ny, nx]`
- Data: all cells for sim 1, then all cells for sim 2, etc.
- Same structure as sgsim - sequential realizations, not interleaved.

### ik3d
- Output: probability per cutoff per cell (order-relations corrected CDF/PDF)
- Header: `[4, ncut, nz, ny, nx]`
- Data: All ncut probabilities for cell 1, then cell 2, etc. (interleaved like kt3d)

### gamv
- Output: variogram statistics per lag per direction per variogram type
- Header: `[4, 5, nvarg, ndir, nlag+2]`
  - ndim=4, nfields=5 (dis, gam, np, hm, tm), nvarg variograms, ndir directions, nlag+2 lags
- Data: For each variogram, for each direction, for each lag: 5 float32 values
- **Note**: gamv has different output structure than grid-based programs
- **Note**: nlag+2 lags are written (lag 0 through nlag+1), Python typically uses lags 1..nlag

---

## Troubleshooting

### "Cannot reshape array" error
- Check that dtype matches what Fortran wrote (usually float32)
- Verify header values match expected dimensions

### Results don't match between binary/ASCII
- Check data ordering (Fortran is column-major)
- Verify both modes use same random seed (for simulation)

### File size wrong
- Header: `(1 + ndim) * 4` bytes
- Data: `product(shape) * sizeof(dtype)` bytes
- float32 = 4 bytes, float64 = 8 bytes
