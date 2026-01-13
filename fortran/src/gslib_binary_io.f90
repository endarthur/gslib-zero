!-----------------------------------------------------------------------
!
!                    Binary I/O Module for GSLIB-Zero
!                    ********************************
!
! This module provides binary file I/O for GSLIB programs, enabling
! 10-100x speedup over ASCII I/O for large datasets.
!
! Binary format:
!   - Header: ndim (int32), then shape (ndim x int32)
!   - Data: float64 array in column-major (Fortran) order
!
! This format matches gslib_zero.core.BinaryIO in Python.
!
!-----------------------------------------------------------------------
module gslib_binary_io

    implicit none

    private
    public :: write_binary_array_1d, write_binary_array_3d
    public :: read_binary_array_1d, read_binary_header
    public :: write_binary_grid_output

contains

    !-------------------------------------------------------------------
    ! Write a 1D array to binary file
    !-------------------------------------------------------------------
    subroutine write_binary_array_1d(lun, filename, array, n)
        integer, intent(in) :: lun
        character(len=*), intent(in) :: filename
        real*8, intent(in) :: array(*)
        integer, intent(in) :: n

        integer :: ndim

        ndim = 1

        open(lun, file=filename, status='UNKNOWN', access='STREAM', &
             form='UNFORMATTED')

        ! Write header
        write(lun) ndim
        write(lun) n

        ! Write data
        write(lun) array(1:n)

        close(lun)

    end subroutine write_binary_array_1d

    !-------------------------------------------------------------------
    ! Write a 3D array to binary file (for gridded output)
    !-------------------------------------------------------------------
    subroutine write_binary_array_3d(lun, filename, array, nx, ny, nz)
        integer, intent(in) :: lun
        character(len=*), intent(in) :: filename
        real*8, intent(in) :: array(*)
        integer, intent(in) :: nx, ny, nz

        integer :: ndim, ntot

        ndim = 3
        ntot = nx * ny * nz

        open(lun, file=filename, status='UNKNOWN', access='STREAM', &
             form='UNFORMATTED')

        ! Write header
        write(lun) ndim
        write(lun) nx, ny, nz

        ! Write data (already in Fortran column-major order)
        write(lun) array(1:ntot)

        close(lun)

    end subroutine write_binary_array_3d

    !-------------------------------------------------------------------
    ! Write gridded output with multiple variables (e.g., estimate + variance)
    ! Format: 4D array (nvars, nx, ny, nz)
    !-------------------------------------------------------------------
    subroutine write_binary_grid_output(lun, filename, arrays, nvars, nx, ny, nz)
        integer, intent(in) :: lun
        character(len=*), intent(in) :: filename
        real*8, intent(in) :: arrays(*)  ! Interleaved: var1[0], var2[0], var1[1], ...
        integer, intent(in) :: nvars, nx, ny, nz

        integer :: ndim, ntot

        ndim = 4
        ntot = nvars * nx * ny * nz

        open(lun, file=filename, status='UNKNOWN', access='STREAM', &
             form='UNFORMATTED')

        ! Write header
        write(lun) ndim
        write(lun) nvars, nx, ny, nz

        ! Write data
        write(lun) arrays(1:ntot)

        close(lun)

    end subroutine write_binary_grid_output

    !-------------------------------------------------------------------
    ! Read binary header to get dimensions
    !-------------------------------------------------------------------
    subroutine read_binary_header(lun, filename, ndim, dims)
        integer, intent(in) :: lun
        character(len=*), intent(in) :: filename
        integer, intent(out) :: ndim
        integer, intent(out) :: dims(*)  ! Caller must allocate enough space

        integer :: i

        open(lun, file=filename, status='OLD', access='STREAM', &
             form='UNFORMATTED')

        read(lun) ndim
        read(lun) (dims(i), i=1, ndim)

        ! Leave file open for subsequent read_binary_array_1d call

    end subroutine read_binary_header

    !-------------------------------------------------------------------
    ! Read a 1D array from binary file (after header already read)
    ! File must already be open from read_binary_header
    !-------------------------------------------------------------------
    subroutine read_binary_array_1d(lun, array, n)
        integer, intent(in) :: lun
        real*8, intent(out) :: array(*)
        integer, intent(in) :: n

        read(lun) array(1:n)
        close(lun)

    end subroutine read_binary_array_1d

end module gslib_binary_io
