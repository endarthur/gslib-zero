@echo off
REM Build script for GSLIB-Zero Fortran programs on Windows
REM
REM Prerequisites:
REM   conda install -c conda-forge m2w64-gcc-fortran
REM
REM Usage:
REM   build.bat           - Build all programs
REM   build.bat clean     - Remove build artifacts
REM   build.bat kt3d      - Build specific program

setlocal enabledelayedexpansion

set FC=gfortran
set FFLAGS=-O2 -std=legacy -w
set SRCDIR=src
set GSLIBDIR=src\gslib
set BINDIR=..\bin

REM Create bin directory if it doesn't exist
if not exist "%BINDIR%" mkdir "%BINDIR%"

REM Handle clean
if "%1"=="clean" (
    echo Cleaning build artifacts...
    del /Q %SRCDIR%\*.o 2>nul
    del /Q %SRCDIR%\*.mod 2>nul
    del /Q %GSLIBDIR%\*.o 2>nul
    del /Q %BINDIR%\*.exe 2>nul
    echo Done.
    goto :eof
)

REM Compile GSLIB common subroutines
echo Compiling GSLIB common subroutines...
for %%f in (%GSLIBDIR%\*.for) do (
    echo   Compiling %%~nf.for
    %FC% %FFLAGS% -c "%%f" -o "%GSLIBDIR%\%%~nf.o"
    if errorlevel 1 goto :error
)

REM List of programs to build
set PROGRAMS=nscore backtr declus gamv kt3d ik3d sgsim sisim

REM Build specific program or all
if not "%1"=="" (
    set PROGRAMS=%1
)

REM Compile and link each program
for %%p in (%PROGRAMS%) do (
    echo Building %%p...

    REM Compile main program
    %FC% %FFLAGS% -c "%SRCDIR%\%%p.for" -o "%SRCDIR%\%%p.o"
    if errorlevel 1 goto :error

    REM Link with GSLIB subroutines
    %FC% %FFLAGS% -o "%BINDIR%\%%p.exe" "%SRCDIR%\%%p.o" %GSLIBDIR%\*.o
    if errorlevel 1 goto :error

    echo   Created %BINDIR%\%%p.exe
)

echo.
echo Build completed successfully!
goto :eof

:error
echo.
echo BUILD FAILED!
exit /b 1
