#!/bin/bash
#
# Build script for GSLIB-Zero Fortran programs on Linux/macOS
#
# Prerequisites:
#   - gfortran (apt install gfortran, brew install gcc, or conda install -c conda-forge gfortran)
#
# Usage:
#   ./build.sh           - Build all programs
#   ./build.sh clean     - Remove build artifacts
#   ./build.sh kt3d      - Build specific program

set -e

FC=gfortran
FFLAGS="-O2 -fallow-argument-mismatch -std=legacy"
SRCDIR=src
GSLIBDIR=src/gslib
BINDIR=../bin

# Create bin directory if it doesn't exist
mkdir -p "$BINDIR"

# Handle clean
if [ "$1" = "clean" ]; then
    echo "Cleaning build artifacts..."
    rm -f $SRCDIR/*.o $SRCDIR/*.mod $GSLIBDIR/*.o
    rm -f $BINDIR/nscore $BINDIR/backtr $BINDIR/declus $BINDIR/gamv
    rm -f $BINDIR/kt3d $BINDIR/ik3d $BINDIR/sgsim $BINDIR/sisim
    echo "Done."
    exit 0
fi

# Compile GSLIB common subroutines
echo "Compiling GSLIB common subroutines..."
for f in $GSLIBDIR/*.for; do
    name=$(basename "$f" .for)
    echo "  Compiling $name.for"
    $FC $FFLAGS -c "$f" -o "$GSLIBDIR/$name.o"
done

# List of programs to build
PROGRAMS="nscore backtr declus gamv kt3d ik3d sgsim sisim"

# Build specific program or all
if [ -n "$1" ]; then
    PROGRAMS="$1"
fi

# Compile and link each program
for prog in $PROGRAMS; do
    echo "Building $prog..."

    # Compile main program
    $FC $FFLAGS -c "$SRCDIR/$prog.for" -o "$SRCDIR/$prog.o"

    # Link with GSLIB subroutines
    $FC $FFLAGS -o "$BINDIR/$prog" "$SRCDIR/$prog.o" $GSLIBDIR/*.o

    echo "  Created $BINDIR/$prog"
done

echo ""
echo "Build completed successfully!"
