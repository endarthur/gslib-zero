# Makefile for GSLIB-Zero Fortran programs
#
# Usage:
#   make all        - Build all programs
#   make kt3d       - Build specific program
#   make clean      - Remove build artifacts
#   make binary     - Build binary I/O versions (gslib-zero modified)
#
# Requirements:
#   - gfortran (install via: conda install -c conda-forge m2w64-gcc-fortran)

FC = gfortran
FFLAGS = -O2 -std=legacy -w
FFLAGS_F90 = -O2

# Directories
SRCDIR = src
GSLIBDIR = src/gslib
BINDIR = ../bin

# Common GSLIB subroutines
GSLIB_SRCS = $(wildcard $(GSLIBDIR)/*.for)
GSLIB_OBJS = $(GSLIB_SRCS:.for=.o)

# Main programs we need for gslib-zero
PROGRAMS = nscore backtr declus gamv kt3d ik3d sgsim sisim

# Binary I/O module
BINARY_IO_MOD = $(SRCDIR)/gslib_binary_io.f90
BINARY_IO_OBJ = $(SRCDIR)/gslib_binary_io.o

.PHONY: all clean binary original $(PROGRAMS)

# Default: build original ASCII versions
all: original

original: $(PROGRAMS)

# Build individual programs
nscore: $(BINDIR)/nscore$(EXE)
backtr: $(BINDIR)/backtr$(EXE)
declus: $(BINDIR)/declus$(EXE)
gamv: $(BINDIR)/gamv$(EXE)
kt3d: $(BINDIR)/kt3d$(EXE)
ik3d: $(BINDIR)/ik3d$(EXE)
sgsim: $(BINDIR)/sgsim$(EXE)
sisim: $(BINDIR)/sisim$(EXE)

# Ensure bin directory exists
$(BINDIR):
	mkdir -p $(BINDIR)

# Compile GSLIB common subroutines
$(GSLIBDIR)/%.o: $(GSLIBDIR)/%.for
	$(FC) $(FFLAGS) -c $< -o $@

# Compile main programs
$(SRCDIR)/%.o: $(SRCDIR)/%.for
	$(FC) $(FFLAGS) -c $< -o $@

# Compile F90 modules
$(SRCDIR)/%.o: $(SRCDIR)/%.f90
	$(FC) $(FFLAGS_F90) -c $< -o $@

# Link programs
$(BINDIR)/nscore$(EXE): $(SRCDIR)/nscore.o $(GSLIB_OBJS) | $(BINDIR)
	$(FC) $(FFLAGS) -o $@ $^

$(BINDIR)/backtr$(EXE): $(SRCDIR)/backtr.o $(GSLIB_OBJS) | $(BINDIR)
	$(FC) $(FFLAGS) -o $@ $^

$(BINDIR)/declus$(EXE): $(SRCDIR)/declus.o $(GSLIB_OBJS) | $(BINDIR)
	$(FC) $(FFLAGS) -o $@ $^

$(BINDIR)/gamv$(EXE): $(SRCDIR)/gamv.o $(GSLIB_OBJS) | $(BINDIR)
	$(FC) $(FFLAGS) -o $@ $^

$(BINDIR)/kt3d$(EXE): $(SRCDIR)/kt3d.o $(GSLIB_OBJS) | $(BINDIR)
	$(FC) $(FFLAGS) -o $@ $^

$(BINDIR)/ik3d$(EXE): $(SRCDIR)/ik3d.o $(GSLIB_OBJS) | $(BINDIR)
	$(FC) $(FFLAGS) -o $@ $^

$(BINDIR)/sgsim$(EXE): $(SRCDIR)/sgsim.o $(GSLIB_OBJS) | $(BINDIR)
	$(FC) $(FFLAGS) -o $@ $^

$(BINDIR)/sisim$(EXE): $(SRCDIR)/sisim.o $(GSLIB_OBJS) | $(BINDIR)
	$(FC) $(FFLAGS) -o $@ $^

# Clean build artifacts
clean:
	rm -f $(SRCDIR)/*.o $(SRCDIR)/*.mod $(GSLIBDIR)/*.o
	rm -f $(BINDIR)/nscore$(EXE) $(BINDIR)/backtr$(EXE)
	rm -f $(BINDIR)/declus$(EXE) $(BINDIR)/gamv$(EXE)
	rm -f $(BINDIR)/kt3d$(EXE) $(BINDIR)/ik3d$(EXE)
	rm -f $(BINDIR)/sgsim$(EXE) $(BINDIR)/sisim$(EXE)

# Windows executable extension
ifeq ($(OS),Windows_NT)
    EXE = .exe
else
    EXE =
endif
