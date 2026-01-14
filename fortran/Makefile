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

# Windows executable extension and static linking (must be before targets)
ifeq ($(OS),Windows_NT)
    EXE = .exe
    LDFLAGS = -static
else
    EXE =
    LDFLAGS =
endif

FC = gfortran
FFLAGS = -O2 -std=legacy -w
FFLAGS_F64 = -O2 -std=legacy -w -freal-4-real-8
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

.PHONY: all clean binary original f64 $(PROGRAMS)

# Default: build original ASCII versions
all: original

original: $(PROGRAMS)

# Double precision (float64) builds
PROGRAMS_F64 = $(addsuffix _f64,$(PROGRAMS))
f64: $(PROGRAMS_F64)

# F64 program targets
nscore_f64: $(BINDIR)/nscore_f64$(EXE)
backtr_f64: $(BINDIR)/backtr_f64$(EXE)
declus_f64: $(BINDIR)/declus_f64$(EXE)
gamv_f64: $(BINDIR)/gamv_f64$(EXE)
kt3d_f64: $(BINDIR)/kt3d_f64$(EXE)
ik3d_f64: $(BINDIR)/ik3d_f64$(EXE)
sgsim_f64: $(BINDIR)/sgsim_f64$(EXE)
sisim_f64: $(BINDIR)/sisim_f64$(EXE)

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

# Link programs (LDFLAGS includes -static on Windows for standalone binaries)
$(BINDIR)/nscore$(EXE): $(SRCDIR)/nscore.o $(GSLIB_OBJS) | $(BINDIR)
	$(FC) $(FFLAGS) $(LDFLAGS) -o $@ $^

$(BINDIR)/backtr$(EXE): $(SRCDIR)/backtr.o $(GSLIB_OBJS) | $(BINDIR)
	$(FC) $(FFLAGS) $(LDFLAGS) -o $@ $^

$(BINDIR)/declus$(EXE): $(SRCDIR)/declus.o $(GSLIB_OBJS) | $(BINDIR)
	$(FC) $(FFLAGS) $(LDFLAGS) -o $@ $^

$(BINDIR)/gamv$(EXE): $(SRCDIR)/gamv.o $(GSLIB_OBJS) | $(BINDIR)
	$(FC) $(FFLAGS) $(LDFLAGS) -o $@ $^

$(BINDIR)/kt3d$(EXE): $(SRCDIR)/kt3d.o $(GSLIB_OBJS) | $(BINDIR)
	$(FC) $(FFLAGS) $(LDFLAGS) -o $@ $^

$(BINDIR)/ik3d$(EXE): $(SRCDIR)/ik3d.o $(GSLIB_OBJS) | $(BINDIR)
	$(FC) $(FFLAGS) $(LDFLAGS) -o $@ $^

$(BINDIR)/sgsim$(EXE): $(SRCDIR)/sgsim.o $(GSLIB_OBJS) | $(BINDIR)
	$(FC) $(FFLAGS) $(LDFLAGS) -o $@ $^

$(BINDIR)/sisim$(EXE): $(SRCDIR)/sisim.o $(GSLIB_OBJS) | $(BINDIR)
	$(FC) $(FFLAGS) $(LDFLAGS) -o $@ $^

# F64 builds - compile and link in one step to avoid object file conflicts
$(BINDIR)/nscore_f64$(EXE): $(SRCDIR)/nscore.for $(GSLIB_SRCS) | $(BINDIR)
	$(FC) $(FFLAGS_F64) $(LDFLAGS) -o $@ $^

$(BINDIR)/backtr_f64$(EXE): $(SRCDIR)/backtr.for $(GSLIB_SRCS) | $(BINDIR)
	$(FC) $(FFLAGS_F64) $(LDFLAGS) -o $@ $^

$(BINDIR)/declus_f64$(EXE): $(SRCDIR)/declus.for $(GSLIB_SRCS) | $(BINDIR)
	$(FC) $(FFLAGS_F64) $(LDFLAGS) -o $@ $^

$(BINDIR)/gamv_f64$(EXE): $(SRCDIR)/gamv.for $(GSLIB_SRCS) | $(BINDIR)
	$(FC) $(FFLAGS_F64) $(LDFLAGS) -o $@ $^

$(BINDIR)/kt3d_f64$(EXE): $(SRCDIR)/kt3d.for $(GSLIB_SRCS) | $(BINDIR)
	$(FC) $(FFLAGS_F64) $(LDFLAGS) -o $@ $^

$(BINDIR)/ik3d_f64$(EXE): $(SRCDIR)/ik3d.for $(GSLIB_SRCS) | $(BINDIR)
	$(FC) $(FFLAGS_F64) $(LDFLAGS) -o $@ $^

$(BINDIR)/sgsim_f64$(EXE): $(SRCDIR)/sgsim.for $(GSLIB_SRCS) | $(BINDIR)
	$(FC) $(FFLAGS_F64) $(LDFLAGS) -o $@ $^

$(BINDIR)/sisim_f64$(EXE): $(SRCDIR)/sisim.for $(GSLIB_SRCS) | $(BINDIR)
	$(FC) $(FFLAGS_F64) $(LDFLAGS) -o $@ $^

# Clean build artifacts
clean:
	rm -f $(SRCDIR)/*.o $(SRCDIR)/*.mod $(GSLIBDIR)/*.o
	rm -f $(BINDIR)/nscore$(EXE) $(BINDIR)/backtr$(EXE)
	rm -f $(BINDIR)/declus$(EXE) $(BINDIR)/gamv$(EXE)
	rm -f $(BINDIR)/kt3d$(EXE) $(BINDIR)/ik3d$(EXE)
	rm -f $(BINDIR)/sgsim$(EXE) $(BINDIR)/sisim$(EXE)
	rm -f $(BINDIR)/nscore_f64$(EXE) $(BINDIR)/backtr_f64$(EXE)
	rm -f $(BINDIR)/declus_f64$(EXE) $(BINDIR)/gamv_f64$(EXE)
	rm -f $(BINDIR)/kt3d_f64$(EXE) $(BINDIR)/ik3d_f64$(EXE)
	rm -f $(BINDIR)/sgsim_f64$(EXE) $(BINDIR)/sisim_f64$(EXE)
