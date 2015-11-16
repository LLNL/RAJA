# Makefile for src-openmp

SHELL = /bin/bash

OPT_DEBUG = opt
#OPT_DEBUG = debug

#rzalastor
RAJA_ARCH = x86_sse_icc
#RAJA_ARCH = x86_sse_gnu

#rzmerl
#RAJA_ARCH = x86_avx_icc
#RAJA_ARCH = x86_avx_gnu

#rzuseq
#RAJA_ARCH = bgq_xlc12
#RAJA_ARCH = bgq_clang
#RAJA_ARCH = bgq_gnu

#rzmic
#RAJA_ARCH = MIC

#rzmist
#RAJA_ARCH = rzmist_clang
#RAJA_ARCH = rzmist_xlc

RAJA_TOPDIR    = ../../..
RAJA_INC       = $(RAJA_TOPDIR)/include/
RAJA_SRC       = $(RAJA_TOPDIR)/src/

##
## The RAJA_rules.mk file defines macro variables that specify RAJA behavior.
## To change the rules, the file in the RAJA include directory can be edited
## or it can be replaced with custom version here.
##
## Note that all files should be compiled using these definitions, which are
## included in the "RAJA_RULES" variable.  If files are not consistently
## compiled, an interesting debugging exercise could result.
##
include $(RAJA_TOPDIR)/build/RAJA_rules.mk

##
## Include file containing compiler version and options triggered by
## "RAJA_ARCH" setting above.
##
include ../../compilers.mk


### CoMD can be built in either double or single precision and with or
### without MPI.  Select desired precision and MPI here.

# double precision (ON/OFF)
DOUBLE_PRECISION = ON
# MPI for parallel (ON/OFF)
##DO_MPI = ON
DO_MPI = OFF

###
### Comment from original makefile. Not sure what still applies...
###
### Set your desired C compiler and any necessary flags.  Note that CoMD
### uses some c99 features.  You can also set flags for optimization and
### specify paths to include files that the compiler can't find on its
### own.  If you need any -L or -l switches to get C standard libraries
### (such as -lm for the math library) put them in C_LIB.
###

##
## Intel build on TLCC2
##
#CC = mpiicpc
#CFLAGS =  -align -inline-max-total-size=20000 -inline-forceinline -ansi-alias -std=c++0x -openmp -static-intel  -DRAJA_PLATFORM_X86_SSE -DRAJA_COMPILER_GNU -DRAJA_USE_DOUBLE -DRAJA_USE_RESTRICT_PTR
#OPTFLAGS = -g -O3 -mavx
#OPTFLAGS = -g -O5
#OPTFLAGS = -g -O0

INCLUDES = -I${RAJA_INC}
C_LIB = -lm


### If you need to specify include paths, library paths, or link flags
### for MPI, put them here.  Put both -L and -l switches into MPI_LIB.
MPI_LIB =
MPI_INCLUDE =

### A place to specify any other include or library switches your
### platform requires.
OTHER_LIB =
OTHER_INCLUDE =




#########################################
### Below here, it is pitch black.  
### You are likely to be eaten by a grue.
##########################################

# clear all suffixes
.SUFFIXES:
# list only those that we use 
.SUFFIXES: .c .o

.PHONY: DEFAULT clean distclean depend

BIN_DIR=../bin

# Check for double precision
ifeq ($(DOUBLE_PRECISION), ON)
CFLAGS += -DDOUBLE
else
CFLAGS += -DSINGLE
endif

# Set executable name and add includes & libraries for MPI if needed.
CoMD_VARIANT = CoMD-local
ifeq ($(DO_MPI), ON)
CoMD_VARIANT = CoMD-openmp-mpi
INCLUDES += ${MPI_INCLUDE}
CFLAGS += -DDO_MPI
LDFLAGS += ${MPI_LIB}
endif
CoMD_EXE = ${BIN_DIR}/${CoMD_VARIANT}

LDFLAGS += ${C_LIB} ${OTHER_LIB}
CFLAGS  += ${INCLUDES} ${OTHER_INCLUDE}


COMD_SOURCES=$(wildcard *.c)
COMD_OBJECTS=$(COMD_SOURCES:.c=.o)

RAJAOBJS := $(patsubst %.cxx,%.o,$(wildcard ${RAJA_SRC}/*.cxx))

CXXFLAGS_BUILD = -I. -I${RAJA_INC} ${CFLAGS} ${CXXFLAGS} ${RAJA_RULES}

DEFAULT: ${CoMD_EXE}

##
## Rule to build *.cxx RAJA files
%.o : %.cxx ; ${CXX_COMPILE} -c -o $@ $< ${CXXFLAGS_BUILD}

##
## Rule to build *.c CoMD files
%.o : %.c ; ${CXX_COMPILE} -c -o $@ $< ${CXXFLAGS_BUILD}

${CoMD_EXE}: ${BIN_DIR} CoMD_info.h ${COMD_OBJECTS} ${RAJAOBJS}
	${CXX_COMPILE} ${CFLAGS} -o ${CoMD_EXE} ${COMD_OBJECTS} ${RAJAOBJS} ${LDFLAGS}

CoMD_info.h: Makefile
	./generate_info_header ${CoMD_VARIANT} "$(CXX)" "$(CXX_COMPILE)" "$(LDFLAGS)"

${BIN_DIR}:
	@if [ ! -d ${BIN_DIR} ]; then mkdir -p ${BIN_DIR} ; fi

clean:
	rm -f *.o CoMD_info.h .depend
	rm -f ${RAJA_SRC}/*.o 

distclean: clean
	rm -f ${CoMD_EXE} .depend.bak
	rm -rf html latex

.depend: $(COMD_SOURCES)
	@touch .depend
	@$(MAKE) --no-print-directory depend

depend:
	@echo "Rebuilding dependencies..."
	@$(MAKE) CoMD_info.h
	@makedepend -f .depend -Y. --$(CFLAGS)-- $(COMD_SOURCES) 2> /dev/null


-include .depend
