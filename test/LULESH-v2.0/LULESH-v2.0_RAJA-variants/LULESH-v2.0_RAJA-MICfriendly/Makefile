#  
# THIS IS NOT OPEN SOURCE OR PUBLIC DOMAIN SOFTWARE
#
# See README-RAJA_license.txt for access and distribution restrictions
#

#
#  Modify stuff in this Makefile to fit your machine and compiler.
#
#  Uncomment ONE AND ONLY ONE opt/debug option and "RAJA_ARCH" 
#  variable line for the platform you want. 
#
# IMPORTANT:  The specific compiler version and options are set in
#             the file ../compilers.mk.
#

OPT_DEBUG = opt
#OPT_DEBUG = debug

##
## Set option to report basic runtime information.
##
##LULESH_TIMER_OPTS = -DRAJA_USE_CYCLE
#LULESH_TIMER_OPTS = -DRAJA_USE_CLOCK
LULESH_TIMER_OPTS = -DRAJA_USE_GETTIME

# This is needed for RAJA_USE_GETTIME
LDTIMER = -lrt


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

#ipa-cuda
#RAJA_ARCH = ipa_cuda

#rzmist
#RAJA_ARCH = rzmist_clang
#RAJA_ARCH = rzmist_xlc

RAJA_TOPDIR    = ../../../..
RAJA_INC       = $(RAJA_TOPDIR)/include/
RAJA_SRC       = $(RAJA_TOPDIR)/src/


##
## The RAJA_rules.mk file defines macro variables that specify RAJA behavior.
## To change the rules, the file in the RAJA include directory can be edited
## or it can be replaced with custom version here.
##
include $(RAJA_TOPDIR)/build/RAJA_rules.mk


##
## Options to turn on/off MPI and OpenMP
##
LULESH_OPTS = -DLULESH_HEADER=2 -DUSE_MPI=0 -DUSE_OMP=1 $(LULESH_TIMER_OPTS)


##
## Include file containing compiler version and options.
##
include ../../../compilers.mk


RAJAOBJS := $(patsubst %.cxx,%.o,$(wildcard $(RAJA_SRC)/*.cxx))


##-----------------------------------------------------------
## This section is a complete hack....
##-----------------------------------------------------------

##
## If compiling with nvcc on ipa (see RAJA_ARCH defn above)
##
ifeq ($(RAJA_ARCH),ipa_cuda)

SILO_INCDIR =
SILO_LIBDIR =

CXX = nvcc

### Testing non-GPU build with nvcc...
#PLATFORM        = -DRAJA_PLATFORM_X86_SSE -DRAJA_COMPILER_GNU
PLATFORM        = -DRAJA_PLATFORM_X86_SSE -DRAJA_COMPILER_GNU -DRAJA_USE_CUDA

CXXFLAGS = -O2 --expt-extended-lambda -restrict -arch compute_35 -std=c++11 -Xcompiler -fopenmp
# to debug
##CXXFLAGS = -O2 --expt-extended-lambda -restrict -arch compute_35 -std=c++11 -Xcompiler -fopenmp -v -keep

LDFLAGS = -Xcompiler -fopenmp

LULESHOBJS_CC := $(patsubst %.cc,%.o,$(wildcard ./*.cc))
LULESHOBJS_CU := $(patsubst %.cu,%.o,$(wildcard ./*.cu))

LULESHOBJS = $(LULESHOBJS_CC) $(LULESHOBJS_CU)

LDFLAGS_FINAL = $(LDFLAGS) $(LDTIMER) $(LDPATH)

CXXFLAGS_BUILD = -I. -I../../../include -I$(RAJA_INC) $(PLATFORM) $(RAJA_RULES) $(LULESH_OPTS)

else
##
## Pull in CXX_COMPILE and CXXFLAGS from ../../compilers.mk
## (see RAJA_ARCH defn above)
##

#LULESH_OPTS += -DVIZ_MESH
SILO_INCDIR = /usr/gapps/silo/current/chaos_5_x86_64_ib/include
#SILO_LIBDIR = /usr/gapps/silo/current/chaos_5_x86_64_ib/lib

LULESHOBJS := $(patsubst %.cc,%.o,$(wildcard ./*.cc))

# LDFLAGS_FINAL = $(LDPATH) $(LDTIMER) $(LDPATH) ${SILO_LIBDIR}/libsiloh5.so
LDFLAGS_FINAL = $(LDPATH) $(LDTIMER) $(LDPATH) 

CXXFLAGS_BUILD = -I. -I../../../include -I$(RAJA_INC) -I$(SILO_INCDIR) $(PLATFORM) $(RAJA_RULES) $(LULESH_OPTS)

endif

##-----------------------------------------------------------

OBJFILES = $(LULESHOBJS) $(RAJAOBJS)

default: $(OBJFILES)
	$(CXX) $(LDFLAGS) $(OBJFILES) $(LDFLAGS_FINAL) -o lulesh2.0_RAJA-ISet.exe

## Uncomment these lines to generate an executable to run...
%.o : %.cu ; $(CXX) $(CXXFLAGS) -c -o $@ $< $(CXXFLAGS_BUILD)
%.o : %.cxx ; $(CXX) $(CXXFLAGS) -c -o $@ $< $(CXXFLAGS_BUILD) 
%.o : %.cc ; $(CXX) $(CXXFLAGS) -c -o $@ $< $(CXXFLAGS_BUILD) 

## The following can be used to generate vectorization report and 
## assmbly code output for Intel compilers...
#%.o : %.cxx ; $(CXX) $(CXXFLAGS) -S -o $@ $< $(CXXFLAGS_BUILD)
#%.o : %.cxx ; $(CXX) $(CXXFLAGS) -S -vec-report3 -o $@ $< $(CXXFLAGS_BUILD)
#%.o : %.cxx ; $(CXX) $(CXXFLAGS) -S -fcode-asm -vec-report3 -o $@ $< $(CXXFLAGS_BUILD)

## The following can be used to generate vectorization report and 
## assmbly code output for IBM XLC compilers...
## See compiler flag options in the appropriate section above.
#%.o : %.cxx ; $(CXX) $(CXXFLAGS) -c -o $@ $< $(CXXFLAGS_BUILD)


clean-obj: 
	rm -rf *.o $(RAJA_SRC)/*.o

clean: clean-obj
	rm -rf *.cpp*.i* *.cuda*.* *.fatbin* *.hash* *.module_id* *.ptx *.reg.c *.s *.lst *.exe
