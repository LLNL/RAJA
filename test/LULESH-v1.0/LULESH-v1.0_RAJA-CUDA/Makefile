#
#  This is cheesy, but it's simple and easy enough to muck with for testing...
#

#############################################
#
# To build and run on IPA:
#   %module load cudatoolkit/7.5
#   %use gcc-4.9.0p
#   %make
#   %salloc -p nvidia -N1
#   %setenv CUDA_VISIBLE_DEVICES 0
#   %<run code here>
#
#############################################


##
## Set option to report basic runtime information.
##
##LULESH_OPTS = -DRAJA_USE_CYCLE
#LULESH_OPTS = -DRAJA_USE_CLOCK
LULESH_OPTS = -DRAJA_USE_GETTIME

# This is needed for RAJA_USE_GETTIME
LDTIMER = -lrt


RAJA_TOPDIR    = ../../..
RAJA_INC       = $(RAJA_TOPDIR)/includes/
RAJA_SRC       = $(RAJA_TOPDIR)/sources/

##
## The RAJA_rules.mk file defines macro variables that specify RAJA behavior.
## To change the rules, the file in the RAJA include directory can be edited
## or it can be replaced with custom version here.
##
include $(RAJA_TOPDIR)/build/RAJA_rules.mk

##
## Include file containing compiler version and options.
##
#include ../../compilers.mk

CXXFLAGS        = -DRAJA_PLATFORM_X86_SSE -DRAJA_COMPILER_GNU -DRAJA_USE_CUDA

#
# This is needed, or else the code dies with a bus error. Why???
#
LULESH_OPTS += -DRAJA_SYNC



CXX_COMPILE = nvcc -O2 --expt-extended-lambda -arch compute_35 -std=c++11 -Xcompiler -fopenmp -v -keep
#CXX_COMPILE = nvcc -O2 --extended-lambda -arch compute_35 -std=c++11 -v -keep
#CXX_COMPILE = g++ -g -fpermissive -fopenmp -O0 -std=c++11 -fopenmp

CXXFLAGS_BUILD = -I. -I../../includes -I$(RAJA_INC) $(CXXFLAGS) $(RAJA_RULES) $(LULESH_OPTS)


RAJAOBJS := $(patsubst %.cxx,%.o,$(wildcard $(RAJA_SRC)/*.cxx))

TESTOBJS = luleshRAJA.o $(RAJAOBJS)

lulesh: $(TESTOBJS) 
	$(CXX_COMPILE) $(TESTOBJS) $(LDFLAGS) $(LDTIMER) $(LDPATH) -o  lulesh-RAJA-CUDA.exe

%.o : %.cu ; $(CXX_COMPILE) -c -o $@ $< $(CXXFLAGS_BUILD)
%.o : %.cxx ; $(CXX_COMPILE) -c -o $@ $< $(CXXFLAGS_BUILD)

#%.o : %.cxx ; $(CXX_COMPILE) -S -o $@ $< 
## The following assembly options are for Intel compiler...
#%.o : %.cxx ; $(CXX_COMPILE) -S -fcode-asm -vec-report3 -o $@ $< $(CXXFLAGS)

clean-obj: 
	rm -rf *.o $(RAJA_SRC)/*.o

clean-all: clean-obj 
	rm -rf *.cpp*.i* *.cuda*.* *.fatbin* *.hash* *.module_id* *.ptx *.reg.c *.exe


