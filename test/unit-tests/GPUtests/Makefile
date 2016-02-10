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
##TEST_OPTS = -DRAJA_USE_CYCLE
#TEST_OPTS = -DRAJA_USE_CLOCK
TEST_OPTS = -DRAJA_USE_GETTIME

# This is needed for RAJA_USE_GETTIME
LDTIMER = -lrt


RAJA_TOPDIR    = ../../..
RAJA_INC       = $(RAJA_TOPDIR)/include/
RAJA_SRC       = $(RAJA_TOPDIR)/src/

##
## The RAJA_rules.mk file defines macro variables that specify RAJA behavior.
## To change the rules, the file in the RAJA include directory can be edited
## or it can be replaced with custom version here.
##
###include $(RAJA_TOPDIR)/build/RAJA_rules.mk

##
## Include file containing compiler version and options.
##
#include ../../../compilers.mk

PLATFORM        = -DRAJA_PLATFORM_X86_SSE -DRAJA_COMPILER_GNU -DRAJA_USE_CUDA -DRAJA_USE_DOUBLE -DRAJA_USE_BARE_PTR

CXX = nvcc

#CXXFLAGS = -O2 --expt-extended-lambda -arch compute_35 -std=c++11 -Xcompiler -fopenmp --x cu -v -keep
CXXFLAGS = -O2 --expt-extended-lambda -arch compute_35 -std=c++11 -Xcompiler -fopenmp --x cu
#CXXFLAGS = -g -G -O0 --expt-extended-lambda -arch compute_35 -std=c++11 -Xcompiler -fopenmp --x cu
#CXXFLAGS = -O2 --extended-lambda -arch compute_35 -std=c++11 --x cu -v -keep
#CXXFLAGS = g++ -g -fpermissive -fopenmp -O0 -std=c++11 -fopenmp

CXXFLAGS_BUILD = -I. -I../../include -I$(RAJA_INC) $(PLATFORM) $(TEST_OPTS)

LDFLAGS = -Xcompiler -fopenmp

RAJAOBJS := $(patsubst %.cxx,%.o,$(wildcard $(RAJA_SRC)/*.cxx))

reducemin: ReduceMin.o $(RAJAOBJS) 
	$(CXX) $(LDFLAGS) ReduceMin.o $(RAJAOBJS) $(LDFLAGS) $(LDTIMER) $(LDPATH) -o  reducemin.exe

reducemax: ReduceMax.o $(RAJAOBJS) 
	$(CXX) $(LDFLAGS) ReduceMax.o $(RAJAOBJS) $(LDFLAGS) $(LDTIMER) $(LDPATH) -o  reducemax.exe

reducesum: ReduceSum.o $(RAJAOBJS) 
	$(CXX) $(LDFLAGS) ReduceSum.o $(RAJAOBJS) $(LDFLAGS) $(LDTIMER) $(LDPATH) -o  reducesum.exe

traversal: Traversal.o $(RAJAOBJS) 
	$(CXX) $(LDFLAGS) Traversal.o $(RAJAOBJS) $(LDFLAGS) $(LDTIMER) $(LDPATH) -o  traversal.exe

#%.o : %.cu ; $(CXX) $(CXXFLAGS) -c -o $@ $< $(CXXFLAGS_BUILD)
%.o : %.cxx ; $(CXX) $(CXXFLAGS) -c -o $@ $< $(CXXFLAGS_BUILD)

#%.o : %.cxx ; $(CXX) $(CXXFLAGS) -S -o $@ $< 
## The following assembly options are for Intel compiler...
#%.o : %.cxx ; $(CXX) $(CXXFLAGS) -S -fcode-asm -vec-report3 -o $@ $< $(CXXFLAGS)

clean-obj: 
	rm -rf *.o $(RAJA_SRC)/*.o

clean: clean-obj 
	rm -rf *.cpp*.i* *.cuda*.* *.fatbin* *.hash* *.module_id* *.ptx *.reg.c *.exe


