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
#             the file ../../compilers.mk.
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

#ipa-nvcc
#RAJA_ARCH = ipa_nvcc

#rzmist
#RAJA_ARCH = rzmist_clang
#RAJA_ARCH = rzmist_xlc

RAJA_TOPDIR    = ../../..
RAJA_INC       = $(RAJA_TOPDIR)/include/
RAJA_SRC       = $(RAJA_TOPDIR)/sources/


##
## The RAJA_rules.mk file defines macro variables that specify RAJA behavior.
## To change the rules, the file in the RAJA include directory can be edited
## or it can be replaced with custom version here.
##
include $(RAJA_TOPDIR)/build/RAJA_rules.mk

##
## Options for LULESH
##
LULESH_OPTS = $(LULESH_TIMER_OPTS)


##
## Include file containing compiler version and options.
##
include ../../compilers.mk


RAJAOBJS := $(patsubst %.cxx,%.o,$(wildcard $(RAJA_SRC)/*.cxx))


##-----------------------------------------------------------
## This section is a complete hack....
##-----------------------------------------------------------

##
## If compiling with nvcc on ipa (see RAJA_ARCH defn above)
##
ifeq ($(RAJA_ARCH),ipa_nvcc)

### Test nvcc build and run non-GPU variant...
#CXXFLAGS        = -DRAJA_PLATFORM_X86_SSE -DRAJA_COMPILER_GNU

### nvcc build and run GPU variant...
CXXFLAGS        = -DRAJA_PLATFORM_X86_SSE -DRAJA_COMPILER_GNU -DRAJA_USE_CUDA


CXX_COMPILE = nvcc -O2 --expt-extended-lambda -arch compute_35 -std=c++11 -Xcompiler -fopenmp
## To debug
##CXX_COMPILE = nvcc -g -G -O0 --expt-extended-lambda -arch compute_35 -std=c++11 -Xcompiler -fopenmp -v -keep

RAJA_RULES = -DRAJA_USE_DOUBLE -DRAJA_USE_BARE_PTR

else
##
## Pull in CXX_COMPILE and CXXFLAGS from ../../compilers.mk
## (see RAJA_ARCH defn above)
##
endif

##-----------------------------------------------------------


CXXFLAGS_BUILD = -I. -I../../include -I$(RAJA_INC) $(CXXFLAGS) $(RAJA_RULES) $(LULESH_OPTS)


OBJFILES_PAR_FT = luleshRAJA-parallel-FT.o $(RAJAOBJS)
OBJFILES_PAR    = luleshRAJA-parallel.o $(RAJAOBJS)
OBJFILES_SER    = luleshRAJA-serial.o $(RAJAOBJS)

default:
	echo "Please specify version to build make serial, make parallel, or make parallel-FT"

parallel-FT: CXXFLAGS_BUILD += -DRAJA_USE_FT
parallel-FT: $(OBJFILES_PAR_FT)
	$(CXX_COMPILE) $(OBJFILES_PAR_FT) $(LDFLAGS) $(LDTIMER) $(LDPATH) -o  lulesh-RAJA-parallel-FT.exe

parallel: $(OBJFILES_PAR)
	$(CXX_COMPILE) $(OBJFILES_PAR) $(LDFLAGS) $(LDTIMER) $(LDPATH) -o  lulesh-RAJA-parallel.exe

serial: $(OBJFILES_SER)
	$(CXX_COMPILE) $(OBJFILES_SER) $(LDFLAGS) $(LDTIMER) $(LDPATH) -o  lulesh-RAJA-serial.exe

## Uncomment this line to generate an executable to run...
%.o : %.cxx ; $(CXX_COMPILE) -c -o $@ $< $(CXXFLAGS_BUILD) 
%.o : %.cu ; $(CXX_COMPILE) -c -o $@ $< $(CXXFLAGS_BUILD) 

## The following can be used to generate vectorization report and 
## assmbly code output for Intel compilers...
#%.o : %.cxx ; $(CXX_COMPILE) -S -o $@ $< $(CXXFLAGS_BUILD)
#%.o : %.cxx ; $(CXX_COMPILE) -S -vec-report3 -o $@ $< $(CXXFLAGS_BUILD)
#%.o : %.cxx ; $(CXX_COMPILE) -S -fcode-asm -vec-report3 -o $@ $< $(CXXFLAGS_BUILD)

## The following can be used to generate vectorization report and 
## assmbly code output for IBM XLC compilers...
## See compiler flag options in the appropriate section above.
#%.o : %.cxx ; $(CXX_COMPILE) -c -o $@ $< $(CXXFLAGS_BUILD)


clean-obj: 
	rm -rf *.o $(RAJA_SRC)/*.o

clean: clean-obj
	rm -rf luleshRAJA-parallel.cxx luleshRAJA-parallel.cu *.cpp*.i* *.cuda*.* *.fatbin* *.hash* *.module_id* *.ptx *.reg.c *.s *.lst *.exe 
