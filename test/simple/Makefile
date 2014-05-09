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


RAJA_TOPDIR    = ../..
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
include ../compilers.mk


CXXFLAGS_BUILD = -I. -I$(RAJA_INC) $(CXXFLAGS) $(RAJA_RULES)


TESTOBJS := $(patsubst %.cxx,%.o,$(wildcard ./*.cxx))

RAJAOBJS := $(patsubst %.cxx,%.o,$(wildcard $(RAJA_SRC)/*.cxx))

OBJFILES = $(TESTOBJS) $(RAJAOBJS)

simple-test: $(OBJFILES)
	$(CXX_COMPILE) $(OBJFILES) $(LDFLAGS) $(LDPATH) -o  $@.exe

## Uncomment this line to generate an executable to run...
%.o : %.cxx ; $(CXX_COMPILE) -c -o $@ $< $(CXXFLAGS_BUILD) 

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
	rm -rf *.s *.lst *.exe 
