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
# IMPORTANT:  Make sure CXXFLAGS are what you want. They are used in
#             the source code, RAJA header files in particular, to
#             set code compilation options.
#

OPT_DEBUG = opt
#OPT_DEBUG = debug

#rzalastor
#RAJA_ARCH = x86_sse_icc
#RAJA_ARCH = x86_sse_gnu

#rzmerl
#RAJA_ARCH = x86_avx_icc
#RAJA_ARCH = x86_avx_gnu

#rzuseq
RAJA_ARCH = bgq_xlc12
#RAJA_ARCH = bgq_clang
#RAJA_ARCH = bgq_gnu

#rzmic
#RAJA_ARCH = MIC

LULESH_OPTS = -DUSE_MPI=0 -DUSE_OMP=1

#common places you might find silo on the Livermore machines.
#SILO_INCDIR = /opt/local/include
#SILO_LIBDIR = /opt/local/lib
#SILO_INCDIR = ./silo/4.9/1.8.10.1/include
#SILO_LIBDIR = ./silo/4.9/1.8.10.1/lib

#If you do not have silo and visit you can get them at:
#silo:  https://wci.llnl.gov/codes/silo/downloads.html
#visit: https://wci.llnl.gov/codes/visit/download.html


##
## Include file containing compiler version and options.
##
include ../../compilers.mk


CXXFLAGS_BUILD = -I. $(CXXFLAGS) $(LULESH_OPTS)

OBJFILES := $(patsubst %.cc,%.o,$(wildcard ./*.cc))


default: $(OBJFILES)
	$(CXX_COMPILE) $(OBJFILES) $(LDFLAGS) $(LDPATH) -o  lulesh-v2.0.exe
	
## Uncomment this line to generate an executable to run...
%.o : %.cc ; $(CXX_COMPILE) -c -o $@ $< $(CXXFLAGS_BUILD)

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
	rm -rf *.o

clean: clean-obj
	rm -rf *.s *.lst *.exe
