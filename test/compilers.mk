# 
# THIS IS NOT OPEN SOURCE OR PUBLIC DOMAIN SOFTWARE
#
# See README-RAJA_license.txt for access and distribution restrictions
#

#
#  Edit this file to choose compiler versions and options.
#
#  Make sure ONE AND ONLY ONE 'CXX' and 'CXXFLAGS' variable is set 
#  in the 'RAJA_ARCH' section for the desired compiler.
#
# IMPORTANT:  Make sure PLATFORM is what you want. It is used in 
#             the source code, RAJA header files in particular, to
#             set code compilation options. 
#
#             The RAJA_ARCH variable is set in the Makefile in the
#             directory containing the source code you want to build. 
#


#
#  Intel compiler on MIC 
# 
ifeq ($(RAJA_ARCH),MIC)
CXX 		= icc

ifeq ($(OPT_DEBUG),opt)
CXXFLAGS = -g -O3 -mmic -vec-report3  -inline-max-total-size=20000 -inline-forceinline -ansi-alias -std=c++0x -openmp
LDFLAGS	= $(CXXFLAGS)
endif

ifeq ($(OPT_DEBUG),debug)
CXXFLAGS = -g -mmic  -O0 -std=c++0x -openmp
LDFLAGS	= $(CXXFLAGS)
endif

PLATFORM	= -DRAJA_PLATFORM_X86_SSE -DRAJA_COMPILER_ICC
LDPATH		=

endif 


#
#  Intel compiler on x86 with SSE 4.1
# 
ifeq ($(RAJA_ARCH),x86_sse_icc)
#CXX 		= /usr/local/tools/icc-12.1.339/bin/icpc
#CXX 		= /usr/local/tools/ic-13.0.117/bin/icpc
#CXX 		= /usr/local/tools/ic-13.0.146/bin/icpc
#CXX 		= /usr/local/tools/ic-13.1.163/bin/icpc
#CXX 		= /usr/local/tools/ic-14.0.080/bin/icpc
#CXX 		= /usr/local/tools/ic-14.0.097/bin/icpc
#CXX            = /usr/local/tools/ic-14.0.106/bin/icpc
#CXX            = /usr/local/tools/ic-14.0.144/bin/icpc
#CXX            = /usr/local/tools/ic-14.0.174/bin/icpc
#CXX            = /usr/local/tools/ic-14.0.211/bin/icpc
#CXX             = /usr/local/tools/ic-15.0.024-beta/bin/icpc
#CXX             = /usr/local/tools/ic-15.0.090/bin/icpc
CXX             = /usr/local/tools/ic-15.0.133/bin/icpc

ifeq ($(OPT_DEBUG),opt)
#CXXFLAGS = -O3 -msse4.1 -inline-max-total-size=20000 -inline-forceinline -opt-streaming-stores always -ansi-alias -std=c++0x -openmp
CXXFLAGS = -O3 -msse4.1 -inline-max-total-size=20000 -inline-forceinline -ansi-alias -std=c++0x -openmp
#CXXFLAGS = -O2 -msse4.1 -inline-max-total-size=20000 -inline-forceinline -ansi-alias -std=c++0x -openmp
#CXXFLAGS = -O1 -g -msse4.1 -inline-max-total-size=20000 -inline-forceinline -ansi-alias -std=c++0x -openmp
##CXXFLAGS = -O3 -msse4.1 -inline-max-total-size=20000 -inline-forceinline -ansi-alias -std=c++0x 
###CXXFLAGS = -O3 -msse4.1 -inline-max-size=20000 -inline-max-total-size=20000 -inline-forceinline -ansi-alias -std=c++0x -openmp
LDFLAGS	= $(CXXFLAGS)
endif

ifeq ($(OPT_DEBUG),debug)
CXXFLAGS = -g -O0 -std=c++0x -openmp
LDFLAGS	= $(CXXFLAGS)
endif

PLATFORM	= -DRAJA_PLATFORM_X86_SSE -DRAJA_COMPILER_ICC
LDPATH		=

endif 


#
#  GNU compiler on x86 with SSE 4.1
# 
ifeq ($(RAJA_ARCH),x86_sse_gnu)
#CXX 		= /usr/local/bin/g++-4.4.6
#CXX 		= /usr/local/bin/g++-4.6.1
#CXX 		= /usr/apps/gnu/4.7.1/bin/g++
#CXX 		= /usr/apps/gnu/4.8.0/bin/g++
CXX 		= /usr/apps/gnu/4.9.0/bin/g++

ifeq ($(OPT_DEBUG),opt)
#
# Use this with GNU 4.7X and later
CXXFLAGS = -Ofast -msse4.1 -finline-functions -finline-limit=20000 -std=c++11 -fopenmp
#CXXFLAGS = -O3 -msse4.1 -finline-functions -finline-limit=20000 -std=c++0x -openmp
##CXXFLAGS = -O3 -msse4.1  -ansi-alias -std=c++0x
## inline flags...
#CXXFLAGS = -O3 -msse4.1  -finline-functions -finline-limit=20000 -ansi-alias -std=c++0x
## inline flags + others...
##CXXFLAGS = -O3 -msse4.1  -finline-functions -finline-limit=20000 -fomit-frame-pointer -minline-all-stringops -malign-double -ftree-vectorize -floop-block -ansi-alias -std=c++0x -openmp
##CXXFLAGS = -O3 -msse4.1  -finline-functions -finline-limit=20000 -fomit-frame-pointer -malign-double -ftree-vectorize -floop-block -ansi-alias -std=c++0x
LDFLAGS	= $(CXXFLAGS)
endif

ifeq ($(OPT_DEBUG),debug)
CXXFLAGS = -g -O0 -std=c++0x -fopenmp
LDFLAGS	= $(CXXFLAGS)
endif

PLATFORM 	= -DRAJA_PLATFORM_X86_SSE -DRAJA_COMPILER_GNU
LDPATH		=

endif 


#
#  Intel compiler on x86 with AVX 2
# 
ifeq ($(RAJA_ARCH),x86_avx_icc)
#CXX 		= /usr/local/tools/icc-12.1.339/bin/icpc
#CXX 		= /usr/local/tools/ic-13.0.117/bin/icpc

#CXX 		= /usr/local/tools/ic-13.0.146/bin/icpc
#CXX 		= /usr/local/tools/ic-13.1.163/bin/icpc
#CXX 		= /usr/local/tools/ic-14.0.080/bin/icpc
#CXX 		= /usr/local/tools/ic-14.0.097/bin/icpc
#CXX 		= /usr/local/tools/ic-14.0.106/bin/icpc
#CXX            = /usr/local/tools/ic-14.0.144/bin/icpc
#CXX            = /usr/local/tools/ic-14.0.174/bin/icpc
#CXX             = /usr/local/tools/ic-15.0.024-beta/bin/icpc
#CXX             = /usr/local/tools/ic-15.0.090/bin/icpc
CXX             = /usr/local/tools/ic-15.0.133/bin/icpc

ifeq ($(OPT_DEBUG),opt)
#CXXFLAGS = -O3 -mavx -inline-max-total-size=20000 -inline-forceinline -opt-streaming-stores always -ansi-alias -std=c++0x -openmp
CXXFLAGS = -g -O3 -mavx -inline-max-total-size=20000 -inline-forceinline -ansi-alias -std=c++0x -openmp -static-intel
##CXXFLAGS = -O3 -mavx -inline-max-total-size=20000 -inline-forceinline -ansi-alias -std=c++0x 
###CXXFLAGS = -O3 -mavx -inline-max-size=20000 -inline-max-total-size=20000 -inline-forceinline -ansi-alias -std=c++0x -openmp
LDFLAGS	= $(CXXFLAGS)
endif

ifeq ($(OPT_DEBUG),debug)
CXXFLAGS = -g -O0 -std=c++0x -openmp
LDFLAGS	= $(CXXFLAGS)
endif

PLATFORM 	= -DRAJA_PLATFORM_X86_AVX -DRAJA_COMPILER_ICC
LDPATH		=

endif



#
#  GNU compiler on x86 with AVX 2
# 
ifeq ($(RAJA_ARCH),x86_avx_gnu)
#CXX 		= /usr/local/bin/g++-4.6.1
#CXX 		= /usr/apps/gnu/4.7.1/bin/g++
#CXX 		= /usr/apps/gnu/4.8.0/bin/g++
CXX 		= /usr/apps/gnu/4.9.0/bin/g++

ifeq ($(OPT_DEBUG),opt)
#
# Use this with GNU 4.7X and later
CXXFLAGS = -Ofast -mavx -finline-functions -finline-limit=20000 -std=c++11 -fopenmp
#
# These should work with older compiler versions...
#CXXFLAGS = -O3 -mavx -ansi-alias -std=c++0x -openmp
#CXXFLAGS = -O3 -mavx -std=c++0x
##CXXFLAGS = -O3 -mavx -ansi-alias -std=c++0x
##CXXFLAGS = -O3 -mavx -finline-functions -finline-limit=20000 -fomit-frame-pointer -malign-double -ftree-vectorize -floop-block -ansi-alias -std=c++0x
LDFLAGS	= $(CXXFLAGS)
endif

ifeq ($(OPT_DEBUG),debug)
CXXFLAGS = -g -O0 -std=c++0x -fopenmp
LDFLAGS	= $(CXXFLAGS)
endif

PLATFORM 	= -DRAJA_PLATFORM_X86_AVX -DRAJA_COMPILER_GNU
LDPATH		=

endif


#
#  Clang compiler on x86 
#
ifeq ($(RAJA_ARCH),x86_clang)
CXX             = /usr/global/tools/clang/chaos_5_x86_64_ib/clang-omp-3.5.0/bin/clang++

ifeq ($(OPT_DEBUG),opt)
#
CXXFLAGS = -O3 -std=c++11 -fopenmp
LDFLAGS = $(CXXFLAGS)
endif

ifeq ($(OPT_DEBUG),debug)
CXXFLAGS = -g -O0 -std=c++0x -fopenmp
LDFLAGS = -std=c++11 -g -O0 -Wl,--export-dynamic
endif

PLATFORM        = -DRAJA_PLATFORM_X86_AVX -DRAJA_COMPILER_CLANG
LDPATH          =

endif

#
#  Clang compiler for rzmist
#
ifeq ($(RAJA_ARCH),rzmist_clang)
CXX             = clang++

ifeq ($(OPT_DEBUG),opt)
#
CXXFLAGS = -O3 -mllvm -inline-threshold=10000 -std=c++11 -fopenmp
LDFLAGS = $(CXXFLAGS)
endif

ifeq ($(OPT_DEBUG),debug)
CXXFLAGS = -g -O0 -std=c++0x -fopenmp
LDFLAGS = -std=c++11 -g -O0 -Wl,--export-dynamic
endif

PLATFORM        = -DRAJA_PLATFORM_X86_AVX -DRAJA_COMPILER_CLANG
LDPATH          =

endif



#
#  XLC compiler for rzmist
# 
ifeq ($(RAJA_ARCH),rzmist_xlc)
CXX 		=  xlC

ifeq ($(OPT_DEBUG),opt)
CXXFLAGS = -O3 -qhot=novector -qsimd=auto -qlanglvl=extended0x -qnostrict -qinline=auto:level=10 -qsmp=omp
LDFLAGS	= 
## The following is needed for lompbeta1
#LDFLAGS = -O3 -qsmp=omp -qdebug=lompinterface
endif

ifeq ($(OPT_DEBUG),debug)
CXXFLAGS = -g -O0 -qlanglvl=extended0x -qsmp=omp
LDFLAGS	=  $(CXXFLAGS)
endif

PLATFORM 	= -DRAJA_PLATFORM_BGQ -DRAJA_COMPILER_XLC12 -DRAJA_COMPILER_XLC_POWER8
LDPATH		=

endif


#
#  XLC 12 compiler  for BG/Q
# 
ifeq ($(RAJA_ARCH),bgq_xlc12)
#CXX 		= /usr/local/tools/compilers/ibm/bgxlc++-12.1.0.2a
#CXX 		= /usr/local/tools/compilers/ibm/bgxlc++-12.1.0.3
#CXX 		= /usr/local/tools/compilers/ibm/mpixlcxx-12.1.0.1d
#CXX 		= /usr/local/tools/compilers/ibm/mpixlcxx-12.1.0.2
#CXX 		= /usr/local/tools/compilers/ibm/mpixlcxx-12.1.0.3
#CXX 		= mpixlcxx_r
CXX 		= /usr/local/tools/compilers/ibm/mpixlcxx_r-lompbeta2-fastmpi
#CXX 		= /usr/local/tools/compilers/ibm/mpixlcxx_r-lompbeta1-fastmpi

ifeq ($(OPT_DEBUG),opt)
#CXXFLAGS = -O3 -qarch=qp -qhot=novector -qsimd=auto -qlanglvl=extended0x -qstrict -qinline=20000 -qsmp=omp
#CXXFLAGS = -O3 -qarch=qp -qhot=novector -qsimd=auto -qlanglvl=extended0x -qnostrict -qinline=20000 -qsmp=omp
CXXFLAGS = -O3 -qarch=qp -qhot=novector -qsimd=auto -qlanglvl=extended0x -qnostrict -qinline=auto:level=10 -qsmp=omp
##
## USE THESE LINE TO GENERATE VECTORIZATION REPORT INFO...
#CXXFLAGS = -O3 -qarch=qp -qhot=novector -qsimd=auto -qlanglvl=extended0x -qstrict -qinline=20000  -qlist -qsource -qlistopt -qreport
#CXXFLAGS = -O3 -qarch=qp -qhot=novector -qsimd=auto -qlanglvl=extended0x -qnostrict -qinline=20000  -qlist -qsource -qlistopt -qreport
#CXXFLAGS = -O3 -qarch=qp -qhot=novector -qsimd=auto -qlanglvl=extended0x -qinline=100 -qlistfmt=html=inlines
LDFLAGS	= 
## The following is needed for lompbeta1
#LDFLAGS = -O3 -qsmp=omp -qdebug=lompinterface
endif

ifeq ($(OPT_DEBUG),debug)
CXXFLAGS = -g -O0 -qarch=qp -qlanglvl=extended0x -qsmp=omp
LDFLAGS	=  $(CXXFLAGS)
endif

PLATFORM 	= -DRAJA_PLATFORM_BGQ -DRAJA_COMPILER_XLC12
LDPATH		=

endif


#
#  Clang C++ compiler for BG/Q
# 
ifeq ($(RAJA_ARCH),bgq_clang)
#CXX 		= /usr/apps/gnu/clang/bin/mpiclang++11
#CXX 		= /usr/apps/gnu/clang/bin/bgclang++11

#Specific versions
#CXX            = /usr/apps/gnu/clang/r176829-20130309/bin/bgclang++11
#CXX             = /usr/apps/gnu/clang/r176751-20130307/bin/bgclang++11
#CXX             = /usr/apps/gnu/clang/r181589-20130510/bin/bgclang++11
CXX             = /usr/apps/gnu/clang/r189357-20130827/bin/bgclang++11

ifeq ($(OPT_DEBUG),opt)
#CXXFLAGS = -O3 -finline-functions -finline-limit=20000 -fomit-frame-pointer -minline-all-stringops -malign-double -ftree-vectorize -floop-block -ansi-alias -std=c++0x
#Opt 3
#CXXFLAGS = -O3 -finline-functions -finline-limit=20000 -malign-double -std=c++0x
#Opt 2
CXXFLAGS = -O3 -finline-functions  -ffast-math -std=c++0x
#Opt 1
#CXXFLAGS = -O3 -finline-functions  -std=c++0x
#Opt 0
#CXXFLAGS = -O0 -finline-functions  -std=c++0x
LDFLAGS	= $(CXXFLAGS)
endif

ifeq ($(OPT_DEBUG),debug)
CXXFLAGS = -g -O0 -std=c++0x
LDFLAGS	= $(CXXFLAGS)
endif

PLATFORM 	= -DRAJA_PLATFORM_BGQ -DRAJA_COMPILER_CLANG
LDPATH		=

endif 


#
#  GNU compiler for BG/Q
#
ifeq ($(RAJA_ARCH),bgq_gnu)
#CXX             = /bgsys/drivers/ppcfloor/gnu-linux/powerpc64-bgq-linux/bin/g++
CXX             = /usr/local/tools/compilers/ibm/mpicxx-4.7.2

#Previous versions

ifeq ($(OPT_DEBUG),opt)
#CXXFLAGS = -O3 -finline-functions -finline-limit=20000 -std=c++0x -fopenmp
CXXFLAGS = -O3 -mcpu=a2 -mtune=a2 -finline-functions -finline-limit=20000 -std=c++11 -fopenmp
#CXXFLAGS = -O3 -mcpu=a2 -mtune=a2 -finline-functions -finline-limit=20000 -std=c++11

##LDFLAGS = -lmass
LDFLAGS = -std=c++11  -O3 -fopenmp 
#LDFLAGS = -std=c++11  -O3 
endif

ifeq ($(OPT_DEBUG),debug)
CXXFLAGS = -g -O0 -std=c++0x -fopenmp
LDFLAGS = $(CXXFLAGS)
endif

PLATFORM        = -DRAJA_PLATFORM_BGQ -DRAJA_COMPILER_GNU
LDPATH          =

endif
