# 
# THIS IS NOT OPEN SOURCE OR PUBLIC DOMAIN SOFTWARE
#
# See README-RAJA_license.txt for access and distribution restrictions
#

#
#  Edit this file to choose compiler versions and options.
#
#  Make sure ONE AND ONLY ONE 'CXX' and 'CXX_COMPILE' variable is set 
#  in the 'RAJA_ARCH' section for the desired compiler.
#
# IMPORTANT:  Make sure CXXFLAGS are what you want. They are used in 
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
CXX_COMPILE = $(CXX) -g -O3 -mmic -vec-report3  -inline-max-total-size=20000 -inline-forceinline -ansi-alias -std=c++0x -openmp
LDFLAGS	=
endif

ifeq ($(OPT_DEBUG),debug)
CXX_COMPILE = $(CXX) -g -mmic  -O0 -std=c++0x -openmp
LDFLAGS	=
endif

CXXFLAGS	= -DRAJA_PLATFORM_X86_SSE -DRAJA_COMPILER_ICC
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
CXX             = /usr/local/tools/ic-15.0.024-beta/bin/icpc

ifeq ($(OPT_DEBUG),opt)
#CXX_COMPILE = $(CXX) -O3 -msse4.1 -inline-max-total-size=20000 -inline-forceinline -opt-streaming-stores always -ansi-alias -std=c++0x -openmp
CXX_COMPILE = $(CXX) -O3 -msse4.1 -inline-max-total-size=20000 -inline-forceinline -ansi-alias -std=c++0x -openmp
#CXX_COMPILE = $(CXX) -O2 -msse4.1 -inline-max-total-size=20000 -inline-forceinline -ansi-alias -std=c++0x -openmp
#CXX_COMPILE = $(CXX) -O1 -g -msse4.1 -inline-max-total-size=20000 -inline-forceinline -ansi-alias -std=c++0x -openmp
##CXX_COMPILE = $(CXX) -O3 -msse4.1 -inline-max-total-size=20000 -inline-forceinline -ansi-alias -std=c++0x 
LDFLAGS	=
endif

ifeq ($(OPT_DEBUG),debug)
CXX_COMPILE = $(CXX) -g -O0 -std=c++0x -openmp
LDFLAGS	=
endif

CXXFLAGS	= -DRAJA_PLATFORM_X86_SSE -DRAJA_COMPILER_ICC
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
CXX_COMPILE = $(CXX) -Ofast -msse4.1 -finline-functions -finline-limit=20000 -std=c++11 -fopenmp
#CXX_COMPILE = $(CXX) -O3 -msse4.1 -finline-functions -finline-limit=20000 -std=c++0x -openmp
##CXX_COMPILE = $(CXX) -O3 -msse4.1  -ansi-alias -std=c++0x
## inline flags...
#CXX_COMPILE = $(CXX) -O3 -msse4.1  -finline-functions -finline-limit=20000 -ansi-alias -std=c++0x
## inline flags + others...
##CXX_COMPILE = $(CXX) -O3 -msse4.1  -finline-functions -finline-limit=20000 -fomit-frame-pointer -minline-all-stringops -malign-double -ftree-vectorize -floop-block -ansi-alias -std=c++0x -openmp
##CXX_COMPILE = $(CXX) -O3 -msse4.1  -finline-functions -finline-limit=20000 -fomit-frame-pointer -malign-double -ftree-vectorize -floop-block -ansi-alias -std=c++0x
LDFLAGS	=
endif

ifeq ($(OPT_DEBUG),debug)
CXX_COMPILE = $(CXX) -g -O0 -std=c++0x -openmp
LDFLAGS	=
endif

CXXFLAGS 	= -DRAJA_PLATFORM_X86_SSE -DRAJA_COMPILER_GNU
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
CXX            = /usr/local/tools/ic-14.0.174/bin/icpc
#CXX             = /usr/local/tools/ic-15.0.024-beta/bin/icpc

ifeq ($(OPT_DEBUG),opt)
#CXX_COMPILE = $(CXX) -O3 -mavx -inline-max-total-size=20000 -inline-forceinline -opt-streaming-stores always -ansi-alias -std=c++0x -openmp
CXX_COMPILE = $(CXX) -g -O3 -mavx -inline-max-total-size=20000 -inline-forceinline -ansi-alias -std=c++0x -openmp -static-intel
##CXX_COMPILE = $(CXX) -O3 -mavx -inline-max-total-size=20000 -inline-forceinline -ansi-alias -std=c++0x 
LDFLAGS	=
endif

ifeq ($(OPT_DEBUG),debug)
CXX_COMPILE = $(CXX) -g -O0 -std=c++0x -openmp
LDFLAGS	=
endif

CXXFLAGS 	= -DRAJA_PLATFORM_X86_AVX -DRAJA_COMPILER_ICC
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
CXX_COMPILE = $(CXX) -Ofast -mavx -finline-functions -finline-limit=20000 -std=c++11 -fopenmp
#
# These should work with older compiler versions...
#CXX_COMPILE = $(CXX) -O3 -mavx -ansi-alias -std=c++0x -openmp
#CXX_COMPILE = $(CXX) -O3 -mavx -std=c++0x
##CXX_COMPILE = $(CXX) -O3 -mavx -ansi-alias -std=c++0x
##CXX_COMPILE = $(CXX) -O3 -mavx -finline-functions -finline-limit=20000 -fomit-frame-pointer -malign-double -ftree-vectorize -floop-block -ansi-alias -std=c++0x
LDFLAGS	=
endif

ifeq ($(OPT_DEBUG),debug)
CXX_COMPILE = $(CXX) -g -O0 -std=c++0x -openmp
LDFLAGS	=
endif

CXXFLAGS 	= -DRAJA_PLATFORM_X86_AVX -DRAJA_COMPILER_GNU
LDPATH		=

endif


#
#  XLC 12 compiler  for BG/Q
# 
ifeq ($(RAJA_ARCH),bgq_xlc12)
#CXX 		= /usr/local/tools/compilers/ibm/bgxlc++-12.1.0.2a
#CXX 		= /usr/local/tools/compilers/ibm/bgxlc++-12.1.0.3
CXX 		= mpixlcxx_r
#CXX 		= /usr/local/tools/compilers/ibm/mpixlcxx-12.1.0.1d
#CXX 		= /usr/local/tools/compilers/ibm/mpixlcxx-12.1.0.2
#CXX 		= /usr/local/tools/compilers/ibm/mpixlcxx-12.1.0.3

ifeq ($(OPT_DEBUG),opt)
#CXX_COMPILE = $(CXX) -O3 -qarch=qp -qhot=novector -qsimd=auto -qlanglvl=extended0x -qstrict -qinline=20000 -qsmp=omp
#CXX_COMPILE = $(CXX) -O3 -qarch=qp -qhot=novector -qsimd=auto -qlanglvl=extended0x -qnostrict -qinline=20000 -qsmp=omp
CXX_COMPILE = $(CXX) -O3 -qarch=qp -qhot=novector -qsimd=auto -qlanglvl=extended0x -qnostrict -qinline=auto:level=10 -qsmp=omp
##
## USE THESE LINE TO GENERATE VECTORIZATION REPORT INFO...
#CXX_COMPILE = $(CXX) -O3 -qarch=qp -qhot=novector -qsimd=auto -qlanglvl=extended0x -qstrict -qinline=20000  -qlist -qsource -qlistopt -qreport
#CXX_COMPILE = $(CXX) -O3 -qarch=qp -qhot=novector -qsimd=auto -qlanglvl=extended0x -qnostrict -qinline=20000  -qlist -qsource -qlistopt -qreport
#CXX_COMPILE = $(CXX) -O3 -qarch=qp -qhot=novector -qsimd=auto -qlanglvl=extended0x -qinline=100 -qlistfmt=html=inlines
LDFLAGS	= 
endif

ifeq ($(OPT_DEBUG),debug)
CXX_COMPILE = $(CXX) -g -O0 -qarch=qp -qlanglvl=extended0x -qsmp=omp
LDFLAGS	= 
endif

CXXFLAGS 	= -DRAJA_PLATFORM_BGQ -DRAJA_COMPILER_XLC12
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
#CXX_COMPILE = $(CXX) -O3 -finline-functions -finline-limit=20000 -fomit-frame-pointer -minline-all-stringops -malign-double -ftree-vectorize -floop-block -ansi-alias -std=c++0x
#Opt 3
#CXX_COMPILE = $(CXX) -O3 -finline-functions -finline-limit=20000 -malign-double -std=c++0x
#Opt 2
CXX_COMPILE = $(CXX) -O3 -finline-functions  -ffast-math -std=c++0x
#Opt 1
#CXX_COMPILE = $(CXX) -O3 -finline-functions  -std=c++0x
#Opt 0
#CXX_COMPILE = $(CXX) -O0 -finline-functions  -std=c++0x
LDFLAGS	=
endif

ifeq ($(OPT_DEBUG),debug)
CXX_COMPILE = $(CXX) -g -O0 -std=c++0x
LDFLAGS	=
endif

CXXFLAGS 	= -DRAJA_PLATFORM_BGQ -DRAJA_COMPILER_CLANG
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
#CXX_COMPILE = $(CXX) -O3 -finline-functions -finline-limit=20000 -std=c++0x -fopenmp
CXX_COMPILE = $(CXX) -O3 -mcpu=a2 -mtune=a2 -finline-functions -finline-limit=20000 -std=c++11 -fopenmp
#CXX_COMPILE = $(CXX) -O3 -mcpu=a2 -mtune=a2 -finline-functions -finline-limit=20000 -std=c++11

##LDFLAGS = -lmass
LDFLAGS = -std=c++11  -O3 -fopenmp 
#LDFLAGS = -std=c++11  -O3 
endif

ifeq ($(OPT_DEBUG),debug)
CXX_COMPILE = $(CXX) -g -O0 -std=c++0x -fopenmp
LDFLAGS =
endif

CXXFLAGS        = -DRAJA_PLATFORM_BGQ -DRAJA_COMPILER_GNU
LDPATH          =

endif
