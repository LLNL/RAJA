###############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
###############################################################################

set(RAJA_COMPILER "RAJA_COMPILER_XLC" CACHE STRING "")

set(CMAKE_CXX_COMPILER "/usr/tce/packages/xl/xl-2018.11.26/bin/xlc++_r" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/tce/packages/xl/xl-2018.11.26/bin/xlC_r" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -qxlcompatmacros -qlanglvl=extended0x -qalias=noansi -qsmp=omp -qhot -qnoeh" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -qxlcompatmacros -qlanglvl=extended0x -qalias=noansi -qsmp=omp -qhot -qnoeh" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -qsmp=omp:noopt " CACHE STRING "")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-z,muldefs" CACHE STRING "")

set(RAJA_RANGE_ALIGN 4 CACHE INT "")
set(RAJA_RANGE_MIN_LENGTH 32 CACHE INT "")
set(RAJA_DATA_ALIGN 64 CACHE INT "")

set(RAJA_HOST_CONFIG_LOADED On CACHE Bool "")

