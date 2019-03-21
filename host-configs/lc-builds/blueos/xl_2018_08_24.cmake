##
## Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-689114
##
## All rights reserved.
##
## This file is part of RAJA.
##
## For details about use and distribution, please read RAJA/LICENSE.
##

set(RAJA_COMPILER "RAJA_COMPILER_XLC" CACHE STRING "")

set(CMAKE_CXX_COMPILER "/usr/tce/packages/xl/xl-beta-2018.08.24/bin/xlc++_r" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/tce/packages/xl/xl-beta-2018.08.24/bin/xlC_r" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 " CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g " CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -qsmp=omp:noopt " CACHE STRING "")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-z,muldefs" CACHE STRING "")

set(RAJA_RANGE_ALIGN 4 CACHE INT "")
set(RAJA_RANGE_MIN_LENGTH 32 CACHE INT "")
set(RAJA_DATA_ALIGN 64 CACHE INT "")

set(RAJA_HOST_CONFIG_LOADED On CACHE Bool "")

