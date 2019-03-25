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

set(CMAKE_CXX_COMPILER "/usr/tce/packages/xl/xl-test-2019.03.12/bin/xlc++_r" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g9" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -qsmp=omp:noopt" CACHE STRING "")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-z,muldefs" CACHE STRING "")

set(HOST_OPT_FLAGS "-Xcompiler -O3 -Xcompiler -qxlcompatmacros -Xcompiler -qlanglvl=extended0x  -Xcompiler -qalias=noansi -Xcompiler -qsmp=omp -Xcompiler -qhot -Xcompiler -qnoeh -Xcompiler -qsuppress=1500-029 -Xcompiler -qsuppress=1500-036")

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 ${HOST_OPT_FLAGS}" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-g -lineinfo -O3 ${HOST_OPT_FLAGS}" CACHE STRING "")

set(RAJA_RANGE_ALIGN 4 CACHE INT "")
set(RAJA_RANGE_MIN_LENGTH 32 CACHE INT "")
set(RAJA_DATA_ALIGN 64 CACHE INT "")

set(RAJA_HOST_CONFIG_LOADED On CACHE Bool "")

