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

set(RAJA_COMPILER "RAJA_COMPILER_GNU" CACHE STRING "")

set(CMAKE_CXX_COMPILER "/usr/tce/packages/gcc/gcc-7.3.1/bin/g++" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -finline-functions" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-Ofast -g -finline-functions" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "")

set(HOST_OPT_FLAGS "-Xcompiler -O3 -Xcompiler -finline-functions -Xcompiler -fopenmp")

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 ${HOST_OPT_FLAGS}" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-g -lineinfo -O3 ${HOST_OPT_FLAGS}" CACHE STRING "")

set(RAJA_RANGE_ALIGN 4 CACHE INT "")
set(RAJA_RANGE_MIN_LENGTH 32 CACHE INT "")
set(RAJA_DATA_ALIGN 64 CACHE INT "")

set(RAJA_HOST_CONFIG_LOADED On CACHE Bool "")
