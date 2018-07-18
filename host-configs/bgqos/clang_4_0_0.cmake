##
## Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-738930
##
## All rights reserved.
##
## This file is part of the RAJA Performance Suite.
##
## For details about use and distribution, please read raja-perfsuite/LICENSE.
##

set(RAJA_COMPILER "RAJA_COMPILER_CLANG" CACHE STRING "")

set(CMAKE_CXX_COMPILER "/usr/apps/gnu/clang/r284961-stable/bin/bgclang++11" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/apps/gnu/clang/r284961-stable/bin/bgclang" CACHE PATH "")

set(TEST_DRIVER srun CACHE STRING "use slurm to launch on BGQ")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -std=c++11 -stdlib=libc++" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -ffast-math -std=c++11 -stdlib=libc++" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -std=c++11 -stdlib=libc++" CACHE STRING "")

set(RAJA_RANGE_ALIGN 4 CACHE INT "")
set(RAJA_RANGE_MIN_LENGTH 32 CACHE INT "")
set(RAJA_DATA_ALIGN 64 CACHE INT "")
set(RAJA_COHERENCE_BLOCK_SIZE 64 CACHE INT "")

set(RAJA_HOST_CONFIG_LOADED On CACHE Bool "")
