###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(RAJA_COMPILER "RAJA_COMPILER_CLANG" CACHE STRING "")

set(CMAKE_CXX_COMPILER "/usr/apps/gnu/clang/r284961-stable/bin/bgclang++11" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/apps/gnu/clang/r284961-stable/bin/bgclang" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -std=c++11 -stdlib=libc++" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -ffast-math -std=c++11 -stdlib=libc++" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -std=c++11 -stdlib=libc++" CACHE STRING "")

set(MPIEXEC              "/usr/bin/srun" CACHE PATH "")
set(MPIEXEC_NUMPROC_FLAG "-n" CACHE PATH "")

set(ENABLE_WRAP_ALL_TESTS_WITH_MPIEXEC TRUE CACHE BOOL "Ensures that tests will be wrapped with srun to run on the backend nodes")

set(RAJA_DATA_ALIGN 64 CACHE INT "")

set(RAJA_HOST_CONFIG_LOADED On CACHE Bool "")
