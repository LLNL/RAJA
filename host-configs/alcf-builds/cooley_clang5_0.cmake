###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(RAJA_COMPILER "RAJA_COMPILER_CLANG" CACHE STRING "")

set(CMAKE_CXX_COMPILER "/soft/compilers/llvm/5.0/bin/clang++" CACHE PATH "")
set(CMAKE_C_COMPILER "/soft/compilers/llvm/5.0/bin/clang" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g" CACHE STRING "")

set(RAJA_RANGE_ALIGN 4 CACHE INT "")
set(RAJA_RANGE_MIN_LENGTH 32 CACHE INT "")
set(RAJA_DATA_ALIGN 64 CACHE INT "")

set(RAJA_HOST_CONFIG_LOADED On CACHE Bool "")
