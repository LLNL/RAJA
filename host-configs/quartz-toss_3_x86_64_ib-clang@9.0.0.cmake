###################
# Generated host-config - Edit at own risk!
###################
# Copyright (c) 2020, Lawrence Livermore National Security, LLC and
# other Umpire Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 
###################

#------------------
# SYS_TYPE: toss_3_x86_64_ib
# Compiler Spec: clang@9.0.0
# CMake executable path: /usr/tce/packages/cmake/cmake-3.14.5/bin/cmake
#------------------

#------------------
# Compilers
#------------------

set(CMAKE_C_COMPILER "/usr/tce/packages/clang/clang-9.0.0/bin/clang" CACHE PATH "")

set(CMAKE_CXX_COMPILER "/usr/tce/packages/clang/clang-9.0.0/bin/clang++" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -msse4.2 -funroll-loops -finline-functions" CACHE STRING "")

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -msse4.2 -funroll-loops -finline-functions" CACHE STRING "")

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "")

set(ENABLE_CUDA OFF CACHE BOOL "")

#------------------------------------------------------------------------------
# Other
#------------------------------------------------------------------------------

set(RAJA_RANGE_ALIGN "4" CACHE STRING "")

set(RAJA_RANGE_MIN_LENGTH "32" CACHE STRING "")

set(RAJA_DATA_ALIGN "64" CACHE STRING "")

set(RAJA_HOST_CONFIG_LOADED ON CACHE BOOL "")

