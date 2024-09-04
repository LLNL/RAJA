###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(RAJA_COMPILER "RAJA_COMPILER_CLANG" CACHE STRING "")

set(CMAKE_CXX_FLAGS_RELEASE "--gcc-toolchain=/usr/tce/packages/gcc/gcc-10.3.1 -O3 -march=native -funroll-loops -finline-functions" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "--gcc-toolchain=/usr/tce/packages/gcc/gcc-10.3.1 -O3 -g -march=native -funroll-loops -finline-functions" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "--gcc-toolchain=/usr/tce/packages/gcc/gcc-10.3.1 -O0 -g -fsanitize=address -fno-omit-frame-pointer" CACHE STRING "")

set(RAJA_HOST_CONFIG_LOADED On CACHE BOOL "")
