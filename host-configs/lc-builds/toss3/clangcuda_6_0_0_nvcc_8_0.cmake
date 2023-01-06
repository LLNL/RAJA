###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(RAJA_COMPILER "RAJA_COMPILER_CLANG" CACHE STRING "")

set(CMAKE_CXX_COMPILER "/usr/tce/packages/clang/clang-6.0.0/bin/clang++" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/tce/packages/clang/clang-6.0.0/bin/clang" CACHE PATH "")

set(CMAKE_EXE_LINKER_FLAGS "-L/usr/tce/packages/cuda/cuda-8.0/lib64 -lcudart_static -lcudadevrt -lrt -ldl -lnvToolsExt -pthread -Wl,-rpath=/usr/tce/packages/clang/clang-6.0.0/lib" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -msse4.2 -funroll-loops -finline-functions" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -msse4.2 -funroll-loops -finline-functions" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "")

set(RAJA_DATA_ALIGN 64 CACHE INT "")

set(RAJA_HOST_CONFIG_LOADED On CACHE BOOL "")
