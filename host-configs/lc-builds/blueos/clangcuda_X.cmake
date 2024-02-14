###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(RAJA_COMPILER "RAJA_COMPILER_CLANG" CACHE STRING "")

set(CMAKE_EXE_LINKER_FLAGS "-L${CUDA_TOOLKIT_ROOT_DIR}/lib64 -lcudart_static -lcudadevrt -lrt -ldl -lnvToolsExt -pthread -Wl,-rpath=${RAJA_CXX_COMPILER_ROOT_DIR}/lib" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "")

set(RAJA_DATA_ALIGN 64 CACHE STRING "")

set(RAJA_HOST_CONFIG_LOADED On CACHE BOOL "")
