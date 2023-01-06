###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(RAJA_COMPILER "RAJA_COMPILER_ICC" CACHE STRING "")

set(COMMON_FLAGS "-gxx-name=/usr/tce/packages/gcc/gcc-8.1.0/bin/g++")

set(CMAKE_CXX_FLAGS_RELEASE "${COMMON_FLAGS} -O3 -march=native -ansi-alias -diag-disable cpu-dispatch" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${COMMON_FLAGS} -O3 -g -march=native -ansi-alias -diag-disable cpu-dispatch" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "${COMMON_FLAGS} -O0 -g" CACHE STRING "")

set(RAJA_DATA_ALIGN 64 CACHE STRING "")

set(RAJA_HOST_CONFIG_LOADED On CACHE BOOL "")
