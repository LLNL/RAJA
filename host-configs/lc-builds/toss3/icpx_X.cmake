###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(CMAKE_CXX_FLAGS_RELEASE "${COMMON_FLAGS} -O3 -march=native -finline-functions --gcc-toolchain=/usr/tce/packages/gcc/gcc-8.1.0" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${COMMON_FLAGS} -O3 -g -march=native -finline-functions --gcc-toolchain=/usr/tce/packages/gcc/gcc-8.1.0" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "${COMMON_FLAGS} -O0 -g --gcc-toolchain=/usr/tce/packages/gcc/gcc-8.1.0" CACHE STRING "")

set(RAJA_HOST_CONFIG_LOADED On CACHE BOOL "")
