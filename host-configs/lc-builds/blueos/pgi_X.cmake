###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(RAJA_COMPILER "RAJA_COMPILER_PGI" CACHE STRING "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fast -mp" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-fast -g -mp" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -mp" CACHE STRING "")

set(RAJA_DATA_ALIGN 64 CACHE STRING "")

set(RAJA_HOST_CONFIG_LOADED On CACHE BOOL "")
