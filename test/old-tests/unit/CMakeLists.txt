###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-sharedmem
  SOURCES test-sharedmem.cpp)

raja_add_test(
  NAME test-simd
  SOURCES test-simd.cpp)

add_subdirectory(cpu)

if(RAJA_ENABLE_CUDA)
  add_subdirectory(cuda)
endif(RAJA_ENABLE_CUDA)
