###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-kernel
  SOURCES test-kernel.cpp)

raja_add_test(
  NAME test-sharedmem
  SOURCES test-sharedmem.cpp)

raja_add_test(
  NAME test-simd
  SOURCES test-simd.cpp)


raja_add_test(
  NAME test-register
  SOURCES test-register.cpp)
  
raja_add_test(
  NAME test-vector
  SOURCES test-vector.cpp)
  
raja_add_test(
  NAME test-matrix
  SOURCES test-matrix.cpp)

add_subdirectory(cpu)

if(RAJA_ENABLE_CUDA)
  add_subdirectory(cuda)
endif(RAJA_ENABLE_CUDA)
