###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-kernel-lambda-args
  SOURCES test-kernel-lambda-args.cpp)

raja_add_test(
  NAME test-kernel
  SOURCES test-kernel.cpp)

raja_add_test(
  NAME test-multipolicy
  SOURCES test-multipolicy.cpp)

raja_add_test(
  NAME test-sharedmem
  SOURCES test-sharedmem.cpp)

raja_add_test(
  NAME test-simd
  SOURCES test-simd.cpp)

raja_add_test_subdirectory(cpu)

if(ENABLE_CUDA)
  raja_add_test_subdirectory(cuda)
endif(ENABLE_CUDA)

if(ENABLE_TARGET_OPENMP)
  raja_add_test_subdirectory(omp-target)
endif(ENABLE_TARGET_OPENMP)
