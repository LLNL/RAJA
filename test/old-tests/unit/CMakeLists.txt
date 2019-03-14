###############################################################################
#
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-689114
#
# All rights reserved.
#
# This file is part of RAJA.
#
# For details about use and distribution, please read RAJA/LICENSE.
#
###############################################################################

raja_add_test(
  NAME camp_tuple
  SOURCES camp_tuple.cpp)

raja_add_test(
  NAME test-atomic
  SOURCES test-atomic.cpp)

raja_add_test(
  NAME test-region
  SOURCES test-region.cpp)

raja_add_test(
  NAME test-layout
  SOURCES test-layout.cpp)

raja_add_test(
  NAME test-view
  SOURCES test-view.cpp)

raja_add_test(
  NAME test-timer
  SOURCES test-timer.cpp)

raja_add_test(
  NAME test-integral-limits
  SOURCES test-integral-limits.cpp)

raja_add_test(
  NAME test-rajavec
  SOURCES test-rajavec.cpp)

raja_add_test(
  NAME test-iterators
  SOURCES test-iterators.cpp)

raja_add_test(
  NAME test-indexvalue
  SOURCES test-indexvalue.cpp)

raja_add_test(
  NAME test-span
  SOURCES test-span.cpp)

raja_add_test(
  NAME test-kernel
  SOURCES test-kernel.cpp)

raja_add_test(
  NAME test-multipolicy
  SOURCES test-multipolicy.cpp)

raja_add_test(
  NAME test-simd
  SOURCES test-simd.cpp)

raja_add_test(
  NAME test-sharedmem
  SOURCES test-sharedmem.cpp)

add_subdirectory(cpu)

if(ENABLE_CUDA)
  add_subdirectory(cuda)
endif(ENABLE_CUDA)

if(ENABLE_TARGET_OPENMP)
  add_subdirectory(omp-target)
endif(ENABLE_TARGET_OPENMP)
