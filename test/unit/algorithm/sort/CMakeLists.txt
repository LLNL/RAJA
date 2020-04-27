###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-sort-seq
  SOURCES test-sort-seq.cpp)

if(RAJA_ENABLE_OPENMP)
  raja_add_test(
    NAME test-sort-openmp
    SOURCES test-sort-openmp.cpp)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  # reserved for futute compatability
endif()

if(RAJA_ENABLE_TBB)
  raja_add_test(
    NAME test-sort-tbb
    SOURCES test-sort-tbb.cpp)
endif()

if(RAJA_ENABLE_CUDA)
  raja_add_test(
    NAME test-sort-cuda
    SOURCES test-sort-cuda.cpp)
endif()

if(RAJA_ENABLE_HIP)
  raja_add_test(
    NAME test-sort-hip
    SOURCES test-sort-hip.cpp)
endif()

