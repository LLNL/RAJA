###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-pol-stable-sort-seq
  SOURCES test-pol-stable-sort-seq.cpp)

if(RAJA_ENABLE_OPENMP)
  raja_add_test(
    NAME test-pol-stable-sort-openmp
    SOURCES test-pol-stable-sort-openmp.cpp)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  # reserved for futute compatability
endif()

if(RAJA_ENABLE_TBB)
  raja_add_test(
    NAME test-pol-stable-sort-tbb
    SOURCES test-pol-stable-sort-tbb.cpp)
endif()

if(RAJA_ENABLE_CUDA)
  raja_add_test(
    NAME test-pol-stable-sort-cuda
    SOURCES test-pol-stable-sort-cuda.cpp)
endif()

if(RAJA_ENABLE_HIP)
  raja_add_test(
    NAME test-pol-stable-sort-hip
    SOURCES test-pol-stable-sort-hip.cpp)
endif()

