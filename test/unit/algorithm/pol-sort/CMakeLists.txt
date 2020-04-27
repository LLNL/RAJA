###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-pol-sort-seq
  SOURCES test-pol-sort-seq.cpp)

if(RAJA_ENABLE_OPENMP)
  raja_add_test(
    NAME test-pol-sort-openmp
    SOURCES test-pol-sort-openmp.cpp)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  # reserved for futute compatability
endif()

if(RAJA_ENABLE_TBB)
  raja_add_test(
    NAME test-pol-sort-tbb
    SOURCES test-pol-sort-tbb.cpp)
endif()

if(RAJA_ENABLE_CUDA)
  raja_add_test(
    NAME test-pol-sort-cuda
    SOURCES test-pol-sort-cuda.cpp)
endif()

if(RAJA_ENABLE_HIP)
  raja_add_test(
    NAME test-pol-sort-hip
    SOURCES test-pol-sort-hip.cpp)
endif()

