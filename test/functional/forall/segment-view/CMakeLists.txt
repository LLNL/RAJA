###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-forall-segment-view-seq
  SOURCES test-forall-segment-view-seq.cpp)

if(RAJA_ENABLE_OPENMP)
  raja_add_test(
    NAME test-forall-segment-view-openmp
    SOURCES test-forall-segment-view-openmp.cpp)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  raja_add_test(
    NAME test-forall-segment-view-openmp-target
    SOURCES test-forall-segment-view-openmp-target.cpp)
endif()

if(RAJA_ENABLE_TBB)
  raja_add_test(
    NAME test-forall-segment-view-tbb
    SOURCES test-forall-segment-view-tbb.cpp)
endif()

if(RAJA_ENABLE_CUDA)
  raja_add_test(
    NAME test-forall-segment-view-cuda
    SOURCES test-forall-segment-view-cuda.cpp)
endif()

if(RAJA_ENABLE_HIP)
  raja_add_test(
    NAME test-forall-segment-view-hip
    SOURCES test-forall-segment-view-hip.cpp)
endif()

if(RAJA_ENABLE_SYCL)
  raja_add_test(
    NAME test-forall-segment-view-sycl
    SOURCES test-forall-segment-view-sycl.cpp)
endif()
