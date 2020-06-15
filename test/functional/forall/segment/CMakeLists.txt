###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-forall-segment-seq
  SOURCES test-forall-segment-seq.cpp)

if(RAJA_ENABLE_OPENMP)
  raja_add_test(
    NAME test-forall-segment-openmp
    SOURCES test-forall-segment-openmp.cpp)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  raja_add_test(
    NAME test-forall-segment-openmp-target
    SOURCES test-forall-segment-openmp-target.cpp)
endif()

if(RAJA_ENABLE_TBB)
  raja_add_test(
    NAME test-forall-segment-tbb
    SOURCES test-forall-segment-tbb.cpp)
endif()

if(RAJA_ENABLE_CUDA)
  raja_add_test(
    NAME test-forall-segment-cuda
    SOURCES test-forall-segment-cuda.cpp)
endif()

if(RAJA_ENABLE_HIP)
  raja_add_test(
    NAME test-forall-segment-hip
    SOURCES test-forall-segment-hip.cpp)
endif()

if(RAJA_ENABLE_SYCL)
  raja_add_test(
    NAME test-forall-segment-sycl
    SOURCES test-forall-segment-sycl.cpp)
endif()
