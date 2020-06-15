###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-forall-indexset-view-seq
  SOURCES test-forall-indexset-view-seq.cpp)

if(RAJA_ENABLE_OPENMP)
  raja_add_test(
    NAME test-forall-indexset-view-openmp
    SOURCES test-forall-indexset-view-openmp.cpp)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  raja_add_test(
    NAME test-forall-indexset-view-openmp-target
    SOURCES test-forall-indexset-view-openmp-target.cpp)
endif()

if(RAJA_ENABLE_TBB)
  raja_add_test(
    NAME test-forall-indexset-view-tbb
    SOURCES test-forall-indexset-view-tbb.cpp)
endif()

if(RAJA_ENABLE_CUDA)
  raja_add_test(
    NAME test-forall-indexset-view-cuda
    SOURCES test-forall-indexset-view-cuda.cpp)
endif()

if(RAJA_ENABLE_HIP)
  raja_add_test(
    NAME test-forall-indexset-view-hip
    SOURCES test-forall-indexset-view-hip.cpp)
endif()

if(RAJA_ENABLE_SYCL)
  raja_add_test(
    NAME test-forall-indexset-view-sycl
    SOURCES test-forall-indexset-view-sycl.cpp)
endif()
