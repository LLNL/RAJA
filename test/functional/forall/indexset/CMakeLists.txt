###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-forall-indexset-seq
  SOURCES test-forall-indexset-seq.cpp)

if(RAJA_ENABLE_OPENMP)
  raja_add_test(
    NAME test-forall-indexset-openmp
    SOURCES test-forall-indexset-openmp.cpp)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  raja_add_test(
    NAME test-forall-indexset-openmp-target
    SOURCES test-forall-indexset-openmp-target.cpp)
endif()

if(RAJA_ENABLE_TBB)
  raja_add_test(
    NAME test-forall-indexset-tbb
    SOURCES test-forall-indexset-tbb.cpp)
endif()

if(RAJA_ENABLE_CUDA)
  raja_add_test(
    NAME test-forall-indexset-cuda
    SOURCES test-forall-indexset-cuda.cpp)
endif()

if(RAJA_ENABLE_HIP)
  raja_add_test(
    NAME test-forall-indexset-hip
    SOURCES test-forall-indexset-hip.cpp)
endif()

if(RAJA_ENABLE_SYCL)
  raja_add_test(
    NAME test-forall-indexset-sycl
    SOURCES test-forall-indexset-sycl.cpp)
endif()
