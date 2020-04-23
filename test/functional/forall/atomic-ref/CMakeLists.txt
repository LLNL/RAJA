###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-forall-atomic-ref-math-seq
  SOURCES test-forall-atomic-ref-math-seq.cpp)

raja_add_test(
  NAME test-forall-atomic-ref-other-seq
  SOURCES test-forall-atomic-ref-other-seq.cpp)

if(RAJA_ENABLE_OPENMP)
  raja_add_test(
    NAME test-forall-atomic-ref-math-openmp
    SOURCES test-forall-atomic-ref-math-openmp.cpp)

  raja_add_test(
    NAME test-forall-atomic-ref-other-openmp
    SOURCES test-forall-atomic-ref-other-openmp.cpp)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  raja_add_test(
    NAME test-forall-atomic-ref-math-openmp-target
    SOURCES test-forall-atomic-ref-math-openmp-target.cpp)

  raja_add_test(
    NAME test-forall-atomic-ref-other-openmp-target
    SOURCES test-forall-atomic-ref-other-openmp-target.cpp)
endif()

if(RAJA_ENABLE_TBB)
  # reserved for future compatability
endif()

if(RAJA_ENABLE_CUDA)
  raja_add_test(
    NAME test-forall-atomic-ref-math-cuda
    SOURCES test-forall-atomic-ref-math-cuda.cpp)

  raja_add_test(
    NAME test-forall-atomic-ref-other-cuda
    SOURCES test-forall-atomic-ref-other-cuda.cpp)
endif()

if(RAJA_ENABLE_HIP)
  raja_add_test(
    NAME test-forall-atomic-ref-math-hip
    SOURCES test-forall-atomic-ref-math-hip.cpp)

  raja_add_test(
    NAME test-forall-atomic-ref-other-hip
    SOURCES test-forall-atomic-ref-other-hip.cpp)
endif()
