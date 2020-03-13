###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-forall-seq
  SOURCES test-forall-seq.cpp)

if(RAJA_ENABLE_OPENMP)
  raja_add_test(
    NAME test-forall-openmp
    SOURCES test-forall-openmp.cpp)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  raja_add_test(
    NAME test-forall-target-openmp
    SOURCES test-forall-target-openmp.cpp)
endif()

if(RAJA_ENABLE_TBB)
  raja_add_test(
    NAME test-forall-tbb
    SOURCES test-forall-tbb.cpp)
endif()

if(RAJA_ENABLE_CUDA)
  raja_add_test(
    NAME test-forall-cuda
    SOURCES test-forall-cuda.cpp)
endif()

if(RAJA_ENABLE_HIP)
  raja_add_test(
    NAME test-forall-hip
    SOURCES test-forall-hip.cpp)
endif()
