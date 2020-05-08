###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-forall-atomic-view-seq
  SOURCES test-forall-atomic-view-seq.cpp)

if(RAJA_ENABLE_OPENMP)
  raja_add_test(
    NAME test-forall-atomic-view-openmp
    SOURCES test-forall-atomic-view-openmp.cpp)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  raja_add_test(
    NAME test-forall-atomic-view-openmp-target
    SOURCES test-forall-atomic-view-openmp-target.cpp)
endif()

if(RAJA_ENABLE_TBB)
  # reserved for future compatability
endif()

if(RAJA_ENABLE_CUDA)
  raja_add_test(
    NAME test-forall-atomic-view-cuda
    SOURCES test-forall-atomic-view-cuda.cpp)
endif()

if(RAJA_ENABLE_HIP)
  raja_add_test(
    NAME test-forall-atomic-view-hip
    SOURCES test-forall-atomic-view-cuda.cpp)
endif()
