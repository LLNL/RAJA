###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-forall-atomic-basic-seq
  SOURCES test-forall-atomic-basic-seq.cpp)

if(RAJA_ENABLE_OPENMP)
  raja_add_test(
    NAME test-forall-atomic-basic-openmp
    SOURCES test-forall-atomic-basic-openmp.cpp)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  raja_add_test(
    NAME test-forall-atomic-basic-openmp-target
    SOURCES test-forall-atomic-basic-openmp-target.cpp)
endif()

if(RAJA_ENABLE_TBB)
  # reserved for future compatability
endif()

if(RAJA_ENABLE_CUDA)
  raja_add_test(
    NAME test-forall-atomic-basic-cuda
    SOURCES test-forall-atomic-basic-cuda.cpp)
endif()

if(RAJA_ENABLE_HIP)
  raja_add_test(
    NAME test-forall-atomic-basic-hip
    SOURCES test-forall-atomic-basic-cuda.cpp)
endif()
