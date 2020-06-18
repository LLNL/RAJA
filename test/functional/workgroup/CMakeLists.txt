###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-workgroup-ordered-seq
  SOURCES test-workgroup-ordered-seq.cpp)

if(RAJA_ENABLE_TBB)
raja_add_test(
  NAME test-workgroup-ordered-tbb
  SOURCES test-workgroup-ordered-tbb.cpp)
endif()

if(RAJA_ENABLE_OPENMP)
raja_add_test(
  NAME test-workgroup-ordered-openmp
  SOURCES test-workgroup-ordered-openmp.cpp)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
raja_add_test(
  NAME test-workgroup-ordered-openmp-target
  SOURCES test-workgroup-ordered-openmp-target.cpp)
endif()

if(RAJA_ENABLE_CUDA)
raja_add_test(
  NAME test-workgroup-ordered-cuda
  SOURCES test-workgroup-ordered-cuda.cpp)
endif()

if(RAJA_ENABLE_HIP)
raja_add_test(
  NAME test-workgroup-ordered-hip
  SOURCES test-workgroup-ordered-hip.cpp)
endif()
