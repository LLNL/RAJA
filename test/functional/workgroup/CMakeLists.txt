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


raja_add_test(
  NAME test-workgroup-unordered-seq
  SOURCES test-workgroup-unordered-seq.cpp)

if(RAJA_ENABLE_TBB)
raja_add_test(
  NAME test-workgroup-unordered-tbb
  SOURCES test-workgroup-unordered-tbb.cpp)
endif()

if(RAJA_ENABLE_OPENMP)
raja_add_test(
  NAME test-workgroup-unordered-openmp
  SOURCES test-workgroup-unordered-openmp.cpp)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
raja_add_test(
  NAME test-workgroup-unordered-openmp-target
  SOURCES test-workgroup-unordered-openmp-target.cpp)
endif()

if(RAJA_ENABLE_CUDA)
raja_add_test(
  NAME test-workgroup-unordered-cuda
  SOURCES test-workgroup-unordered-cuda.cpp)
endif()

if(RAJA_ENABLE_HIP)
raja_add_test(
  NAME test-workgroup-unordered-hip
  SOURCES test-workgroup-unordered-hip.cpp)
endif()
