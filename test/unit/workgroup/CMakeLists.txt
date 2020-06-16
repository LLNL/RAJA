###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-workgroup-constructors-seq
  SOURCES test-workgroup-constructors-seq.cpp)

if(RAJA_ENABLE_TBB)
raja_add_test(
  NAME test-workgroup-constructors-tbb
  SOURCES test-workgroup-constructors-tbb.cpp)
endif()

if(RAJA_ENABLE_OPENMP)
raja_add_test(
  NAME test-workgroup-constructors-openmp
  SOURCES test-workgroup-constructors-openmp.cpp)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
raja_add_test(
  NAME test-workgroup-constructors-openmp-target
  SOURCES test-workgroup-constructors-openmp-target.cpp)
endif()

if(RAJA_ENABLE_CUDA)
raja_add_test(
  NAME test-workgroup-constructors-cuda
  SOURCES test-workgroup-constructors-cuda.cpp)
endif()

if(RAJA_ENABLE_HIP)
raja_add_test(
  NAME test-workgroup-constructors-hip
  SOURCES test-workgroup-constructors-hip.cpp)
endif()


raja_add_test(
  NAME test-workgroup-vtable-seq
  SOURCES test-workgroup-vtable-seq.cpp)

if(RAJA_ENABLE_TBB)
raja_add_test(
  NAME test-workgroup-vtable-tbb
  SOURCES test-workgroup-vtable-tbb.cpp)
endif()

if(RAJA_ENABLE_OPENMP)
raja_add_test(
  NAME test-workgroup-vtable-openmp
  SOURCES test-workgroup-vtable-openmp.cpp)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
raja_add_test(
  NAME test-workgroup-vtable-openmp-target
  SOURCES test-workgroup-vtable-openmp-target.cpp)
endif()

if(RAJA_ENABLE_CUDA)
raja_add_test(
  NAME test-workgroup-vtable-cuda
  SOURCES test-workgroup-vtable-cuda.cpp)
endif()

if(RAJA_ENABLE_HIP AND DEFINED RAJA_ENABLE_HIP_INDIRECT_FUNCTION_CALL)
raja_add_test(
  NAME test-workgroup-vtable-hip
  SOURCES test-workgroup-vtable-hip.cpp)
endif()


raja_add_test(
  NAME test-workgroup-workstorage
  SOURCES test-workgroup-workstorage.cpp)
