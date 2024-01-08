###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-reducer-constructors-seq
  SOURCES test-reducer-constructors-seq.cpp)

raja_add_test(
  NAME test-reducer-reset-seq
  SOURCES test-reducer-reset-seq.cpp)

if(RAJA_ENABLE_OPENMP)
raja_add_test(
  NAME test-reducer-constructors-openmp
  SOURCES test-reducer-constructors-openmp.cpp)

raja_add_test(
  NAME test-reducer-reset-openmp
  SOURCES test-reducer-reset-openmp.cpp)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
raja_add_test(
  NAME test-reducer-constructors-openmp-target
  SOURCES test-reducer-constructors-openmp-target.cpp)

raja_add_test(
  NAME test-reducer-reset-openmp-target
  SOURCES test-reducer-reset-openmp-target.cpp)
endif()

if(RAJA_ENABLE_CUDA)
raja_add_test(
  NAME test-reducer-constructors-cuda
  SOURCES test-reducer-constructors-cuda.cpp)

raja_add_test(
  NAME test-reducer-reset-cuda
  SOURCES test-reducer-reset-cuda.cpp)
endif()

if(RAJA_ENABLE_HIP)
raja_add_test(
  NAME test-reducer-constructors-hip
  SOURCES test-reducer-constructors-hip.cpp)

raja_add_test(
  NAME test-reducer-reset-hip
  SOURCES test-reducer-reset-hip.cpp)
endif()
