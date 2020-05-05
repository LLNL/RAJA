###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-reducer-constructors-host
  SOURCES test-reducer-constructors-host.cpp)

raja_add_test(
  NAME test-reducer-reset-host
  SOURCES test-reducer-reset-host.cpp)

if(RAJA_ENABLE_CUDA)
raja_add_test(
  NAME test-reducer-constructors-cuda
  SOURCES test-reducer-constructors-cuda.cpp)

raja_add_test(
  NAME test-reducer-reset-cuda
  SOURCES test-reducer-reset-cuda.cpp)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
raja_add_test(
  NAME test-reducer-constructors-openmp-target
  SOURCES test-reducer-constructors-openmp-target.cpp)

raja_add_test(
  NAME test-reducer-reset-openmp-target
  SOURCES test-reducer-reset-openmp-target.cpp)
endif()
