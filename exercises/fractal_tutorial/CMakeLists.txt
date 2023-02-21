###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if(RAJA_ENABLE_CUDA)
raja_add_executable(
  NAME fractal-CUDA 
  SOURCES fractal-CUDA.cpp
  DEPENDS_ON cuda)
endif()

if(RAJA_ENABLE_OPENMP)
raja_add_executable(
  NAME fractal-OpenMP 
  SOURCES fractal-OpenMP.cpp
  DEPENDS_ON openmp)
endif()
