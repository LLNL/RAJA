###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
################################################################################

if(ENABLE_CLANG_CUDA)
  add_subdirectory(clangcuda)
endif(ENABLE_CLANG_CUDA)

if(ENABLE_CUDA)
  add_subdirectory(nvcc)
endif(ENABLE_CUDA)

add_subdirectory(openmp-target)
