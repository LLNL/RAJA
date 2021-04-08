###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
################################################################################

if(ENABLE_CLANG_CUDA)
  add_subdirectory(clangcuda)
endif(ENABLE_CLANG_CUDA)

if(ENABLE_TARGET_OPENMP)
add_subdirectory(openmp-target)
endif(ENABLE_TARGET_OPENMP)

if(ENABLE_HIP)
  add_subdirectory(hip)
endif(ENABLE_HIP)
