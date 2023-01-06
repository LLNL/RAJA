###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
################################################################################

if(RAJA_ENABLE_CLANG_CUDA)
  add_subdirectory(clangcuda)
endif(RAJA_ENABLE_CLANG_CUDA)

add_subdirectory(openmp-target)
