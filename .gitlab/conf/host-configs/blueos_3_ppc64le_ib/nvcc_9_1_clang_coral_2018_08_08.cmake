###############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(CMAKE_CXX_COMPILER "/usr/tce/packages/clang/clang-coral-2018.08.08/bin/clang++" CACHE PATH "")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/tce/packages/cuda/cuda-9.1.85" CACHE PATH "")
set(CMAKE_CUDA_COMPILER "/usr/tce/packages/cuda/cuda-9.1.85/bin/nvcc" CACHE PATH "")
