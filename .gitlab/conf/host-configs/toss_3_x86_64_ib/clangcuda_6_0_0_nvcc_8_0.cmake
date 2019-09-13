###############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(CMAKE_CXX_COMPILER "/usr/tce/packages/clang/clang-6.0.0/bin/clang++" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/tce/packages/clang/clang-6.0.0/bin/clang" CACHE PATH "")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/tce/packages/cuda/cuda-8.0" CACHE PATH "")
set(CMAKE_EXE_LINKER_FLAGS "-L/usr/tce/packages/cuda/cuda-8.0/lib64 -lcudart_static -lcudadevrt -lrt -ldl -lnvToolsExt -pthread -Wl,-rpath=/usr/tce/packages/clang/clang-6.0.0/lib" CACHE PATH "")
