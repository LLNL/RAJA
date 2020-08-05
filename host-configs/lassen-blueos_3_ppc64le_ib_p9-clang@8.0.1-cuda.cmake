###################
# Generated host-config - Edit at own risk!
###################
# Copyright (c) 2020, Lawrence Livermore National Security, LLC and
# other Umpire Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 
###################

#------------------
# SYS_TYPE: blueos_3_ppc64le_ib_p9
# Compiler Spec: clang@8.0.1
# CMake executable path: /usr/tce/packages/cmake/cmake-3.14.5/bin/cmake
#------------------

#------------------
# Compilers
#------------------

set(CMAKE_C_COMPILER "/usr/tce/packages/clang/clang-8.0.1/bin/clang" CACHE PATH "")

set(CMAKE_CXX_COMPILER "/usr/tce/packages/clang/clang-8.0.1/bin/clang++" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3" CACHE STRING "")

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g" CACHE STRING "")

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "")

#------------------------------------------------------------------------------
# Cuda
#------------------------------------------------------------------------------

set(ENABLE_CUDA ON CACHE BOOL "")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/tce/packages/cuda/cuda-10.1.243" CACHE PATH "")

set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" CACHE PATH "")

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -Xcompiler -O3 -Xcompiler -fopenmp" CACHE STRING "")

set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O3 -g -lineinfo -Xcompiler -O3 -Xcompiler -fopenmp" CACHE STRING "")

set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g -G" CACHE STRING "")

#------------------------------------------------------------------------------
# Other
#------------------------------------------------------------------------------

set(RAJA_RANGE_ALIGN "4" CACHE STRING "")

set(RAJA_RANGE_MIN_LENGTH "32" CACHE STRING "")

set(RAJA_DATA_ALIGN "64" CACHE STRING "")

set(RAJA_HOST_CONFIG_LOADED ON CACHE BOOL "")

