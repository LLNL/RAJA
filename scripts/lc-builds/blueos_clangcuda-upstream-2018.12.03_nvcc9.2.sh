#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

BUILD_SUFFIX=lc_blueos-clangcuda-upstream-2018.12.03_nvcc-9.2

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.14.5

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ../host-configs/lc-builds/blueos/clangcuda_upstream_2018_12_03_nvcc_9_2.cmake \
  -DENABLE_OPENMP=Off \
  -DENABLE_CLANG_CUDA=On \
  -DBLT_CLANG_CUDA_ARCH=sm_60 \
  -DENABLE_CUDA=On \
  -DCUDA_ARCH=sm_60 \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/tce/packages/cuda/cuda-9.2.148 \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
