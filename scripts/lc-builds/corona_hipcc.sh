#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

BUILD_SUFFIX=lc_corona-hipcc

rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

#============= For LC Corona ONLY =============
export PATH=/usr/workspace/wsb/raja-dev/opt/hip-clang/bin:$PATH
export HIP_CLANG_PATH=/usr/workspace/wsb/raja-dev/opt/llvm/bin
export DEVICE_LIB_PATH=/usr/workspace/wsb/raja-dev/opt/lib
export HCC_AMDGPU_TARGET=gfx900
module load opt
module load dts/7.1
module load rocm
#==============================================

module load cmake/3.14.5

cmake \
  -C ../host-configs/hip.cmake \
  ..
