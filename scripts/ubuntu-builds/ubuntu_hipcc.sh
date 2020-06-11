#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

BUILD_SUFFIX=ubuntu-hipcc

rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

#============= For ubuntu ONLY =============
export PATH=/opt/rocm-3.5.0/hip/bin:$PATH
export HIP_CLANG_PATH=/opt/rocm-3.5.0/llvm/bin
export DEVICE_LIB_PATH=/opt/rocm-3.5.0/lib
export HCC_AMDGPU_TARGET=gfx900
#==============================================

cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  -C ../host-configs/lc-builds/toss3/hip.cmake \
  ..
