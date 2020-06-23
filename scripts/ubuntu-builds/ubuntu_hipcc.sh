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

ROCM_DIR=/opt/rocm-3.5.1

#============= For ubuntu ONLY =============
export PATH=${ROCM_DIR}/hip/bin:$PATH
export HIP_CLANG_PATH=${ROCM_DIR}/llvm/bin
export DEVICE_LIB_PATH=${ROCM_DIR}/lib
export HCC_AMDGPU_TARGET=gfx900
#==============================================

cmake \
  -DROCM_DIR=${ROCM_DIR} \
  -DCMAKE_BUILD_TYPE=Debug \
  -C ../host-configs/ubuntu-builds/hip.cmake \
  ..
