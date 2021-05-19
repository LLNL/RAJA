#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [[ $# -ne 3 ]]; then
  echo
  echo "You must pass 3 arguments to the script (in this order): "
  echo "   1) compiler version number for nvcc"  
  echo "   2) CUDA compute architecture"
  echo "   3) compiler version number for clang. "
  echo 
  echo "For example: "
  echo "    blueos_nvcc_clang.sh 10.2.89 sm_70 10.0.1"
  exit
fi

COMP_NVCC_VER=$1
COMP_ARCH=$2
COMP_CLANG_VER=$3

BUILD_SUFFIX=lc_blueos-nvcc${COMP_NVCC_VER}-${COMP_ARCH}-clang${COMP_CLANG_VER}

echo
echo "Creating build directory ${BUILD_SUFFIX} and generating configuration in it"
echo

rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.14.5

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/clang/clang-${COMP_CLANG_VER}/bin/clang++ \
  -DBLT_CXX_STD=c++11 \
  -C ../host-configs/lc-builds/blueos/nvcc_clang_X.cmake \
  -DENABLE_OPENMP=On \
  -DENABLE_CUDA=On \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/tce/packages/cuda/cuda-${COMP_NVCC_VER} \
  -DCMAKE_CUDA_COMPILER=/usr/tce/packages/cuda/cuda-${COMP_NVCC_VER}/bin/nvcc \
  -DCUDA_ARCH=${COMP_ARCH} \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
