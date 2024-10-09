#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [[ $# -lt 3 ]]; then
  echo
  echo "You must pass 3 arguments to the script (in this order): "
  echo "   1) compiler version number for clang"
  echo "   2) toolkit version number for cuda"
  echo "   3) CUDA compute architecture"
  echo
  echo "For example: "
  echo "    blueos_nvcc_clangcuda.sh 10.0.1 10.2.89 sm_70"
  exit
fi

COMP_CLANG_VER=$1
TOOLKIT_CUDA_VER=$2
CUDA_ARCH=$3
shift 3

BUILD_SUFFIX=lc_blueos-clangcuda${COMP_CLANG_VER}_cuda${TOOLKIT_CUDA_VER}-${CUDA_ARCH}

echo
echo "Creating build directory build_${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.23.1

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DRAJA_CXX_COMPILER_ROOT_DIR=/usr/tce/packages/clang/clang-${COMP_CLANG_VER} \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/clang/clang-${COMP_CLANG_VER}/bin/clang++ \
  -DCMAKE_C_COMPILER=/usr/tce/packages/clang/clang-${COMP_CLANG_VER}/bin/clang \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/tce/packages/cuda/cuda-${TOOLKIT_CUDA_VER} \
  -DBLT_CXX_STD=c++14 \
  -C ../host-configs/lc-builds/blueos/clangcuda_X.cmake \
  -DENABLE_OPENMP=Off \
  -DENABLE_CLANG_CUDA=On \
  -DBLT_CLANG_CUDA_ARCH=${CUDA_ARCH} \
  -DENABLE_CUDA=On \
  -DCUDA_ARCH=${CUDA_ARCH} \
  -DENABLE_BENCHMARKS=ON \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
