#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [[ $# -lt 2 ]]; then
  echo
  echo "You must pass 2 or more arguments to the script (in this order): "
  echo "   1) compiler version number"
  echo "   2) HIP compute architecture"
  echo "   3...) optional arguments to cmake"
  echo
  echo "For example: "
  echo "    toss3_hipcc.sh 4.1.0 gfx906"
  echo "    toss3_hipcc.sh 4.1.0 gfx906 -DBLT_CXX_STD=c++11"
  exit
fi

COMP_VER=$1
COMP_ARCH=$2
shift 2

BUILD_SUFFIX=lc_toss3-hipcc-${COMP_VER}-${COMP_ARCH}

echo
echo "Creating build directory ${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}


module load cmake/3.14.5


cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DROCM_ROOT_DIR="/opt/rocm-${COMP_VER}" \
  -DHIP_ROOT_DIR="/opt/rocm-${COMP_VER}/hip" \
  -DHIP_CLANG_PATH=/opt/rocm-${COMP_VER}/llvm/bin \
  -DCMAKE_C_COMPILER=/opt/rocm-${COMP_VER}/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm-${COMP_VER}/llvm/bin/clang++ \
  -DHIP_HIPCC_FLAGS=--offload-arch=${COMP_ARCH} \
  -C ../host-configs/lc-builds/toss3/hip_link_X.cmake \
  -DENABLE_HIP=ON \
  -DENABLE_OPENMP=OFF \
  -DENABLE_CUDA=OFF \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
