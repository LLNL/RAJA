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

HIP_CLANG_FLAGS="--offload-arch=${COMP_ARCH}"
HOSTCONFIG="hip_3_X"

if [[ ${COMP_VER} == 4.5.* ]]
then
  HIP_CLANG_FLAGS="${HIP_CLANG_FLAGS} -mllvm -amdgpu-fixed-function-abi=1"
  HOSTCONFIG="hip_4_5_link_X"
elif [[ ${COMP_VER} == 4.* ]]
then
  HOSTCONFIG="hip_4_link_X"
elif [[ ${COMP_VER} == 3.* ]]
then
  HOSTCONFIG="hip_3_X"
else
  echo "Unknown hip version, using ${HOSTCONFIG} host-config"
fi

BUILD_SUFFIX=lc_toss3-hipcc-${COMP_VER}-${COMP_ARCH}

echo
echo "Creating build directory ${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}


module load cmake/3.14.5

# unload rocm to avoid configuration problems where the loaded rocm and COMP_VER
# are inconsistent causing the rocprim from the module to be used unexpectedly
module unload rocm


cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DROCM_ROOT_DIR="/opt/rocm-${COMP_VER}" \
  -DHIP_ROOT_DIR="/opt/rocm-${COMP_VER}/hip" \
  -DHIP_CLANG_PATH=/opt/rocm-${COMP_VER}/llvm/bin \
  -DCMAKE_C_COMPILER=/opt/rocm-${COMP_VER}/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm-${COMP_VER}/llvm/bin/clang++ \
  -DHIP_CLANG_FLAGS="${HIP_CLANG_FLAGS}" \
  -C "../host-configs/lc-builds/toss3/${HOSTCONFIG}.cmake" \
  -DENABLE_HIP=ON \
  -DENABLE_OPENMP=ON \
  -DENABLE_CUDA=OFF \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..

echo
echo "***********************************************************************"
echo
echo "cd into directory build_${BUILD_SUFFIX} and run make to build RAJA"
echo
echo "  Please note that you have to have a consistent build environment"
echo "  when you make RAJA as cmake may reconfigure; unload the rocm module"
echo "  or load the appropriate rocm module (${COMP_VER}) when building."
echo
echo "    module unload rocm"
echo "    srun -n1 make"
echo
echo "***********************************************************************"
