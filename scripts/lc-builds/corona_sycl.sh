#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

BUILD_SUFFIX=corona-sycl
: ${BUILD_TYPE:=RelWithDebInfo}
RAJA_HOSTCONFIG=../host-configs/lc-builds/toss4/corona_sycl.cmake

rm -rf build_${BUILD_SUFFIX}_${USER} >/dev/null
mkdir build_${BUILD_SUFFIX}_${USER} && cd build_${BUILD_SUFFIX}_${USER}

DATE=$(printf '%(%Y-%m-%d)T\n' -1)

## NOTE: RAJA tests are turned off due to compilation issues.

cmake \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -C ${RAJA_HOSTCONFIG} \
  -DENABLE_OPENMP=Off \
  -DENABLE_CUDA=Off \
  -DRAJA_ENABLE_TARGET_OPENMP=Off \
  -DENABLE_ALL_WARNINGS=Off \
  -DRAJA_ENABLE_SYCL=On \
  -DCMAKE_LINKER=clang++ \
  -DCMAKE_CXX_STANDARD=17 \
  -DENABLE_TESTS=Off \
  -DENABLE_EXAMPLES=On \
  "$@" \
  ..

