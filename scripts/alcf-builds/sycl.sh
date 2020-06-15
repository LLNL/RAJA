#!/usr/bin/env bash

##
## Copyright (c) 2017-19, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-738930
##
## All rights reserved.
##
## This file is part of the RAJA Performance Suite.
##
## For details about use and distribution, please read RAJAPerf/LICENSE.
##

BUILD_SUFFIX=sycl
: ${BUILD_TYPE:=RelWithDebInfo}
RAJA_HOSTCONFIG=../host-configs/alcf-builds/sycl.cmake

rm -rf build_${BUILD_SUFFIX}_${USER} >/dev/null
mkdir build_${BUILD_SUFFIX}_${USER} && cd build_${BUILD_SUFFIX}_${USER}

## NOTE: RAJA tests are turned off due to compilation issues.

cmake \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -C ${RAJA_HOSTCONFIG} \
  -DENABLE_OPENMP=Off \
  -DENABLE_CUDA=Off \
  -DENABLE_TARGET_OPENMP=Off \
  -DENABLE_ALL_WARNINGS=Off \
  -DENABLE_NO_LIBS=On \
  -DENABLE_SYCL=On \
  -DCMAKE_LINKER=clang++ \
  -DENABLE_TESTS=On \
  -DENABLE_EXAMPLES=On \
  "$@" \
  ..

make -j 2
