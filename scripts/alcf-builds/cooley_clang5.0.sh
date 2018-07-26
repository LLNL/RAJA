#!/bin/bash

##
## Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-689114
##
## All rights reserved.
##
## This file is part of RAJA.
##
## For details about use and distribution, please read RAJA/LICENSE.
##

##
## Execute these commands before running this script to
## set up your build environment:
##
##  > soft add +cmake-3.9.1 
##  > soft add +clang-5.0
##

RAJA_DIR=$(git rev-parse --show-toplevel)

BUILD_SUFFIX=alcf-cooley-clang5.0

rm -rf ${RAJA_DIR}/build-${BUILD_SUFFIX} 2>/dev/null
mkdir ${RAJA_DIR}/build-${BUILD_SUFFIX} && cd ${RAJA_DIR}/build-${BUILD_SUFFIX}

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${RAJA_DIR}/host-configs/alcf-builds/cooley_clang5_0.cmake \
  -DENABLE_OPENMP=On \
  -DCMAKE_INSTALL_PREFIX=${RAJA_DIR}/install-${BUILD_SUFFIX} \
  "$@" \
  ${RAJA_DIR}
