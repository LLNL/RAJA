#!/bin/bash

##
## Copyright (c) 2016, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-689114
##
## All rights reserved.
##
## For additional details and restrictions, please see RAJA/LICENSE.txt
##

BUILD_DIR='build-hcc-release'

rm -rf ${BUILD_DIR} 2>/dev/null
mkdir ${BUILD_DIR} && cd ${BUILD_DIR}


RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_ROCM=ON -DENABLE_OPENMP=OFF -DBLT_SOURCE_DIR=${BLT_DIR} \
  -DROCM_ARCH=gfx900 \
  -C ${RAJA_DIR}/host-configs/linux/rocm.cmake \
  -DCMAKE_INSTALL_PREFIX=../install-hcc-release \
  "$@" \
  ${RAJA_DIR}
