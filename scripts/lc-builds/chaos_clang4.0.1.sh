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

rm -rf build_lc_chaos-clang-4.0.1 2>/dev/null
mkdir build_lc_chaos-clang-4.0.1 && cd build_lc_chaos-clang-4.0.1
. /usr/local/tools/dotkit/init.sh && use cmake-3.4.1

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${RAJA_DIR}/host-configs/lc-builds/chaos/clang_4_0_1.cmake \
  -DENABLE_OPENMP=On \
  -DCMAKE_INSTALL_PREFIX=${RAJA_DIR}/install_lc_chaos-clang-4.0.1 \
  "$@" \
  ${RAJA_DIR}
