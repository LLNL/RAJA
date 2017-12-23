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

rm -rf build-clang-3.8.0 2>/dev/null
mkdir build-clang-3.8.0 && cd build-clang-3.8.0

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${RAJA_DIR}/host-configs/chaos/clang_3_8_0.cmake \
  -DCMAKE_INSTALL_PREFIX=../install-clang-3.8.0 \
  "$@" \
  ${RAJA_DIR}
