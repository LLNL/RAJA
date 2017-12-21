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

rm -rf build-gcc-release 2>/dev/null
mkdir build-gcc-release && cd build-gcc-release

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${RAJA_DIR}/host-configs/chaos/gcc.cmake \
  -DCMAKE_INSTALL_PREFIX=../install-gcc-release \
  "$@" \
  ${RAJA_DIR}
