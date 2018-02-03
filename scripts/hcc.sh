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

rm -rf build-hcc-release 2>/dev/null
mkdir build-hcc-release && cd build-hcc-release

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${RAJA_DIR}/host-configs/linux/hcc.cmake \
  -DCMAKE_INSTALL_PREFIX=../install-hcc-release \
  "$@" \
  ${RAJA_DIR}
