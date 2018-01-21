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

rm -rf build-icpc-16.0.258 2>/dev/null
mkdir build-icpc-16.0.258 && cd build-icpc-16.0.258
. /usr/local/tools/dotkit/init.sh && use gcc-4.9.3p

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${RAJA_DIR}/host-configs/chaos/icpc_16_0_258.cmake \
  -DCMAKE_INSTALL_PREFIX=../install-icpc-16.0.258 \
  "$@" \
  ${RAJA_DIR}
