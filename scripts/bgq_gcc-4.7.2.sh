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

rm -rf build-bgq_gcc-4.7.2 2>/dev/null
mkdir build-bgq_gcc-4.7.2 && cd build-bgq_gcc-4.7.2
. /usr/local/tools/dotkit/init.sh && use cmake-3.4.3

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${RAJA_DIR}/host-configs/bgqos/gcc_4_7_2.cmake \
  -DCMAKE_INSTALL_PREFIX=../install-bgq_gcc-4.7.2 \
  "$@" \
  ${RAJA_DIR}
